"""Training pipeline for the RoBERTa-BiLSTM hybrid architecture."""

import os
import logging
from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup

from src.model import RoBERTaBiLSTM
from src.utils import seed_everything, load_dataset, NewsDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configs/config.yaml. Previously this file existed but nothing
    read it — every hyperparameter in train() was hardcoded separately,
    so editing the YAML had no effect. This makes the YAML the actual
    source of truth, with train()'s arguments only as a fallback when no
    config file is present."""
    if not os.path.exists(config_path):
        logger.warning(f"No config file found at {config_path}; using train() defaults.")
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def train_epoch(model, dataloader, optimizer, scheduler, device, use_amp: bool):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # torch.autocast is safe on both CPU and CUDA; the previous version
        # hardcoded torch.amp.autocast('cuda'), which raises on a CPU-only
        # machine regardless of the use_amp intent.
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_val(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def train(
    config_path: str = "configs/config.yaml",
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    seed: Optional[int] = None,
):
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    # Function arguments override the YAML when explicitly passed; anything
    # left as None falls back to config.yaml, then to a hardcoded default.
    train_path = train_path or os.path.join(data_cfg.get("processed_dir", "data/processed"), "train.csv")
    val_path = val_path or os.path.join(data_cfg.get("processed_dir", "data/processed"), "val.csv")
    checkpoint_dir = checkpoint_dir or train_cfg.get("final_model_dir", "models/final_hybrid_roberta_bilstm")
    epochs = epochs if epochs is not None else train_cfg.get("epochs", 4)
    batch_size = batch_size if batch_size is not None else train_cfg.get("batch_size", 16)
    max_length = max_length if max_length is not None else data_cfg.get("max_length", 128)
    seed = seed if seed is not None else 42
    lr = train_cfg.get("lr", 2e-5)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    clip_grad_norm = train_cfg.get("clip_grad_norm", 1.0)
    early_stopping_patience = train_cfg.get("early_stopping_patience", 3)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and train_cfg.get("mixed_precision", True)
    logger.info(f"Using device: {device} (mixed precision: {use_amp})")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # load_dataset validates that 'content'/'label' columns actually exist
    # (the schema data_prep.preprocess_and_split writes) and raises a clear
    # error naming the missing column otherwise. This replaces a previous
    # hack that blindly force-renamed columns by position
    # (train_df.columns = ['text'] + ... + ['label']) to work around a
    # schema mismatch instead of fixing it.
    train_df = load_dataset(train_path)
    val_df = load_dataset(val_path)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    train_loader = DataLoader(
        NewsDataset(train_df, tokenizer, max_length), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        NewsDataset(val_df, tokenizer, max_length), batch_size=batch_size, shuffle=False
    )

    model = RoBERTaBiLSTM(
        model_name=model_cfg.get("model_name", "roberta-base"),
        num_labels=model_cfg.get("num_labels", 2),
        lstm_hidden_size=model_cfg.get("lstm_hidden_size", 256),
        lstm_layers=model_cfg.get("lstm_layers", 1),
        dropout=model_cfg.get("dropout", 0.3),
        use_attention=model_cfg.get("use_attention", True),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(warmup_ratio * total_steps), num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, use_amp)
        val_loss, val_acc = evaluate_val(model, val_loader, device)

        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            model.save_pretrained(checkpoint_dir)
            logger.info(f"New best checkpoint saved (val loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            logger.info(
                f"No improvement for {epochs_without_improvement} epoch(s) "
                f"(patience: {early_stopping_patience})"
            )
            # early_stopping_patience was declared in config.yaml but never
            # actually checked anywhere in the training loop; this wires it
            # in so long training runs stop once val loss plateaus.
            if epochs_without_improvement >= early_stopping_patience:
                logger.info("Early stopping triggered.")
                break


if __name__ == "__main__":
    train()
