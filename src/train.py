# src/train.py
"""Training pipeline for the RoBERTa-BiLSTM hybrid architecture."""

import os
import logging

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup

from src.model import RoBERTaBiLSTM
from src.utils import seed_everything, load_dataset, NewsDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/val.csv",
    checkpoint_dir: str = "models/final_hybrid_roberta_bilstm",
    epochs: int = 4,
    batch_size: int = 16,
    max_length: int = 128,
    seed: int = 42,
):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
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

    model = RoBERTaBiLSTM(model_name="roberta-base")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, use_amp)
        val_loss, val_acc = evaluate_val(model, val_loader, device)

        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(checkpoint_dir)
            logger.info(f"New best checkpoint saved (val loss: {val_loss:.4f})")


if __name__ == "__main__":
    train()
