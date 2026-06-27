
"""Production-grade training pipeline for RoBERTa-BiLSTM hybrid architecture with robust column names."""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from src.model import RoBERTaBiLSTM
from tqdm import tqdm

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training Batches"):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
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
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return total_loss / len(dataloader), correct / total

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    train_path = "data/processed/train.csv"
    val_path = "data/processed/val.csv"
    checkpoint_dir = "models/final_hybrid_roberta_bilstm"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # ULTIMATE FIX: DataFrame ke columns ko manually overwrite kar rahe hain takki KeyError impossible ho jaye
    # Pehla column text banega, aakhri column label banega
    train_df.columns = ['text'] + list(train_df.columns[1:-1]) + ['label']
    val_df.columns = ['text'] + list(val_df.columns[1:-1]) + ['label']
    
    logger.info(f"Forced Column Normalization -> Features assigned to 'text' and 'label'")
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    def get_loader(df, shuffle=True):
        encodings = tokenizer(
            df['text'].astype(str).tolist(), 
            truncation=True, 
            padding=True, 
            max_length=128, 
            return_tensors="pt"
        )
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(df['label'].astype(int).tolist()))
        return DataLoader(dataset, batch_size=16, shuffle=shuffle)
        
    train_loader = get_loader(train_df, shuffle=True)
    val_loader = get_loader(val_df, shuffle=False)
    
    model = RoBERTaBiLSTM(model_name="roberta-base")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    epochs = 4
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc = evaluate_val(model, val_loader, device)
        
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(checkpoint_dir)
            logger.info(f"New Best Model Checkpoint Saved with Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    train()