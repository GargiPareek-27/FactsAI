# src/evaluate.py
"""Evaluation module for fake news detection model."""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import RoBERTaBiLSTM
from src.data_prep import preprocess_and_split
from src.utils import seed_everything, load_dataset, NewsDataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model_and_tokenizer(
    model_dir: str,
    tokenizer_name: str = "roberta-base",
) -> tuple:
    """Load model and tokenizer."""
    model = RoBERTaBiLSTM.from_pretrained(model_dir)
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
    return model, tokenizer


def evaluate_model(
    model_dir: str,
    test_csv: str,
    tokenizer_name: str = "roberta-base",
    max_length: int = 512,
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    model, tokenizer = load_model_and_tokenizer(model_dir, tokenizer_name)
    model.to(device)
    model.eval()

    test_df = pd.read_csv(test_csv)
    test_dataset = NewsDataset(test_df, tokenizer, max_length)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_labels = np.array(all_labels).astype(int)
    all_preds = np.array(all_preds).astype(int)
    all_probs = np.array(all_probs)

    # Binary metrics
    acc = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, average="binary")
    r = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    auc = roc_auc_score(all_labels, all_probs[:, 1])

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Save basic metrics dict
    metrics = {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
    }

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
    plt.close(fig)

    # Print metrics
    print("Evaluation results:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {p:.4f}")
    print(f"Recall    : {r:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"AUC       : {auc:.4f}")

    return metrics