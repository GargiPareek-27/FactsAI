# src/evaluate.py

import argparse
import json
import os
import subprocess
import sys

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import RoBERTaBiLSTM
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
    audit_report_path: str = "data/processed/logs/audit_report.json",
    checkpoint_trained_post_dedup_fix: bool = None,
) -> dict:
    """Evaluate a trained checkpoint on the untouched test set and write
    real, reproducible artifacts to results/.

    checkpoint_trained_post_dedup_fix must be set explicitly by the
    caller (True/False). This function does NOT guess whether the
    checkpoint being loaded was trained on data produced by the current
    (deduplicated) src/data_prep.py, because that can't be inferred from
    the checkpoint file itself. Leaving it as None marks METRIC VALIDITY
    as REQUIRES RETRAINING in the generated summary, on purpose -- an
    unverified assumption of validity is exactly the kind of thing this
    audit process exists to prevent.
    """
    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(model_dir, tokenizer_name)
    model.to(device)
    model.eval()

    # load_dataset validates that 'content'/'label' columns exist rather
    # than assuming the CSV schema — see src/utils.py.
    test_df = load_dataset(test_csv)
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

    # Binary metrics (fake=1 as the positive class)
    acc = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, average="binary")
    r = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    auc = roc_auc_score(all_labels, all_probs[:, 1])

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Full classification report (per-class + macro/weighted averages)
    class_report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=["Real", "Fake"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "n_test_samples": int(len(all_labels)),
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1_binary": float(f1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }

    # Save metrics.json (numbers only — no raw arrays, keeps the file
    # small and diffable in git history)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save full classification report separately
    with open(os.path.join(RESULTS_DIR, "classification_report.json"), "w") as f:
        json.dump(class_report_dict, f, indent=2)

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
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {p:.4f}")
    print(f"Recall      : {r:.4f}")
    print(f"F1 (binary) : {f1:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print(f"AUC         : {auc:.4f}")

    write_evaluation_summary(
        metrics=metrics,
        audit_report_path=audit_report_path,
        checkpoint_trained_post_dedup_fix=checkpoint_trained_post_dedup_fix,
        model_dir=model_dir,
        test_csv=test_csv,
    )

    metrics["labels"] = all_labels
    metrics["preds"] = all_preds
    metrics["probs"] = all_probs
    return metrics


def write_evaluation_summary(
    metrics: dict,
    audit_report_path: str,
    checkpoint_trained_post_dedup_fix: bool,
    model_dir: str,
    test_csv: str,
) -> None:
    """Write results/evaluation_summary.md honestly from actual state --
    never a hardcoded PASS/VALID. Data integrity status is read from a
    real audit_data_integrity.py JSON report if one exists; otherwise
    that section is marked WARNING rather than assumed clean.
    """
    data_integrity_status = "WARNING"
    data_integrity_reason = (
        f"No audit report found at {audit_report_path}. Run "
        f"`python scripts/audit_data_integrity.py --json-out {audit_report_path}` "
        f"before trusting these metrics."
    )
    if os.path.exists(audit_report_path):
        with open(audit_report_path) as f:
            audit_report = json.load(f)
        data_integrity_status = audit_report.get("status", "WARNING")
        data_integrity_reason = audit_report.get(
            "reason", "See full audit report for details."
        )

    if checkpoint_trained_post_dedup_fix is None:
        metric_validity = "REQUIRES RETRAINING"
        validity_reason = (
            "checkpoint_trained_post_dedup_fix was not specified. This "
            "evaluation cannot confirm whether the loaded checkpoint was "
            "trained on data produced by the current (deduplicated) "
            "src/data_prep.py. Retrain and re-run with "
            "checkpoint_trained_post_dedup_fix=True to mark metrics VALID."
        )
    elif data_integrity_status == "FAIL":
        metric_validity = "REQUIRES RETRAINING"
        validity_reason = (
            "Data integrity check FAILED (cross-split overlap or label "
            "conflicts present). Metrics computed against a leaking test "
            "set are not trustworthy regardless of checkpoint origin."
        )
    elif checkpoint_trained_post_dedup_fix and data_integrity_status == "PASS":
        metric_validity = "VALID"
        validity_reason = (
            "Checkpoint was trained on deduplicated data and the "
            "corresponding data integrity check passed."
        )
    else:
        metric_validity = "PARTIALLY COMPARABLE"
        validity_reason = (
            "Checkpoint predates the deduplication fix, or data integrity "
            "status could not be fully confirmed. Numbers are reported for "
            "reference only and should not be presented as validating the "
            "current pipeline."
        )

    summary = f"""CODE STATUS: FIXED

DATA INTEGRITY CHECK:
{data_integrity_status}
Reason: {data_integrity_reason}

MODEL STATUS:
Checkpoint evaluated: {model_dir}
Test set: {test_csv}
Test samples: {metrics['n_test_samples']}

METRIC VALIDITY:
{metric_validity}

REASON:
{validity_reason}

MEASURED METRICS (from actual predictions on the above test set):
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1 (binary): {metrics['f1_binary']:.4f}
- Macro F1: {metrics['macro_f1']:.4f}
- Weighted F1: {metrics['weighted_f1']:.4f}
- AUC: {metrics['auc']:.4f}
"""

    with open(os.path.join(RESULTS_DIR, "evaluation_summary.md"), "w") as f:
        f.write(summary)

    print("\n" + summary)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--tokenizer_name", default="roberta-base")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--audit_report_path", default="data/processed/logs/audit_report.json"
    )
    parser.add_argument(
        "--checkpoint_trained_post_dedup_fix",
        action="store_true",
        help="Set this flag only if the checkpoint being loaded was "
        "trained on data produced by the current (deduplicated) "
        "src/data_prep.py. Omitting it marks metrics as requiring "
        "retraining, on purpose.",
    )
    args = parser.parse_args()

    evaluate_model(
        model_dir=args.model_dir,
        test_csv=args.test_csv,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        audit_report_path=args.audit_report_path,
        checkpoint_trained_post_dedup_fix=args.checkpoint_trained_post_dedup_fix or None,
    )


if __name__ == "__main__":
    main()
    
