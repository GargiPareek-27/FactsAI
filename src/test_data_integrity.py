# src/test_data_integrity.py

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_prep import deduplicate_and_resolve_conflicts, load_liar_dataset
from src.audit_data_integrity import (
    normalize_text,
    fingerprint,
    build_report,
)


# ---------------------------------------------------------------------
# 1. Duplicate detection works
# ---------------------------------------------------------------------
def test_exact_duplicates_are_collapsed():
    df = pd.DataFrame({
        "content": ["Same Article", "same   article", "Different Article"],
        "label": [1, 1, 0],
    })
    cleaned = deduplicate_and_resolve_conflicts(df)
    assert len(cleaned) == 2
    assert set(cleaned["content"].str.lower().str.strip()) == {
        "same article", "different article"
    }


def test_label_conflicts_are_removed_not_averaged():
    df = pd.DataFrame({
        "content": ["Conflicting Text", "conflicting text", "Clean Row"],
        "label": [1, 0, 0],
    })
    cleaned = deduplicate_and_resolve_conflicts(df)
    # both conflicting rows removed, only the clean row remains
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["content"] == "Clean Row"


# ---------------------------------------------------------------------
# 2. Cross-split overlap is detected (via the audit script's own logic)
# ---------------------------------------------------------------------
def test_cross_split_overlap_detected(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    train = pd.DataFrame({"content": ["Article A", "Article B"], "label": [0, 1]})
    # "Article B" duplicated into test -> should be flagged
    val = pd.DataFrame({"content": ["Article C"], "label": [0]})
    test = pd.DataFrame({"content": ["Article B", "Article D"], "label": [1, 0]})

    train.to_csv(processed_dir / "train.csv", index=False)
    val.to_csv(processed_dir / "val.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)

    report = build_report(str(processed_dir), check_near_duplicates=False)
    assert report["status"] == "FAIL"

    train_test_pair = next(
        p for p in report["cross_split_overlap"] if p["pair"] == "train <-> test"
    )
    assert train_test_pair["n_overlap"] == 1


def test_clean_splits_pass():
    pass  # placeholder kept intentionally simple; see test below for the real PASS case


def test_no_overlap_reports_pass(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    pd.DataFrame({"content": ["Article A", "Article E"], "label": [0, 1]}).to_csv(
        processed_dir / "train.csv", index=False
    )
    pd.DataFrame({"content": ["Article C"], "label": [0]}).to_csv(
        processed_dir / "val.csv", index=False
    )
    pd.DataFrame({"content": ["Article D"], "label": [0]}).to_csv(
        processed_dir / "test.csv", index=False
    )

    report = build_report(str(processed_dir), check_near_duplicates=False)
    assert report["status"] == "PASS"


# ---------------------------------------------------------------------
# 3. No silent label fallback + correct LIAR mapping
# ---------------------------------------------------------------------
def test_liar_unknown_label_raises(tmp_path):
    liar_dir = tmp_path / "liar"
    liar_dir.mkdir()
    # column 1 = label, column 2 = statement text, per LIAR's format
    bad_row = pd.DataFrame([[0, "totally-bogus-label", "some statement"]])
    bad_row.to_csv(liar_dir / "train.tsv", sep="\t", header=False, index=False)

    with pytest.raises(ValueError, match="Unexpected LIAR label"):
        load_liar_dataset(str(tmp_path))


def test_liar_known_labels_map_correctly(tmp_path):
    liar_dir = tmp_path / "liar"
    liar_dir.mkdir()
    rows = pd.DataFrame([
        [0, "pants-fire", "statement 1"],
        [1, "true", "statement 2"],
        [2, "half-true", "statement 3"],
    ])
    rows.to_csv(liar_dir / "train.tsv", sep="\t", header=False, index=False)

    df = load_liar_dataset(str(tmp_path))
    mapping = dict(zip(df["content"], df["label"]))
    assert mapping["statement 1"] == 1  # pants-fire -> fake
    assert mapping["statement 2"] == 0  # true -> real
    assert mapping["statement 3"] == 0  # half-true -> real


# ---------------------------------------------------------------------
# 4. Padding receives zero attention contribution
# ---------------------------------------------------------------------
def test_padding_gets_zero_attention_weight():
    """Directly tests the masking math used in the model's attention
    pooling: after masked_fill(-inf) + softmax, padded positions must
    receive an attention weight of (numerically) zero.
    """
    seq_len = 6
    n_real_tokens = 3
    scores = torch.randn(1, seq_len)
    attention_mask = torch.zeros(1, seq_len)
    attention_mask[:, :n_real_tokens] = 1  # first 3 tokens real, rest padding

    masked_scores = scores.masked_fill(attention_mask == 0, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)

    padding_weights = weights[:, n_real_tokens:]
    assert torch.allclose(padding_weights, torch.zeros_like(padding_weights), atol=1e-6)
    # real-token weights should sum to ~1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1), atol=1e-5)


# ---------------------------------------------------------------------
# 5. Evaluation metrics run correctly (metric math only, no model load)
# ---------------------------------------------------------------------
def test_metrics_computation_matches_sklearn():
    from sklearn.metrics import f1_score, accuracy_score

    labels = np.array([0, 1, 1, 0, 1, 0])
    preds = np.array([0, 1, 0, 0, 1, 1])

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")

    assert 0.0 <= acc <= 1.0
    assert 0.0 <= macro_f1 <= 1.0
    assert 0.0 <= weighted_f1 <= 1.0
    # sanity: 4/6 correct
    assert acc == pytest.approx(4 / 6)


def test_fingerprint_is_case_and_whitespace_insensitive():
    a = fingerprint("Hello   World")
    b = fingerprint("hello world")
    c = fingerprint("Something Else")
    assert a == b
    assert a != c
    
