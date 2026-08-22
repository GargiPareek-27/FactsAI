# src/utils.py
"""Shared utilities: reproducibility, dataset loading, and the shared
NewsDataset class used by train.py, evaluate.py, and predict.py.

This module did not previously exist even though evaluate.py and
predict.py both imported from it (`from src.utils import ...`) — that
made evaluate.py unrunnable as-is. This file provides those pieces.
"""

import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def seed_everything(seed: int = 42) -> None:
    """Seeds python/numpy/torch RNGs so training and evaluation runs are
    reproducible. Call this once at the start of train.py / evaluate.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Loads a processed train/val/test CSV produced by data_prep.py.

    Requires the 'content' and 'label' columns that
    data_prep.preprocess_and_split actually writes, and fails loudly if
    either is missing — rather than assuming column position and
    silently renaming columns, which was the root cause of the
    'ULTIMATE FIX' column-overwrite hack previously in train.py.
    """
    df = pd.read_csv(csv_path)
    missing = {"content", "label"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required column(s): {sorted(missing)}. "
            f"Found columns: {list(df.columns)}. "
            "Re-run data_prep.preprocess_and_split to regenerate this file "
            "with the expected schema."
        )
    return df


class NewsDataset(Dataset):
    """Tokenizes a (content, label) dataframe for RoBERTaBiLSTM.

    Expects a dataframe with 'content' (article text) and 'label'
    (0 = real, 1 = fake — see src/data_prep.py's load_isot_dataset for
    where that convention is set) columns.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.texts = df["content"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
