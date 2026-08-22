# src/data_prep.py
"""Data preprocessing for fake news detection."""

import os
import pandas as pd
import numpy as np
import unicodedata
import re
import nltk
from typing import List, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords


def load_isot_dataset(
    raw_dir: str,
) -> pd.DataFrame:
    """Load ISOT Fake News dataset (Fake.csv, True.csv) as a single DataFrame."""
    fake_path = os.path.join(raw_dir, "Fake.csv")
    true_path = os.path.join(raw_dir, "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError(
            f"ISOT files not found: {fake_path}, {true_path}"
        )

    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # Give them the same column set if needed
    required_cols = ["title", "text"]
    for col in required_cols:
        if col not in df_fake.columns:
            df_fake[col] = ""
        if col not in df_true.columns:
            df_true[col] = ""

    df_fake["label"] = 1  # fake
    df_true["label"] = 0  # real

    # Combine fields if needed
    df_fake["content"] = df_fake["title"].fillna("") + " " + df_fake["text"].fillna("")
    df_true["content"] = df_true["title"].fillna("") + " " + df_true["text"].fillna("")

    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df[["content", "label"]].dropna(subset=["content"])
    return df


def load_kaggle_fake_news(
    raw_dir: str,
) -> pd.DataFrame:
    """Load Kaggle fake‑news dataset."""
    train_path = os.path.join(raw_dir, "train.csv")
    if not os.path.exists(train_path):
        # older fake‑news dataset with "label" column
        train_path = os.path.join(raw_dir, "fake_news.csv")

    df = pd.read_csv(train_path)

    # Most Kaggle versions: "title", "text", "label"
    if "label" not in df.columns:
        if "fake" in df.columns:
            df["label"] = df["fake"]
        else:
            raise KeyError("Cannot identify label column in Kaggle data.")

    text_cols = [c for c in ["title", "text"] if c in df.columns]
    df["content"] = df[text_cols].fillna("").apply(" ".join, axis=1)

    df = df[["content", "label"]].dropna(subset=["content"])
    return df


def load_liar_dataset(
    raw_dir: str,
) -> pd.DataFrame:
    """Load LIAR dataset (statements with 6-way truthfulness labels),
    binarized to match the real=0/fake=1 convention used by ISOT/Kaggle.

    IMPORTANT — bug fixed here: LIAR's raw label column uses six
    lowercase truthfulness levels (pants-fire, false, barely-true,
    half-true, mostly-true, true). A previous version of this function
    mapped {"TRUE": 0, "FALSE": 1, "NONE": 0} — none of those keys match
    LIAR's actual label strings, so every row silently fell through to
    `.fillna(0)` and every single LIAR example was mapped to label 0
    (real), regardless of its actual truthfulness rating. If LIAR data
    was ever merged into a real training run before this fix, that
    portion of the training set was mislabeled and would have taught the
    model nothing correct about those examples.

    Binarization used here (a documented modeling choice — reasonable
    alternatives exist): {pants-fire, false, barely-true} -> fake (1),
    {half-true, mostly-true, true} -> real (0). Unknown label strings
    raise an error instead of silently defaulting, so a future dataset-
    format change fails loudly rather than quietly mislabeling data again.
    """
    data_dir = os.path.join(raw_dir, "liar")
    train_path = os.path.join(data_dir, "train.tsv")
    test_path = os.path.join(data_dir, "test.tsv")
    val_path = os.path.join(data_dir, "val.tsv")

    fake_labels = {"pants-fire", "false", "barely-true"}
    real_labels = {"half-true", "mostly-true", "true"}

    dfs = []
    for path in [train_path, test_path, val_path]:
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t", header=None)
            # LIAR format: index, label, statement, etc.
            df = df[[
                2,  # statement text
                1,  # label
            ]].rename(columns={2: "content", 1: "label"})

            df["label"] = df["label"].astype(str).str.strip().str.lower()

            unmapped = set(df["label"].unique()) - fake_labels - real_labels
            if unmapped:
                raise ValueError(
                    f"Unexpected LIAR label value(s) found: {sorted(unmapped)}. "
                    "Expected one of: pants-fire, false, barely-true, "
                    "half-true, mostly-true, true. Refusing to silently "
                    "default these to a guessed label — update fake_labels/"
                    "real_labels above if the dataset format has changed."
                )

            df["label"] = df["label"].apply(lambda l: 1 if l in fake_labels else 0)
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"LIAR TSV files not found in {data_dir}")

    df = pd.concat(dfs, ignore_index=True)
    df = df[["content", "label"]].dropna(subset=["content"])
    return df


def clean_text(text: str) -> str:
    """Clean and normalize a single text string."""
    if not isinstance(text, str):
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)

    # Remove URLs, mentions, hashes
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", text)
    text = re.sub(r"\bRT\b", " ", text)
    text = re.sub(r"[\"#$%&'()*+,-/:;<=>?@\\^_`{|}~]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text: str, stop_words: List[str]) -> str:
    """Remove stopwords from text."""
    if not text:
        return text
    tokens = text.split()
    tokens = [tok for tok in tokens if tok.lower() not in stop_words]
    return " ".join(tokens)


def merge_datasets(
    isot_df: pd.DataFrame,
    kaggle_df: Optional[pd.DataFrame] = None,
    liar_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge multiple datasets and shuffle."""
    dfs = [isot_df]
    if kaggle_df is not None:
        dfs.append(kaggle_df)
    if liar_df is not None:
        dfs.append(liar_df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Ensure label is int
    df["label"] = df["label"].astype(int)
    return df


def preprocess_and_split(
    raw_dir: str,
    processed_dir: str,
    clean_stopwords: bool = False,
    # Default is False, not True: RoBERTa's subword tokenizer and
    # self-attention rely on natural sentence structure, and removing
    # stopwords (a bag-of-words-era technique) measurably degrades its
    # contextual embeddings rather than improving them. Kept configurable
    # for anyone who wants to run the comparison explicitly.
) -> None:
    """Main pipeline: load, clean, merge, split, and save."""
    os.makedirs(processed_dir, exist_ok=True)

    # Load datasets
    isot_df = load_isot_dataset(raw_dir)

    # Optionally load Kaggle
    kaggle_df = None
    kaggle_raw = os.path.join(raw_dir, "kaggle")
    if os.path.exists(kaggle_raw):
        try:
            kaggle_df = load_kaggle_fake_news(kaggle_raw)
        except Exception as e:
            print(f"Skipping Kaggle dataset: {e}")

    # Optionally load LIAR
    liar_df = None
    liar_raw = os.path.join(raw_dir, "liar")
    if os.path.exists(liar_raw):
        try:
            liar_df = load_liar_dataset(raw_dir)
        except Exception as e:
            print(f"Skipping LIAR dataset: {e}")

    # Merge
    df = merge_datasets(isot_df, kaggle_df, liar_df)

    # Clean text
    df["content"] = df["content"].apply(clean_text)

    # Optional stopwords removal
    if clean_stopwords:
        stop_words = set(stopwords.words("english"))
        df["content"] = df["content"].apply(
            lambda t: remove_stopwords(t, stop_words)
        )

    # Drop empty
    df = df[df["content"].str.len() > 5].reset_index(drop=True)

    # Train/val/test split
    train, temp = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label"],
        random_state=42,
    )
    val, test = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp["label"],
        random_state=42,
    )

    # Save
    train.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    print(f"Saved train={len(train)}, val={len(val)}, test={len(test)}")
