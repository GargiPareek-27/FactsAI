# src/audit_data_integrity.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import pandas as pd

DEFAULT_PROCESSED_DIR = "data/processed"
SPLIT_FILES = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}


def normalize_text(text: str) -> str:
    """Same normalization used for fingerprinting -- lowercase, collapse
    whitespace, strip punctuation-heavy noise so near-identical formatting
    (extra spaces, case) doesn't hide an exact duplicate."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def fingerprint(text: str) -> str:
    """Short, non-reversible hash used for reporting overlap examples
    without printing the underlying (possibly sensitive) article text."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass
class SplitData:
    name: str
    path: str
    df: pd.DataFrame
    fingerprints: pd.Series = field(init=False)

    def __post_init__(self):
        self.fingerprints = self.df["content"].apply(fingerprint)


def load_split(name: str, path: str) -> SplitData:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing processed split: {path}")
    df = pd.read_csv(path)
    missing = {"content", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
    return SplitData(name=name, path=path, df=df)


def within_split_duplicates(split: SplitData) -> Dict:
    counts = split.fingerprints.value_counts()
    dupe_fps = counts[counts > 1]
    return {
        "split": split.name,
        "n_duplicate_groups": int(len(dupe_fps)),
        "n_duplicate_rows": int(dupe_fps.sum() - len(dupe_fps)),  # extra copies beyond the first
        "example_fingerprints": list(dupe_fps.index[:5]),
    }


def cross_split_overlap(a: SplitData, b: SplitData) -> Dict:
    set_a: Set[str] = set(a.fingerprints)
    set_b: Set[str] = set(b.fingerprints)
    overlap = set_a & set_b
    n = len(overlap)
    denom = min(len(set_a), len(set_b)) or 1
    return {
        "pair": f"{a.name} <-> {b.name}",
        "n_overlap": n,
        "pct_of_smaller_split": round(100.0 * n / denom, 4),
        "example_fingerprints": list(sorted(overlap))[:5],
    }


def label_conflicts(all_df: pd.DataFrame) -> Dict:
    """Same normalized text, different label, anywhere across the
    combined dataset (not just within one split)."""
    tmp = all_df.copy()
    tmp["_fp"] = tmp["content"].apply(fingerprint)
    grouped = tmp.groupby("_fp")["label"].nunique()
    conflicting_fps = grouped[grouped > 1]
    conflict_rows = tmp[tmp["_fp"].isin(conflicting_fps.index)]
    return {
        "n_conflicting_texts": int(len(conflicting_fps)),
        "n_affected_rows": int(len(conflict_rows)),
        "example_fingerprints": list(conflicting_fps.index[:5]),
    }


def near_duplicate_audit(all_df: pd.DataFrame, num_perm: int = 64, threshold: float = 0.85) -> Dict:
    """Optional, approximate. Uses MinHash + LSH over character 3-gram
    shingles so it stays roughly O(N) instead of O(N^2). Reported as a
    diagnostic only -- a positive here is a candidate to review, not a
    confirmed duplicate, and a clean run here is not proof that no
    near-duplicates exist.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        return {
            "run": False,
            "reason": "datasketch not installed. `pip install datasketch` to enable "
            "this optional check.",
        }

    def shingles(text: str, k: int = 3) -> Set[str]:
        norm = normalize_text(text)
        if len(norm) < k:
            return {norm} if norm else set()
        return {norm[i : i + k] for i in range(len(norm) - k + 1)}

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    for idx, text in enumerate(all_df["content"].tolist()):
        m = MinHash(num_perm=num_perm)
        for sh in shingles(text):
            m.update(sh.encode("utf-8"))
        minhashes[idx] = m
        lsh.insert(str(idx), m)

    seen_pairs = set()
    candidate_groups = 0
    for idx, m in minhashes.items():
        result = lsh.query(m)
        result = [r for r in result if r != str(idx)]
        if result:
            pair_key = tuple(sorted([idx] + [int(r) for r in result]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                candidate_groups += 1

    return {
        "run": True,
        "threshold": threshold,
        "num_perm": num_perm,
        "n_candidate_near_duplicate_groups": candidate_groups,
        "note": "Approximate candidates only. Review manually before treating as leakage.",
    }


def build_report(processed_dir: str, check_near_duplicates: bool) -> Dict:
    report: Dict = {"processed_dir": processed_dir, "status": None, "issues": []}

    splits: Dict[str, SplitData] = {}
    for name, filename in SPLIT_FILES.items():
        path = os.path.join(processed_dir, filename)
        try:
            splits[name] = load_split(name, path)
        except (FileNotFoundError, ValueError) as e:
            report["issues"].append(str(e))

    if len(splits) < 3:
        report["status"] = "WARNING"
        report["counts"] = {name: len(s.df) for name, s in splits.items()}
        report["reason"] = "One or more processed split files could not be loaded."
        return report

    report["counts"] = {name: len(s.df) for name, s in splits.items()}

    report["within_split_duplicates"] = [
        within_split_duplicates(splits[name]) for name in ["train", "validation", "test"]
    ]

    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    report["cross_split_overlap"] = [
        cross_split_overlap(splits[a], splits[b]) for a, b in pairs
    ]

    all_df = pd.concat([s.df.assign(_split=s.name) for s in splits.values()], ignore_index=True)
    report["label_conflicts"] = label_conflicts(all_df)

    total_overlap = sum(x["n_overlap"] for x in report["cross_split_overlap"])
    total_conflicts = report["label_conflicts"]["n_conflicting_texts"]

    if check_near_duplicates:
        report["near_duplicate_audit"] = near_duplicate_audit(all_df)

    if total_overlap > 0 or total_conflicts > 0:
        report["status"] = "FAIL"
    else:
        report["status"] = "PASS"

    return report


def format_text_report(report: Dict) -> str:
    lines = []
    lines.append("DATA INTEGRITY REPORT")
    lines.append("")
    counts = report.get("counts", {})
    lines.append(f"Train samples: {counts.get('train', 'N/A')}")
    lines.append(f"Validation samples: {counts.get('validation', 'N/A')}")
    lines.append(f"Test samples: {counts.get('test', 'N/A')}")
    lines.append("")

    if "cross_split_overlap" in report:
        for pair in report["cross_split_overlap"]:
            lines.append(f"{pair['pair']} overlap: {pair['n_overlap']} "
                         f"({pair['pct_of_smaller_split']}% of smaller split)")
    else:
        lines.append("Cross-split overlap: NOT COMPUTED (missing split file)")
    lines.append("")

    if "within_split_duplicates" in report:
        for w in report["within_split_duplicates"]:
            lines.append(f"Within-{w['split']} duplicate groups: {w['n_duplicate_groups']} "
                         f"(extra rows: {w['n_duplicate_rows']})")
    lines.append("")

    if "label_conflicts" in report:
        lc = report["label_conflicts"]
        lines.append(f"Label conflicts: {lc['n_conflicting_texts']} distinct texts "
                     f"({lc['n_affected_rows']} rows affected)")
    lines.append("")

    if "near_duplicate_audit" in report:
        nd = report["near_duplicate_audit"]
        if nd.get("run"):
            lines.append(f"Near-duplicate candidates (approximate, diagnostic only): "
                         f"{nd['n_candidate_near_duplicate_groups']} groups")
        else:
            lines.append(f"Near-duplicate check: SKIPPED ({nd.get('reason')})")
        lines.append("")

    lines.append("STATUS:")
    lines.append(report["status"])
    if report.get("reason"):
        lines.append(f"Reason: {report['reason']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit train/val/test split integrity.")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--check-near-duplicates", action="store_true",
                        help="Also run the optional MinHash near-duplicate diagnostic.")
    parser.add_argument("--json-out", default=None,
                        help="Optional path to also write the full report as JSON.")
    args = parser.parse_args()

    report = build_report(args.processed_dir, args.check_near_duplicates)
    print(format_text_report(report))

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2, default=str)

    # Exit non-zero on FAIL so this is CI-friendly.
    sys.exit(1 if report["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
    
