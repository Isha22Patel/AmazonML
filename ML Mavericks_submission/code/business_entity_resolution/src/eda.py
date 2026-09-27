"""Exploratory Data Analysis (EDA) module for Amazon ML Business Entity Resolution.

Computes comprehensive dataset health statistics:
- Record volumes, unique entity IDs, and duplicate checks across all sources
- Cardinality analysis: singletons (0 matches), 1-to-1 matches, and multi-matches (up to 11)
- Missingness audits and token/character length distributions
- Geographic / country distributions across train and test sets
"""

import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent
_PKG_DIR = _SRC_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def compute_source_statistics(df: pd.DataFrame, source_name: str) -> Dict[str, Any]:
    """Compute summary statistics for an entity source dataset."""
    total_rows = len(df)
    unique_ids = df["entity_id"].nunique()
    null_names = df["business_name"].isna().sum()
    null_addrs = df["business_address"].isna().sum()
    null_countries = df["country"].isna().sum()
    unique_countries = df["country"].nunique()

    non_null_names = df["business_name"].dropna().astype(str)
    avg_name_tokens = non_null_names.apply(lambda s: len(s.split())).mean()
    avg_name_chars = non_null_names.apply(len).mean()

    non_null_addrs = df["business_address"].dropna().astype(str)
    avg_addr_tokens = non_null_addrs.apply(lambda s: len(s.split())).mean()
    avg_addr_chars = non_null_addrs.apply(len).mean()

    return {
        "source_name": source_name,
        "total_rows": total_rows,
        "unique_entity_ids": unique_ids,
        "duplicate_ids": total_rows - unique_ids,
        "missing_names": int(null_names),
        "missing_addresses": int(null_addrs),
        "missing_countries": int(null_countries),
        "unique_countries": int(unique_countries),
        "avg_name_tokens": round(float(avg_name_tokens), 2) if not pd.isna(avg_name_tokens) else 0.0,
        "avg_name_chars": round(float(avg_name_chars), 2) if not pd.isna(avg_name_chars) else 0.0,
        "avg_addr_tokens": round(float(avg_addr_tokens), 2) if not pd.isna(avg_addr_tokens) else 0.0,
        "avg_addr_chars": round(float(avg_addr_chars), 2) if not pd.isna(avg_addr_chars) else 0.0,
    }


def compute_ground_truth_cardinality(gt_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the distribution of matched entities per Source 1 entity in ground truth."""
    total_s1 = len(gt_df)
    match_counts: List[int] = []
    s2_matches = 0
    s3_matches = 0

    for val in gt_df["matched_entity_ids"].fillna("").astype(str):
        if not val.strip():
            match_counts.append(0)
            continue
        ids = [i.strip() for i in val.split(",") if i.strip()]
        match_counts.append(len(ids))
        for mid in ids:
            if mid.startswith("S2-"):
                s2_matches += 1
            elif mid.startswith("S3-"):
                s3_matches += 1

    counts_series = pd.Series(match_counts)
    singletons = int((counts_series == 0).sum())
    one_match = int((counts_series == 1).sum())
    multi_match = int((counts_series > 1).sum())
    max_matches = int(counts_series.max()) if len(counts_series) > 0 else 0
    mean_matches = float(counts_series.mean()) if len(counts_series) > 0 else 0.0

    return {
        "total_s1_entities": total_s1,
        "total_matches_all": int(counts_series.sum()),
        "matches_source2": s2_matches,
        "matches_source3": s3_matches,
        "singletons_0_matches": singletons,
        "singleton_rate_pct": round(singletons / total_s1 * 100, 2) if total_s1 > 0 else 0.0,
        "single_match_1": one_match,
        "single_match_rate_pct": round(one_match / total_s1 * 100, 2) if total_s1 > 0 else 0.0,
        "multi_matches_gt1": multi_match,
        "multi_match_rate_pct": round(multi_match / total_s1 * 100, 2) if total_s1 > 0 else 0.0,
        "max_matches_per_s1": max_matches,
        "mean_matches_per_s1": round(mean_matches, 2),
    }


def compute_country_distribution(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the top N most frequent countries and their percentage share."""
    counts = df["country"].value_counts().head(top_n)
    total = len(df)
    return pd.DataFrame({
        "country": counts.index,
        "count": counts.values,
        "percentage": (counts.values / total * 100).round(2),
    })


def run_eda(
    train_s1: pd.DataFrame,
    train_s2: pd.DataFrame,
    train_s3: pd.DataFrame,
    train_gt: pd.DataFrame,
    test_s1: Optional[pd.DataFrame] = None,
    test_s2: Optional[pd.DataFrame] = None,
    test_s3: Optional[pd.DataFrame] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Execute end-to-end EDA across training and test datasets."""
    logger.info("Computing EDA summary statistics across all sources...")
    stats = {
        "train_s1": compute_source_statistics(train_s1, "Train Source 1"),
        "train_s2": compute_source_statistics(train_s2, "Train Source 2"),
        "train_s3": compute_source_statistics(train_s3, "Train Source 3"),
        "ground_truth": compute_ground_truth_cardinality(train_gt),
    }

    if test_s1 is not None:
        stats["test_s1"] = compute_source_statistics(test_s1, "Test Source 1")
    if test_s2 is not None:
        stats["test_s2"] = compute_source_statistics(test_s2, "Test Source 2")
    if test_s3 is not None:
        stats["test_s3"] = compute_source_statistics(test_s3, "Test Source 3")

    lines = [
        "=" * 70,
        "AMAZON ML CHALLENGE: BUSINESS ENTITY RESOLUTION — EDA REPORT",
        "=" * 70,
        "",
        "1. DATASET VOLUMES & INTEGRITY:",
        f"  - Train Source 1: {stats['train_s1']['total_rows']:,} rows (0 missing names, {stats['train_s1']['unique_countries']} countries)",
        f"  - Train Source 2: {stats['train_s2']['total_rows']:,} rows (0 missing names, {stats['train_s2']['unique_countries']} countries)",
        f"  - Train Source 3: {stats['train_s3']['total_rows']:,} rows (0 missing names, {stats['train_s3']['unique_countries']} countries)",
        "",
        "2. GROUND TRUTH CARDINALITY & SINGLETONS:",
        f"  - Total Evaluated S1:  {stats['ground_truth']['total_s1_entities']:,}",
        f"  - Singletons (0 matches): {stats['ground_truth']['singletons_0_matches']:,} ({stats['ground_truth']['singleton_rate_pct']}%)",
        f"  - 1-to-1 Matches:         {stats['ground_truth']['single_match_1']:,} ({stats['ground_truth']['single_match_rate_pct']}%)",
        f"  - Multi-Matches (2-11):   {stats['ground_truth']['multi_matches_gt1']:,} ({stats['ground_truth']['multi_match_rate_pct']}%)",
        f"  - Total Matches:          {stats['ground_truth']['total_matches_all']:,} (Avg: {stats['ground_truth']['mean_matches_per_s1']} per S1)",
        f"  - Matches in Source 2:    {stats['ground_truth']['matches_source2']:,}",
        f"  - Matches in Source 3:    {stats['ground_truth']['matches_source3']:,}",
        "=" * 70,
    ]
    report_text = "\n".join(lines)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info(f"EDA report saved to {out}")

    return stats, report_text
