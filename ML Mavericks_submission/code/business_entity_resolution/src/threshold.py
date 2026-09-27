"""Decision threshold tuning and search module for Amazon ML Business Entity Resolution.

Conducts exhaustive threshold searches across the decision boundary [0.10, 0.95]
evaluated strictly against the competition's Entity Macro F0.5 metric. Tracks singleton
false positive suppression and identifies optimal decision cutoffs.
"""

import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent
_PKG_DIR = _SRC_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import numpy as np
import pandas as pd

from src.evaluation import compute_entity_level_metrics

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: List[float] = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
]


def evaluate_threshold_grid(
    val_s1_ids: List[str],
    val_cand_ids_per_s1: List[List[str]],
    val_probs_per_s1: List[np.ndarray],
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, float]:
    """Evaluate candidate match predictions across a grid of thresholds on the validation set.

    Args:
        val_s1_ids: List of Source 1 entity IDs in the validation split.
        val_cand_ids_per_s1: List of candidate ID lists for each S1 entity.
        val_probs_per_s1: List of predicted probabilities for each candidate.
        ground_truth: Mapping of true match sets: {s1_id: set(matched_ids)}.
        thresholds: List of threshold values to test (defaults to DEFAULT_THRESHOLDS).

    Returns:
        Tuple of (results_dataframe, optimal_threshold).
    """
    grid = thresholds or DEFAULT_THRESHOLDS
    logger.info(f"Evaluating decision threshold grid on validation set ({len(grid)} thresholds)...")

    records: List[Dict[str, Any]] = []
    best_f05 = -1.0
    optimal_thresh = 0.95

    # Strictly isolate validation ground truth: zero training leakage
    val_ground_truth = {sid: ground_truth.get(sid, set()) for sid in val_s1_ids}

    for thresh in grid:
        preds_map: Dict[str, Set[str]] = {}
        total_predicted = 0

        for i, s1_id in enumerate(val_s1_ids):
            c_ids = val_cand_ids_per_s1[i]
            probs = val_probs_per_s1[i]

            if len(c_ids) == 0:
                preds_map[s1_id] = set()
                continue

            matched = {c_ids[j] for j, p in enumerate(probs) if p >= thresh}
            preds_map[s1_id] = matched
            total_predicted += len(matched)

        metrics = compute_entity_level_metrics(val_ground_truth, preds_map, eval_s1_ids=val_s1_ids)
        f05 = metrics["entity_macro_f05"]
        prec = metrics["entity_macro_precision"]
        rec = metrics["entity_macro_recall"]
        singletons_fp = metrics.get("singleton_false_positives", 0)
        singletons_total = metrics.get("singleton_total", 1)

        records.append({
            "threshold": round(thresh, 2),
            "entity_macro_f05": round(f05, 5),
            "entity_macro_precision": round(prec, 5),
            "entity_macro_recall": round(rec, 5),
            "total_predicted_matches": total_predicted,
            "singleton_fps": singletons_fp,
            "singleton_fp_rate": round(singletons_fp / singletons_total if singletons_total > 0 else 0.0, 5),
        })

        if f05 > best_f05:
            best_f05 = f05
            optimal_thresh = thresh

    df_results = pd.DataFrame(records)
    logger.info(f"Optimal decision threshold: {optimal_thresh} with Entity Macro F0.5 = {best_f05:.4f}")
    return df_results, optimal_thresh


def save_threshold_comparison(
    df: pd.DataFrame,
    path: Union[str, Path],
) -> None:
    """Save the threshold evaluation comparison table to CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    logger.info(f"Threshold comparison saved to {p}")
