"""Official evaluation metrics and diagnostics for Amazon ML Business Entity Resolution.

Implements the authoritative competition Macro F0.5 metric:
- Entity Macro F0.5 = macro-average across ALL evaluated Source 1 entities
- F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
- Singleton Handling (0 true matches):
    * 0 predictions made: F0.5 = 1.0 (True Negative)
    * >= 1 prediction made: F0.5 = 0.0 (False Positive Merge)
- Stratified tracking across match cardinality tiers: 0, 1, 2, 3+ matches
- Pairwise ROC-AUC and PR-AUC included strictly as diagnostic references.
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
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_entity_level_metrics(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    eval_s1_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate official Macro F0.5, entity precision, and entity recall across all S1 entities.
    
    If eval_s1_ids is provided, evaluation is strictly restricted to those entities.
    """
    if eval_s1_ids is not None:
        all_s1_ids = list(eval_s1_ids)
    else:
        all_s1_ids = list(ground_truth.keys())
    n_total = len(all_s1_ids)
    if n_total == 0:
        return {
            "entity_macro_f05": 0.0,
            "entity_macro_precision": 0.0,
            "entity_macro_recall": 0.0,
            "total_evaluated_entities": 0,
            "singleton_total": 0,
            "singleton_correct_empty": 0,
            "singleton_false_positives": 0,
            "singleton_accuracy": 1.0,
        }

    f05_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    tp_global = 0
    fp_global = 0
    fn_global = 0

    singleton_count = 0
    singleton_correct_empty = 0
    singleton_false_positives = 0

    group_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 0, 1, 2, 3+ matches
    group_f05: Dict[int, List[float]] = {0: [], 1: [], 2: [], 3: []}

    for s1_id in all_s1_ids:
        true_ids = ground_truth.get(s1_id, set())
        pred_ids = predictions.get(s1_id, set())

        n_true = len(true_ids)
        grp = 0 if n_true == 0 else (1 if n_true == 1 else (2 if n_true == 2 else 3))
        group_counts[grp] += 1

        if n_true == 0:
            # Singleton entity (0 true matches in ground truth)
            singleton_count += 1
            if len(pred_ids) == 0:
                score = 1.0
                prec = 1.0
                rec = 1.0
                singleton_correct_empty += 1
            else:
                score = 0.0
                prec = 0.0
                rec = 0.0
                singleton_false_positives += 1
                fp_global += len(pred_ids)
        else:
            # Non-singleton entity
            tp = len(true_ids.intersection(pred_ids))
            fp = len(pred_ids - true_ids)
            fn = len(true_ids - pred_ids)

            tp_global += tp
            fp_global += fp
            fn_global += fn

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if tp == 0 or (0.25 * prec + rec) == 0:
                score = 0.0
            else:
                score = (1.25 * prec * rec) / (0.25 * prec + rec)

        f05_scores.append(score)
        precisions.append(prec)
        recalls.append(rec)
        group_f05[grp].append(score)

    macro_f05 = float(np.mean(f05_scores))
    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))

    return {
        "entity_macro_f05": macro_f05,
        "entity_macro_precision": macro_prec,
        "entity_macro_recall": macro_rec,
        "total_evaluated_entities": n_total,
        "singleton_total": singleton_count,
        "singleton_correct_empty": singleton_correct_empty,
        "singleton_false_positives": singleton_false_positives,
        "singleton_accuracy": singleton_correct_empty / singleton_count if singleton_count > 0 else 1.0,
        "group_counts": group_counts,
        "group_mean_f05": {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in group_f05.items()},
        "global_tp": tp_global,
        "global_fp": fp_global,
        "global_fn": fn_global,
    }


def compute_pairwise_diagnostic_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate standard pairwise machine learning diagnostic metrics."""
    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_pred_probs)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_pred_probs)) if len(np.unique(y_true)) > 1 else 0.0
    prec = float(precision_score(y_true, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_true, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0

    return {
        "pairwise_roc_auc": roc_auc,
        "pairwise_pr_auc": pr_auc,
        "pairwise_precision": prec,
        "pairwise_recall": rec,
        "pairwise_f1": f1,
        "pairwise_f05": f05,
    }
