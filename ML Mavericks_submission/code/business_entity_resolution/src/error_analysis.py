"""Error analysis module for diagnosing false positives and false negatives.

Extracts, structures, and categorizes representative error patterns:
1. SINGLETON_FP: S1 anchor had 0 true matches, but was incorrectly merged.
2. REGULAR_FP: Multi-match entity merged with incorrect duplicate candidate.
3. FN_MISSED: True match was missed by blocking or rejected by high decision threshold.
"""

import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent
_PKG_DIR = _SRC_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import pandas as pd

from src.config import FEATURE_NAMES

logger = logging.getLogger(__name__)


def diagnose_errors(
    val_s1_records: Dict[str, Tuple[str, str, str]],
    val_cand_records: Dict[str, Tuple[str, str, str]],
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    probabilities: Optional[Dict[Tuple[str, str], float]] = None,
    features: Optional[Dict[Tuple[str, str], List[float]]] = None,
    max_cases_per_type: int = 50,
) -> pd.DataFrame:
    """Extract and categorize representative matching errors for inspection."""
    logger.info("Extracting representative error cases...")
    rows: List[Dict[str, Any]] = []
    counts = {"SINGLETON_FP": 0, "REGULAR_FP": 0, "FN_MISSED": 0}

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        s1_tuple = val_s1_records.get(s1_id, ("", "", ""))

        # 1. Singleton False Positives
        if len(true_set) == 0 and len(pred_set) > 0:
            for cid in pred_set:
                if counts["SINGLETON_FP"] >= max_cases_per_type:
                    break
                cand_tuple = val_cand_records.get(cid, ("", "", ""))
                prob = probabilities.get((s1_id, cid), 1.0) if probabilities else None
                feat_vec = features.get((s1_id, cid)) if features else None

                entry = {
                    "error_type": "SINGLETON_FP",
                    "s1_id": s1_id,
                    "s1_name": s1_tuple[0],
                    "s1_address": s1_tuple[1],
                    "s1_country": s1_tuple[2],
                    "cand_id": cid,
                    "cand_name": cand_tuple[0],
                    "cand_address": cand_tuple[1],
                    "cand_country": cand_tuple[2],
                    "model_probability": prob,
                    "is_true_match": False,
                }
                if feat_vec and len(feat_vec) == len(FEATURE_NAMES):
                    for fn, fv in zip(FEATURE_NAMES, feat_vec):
                        entry[f"feat_{fn}"] = round(fv, 4)
                rows.append(entry)
                counts["SINGLETON_FP"] += 1

        # 2. Regular False Positives
        elif len(true_set) > 0:
            fps = pred_set - true_set
            for cid in fps:
                if counts["REGULAR_FP"] >= max_cases_per_type:
                    break
                cand_tuple = val_cand_records.get(cid, ("", "", ""))
                prob = probabilities.get((s1_id, cid), 1.0) if probabilities else None
                feat_vec = features.get((s1_id, cid)) if features else None

                entry = {
                    "error_type": "REGULAR_FP",
                    "s1_id": s1_id,
                    "s1_name": s1_tuple[0],
                    "s1_address": s1_tuple[1],
                    "s1_country": s1_tuple[2],
                    "cand_id": cid,
                    "cand_name": cand_tuple[0],
                    "cand_address": cand_tuple[1],
                    "cand_country": cand_tuple[2],
                    "model_probability": prob,
                    "is_true_match": False,
                }
                if feat_vec and len(feat_vec) == len(FEATURE_NAMES):
                    for fn, fv in zip(FEATURE_NAMES, feat_vec):
                        entry[f"feat_{fn}"] = round(fv, 4)
                rows.append(entry)
                counts["REGULAR_FP"] += 1

            # 3. False Negatives (Missed Matches)
            fns = true_set - pred_set
            for cid in fns:
                if counts["FN_MISSED"] >= max_cases_per_type:
                    break
                cand_tuple = val_cand_records.get(cid, ("", "", ""))
                prob = probabilities.get((s1_id, cid), 0.0) if probabilities else None
                feat_vec = features.get((s1_id, cid)) if features else None

                entry = {
                    "error_type": "FN_MISSED",
                    "s1_id": s1_id,
                    "s1_name": s1_tuple[0],
                    "s1_address": s1_tuple[1],
                    "s1_country": s1_tuple[2],
                    "cand_id": cid,
                    "cand_name": cand_tuple[0],
                    "cand_address": cand_tuple[1],
                    "cand_country": cand_tuple[2],
                    "model_probability": prob,
                    "is_true_match": True,
                }
                if feat_vec and len(feat_vec) == len(FEATURE_NAMES):
                    for fn, fv in zip(FEATURE_NAMES, feat_vec):
                        entry[f"feat_{fn}"] = round(fv, 4)
                rows.append(entry)
                counts["FN_MISSED"] += 1

    df_errors = pd.DataFrame(rows)
    logger.info(
        f"Diagnosed {len(df_errors)} error cases (Singleton FPs: {counts['SINGLETON_FP']}, "
        f"Regular FPs: {counts['REGULAR_FP']}, Missed FNs: {counts['FN_MISSED']})"
    )
    return df_errors


def export_error_analysis(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save error analysis report to CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    logger.info(f"Error analysis exported to {p}")
