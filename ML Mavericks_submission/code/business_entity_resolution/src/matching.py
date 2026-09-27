"""Matching and candidate prediction module for Amazon ML Business Entity Resolution.

Scores blocked candidate pairs using trained classifiers and applies calibrated decision
thresholds (optimal tau = 0.95 for Entity Macro F0.5) to separate true partners from negatives.
"""

import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent
_PKG_DIR = _SRC_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.config import OPTIMAL_THRESHOLD
from src.features import compute_pair_features


def predict_matches_for_s1(
    trained_model: Any,
    s1_id: str,
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    candidate_records: Optional[List[Tuple[str, str, str, str]]] = None,
    candidate_ids: Optional[List[str]] = None,
    candidate_metadata: Optional[Dict[str, Tuple[str, str, str]]] = None,
    threshold: float = OPTIMAL_THRESHOLD,
) -> Tuple[List[str], List[float]]:
    """Score candidate pairs for a single Source 1 entity and filter by decision threshold.

    Args:
        trained_model: Trained matcher with predict_proba method.
        s1_id: Source 1 entity identifier.
        s1_name: Normalized Source 1 business name.
        s1_addr: Normalized Source 1 business address.
        s1_country: Source 1 country code.
        candidate_records: Optional list of (cand_id, cand_name, cand_addr, cand_country).
        candidate_ids: Optional list of blocked candidate IDs for this S1 entity.
        candidate_metadata: Optional mapping of {cand_id: (cand_name, cand_addr, cand_country)}.
        threshold: Decision threshold for positive match (default: 0.95).

    Returns:
        Tuple of (matched_entity_ids, predicted_probabilities).
    """
    X_s1 = []
    valid_cands = []

    if candidate_records is not None:
        if not candidate_records:
            return [], []
        for cid, c_name, c_addr, c_cntry in candidate_records:
            feat = compute_pair_features(
                s1_name=s1_name,
                cand_name=c_name,
                s1_addr=s1_addr,
                cand_addr=c_addr,
                s1_country=s1_country,
                cand_country=c_cntry,
            )
            X_s1.append(feat)
            valid_cands.append(cid)
    elif candidate_ids:
        meta_dict = candidate_metadata or {}
        for cid in candidate_ids:
            cand_meta = meta_dict.get(cid, ("", "", ""))
            feat = compute_pair_features(
                s1_name=s1_name,
                cand_name=cand_meta[0],
                s1_addr=s1_addr,
                cand_addr=cand_meta[1],
                s1_country=s1_country,
                cand_country=cand_meta[2],
            )
            X_s1.append(feat)
            valid_cands.append(cid)
    else:
        return [], []

    if not X_s1:
        return [], []

    X_arr = np.array(X_s1, dtype=np.float32)
    probs = trained_model.predict_proba(X_arr)[:, 1]

    matched_ids: List[str] = []
    for cid, prob in zip(valid_cands, probs):
        if prob >= threshold:
            matched_ids.append(cid)

    return matched_ids, list(probs)


def predict_matches_batch(
    trained_model: Any,
    batch_s1_records: List[Tuple[str, str, str, str]],
    batch_candidates_per_s1: List[List[Tuple[str, str, str, str]]],
    threshold: float = OPTIMAL_THRESHOLD,
) -> List[List[str]]:
    """Score candidate pairs in large vector batches for maximum inference throughput.

    Args:
        trained_model: Trained matcher with predict_proba method.
        batch_s1_records: List of (s1_id, s1_name, s1_addr, s1_country).
        batch_candidates_per_s1: List of candidate records per S1 entity.
        threshold: Decision threshold for positive match.

    Returns:
        List of matched_entity_id lists, one per input S1 entity.
    """
    all_features = []
    offsets = [0]
    cand_ids_flat = []

    for i in range(len(batch_s1_records)):
        _, s1_n, s1_a, s1_c = batch_s1_records[i]
        cands = batch_candidates_per_s1[i]
        for cid, cn, ca, cc in cands:
            feat = compute_pair_features(s1_n, cn, s1_a, ca, s1_c, cc)
            all_features.append(feat)
            cand_ids_flat.append(cid)
        offsets.append(len(all_features))

    if not all_features:
        return [[] for _ in batch_s1_records]

    X_arr = np.array(all_features, dtype=np.float32)
    all_probs = trained_model.predict_proba(X_arr)[:, 1]

    results: List[List[str]] = []
    for i in range(len(batch_s1_records)):
        start = offsets[i]
        end = offsets[i + 1]
        if start == end:
            results.append([])
            continue
        p_slice = all_probs[start:end]
        c_slice = cand_ids_flat[start:end]
        matched = [c_slice[j] for j, prob in enumerate(p_slice) if prob >= threshold]
        results.append(matched)

    return results


def generate_direct_matches(candidate_ids: List[str]) -> List[str]:
    """Candidate-as-match baseline: treats all blocked candidates as matches."""
    return list(candidate_ids)
