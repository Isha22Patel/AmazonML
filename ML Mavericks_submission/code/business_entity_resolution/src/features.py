"""Pairwise similarity feature extraction module for Amazon ML Business Entity Resolution.

Computes 17 fine-grained lexical, token-based, structural, and geographic interaction features
for every candidate pair, optimized with RapidFuzz C-extensions.
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
import re
import numpy as np
import pandas as pd

from src.config import FEATURE_NAMES

try:
    from rapidfuzz.distance import Levenshtein, JaroWinkler
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

_NUM_RE = re.compile(r"\b\d+\b")


def extract_numbers_set(s: str) -> Set[str]:
    """Extract set of integer strings from a text."""
    if not s:
        return set()
    return set(_NUM_RE.findall(s))


def compute_pair_features(
    s1_name: str,
    cand_name: str,
    s1_addr: str,
    cand_addr: str,
    s1_country: str,
    cand_country: str,
) -> List[float]:
    """Compute all 17 pairwise similarity features for a single candidate pair."""
    # 1. Geographic Feature
    country_exact = 1.0 if s1_country.lower().strip() == cand_country.lower().strip() else 0.0

    # 2. Name Features
    n1 = s1_name or ""
    n2 = cand_name or ""
    len1 = len(n1)
    len2 = len(n2)
    max_n_len = max(len1, len2)
    min_n_len = min(len1, len2)

    if max_n_len == 0:
        name_exact = 1.0
        name_jaccard = 1.0
        name_common_tokens = 0.0
        name_lev = 0.0
        name_norm_lev = 1.0
        name_len_ratio = 1.0
        name_prefix_sim = 1.0
        name_token_containment = 1.0
    else:
        name_exact = 1.0 if n1 == n2 else 0.0
        name_len_ratio = min_n_len / max_n_len

        # Prefix similarity
        prefix_len = 0
        for c1, c2 in zip(n1, n2):
            if c1 == c2:
                prefix_len += 1
            else:
                break
        name_prefix_sim = prefix_len / max_n_len

        # Edit distance
        if RAPIDFUZZ_AVAILABLE:
            name_lev = float(Levenshtein.distance(n1, n2))
            name_norm_lev = float(Levenshtein.normalized_similarity(n1, n2))
        else:
            # Fallback
            name_lev = float(abs(len1 - len2))
            name_norm_lev = 1.0 - (name_lev / max_n_len)

        # Token features
        toks1 = set(n1.split())
        toks2 = set(n2.split())
        inter = toks1.intersection(toks2)
        union = toks1.union(toks2)
        name_common_tokens = float(len(inter))
        name_jaccard = len(inter) / len(union) if union else 0.0
        min_toks = min(len(toks1), len(toks2))
        name_token_containment = len(inter) / min_toks if min_toks > 0 else 0.0

    # 3. Address Features
    a1 = s1_addr or ""
    a2 = cand_addr or ""
    len_a1 = len(a1)
    len_a2 = len(a2)
    max_a_len = max(len_a1, len_a2)
    min_a_len = min(len_a1, len_a2)

    if max_a_len == 0:
        address_exact = 1.0
        address_jaccard = 1.0
        address_lev = 0.0
        address_norm_lev = 1.0
        address_len_ratio = 1.0
        address_number_match = 1.0
        address_token_overlap = 1.0
    else:
        address_exact = 1.0 if a1 == a2 else 0.0
        address_len_ratio = min_a_len / max_a_len

        if RAPIDFUZZ_AVAILABLE:
            address_lev = float(Levenshtein.distance(a1, a2))
            address_norm_lev = float(Levenshtein.normalized_similarity(a1, a2))
        else:
            address_lev = float(abs(len_a1 - len_a2))
            address_norm_lev = 1.0 - (address_lev / max_a_len)

        # Address tokens
        a_toks1 = set(a1.split())
        a_toks2 = set(a2.split())
        a_inter = a_toks1.intersection(a_toks2)
        a_union = a_toks1.union(a_toks2)
        address_jaccard = len(a_inter) / len(a_union) if a_union else 0.0
        min_a_toks = min(len(a_toks1), len(a_toks2))
        address_token_overlap = len(a_inter) / min_a_toks if min_a_toks > 0 else 0.0

        # Number set overlap
        nums1 = extract_numbers_set(a1)
        nums2 = extract_numbers_set(a2)
        if nums1 and nums2:
            address_number_match = 1.0 if (nums1 & nums2) else 0.0
        elif not nums1 and not nums2:
            address_number_match = 1.0
        else:
            address_number_match = 0.0

    # 4. Interaction Feature
    name_address_agreement = name_norm_lev * address_norm_lev

    return [
        name_exact,
        name_jaccard,
        name_common_tokens,
        name_lev,
        name_norm_lev,
        name_len_ratio,
        name_prefix_sim,
        name_token_containment,
        address_exact,
        address_jaccard,
        address_lev,
        address_norm_lev,
        address_len_ratio,
        address_number_match,
        address_token_overlap,
        country_exact,
        name_address_agreement,
    ]


def batch_compute_features(
    s1_records: List[Tuple[str, str, str]],
    cand_records: List[Tuple[str, str, str]],
) -> np.ndarray:
    """Compute features for a batch of candidate pairs in parallel/C-speed.

    Args:
        s1_records: List of (name, addr, country) for S1.
        cand_records: List of (name, addr, country) for candidate.

    Returns:
        2D numpy array of shape (N, 17).
    """
    n_pairs = len(s1_records)
    if n_pairs == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    X = np.empty((n_pairs, len(FEATURE_NAMES)), dtype=np.float32)
    for i in range(n_pairs):
        s1_n, s1_a, s1_c = s1_records[i]
        c_n, c_a, c_c = cand_records[i]
        X[i] = compute_pair_features(s1_n, c_n, s1_a, c_a, s1_c, c_c)

    return X
