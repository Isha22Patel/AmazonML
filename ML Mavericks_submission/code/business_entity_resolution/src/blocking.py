"""Candidate generation and blocking module for Amazon ML Business Entity Resolution.

Implements a high-recall, sub-linear multi-pass inverted indexing engine with:
1. Strategy A: Exact normalized business name + country (highest precision)
2. Strategy B: Legal-suffix-stripped name + country (symmetric matching)
3. Strategy C: Permutation-invariant sorted name tokens + country (word order flips)
4. Strategy D: Compact name (whitespace/punctuation stripped) + country (spacing/domain variations)
5. Strategy E: Composite name token + address street number (disambiguation)
6. Strategy F: Address street number + street token (co-located/rebranded entities)
7. Strategy G: Controlled name prefix (length 6) + country (controlled typo fallback)

Enforces principled frequency-aware posting caps and relevance-based candidate
ranking before truncation to eliminate arbitrary file-ordering false negatives.
"""

import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent
_PKG_DIR = _SRC_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import pandas as pd

from src.config import (
    MAX_ADDR_NUM_POSTINGS,
    MAX_EXACT_NAME_POSTINGS,
    MAX_NO_SUFFIX_POSTINGS,
    MAX_PREFIX_POSTINGS,
    MAX_TOKEN_POSTINGS,
    MAX_TOTAL_CANDIDATES_PER_S1,
)
from src.normalize import (
    extract_numbers,
    normalize_address,
    normalize_name,
    remove_legal_suffixes,
)

logger = logging.getLogger(__name__)

# Informative business stopwords to exclude from sorted token and single-token keys
BUSINESS_STOPWORDS: Set[str] = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited", "pvt", "private",
    "llc", "llp", "co", "company", "enterprises", "enterprise", "solutions", "services",
    "service", "group", "associates", "technologies", "technology", "international",
    "global", "industries", "industry", "management", "holdings", "holding", "systems",
    "system", "consulting", "logistics", "trading", "commercial", "and", "the", "of",
    "in", "for", "on", "at", "to", "a", "an", "&", "by", "with", "from", "de", "la",
    "le", "da", "di", "del", "der", "und", "et", "y", "e", "gmbh", "sa", "sas", "sarl",
    "bv", "nv", "spa", "srl", "sl", "pty", "lp", "llp", "kft", "ag", "ab", "as", "oy",
}

# Generic address tokens to exclude when extracting distinctive street words
ADDRESS_STOPWORDS: Set[str] = {
    "st", "street", "rd", "road", "ave", "avenue", "dr", "drive", "ln", "lane",
    "hwy", "highway", "blvd", "boulevard", "ct", "court", "cir", "circle",
    "pkwy", "parkway", "pl", "place", "ter", "terrace", "sq", "square",
    "suite", "ste", "apt", "apartment", "unit", "bldg", "building", "fl",
    "floor", "flr", "rm", "room", "no", "house", "plot", "shop", "near",
    "opp", "opposite", "behind", "beside", "ground", "first", "second",
}

# Relevance weights for principled candidate ranking
WEIGHT_EXACT: float = 10.0
WEIGHT_NOSUFF: float = 8.0
WEIGHT_SORTED_TOKS: float = 6.0
WEIGHT_COMPACT: float = 5.0
WEIGHT_NAME_ADDR: float = 4.0
WEIGHT_ADDR_STREET: float = 3.0
WEIGHT_PREFIX6: float = 2.0


# ==============================================================================
# 1. BLOCKING KEY EXTRACTION FUNCTIONS
# ==============================================================================

def block_exact_name(country: str, name_norm: str) -> Optional[str]:
    """Strategy A: Exact normalized business name partitioned by country."""
    if not name_norm or not country:
        return None
    return f"{country}__exact__{name_norm}"


def block_no_suffix_name(country: str, name_norm: str) -> Optional[str]:
    """Strategy B: Business name with legal suffix stripped partitioned by country.
    
    Applied symmetrically: generated whenever suffix-stripped name is non-empty.
    """
    if not name_norm or not country:
        return None
    clean = remove_legal_suffixes(name_norm)
    if not clean:
        return None
    return f"{country}__nosuff__{clean}"


def block_token_name(country: str, name_norm: str) -> Optional[str]:
    """Strategy C: Permutation-invariant sorted name tokens partitioned by country.
    
    Tokens are extracted from suffix-stripped name and filtered against generic stopwords.
    """
    if not name_norm or not country:
        return None
    clean = remove_legal_suffixes(name_norm)
    tokens = [t for t in clean.split() if t not in BUSINESS_STOPWORDS]
    if len(tokens) > 1:
        return f"{country}__toks__{' '.join(sorted(tokens))}"
    elif len(clean.split()) > 1:
        return f"{country}__toks__{' '.join(sorted(clean.split()))}"
    return None


def block_compact_name(country: str, name_norm: str) -> Optional[str]:
    """Strategy D: Whitespace- and punctuation-free character stream partitioned by country.
    
    Also strips trailing internet domain suffixes (.com, .org, .net).
    """
    if not name_norm or not country:
        return None
    clean = remove_legal_suffixes(name_norm)
    compact = clean.replace(" ", "").replace(".com", "").replace(".org", "").replace(".net", "")
    if len(compact) < 3:
        return None
    return f"{country}__compact__{compact}"


def block_name_token_address_number(country: str, name_norm: str, addr_norm: str) -> Optional[str]:
    """Strategy E: Composite most-informative name token + primary address street number."""
    if not name_norm or not addr_norm or not country:
        return None
    clean = remove_legal_suffixes(name_norm)
    nums = extract_numbers(addr_norm)
    if not nums:
        return None
    tokens = [t for t in clean.split() if len(t) >= 4 and t not in BUSINESS_STOPWORDS]
    if not tokens:
        return None
    best_tok = max(tokens, key=len)
    return f"{country}__na__{best_tok}__{nums[0]}"


def block_address_street(country: str, addr_norm: str) -> Optional[str]:
    """Strategy F: Primary address street number + longest distinctive street word."""
    if not addr_norm or not country:
        return None
    nums = extract_numbers(addr_norm)
    if not nums:
        return None
    words = [w for w in addr_norm.split() if len(w) >= 4 and not w.isdigit() and w not in ADDRESS_STOPWORDS]
    if not words:
        return None
    street_word = max(words, key=len)
    return f"{country}__as__{nums[0]}__{street_word}"


def block_prefix(country: str, name_norm: str, length: int = 6) -> Optional[str]:
    """Strategy G: First N characters of suffix-stripped name partitioned by country."""
    if not name_norm or not country:
        return None
    clean = remove_legal_suffixes(name_norm)
    if len(clean) < length:
        return None
    return f"{country}__pfx{length}__{clean[:length]}"


# ==============================================================================
# 2. INVERTED INDEX CANDIDATE GENERATION ENGINE
# ==============================================================================

class MultiStrategyBlockingIndex:
    """High-throughput multi-pass inverted index with principled candidate ranking."""

    def __init__(
        self,
        max_exact_postings: int = MAX_EXACT_NAME_POSTINGS,
        max_nosuff_postings: int = MAX_NO_SUFFIX_POSTINGS,
        max_token_postings: int = MAX_TOKEN_POSTINGS,
        max_compact_postings: int = 35,
        max_name_addr_postings: int = MAX_ADDR_NUM_POSTINGS,
        max_addr_street_postings: int = 30,
        max_prefix_postings: int = MAX_PREFIX_POSTINGS,
        max_candidates_per_s1: int = MAX_TOTAL_CANDIDATES_PER_S1,
    ) -> None:
        self.max_exact = max_exact_postings
        self.max_nosuff = max_nosuff_postings
        self.max_token = max_token_postings
        self.max_compact = max_compact_postings
        self.max_name_addr = max_name_addr_postings
        self.max_addr_street = max_addr_street_postings
        self.max_prefix = max_prefix_postings
        self.max_candidates = max_candidates_per_s1

        # Separate inverted index maps for each strategy: key -> list of candidate record indices
        self.exact_idx: Dict[str, List[int]] = defaultdict(list)
        self.no_suffix_idx: Dict[str, List[int]] = defaultdict(list)
        self.tokens_idx: Dict[str, List[int]] = defaultdict(list)
        self.compact_idx: Dict[str, List[int]] = defaultdict(list)
        self.name_addr_idx: Dict[str, List[int]] = defaultdict(list)
        self.addr_street_idx: Dict[str, List[int]] = defaultdict(list)
        self.prefix_idx: Dict[str, List[int]] = defaultdict(list)

        self.candidate_ids: List[str] = []
        self.candidate_names: List[str] = []
        self.candidate_addrs: List[str] = []
        self.candidate_countries: List[str] = []

    def clear(self) -> None:
        """Clear all inverted indexes and candidate store to reclaim memory."""
        self.exact_idx.clear()
        self.no_suffix_idx.clear()
        self.tokens_idx.clear()
        self.compact_idx.clear()
        self.name_addr_idx.clear()
        self.addr_street_idx.clear()
        self.prefix_idx.clear()
        self.candidate_ids.clear()
        self.candidate_names.clear()
        self.candidate_addrs.clear()
        self.candidate_countries.clear()
        import gc
        gc.collect()

    def index_candidates_chunk(
        self,
        c_ids: List[str],
        c_names: List[str],
        c_addrs: List[str],
        c_countries: List[str],
        filter_country: Optional[str] = None,
    ) -> None:
        """Index a batch of candidate records (from Source 2 or Source 3)."""
        target_country = filter_country.lower().strip() if filter_country else None

        for i in range(len(c_ids)):
            raw_cntry = str(c_countries[i]).lower().strip()
            if target_country is not None and raw_cntry != target_country:
                continue

            rec_idx = len(self.candidate_ids)
            self.candidate_ids.append(c_ids[i])

            cntry = sys.intern(raw_cntry)
            name_norm = normalize_name(c_names[i])
            addr_norm = normalize_address(c_addrs[i])

            self.candidate_names.append(name_norm)
            self.candidate_addrs.append(addr_norm)
            self.candidate_countries.append(cntry)

            # Strategy A: Exact Name + Country
            k_exact = block_exact_name(cntry, name_norm)
            if k_exact and len(self.exact_idx[k_exact]) < self.max_exact:
                self.exact_idx[k_exact].append(rec_idx)

            # Strategy B: No Legal Suffix + Country
            k_nosuff = block_no_suffix_name(cntry, name_norm)
            if k_nosuff and len(self.no_suffix_idx[k_nosuff]) < self.max_nosuff:
                self.no_suffix_idx[k_nosuff].append(rec_idx)

            # Strategy C: Sorted Tokens + Country
            k_toks = block_token_name(cntry, name_norm)
            if k_toks and len(self.tokens_idx[k_toks]) < self.max_token:
                self.tokens_idx[k_toks].append(rec_idx)

            # Strategy D: Compact Name + Country
            k_compact = block_compact_name(cntry, name_norm)
            if k_compact and len(self.compact_idx[k_compact]) < self.max_compact:
                self.compact_idx[k_compact].append(rec_idx)

            # Strategy E: Name Token + Address Number
            k_na = block_name_token_address_number(cntry, name_norm, addr_norm)
            if k_na and len(self.name_addr_idx[k_na]) < self.max_name_addr:
                self.name_addr_idx[k_na].append(rec_idx)

            # Strategy F: Address Number + Street Word
            k_as = block_address_street(cntry, addr_norm)
            if k_as and len(self.addr_street_idx[k_as]) < self.max_addr_street:
                self.addr_street_idx[k_as].append(rec_idx)

            # Strategy G: Prefix-6 + Country
            k_pfx = block_prefix(cntry, name_norm, length=6)
            if k_pfx and len(self.prefix_idx[k_pfx]) < self.max_prefix:
                self.prefix_idx[k_pfx].append(rec_idx)

    def retrieve_candidate_records(
        self,
        s1_name_norm: str,
        s1_addr_norm: str,
        s1_country: str,
        max_candidates: Optional[int] = None,
    ) -> List[Tuple[str, str, str, str]]:
        """Retrieve candidate records (id, name_norm, addr_norm, country) for an S1 entity.
        
        Applies relevance-based accumulation and sorting before truncation.
        """
        cap = max_candidates if max_candidates is not None else self.max_candidates
        cntry = str(s1_country).lower().strip()
        scores: Dict[int, float] = defaultdict(float)

        # Pass 1: Exact Name
        k_exact = block_exact_name(cntry, s1_name_norm)
        if k_exact:
            for idx in self.exact_idx.get(k_exact, []):
                scores[idx] += WEIGHT_EXACT

        # Pass 2: No Suffix
        k_nosuff = block_no_suffix_name(cntry, s1_name_norm)
        if k_nosuff:
            for idx in self.no_suffix_idx.get(k_nosuff, []):
                scores[idx] += WEIGHT_NOSUFF

        # Pass 3: Sorted Tokens
        k_toks = block_token_name(cntry, s1_name_norm)
        if k_toks:
            for idx in self.tokens_idx.get(k_toks, []):
                scores[idx] += WEIGHT_SORTED_TOKS

        # Pass 4: Compact Name
        k_compact = block_compact_name(cntry, s1_name_norm)
        if k_compact:
            for idx in self.compact_idx.get(k_compact, []):
                scores[idx] += WEIGHT_COMPACT

        # Pass 5: Name Token + Address Number
        k_na = block_name_token_address_number(cntry, s1_name_norm, s1_addr_norm)
        if k_na:
            for idx in self.name_addr_idx.get(k_na, []):
                scores[idx] += WEIGHT_NAME_ADDR

        # Pass 6: Address Street Number + Street Word
        k_as = block_address_street(cntry, s1_addr_norm)
        if k_as:
            for idx in self.addr_street_idx.get(k_as, []):
                scores[idx] += WEIGHT_ADDR_STREET

        # Pass 7: Prefix-6 (fallback only if candidates are few)
        if len(scores) < 15:
            k_pfx = block_prefix(cntry, s1_name_norm, length=6)
            if k_pfx:
                for idx in self.prefix_idx.get(k_pfx, []):
                    scores[idx] += WEIGHT_PREFIX6

        if not scores:
            return []

        ranked_indices = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
        selected = ranked_indices[:cap]
        return [
            (
                self.candidate_ids[idx],
                self.candidate_names[idx],
                self.candidate_addrs[idx],
                self.candidate_countries[idx],
            )
            for idx in selected
        ]

    def retrieve_candidates(
        self,
        s1_name_norm: str,
        s1_addr_norm: str,
        s1_country: str,
        max_candidates: Optional[int] = None,
    ) -> List[str]:
        """Retrieve candidate IDs for a single Source 1 entity across all passes."""
        if self.candidate_names and len(self.candidate_names) == len(self.candidate_ids):
            records = self.retrieve_candidate_records(s1_name_norm, s1_addr_norm, s1_country, max_candidates)
            return [rec[0] for rec in records]

        # Standalone ID-only fallback
        cap = max_candidates if max_candidates is not None else self.max_candidates
        cntry = str(s1_country).lower().strip()
        scores: Dict[int, float] = defaultdict(float)

        k_exact = block_exact_name(cntry, s1_name_norm)
        if k_exact:
            for idx in self.exact_idx.get(k_exact, []):
                scores[idx] += WEIGHT_EXACT
        k_nosuff = block_no_suffix_name(cntry, s1_name_norm)
        if k_nosuff:
            for idx in self.no_suffix_idx.get(k_nosuff, []):
                scores[idx] += WEIGHT_NOSUFF
        k_toks = block_token_name(cntry, s1_name_norm)
        if k_toks:
            for idx in self.tokens_idx.get(k_toks, []):
                scores[idx] += WEIGHT_SORTED_TOKS
        k_compact = block_compact_name(cntry, s1_name_norm)
        if k_compact:
            for idx in self.compact_idx.get(k_compact, []):
                scores[idx] += WEIGHT_COMPACT
        k_na = block_name_token_address_number(cntry, s1_name_norm, s1_addr_norm)
        if k_na:
            for idx in self.name_addr_idx.get(k_na, []):
                scores[idx] += WEIGHT_NAME_ADDR
        k_as = block_address_street(cntry, s1_addr_norm)
        if k_as:
            for idx in self.addr_street_idx.get(k_as, []):
                scores[idx] += WEIGHT_ADDR_STREET
        if len(scores) < 15:
            k_pfx = block_prefix(cntry, s1_name_norm, length=6)
            if k_pfx:
                for idx in self.prefix_idx.get(k_pfx, []):
                    scores[idx] += WEIGHT_PREFIX6

        if not scores:
            return []
        ranked_indices = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
        selected = ranked_indices[:cap]
        return [self.candidate_ids[idx] for idx in selected]

    def add_records(
        self,
        c_ids: List[str],
        c_names: List[str],
        c_addrs: List[str],
        c_countries: List[str],
    ) -> None:
        """Compatibility alias for index_candidates_chunk."""
        self.index_candidates_chunk(c_ids, c_names, c_addrs, c_countries)

    def generate_candidates(
        self,
        s1_name_norm: str,
        s1_addr_norm: str,
        s1_country: str,
        max_candidates: Optional[int] = None,
    ) -> List[str]:
        """Compatibility alias for retrieve_candidates."""
        return self.retrieve_candidates(s1_name_norm, s1_addr_norm, s1_country, max_candidates)


# Compatibility aliases
CandidateInvertedIndex = MultiStrategyBlockingIndex
block_address = block_address_street
COMMON_BLOCKING_TOKENS: Set[str] = BUSINESS_STOPWORDS.union(ADDRESS_STOPWORDS)
