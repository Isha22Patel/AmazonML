"""Text normalization module for Amazon ML Business Entity Resolution.

Provides ultra-fast C-speed Unicode normalization, legal suffix stripping,
address standardization, and numeral extraction. Preserves multilingual scripts
(Devanagari, Chinese, Cyrillic, Latin) while standardizing lexical noise.
"""

import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent
_PKG_DIR = _SRC_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from typing import Any, Dict, List, Optional, Set
import re
import unicodedata
import pandas as pd

from src.config import LEGAL_SUFFIXES

# Standard address abbreviation dictionary
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    "street": "st",
    "str": "st",
    "saint": "st",
    "road": "rd",
    "avenue": "ave",
    "av": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "highway": "hwy",
    "apartment": "apt",
    "suite": "ste",
    "building": "bldg",
    "floor": "fl",
    "flr": "fl",
    "room": "rm",
    "court": "ct",
    "circle": "cir",
    "parkway": "pkwy",
    "place": "pl",
    "terrace": "ter",
    "square": "sq",
}

# Precomputed C-speed single-pass translation table:
# 1. European combining diacritical marks (0x0300 - 0x036F) deleted
# 2. Punctuation characters mapped to ' ' (space)
# 3. Preserves Letters (L), Numbers (N), Combining Marks (M), Whitespace (Z)
# 4. & mapped to ' and '
_PUNCT_TABLE: Dict[int, Any] = {i: None for i in range(0x0300, 0x0370)}
for _i in range(1, 0x10000):
    if _i in _PUNCT_TABLE:
        continue
    _cat = unicodedata.category(chr(_i))
    if _cat[0] not in ("L", "N", "M", "Z"):
        _PUNCT_TABLE[_i] = " "
_PUNCT_TABLE[ord("&")] = " and "

_NUM_RE = re.compile(r"\b\d+\b")


def normalize_name(text: Optional[Any]) -> str:
    """Normalize a business name without destroying multilingual scripts or numerals.

    Pipeline:
    1. Safe NaN/None handling.
    2. NFKD Unicode decomposition (isolates accents).
    3. C-speed translation table (strips diacritics, maps punctuation to spaces, '&' -> ' and ').
    4. Lowercase.
    5. Strip and collapse whitespace.
    """
    if text is None or pd.isna(text):
        return ""
    s = str(text)
    if not s.strip():
        return ""
    return " ".join(unicodedata.normalize("NFKD", s).translate(_PUNCT_TABLE).lower().split())


def remove_legal_suffixes(text: Optional[str]) -> str:
    """Remove standard legal suffixes (inc, corp, ltd, pvt, llc, etc.) using token-level removal."""
    if not text:
        return ""
    tokens = [t for t in text.split() if t not in LEGAL_SUFFIXES]
    if not tokens:
        return text
    return " ".join(tokens)


def normalize_address(text: Optional[Any]) -> str:
    """Normalize a business address with standard abbreviation mapping."""
    if text is None or pd.isna(text):
        return ""
    s = str(text)
    if not s.strip():
        return ""
    cleaned = unicodedata.normalize("NFKD", s).translate(_PUNCT_TABLE).lower()
    tokens = [ADDRESS_ABBREVIATIONS.get(t, t) for t in cleaned.split()]
    return " ".join(tokens)


def extract_numbers(text: Optional[Any]) -> List[str]:
    """Extract meaningful numeric tokens from names or addresses as strings."""
    if text is None or pd.isna(text):
        return []
    return _NUM_RE.findall(str(text))


def tokenize(text: Optional[str]) -> List[str]:
    """Tokenize normalized text into clean tokens."""
    if not text:
        return []
    return text.split()


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and add normalized representation columns to a DataFrame without modifying original columns.

    Added columns:
    - name_norm: normalized business name
    - name_compact: whitespace-stripped normalized name
    - name_no_suffix: name with legal suffixes removed
    - address_norm: normalized address
    - address_compact: whitespace-stripped normalized address
    - address_numbers: comma-separated numbers in address
    """
    res = df.copy()

    names = res["business_name"].fillna("").astype(str)
    res["name_norm"] = names.apply(normalize_name)
    res["name_compact"] = res["name_norm"].str.replace(" ", "", regex=False)
    res["name_no_suffix"] = res["name_norm"].apply(remove_legal_suffixes)

    addrs = res["business_address"].fillna("").astype(str)
    res["address_norm"] = addrs.apply(normalize_address)
    res["address_compact"] = res["address_norm"].str.replace(" ", "", regex=False)
    res["address_numbers"] = res["address_norm"].apply(lambda a: ",".join(extract_numbers(a)))

    return res
