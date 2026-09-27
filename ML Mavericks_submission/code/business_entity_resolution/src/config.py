"""Central configuration module for Amazon ML Business Entity Resolution Pipeline.

Defines all paths, schemas, hyperparameters, blocking caps, feature names,
and deterministic evaluation parameters for competition-level execution.
"""

import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent
_PKG_DIR = _SRC_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import os
from pathlib import Path
from typing import Dict, List, Set

# ------------------------------------------------------------------------------
# 1. Directory Structure & Path Resolution
# ------------------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
CODE_DIR = SRC_DIR.parent
PROJECT_ROOT = CODE_DIR.parent.parent  # Points to AmazonML_Final/

# Allow explicit environment overrides
ENV_DATASET_DIR = os.getenv("AMAZON_ML_DATASET_DIR") or os.getenv("DATA_DIR")
ENV_OUTPUT_DIR = os.getenv("AMAZON_ML_OUTPUT_DIR") or os.getenv("OUTPUT_DIR")

# Determine Dataset Directory
if ENV_DATASET_DIR and Path(ENV_DATASET_DIR).is_dir():
    DATASET_DIR = Path(ENV_DATASET_DIR).resolve()
elif (PROJECT_ROOT / "dataset").is_dir():
    DATASET_DIR = (PROJECT_ROOT / "dataset").resolve()
elif (PROJECT_ROOT.parent / "dataset").is_dir():
    DATASET_DIR = (PROJECT_ROOT.parent / "dataset").resolve()
else:
    # Default fallback
    DATASET_DIR = (PROJECT_ROOT / "dataset").resolve()

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Specific Dataset TSV Paths
TRAIN_SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1_PATH = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2_PATH = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3_PATH = TEST_DIR / "test_source3.tsv"

# Determine Output Directory (Organizers require output/ under project root)
if ENV_OUTPUT_DIR:
    OUTPUT_DIR = Path(ENV_OUTPUT_DIR).resolve()
else:
    OUTPUT_DIR = (PROJECT_ROOT / "output").resolve()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Required Competition Deliverables (Strictly Enforced)
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_RESULTS_TMP_PATH = OUTPUT_DIR / "matching_results.tmp.tsv"
CANDIDATE_PAIRS_TMP_PATH = OUTPUT_DIR / "candidate_pairs.tmp.tsv"
FINAL_CANDIDATE_PAIRS_PATH = CANDIDATE_PAIRS_PATH
DEV_CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "dev_candidate_pairs.tsv"

# Diagnostic Reports & Model Artifacts (Isolated to keep official output/ directory clean)
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
THRESHOLD_COMPARISON_PATH = REPORTS_DIR / "threshold_comparison.csv"
FEATURE_ABLATION_PATH = REPORTS_DIR / "feature_ablation.csv"
BLOCKING_COMPARISON_PATH = REPORTS_DIR / "blocking_comparison.csv"
ERROR_ANALYSIS_PATH = REPORTS_DIR / "error_analysis.csv"
EDA_REPORT_PATH = REPORTS_DIR / "eda_report.txt"
BLOCKING_REPORT_PATH = REPORTS_DIR / "blocking_report.txt"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
BEST_LIGHTGBM_MODEL_PATH = MODELS_DIR / "best_lightgbm_model.joblib"

# Big Data / PySpark Configuration
SPARK_APP_NAME = "EntityRelationDeduplication"
SPARK_DRIVER_MEMORY = "8g"
SPARK_EXECUTOR_MEMORY = "8g"

# ------------------------------------------------------------------------------
# 2. Schema Specifications
# ------------------------------------------------------------------------------
EXPECTED_SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

EXPECTED_GROUND_TRUTH_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]

EXPECTED_MATCHING_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]

EXPECTED_CANDIDATE_PAIRS_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_ids",
]

TSV_DELIMITER = "\t"

# ------------------------------------------------------------------------------
# 3. Model & Feature Specifications
# ------------------------------------------------------------------------------
# The 17 Pairwise Similarity Features
FEATURE_NAMES: List[str] = [
    # --- Name Similarity Features (8) ---
    "name_exact",                     # 1.0 if normalized names match identically
    "name_jaccard",                   # Word token Jaccard similarity [0.0 - 1.0]
    "name_common_tokens",             # Count of shared word tokens
    "name_levenshtein",               # Raw Levenshtein edit distance
    "name_normalized_levenshtein",    # Length-normalized edit similarity: 1 - (dist / max_len)
    "name_length_ratio",              # min(len1, len2) / max(len1, len2)
    "name_prefix_similarity",         # Ratio of matching characters from start
    "name_token_containment",         # Token containment ratio: |toks1 & toks2| / min(|toks1|, |toks2|)
    # --- Address Similarity Features (5) ---
    "address_exact",                  # 1.0 if normalized addresses match identically
    "address_jaccard",                # Word token Jaccard similarity
    "address_levenshtein",            # Raw character edit distance
    "address_normalized_levenshtein", # Length-normalized edit similarity
    "address_length_ratio",           # Address length ratio: min_len / max_len
    "address_number_match",           # 1.0 if street/building number sets overlap
    "address_token_overlap",          # Overlap ratio relative to shorter address
    # --- Geographic / Metadata Features (1) ---
    "country_exact",                  # 1.0 if country codes match identically
    # --- Interaction Features (1) ---
    "name_address_agreement",         # Product interaction: (name_norm_lev * address_norm_lev)
]

# ------------------------------------------------------------------------------
# 4. Evaluation & Offline Training Parameters
# ------------------------------------------------------------------------------
DEV_SAMPLE_SIZE: int = 100_000
VAL_RATIO: float = 0.20
RANDOM_SEED: int = 42

# Calibrated Decision Threshold (Optimized for Entity Macro F0.5)
OPTIMAL_THRESHOLD: float = 0.95

# Candidate Generation / Blocking Limits (Posting Caps to prevent OOM)
MAX_EXACT_NAME_POSTINGS: int = 30
MAX_NO_SUFFIX_POSTINGS: int = 30
MAX_TOKEN_POSTINGS: int = 25
MAX_ADDR_NUM_POSTINGS: int = 25
MAX_ADDR_TOKEN_POSTINGS: int = 20
MAX_PREFIX_POSTINGS: int = 20
MAX_TOTAL_CANDIDATES_PER_S1: int = 50
MAX_CANDIDATES_PER_BLOCK: int = 100

# Legal entity abbreviations and suffixes
LEGAL_SUFFIXES: Set[str] = {
    "llc", "inc", "ltd", "corp", "corporation", "co", "company", "limited",
    "incorporated", "pvt", "private", "plc", "gmbh", "sa", "sas", "sarl",
    "bv", "nv", "spa", "srl", "sl", "pty", "lp", "llp", "kft", "z o o",
    "sp z o o", "ag", "ab", "as", "oy", "hf", "cie", "cia", "ltda", "lda",
}
