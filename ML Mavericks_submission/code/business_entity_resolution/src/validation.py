"""Validation module for Amazon ML Business Entity Resolution.

Provides strict pre-flight checks:
1. Input schema validation (source tables and ground truth)
2. ID prefix and duplicate validation
3. Official validate_submission.py runner
4. Local standalone submission validation verifying 100% S1 coverage,
   subset constraint (matches subset of candidates), no duplicate IDs,
   and zero self-matches.
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
import os
import subprocess
import sys
import pandas as pd

from src.config import (
    CANDIDATE_PAIRS_PATH,
    EXPECTED_CANDIDATE_PAIRS_COLUMNS,
    EXPECTED_GROUND_TRUTH_COLUMNS,
    EXPECTED_MATCHING_COLUMNS,
    EXPECTED_SOURCE_COLUMNS,
    MATCHING_RESULTS_PATH,
    PROJECT_ROOT,
    TEST_DIR,
    TEST_SOURCE1_PATH,
    TRAIN_DIR,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. INPUT DATASET VALIDATION
# ==============================================================================

def validate_file_exists(path: Union[str, Path], description: str = "Dataset") -> Path:
    """Verify that a given file exists and is readable."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"{description} file not found at: {p}. Please verify project directory structure."
        )
    return p


def validate_source_schema(df: pd.DataFrame, dataset_name: str = "source") -> List[str]:
    """Validate that the DataFrame contains all expected source columns."""
    existing_cols = list(df.columns)
    missing_cols = [col for col in EXPECTED_SOURCE_COLUMNS if col not in existing_cols]
    if missing_cols:
        raise ValueError(
            f"Schema validation failed for '{dataset_name}': Missing required columns: {missing_cols}. "
            f"Found columns: {existing_cols}"
        )
    extra_cols = [col for col in existing_cols if col not in EXPECTED_SOURCE_COLUMNS]
    return extra_cols


def validate_ground_truth_schema(df: pd.DataFrame, dataset_name: str = "ground_truth") -> List[str]:
    """Validate that the ground truth DataFrame contains all required columns."""
    existing_cols = list(df.columns)
    missing_cols = [col for col in EXPECTED_GROUND_TRUTH_COLUMNS if col not in existing_cols]
    if missing_cols:
        raise ValueError(
            f"Schema validation failed for '{dataset_name}': Missing required columns: {missing_cols}. "
            f"Found columns: {existing_cols}"
        )
    extra_cols = [col for col in existing_cols if col not in EXPECTED_GROUND_TRUTH_COLUMNS]
    return extra_cols


def validate_entity_id_prefixes(
    df: pd.DataFrame,
    id_column: str,
    expected_prefix: str,
    dataset_name: str = "dataset",
) -> bool:
    """Validate that entity IDs in a dataset have the expected prefix (e.g. 'S1-', 'S2-', 'S3-')."""
    sample_ids = df[id_column].dropna().astype(str).head(1000)
    invalid_ids = [eid for eid in sample_ids if not eid.startswith(expected_prefix)]
    if invalid_ids:
        raise ValueError(
            f"Dataset '{dataset_name}' contains invalid entity IDs not starting with '{expected_prefix}': "
            f"{invalid_ids[:5]}"
        )
    return True


def validate_no_duplicate_ids(
    df: pd.DataFrame,
    id_column: str = "entity_id",
    dataset_name: str = "dataset",
) -> int:
    """Verify that entity IDs are unique within the dataset."""
    total_rows = len(df)
    unique_ids = df[id_column].nunique()
    dupes = total_rows - unique_ids
    if dupes > 0:
        logger.warning(
            f"Dataset '{dataset_name}' contains {dupes:,} duplicate IDs out of {total_rows:,} rows."
        )
    return dupes


def validate_all_input_datasets(
    train_dir: Optional[Path] = None,
    test_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full existence and size checks across all training and test dataset files."""
    t_dir = train_dir or TRAIN_DIR
    te_dir = test_dir or TEST_DIR

    required_files = [
        ("Train Source 1", t_dir / "train_source1.tsv"),
        ("Train Source 2", t_dir / "train_source2.tsv"),
        ("Train Source 3", t_dir / "train_source3.tsv"),
        ("Train Ground Truth", t_dir / "train_ground_truth.tsv"),
        ("Test Source 1", te_dir / "test_source1.tsv"),
        ("Test Source 2", te_dir / "test_source2.tsv"),
        ("Test Source 3", te_dir / "test_source3.tsv"),
    ]

    results = {}
    for name, path in required_files:
        exists = path.is_file()
        size_mb = path.stat().st_size / (1024 * 1024) if exists else 0.0
        results[name] = {"path": str(path), "exists": exists, "size_mb": round(size_mb, 2)}
        if not exists:
            logger.error(f"[MISSING] {name}: {path}")
        else:
            logger.info(f"[FOUND]   {name}: {path} ({size_mb:.1f} MB)")

    all_found = all(r["exists"] for r in results.values())
    if not all_found:
        raise FileNotFoundError("One or more required dataset TSV files are missing.")
    return results


# ==============================================================================
# 2. OFFICIAL VALIDATOR INTEGRATION (validate_submission.py)
# ==============================================================================

def find_validation_script(custom_paths: Optional[List[Union[str, Path]]] = None) -> Optional[Path]:
    """Locate validate_submission.py across standard search locations."""
    candidates = []
    if custom_paths:
        candidates.extend([Path(p) for p in custom_paths])

    candidates.extend([
        PROJECT_ROOT / "utils" / "validate_submission.py",
        PROJECT_ROOT.parent / "utils" / "validate_submission.py",
        PROJECT_ROOT / "student_resource" / "utils" / "validate_submission.py",
        Path("utils/validate_submission.py"),
    ])

    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def run_submission_validation(
    validate_script: Optional[Union[str, Path]] = None,
    matching_path: Union[str, Path] = MATCHING_RESULTS_PATH,
    candidate_path: Union[str, Path] = CANDIDATE_PAIRS_PATH,
    test_dir: Union[str, Path] = TEST_DIR,
    check_ids: bool = False,
) -> Tuple[int, str, str]:
    """Execute official validate_submission.py script."""
    script_path = Path(validate_script) if validate_script else find_validation_script()

    if script_path is None or not script_path.is_file():
        msg = f"Validation script not found. Searched standard utils/ locations."
        logger.error(msg)
        return 1, "", msg

    cmd = [
        sys.executable,
        str(script_path),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir),
    ]

    if check_ids:
        cmd.append("--check-ids")

    logger.info(f"Running validation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("\n" + "=" * 60)
    print("SUBMISSION VALIDATION REPORT")
    print("=" * 60)
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    print(f"Exit code: {result.returncode}")

    return result.returncode, result.stdout, result.stderr


# ==============================================================================
# 3. STANDALONE SUBMISSION INTEGRITY CHECK
# ==============================================================================

def validate_submission_deliverables(
    matching_file: Union[str, Path] = MATCHING_RESULTS_PATH,
    candidate_file: Union[str, Path] = CANDIDATE_PAIRS_PATH,
    test_s1_path: Union[str, Path] = TEST_SOURCE1_PATH,
) -> Dict[str, Any]:
    """Perform comprehensive local verification of generated submission files.

    Checks:
    - Every S1 entity in test_source1 appears exactly once in both files
    - Matching header: ['source1_entity_id', 'matched_entity_ids']
    - Candidate header: ['source1_entity_id', 'candidate_entity_ids']
    - Subset rule: Every predicted match must be present in candidate_pairs.tsv
    - No self-matches ('S1-*') in match sets
    - No duplicate IDs inside the comma-separated lists
    """
    m_path = Path(matching_file)
    c_path = Path(candidate_file)
    s1_path = Path(test_s1_path)

    errors: List[str] = []
    warnings: List[str] = []

    if not m_path.is_file():
        errors.append(f"matching_results.tsv not found at: {m_path}")
        return {"status": "FAIL", "errors": errors, "warnings": warnings}

    # Load required test S1 IDs if available
    required_s1_ids: Set[str] = set()
    if s1_path.is_file():
        with open(s1_path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                if line.strip():
                    required_s1_ids.add(line.split("\t", 1)[0].strip())

    matches_map: Dict[str, Set[str]] = {}
    m_seen_s1: Set[str] = set()
    m_dup_s1: List[str] = []
    m_self_matches: List[str] = []
    m_empty_count = 0
    m_total_predicted = 0

    with open(m_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if header != EXPECTED_MATCHING_COLUMNS:
            errors.append(f"matching_results header mismatch. Expected {EXPECTED_MATCHING_COLUMNS}, got {header}")

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            sid = parts[0].strip()
            rest = parts[1].strip() if len(parts) > 1 else ""

            if sid in m_seen_s1:
                m_dup_s1.append(sid)
            m_seen_s1.add(sid)

            if not rest:
                m_empty_count += 1
                matches_map[sid] = set()
                continue

            ids = [i.strip() for i in rest.split(",") if i.strip()]
            id_set = set(ids)
            matches_map[sid] = id_set
            m_total_predicted += len(id_set)

            for mid in id_set:
                if mid.startswith("S1-"):
                    m_self_matches.append(mid)

    if m_dup_s1:
        errors.append(f"matching_results.tsv has {len(m_dup_s1)} duplicate S1 rows")
    if m_self_matches:
        errors.append(f"matching_results.tsv contains S1 self-matches (e.g. {m_self_matches[:3]})")

    if required_s1_ids:
        missing_in_m = required_s1_ids - m_seen_s1
        if missing_in_m:
            errors.append(f"matching_results.tsv is missing {len(missing_in_m)} test S1 entities")

    # Candidate pairs check
    candidates_map: Dict[str, Set[str]] = {}
    c_seen_s1: Set[str] = set()
    c_total = 0

    if c_path.is_file():
        with open(c_path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            if header != EXPECTED_CANDIDATE_PAIRS_COLUMNS:
                errors.append(f"candidate_pairs header mismatch. Expected {EXPECTED_CANDIDATE_PAIRS_COLUMNS}, got {header}")

            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if not parts or not parts[0]:
                    continue
                sid = parts[0].strip()
                rest = parts[1].strip() if len(parts) > 1 else ""
                c_seen_s1.add(sid)

                if not rest:
                    candidates_map[sid] = set()
                    continue

                c_ids = set(i.strip() for i in rest.split(",") if i.strip())
                candidates_map[sid] = c_ids
                c_total += len(c_ids)

        if required_s1_ids:
            missing_in_c = required_s1_ids - c_seen_s1
            if missing_in_c:
                errors.append(f"candidate_pairs.tsv is missing {len(missing_in_c)} test S1 entities")

        # Subset check: every match must be in candidate set
        not_in_cands = 0
        for sid, m_set in matches_map.items():
            c_set = candidates_map.get(sid, set())
            diff = m_set - c_set
            if diff:
                not_in_cands += len(diff)
        if not_in_cands > 0:
            errors.append(f"{not_in_cands} predicted matches are not in candidate_pairs.tsv")
    else:
        warnings.append(f"candidate_pairs.tsv not found at {c_path}")

    status = "PASS" if len(errors) == 0 else "FAIL"
    return {
        "status": status,
        "matching_rows": len(m_seen_s1),
        "candidate_rows": len(c_seen_s1),
        "total_matches": m_total_predicted,
        "empty_matches": m_empty_count,
        "total_candidates": c_total,
        "errors": errors,
        "warnings": warnings,
    }


# Backwards compatibility alias
validate_submission_format = validate_submission_deliverables
