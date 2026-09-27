"""Data loading and streaming ingestion module for Amazon ML Business Entity Resolution.

Supports chunked streaming over multi-million row TSV datasets, schema validation,
and PySpark DataFrame loading for large-scale distributed execution.
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
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union
import logging
import pandas as pd

from src.config import (
    DATASET_DIR,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    TSV_DELIMITER,
)
from src.validation import (
    validate_file_exists,
    validate_ground_truth_schema,
    validate_source_schema,
)

logger = logging.getLogger(__name__)


def load_tsv(
    path: Union[str, Path],
    name: str = "Dataset",
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Read a tab-separated (.tsv) file into a pandas DataFrame."""
    file_path = validate_file_exists(path, description=name)
    logger.info(f"Loading {name} from {file_path.name}...")
    df = pd.read_csv(
        file_path,
        sep=TSV_DELIMITER,
        nrows=nrows,
        usecols=usecols,
        encoding="utf-8",
        dtype=str,
        na_filter=False,
    )
    return df


def stream_tsv_chunks(
    path: Union[str, Path],
    chunksize: int = 500_000,
    name: str = "Dataset",
    usecols: Optional[List[str]] = None,
) -> Iterator[pd.DataFrame]:
    """Stream a massive TSV file in memory-safe chunks using tab-separation."""
    file_path = validate_file_exists(path, description=name)
    logger.info(f"Streaming {name} from {file_path.name} (chunksize={chunksize:,})...")
    for chunk in pd.read_csv(
        file_path,
        sep=TSV_DELIMITER,
        chunksize=chunksize,
        usecols=usecols,
        encoding="utf-8",
        dtype=str,
        na_filter=False,
    ):
        yield chunk


def load_source1(
    path: Optional[Union[str, Path]] = None,
    nrows: Optional[int] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load Source 1 TSV file (Reference / Anchor businesses)."""
    target_path = Path(path) if path is not None else TRAIN_SOURCE1_PATH
    df = load_tsv(target_path, name=f"Source 1 ({target_path.name})", nrows=nrows)
    if validate:
        validate_source_schema(df, dataset_name=f"Source1 ({target_path.name})")
    return df


def load_source2(
    path: Optional[Union[str, Path]] = None,
    nrows: Optional[int] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load Source 2 TSV file (First candidate source)."""
    target_path = Path(path) if path is not None else TRAIN_SOURCE2_PATH
    df = load_tsv(target_path, name=f"Source 2 ({target_path.name})", nrows=nrows)
    if validate:
        validate_source_schema(df, dataset_name=f"Source2 ({target_path.name})")
    return df


def load_source3(
    path: Optional[Union[str, Path]] = None,
    nrows: Optional[int] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load Source 3 TSV file (Second candidate source)."""
    target_path = Path(path) if path is not None else TRAIN_SOURCE3_PATH
    df = load_tsv(target_path, name=f"Source 3 ({target_path.name})", nrows=nrows)
    if validate:
        validate_source_schema(df, dataset_name=f"Source3 ({target_path.name})")
    return df


def load_ground_truth(
    path: Optional[Union[str, Path]] = None,
    nrows: Optional[int] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load Ground Truth TSV file."""
    target_path = Path(path) if path is not None else TRAIN_GROUND_TRUTH_PATH
    df = load_tsv(target_path, name=f"Ground Truth ({target_path.name})", nrows=nrows)
    if validate:
        validate_ground_truth_schema(df, dataset_name=f"GroundTruth ({target_path.name})")
    return df


def load_ground_truth_map(
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Set[str]]:
    """Parse Ground Truth TSV directly into a fast in-memory dictionary:
    {s1_id: set(matched_entity_ids)}.
    """
    target_path = Path(path) if path is not None else TRAIN_GROUND_TRUTH_PATH
    validate_file_exists(target_path, description="Ground Truth")
    logger.info(f"Building Ground Truth mapping from {target_path.name}...")

    gt_map: Dict[str, Set[str]] = {}
    with open(target_path, "r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 1 and parts[0].strip():
                s1_id = parts[0].strip()
                if len(parts) > 1 and parts[1].strip():
                    m_ids = {m.strip() for m in parts[1].split(",") if m.strip()}
                    gt_map[s1_id] = m_ids
                else:
                    gt_map[s1_id] = set()
    logger.info(f"Loaded ground truth mapping for {len(gt_map):,} Source 1 entities.")
    return gt_map


def load_training_data(
    nrows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all training datasets: train_source1, train_source2, train_source3, train_ground_truth."""
    train_s1 = load_source1(TRAIN_SOURCE1_PATH, nrows=nrows)
    train_s2 = load_source2(TRAIN_SOURCE2_PATH, nrows=nrows)
    train_s3 = load_source3(TRAIN_SOURCE3_PATH, nrows=nrows)
    train_gt = load_ground_truth(TRAIN_GROUND_TRUTH_PATH, nrows=nrows)
    return train_s1, train_s2, train_s3, train_gt


def load_test_data(
    nrows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all test datasets: test_source1, test_source2, test_source3."""
    test_s1 = load_source1(TEST_SOURCE1_PATH, nrows=nrows)
    test_s2 = load_source2(TEST_SOURCE2_PATH, nrows=nrows)
    test_s3 = load_source3(TEST_SOURCE3_PATH, nrows=nrows)
    return test_s1, test_s2, test_s3


# ------------------------------------------------------------------------------
# PySpark Ingestion Support (Big Data / Databricks Compatibility)
# ------------------------------------------------------------------------------
def load_all_spark_datasets(spark: Any, data_dir: Union[str, Path] = DATASET_DIR) -> Dict[str, Any]:
    """Load all TSVs into Spark DataFrames for distributed runs."""
    d = Path(data_dir)
    train_d = d / "train" if (d / "train").is_dir() else d
    test_d = d / "test" if (d / "test").is_dir() else d

    paths = {
        "train_s1": train_d / "train_source1.tsv",
        "train_s2": train_d / "train_source2.tsv",
        "train_s3": train_d / "train_source3.tsv",
        "train_gt": train_d / "train_ground_truth.tsv",
        "test_s1": test_d / "test_source1.tsv",
        "test_s2": test_d / "test_source2.tsv",
        "test_s3": test_d / "test_source3.tsv",
    }

    dfs = {}
    for k, p in paths.items():
        if p.is_file():
            p_str = str(p).replace("\\", "/")
            dfs[k] = (
                spark.read
                .option("header", True)
                .option("sep", TSV_DELIMITER)
                .option("escape", '"')
                .csv(p_str)
            )
    return dfs
