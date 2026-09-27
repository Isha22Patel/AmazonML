"""Submission generation and packaging module for Amazon ML Business Entity Resolution.

Handles generation of the official submission deliverables and creates
the validated competition zip package following organizer rules:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   └── [production source files & notebook]
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
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
from typing import Dict, List, Optional, Set, Tuple, Union
import logging
import zipfile
import pandas as pd

from src.config import (
    CANDIDATE_PAIRS_PATH,
    MATCHING_RESULTS_PATH,
    OUTPUT_DIR,
    PROJECT_ROOT,
    TSV_DELIMITER,
)
from src.validation import validate_submission_deliverables

logger = logging.getLogger(__name__)


def write_submission_deliverables(
    all_s1_ids: List[str],
    predictions_map: Dict[str, List[str]],
    candidates_map: Dict[str, List[str]],
    matching_path: Union[str, Path] = MATCHING_RESULTS_PATH,
    candidate_path: Union[str, Path] = CANDIDATE_PAIRS_PATH,
) -> Tuple[Path, Path]:
    """Write matching_results.tsv and candidate_pairs.tsv ensuring 100% S1 coverage.

    Guarantees:
    - Every S1 entity in all_s1_ids appears exactly once in both files
    - Empty match list for singletons/unmatched entities (tab followed by newline)
    - Comma-separated list with NO quotes
    - Every predicted match is strictly present in candidate_pairs.tsv
    """
    m_p = Path(matching_path)
    c_p = Path(candidate_path)
    m_p.parent.mkdir(parents=True, exist_ok=True)
    c_p.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing {m_p.name} ({len(all_s1_ids):,} entities)...")
    logger.info(f"Writing {c_p.name} ({len(all_s1_ids):,} entities)...")

    total_matches = 0
    total_candidates = 0
    empty_matches = 0

    with open(m_p, "w", encoding="utf-8") as f_match, open(c_p, "w", encoding="utf-8") as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for sid in all_s1_ids:
            matches = predictions_map.get(sid, [])
            cands = candidates_map.get(sid, [])

            # Strict candidate-pair semantics: Every predicted match must be in candidate set
            cand_set = set(cands)
            missing = set(matches) - cand_set
            if missing:
                raise ValueError(
                    f"Pipeline consistency violation for entity {sid}: "
                    f"{len(missing)} predicted matches were not in the candidate set! "
                    f"Sample offending matches: {list(missing)[:5]}"
                )

            # Write matching row
            if matches:
                total_matches += len(matches)
                f_match.write(f"{sid}\t{','.join(matches)}\n")
            else:
                empty_matches += 1
                f_match.write(f"{sid}\t\n")

            # Write candidate row
            if cands:
                total_candidates += len(cands)
                f_cand.write(f"{sid}\t{','.join(cands)}\n")
            else:
                f_cand.write(f"{sid}\t\n")

    logger.info(
        f"Deliverables written successfully: {len(all_s1_ids):,} S1 entities | "
        f"{total_matches:,} matches ({empty_matches:,} singletons) | {total_candidates:,} candidates"
    )
    return m_p, c_p


def create_submission_zip(
    team_name: str = "EntityResolution_Champions",
    output_zip_path: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Package the final clean project into the official submission zip.

    Structure:
    <team_name>_submission/
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/
    │       ├── README.md
    │       └── requirements.txt
    └── Documentation_template.md
    """
    root = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    zip_dest = Path(output_zip_path) if output_zip_path else (root.parent / f"{team_name}_submission.zip")

    logger.info(f"Creating competition submission package: {zip_dest}...")

    # Files to include in zip archive
    files_to_pack = [
        # Output deliverables
        (root / "output" / "matching_results.tsv", "output/matching_results.tsv"),
        (root / "output" / "candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        # Documentation
        (root / "Documentation_template.md", "Documentation_template.md"),
        # Code package
        (root / "code" / "business_entity_resolution" / "README.md", "code/business_entity_resolution/README.md"),
        (root / "code" / "business_entity_resolution" / "requirements.txt", "code/business_entity_resolution/requirements.txt"),
    ]

    # Add all clean source files and notebook under src/
    src_dir = root / "code" / "business_entity_resolution" / "src"
    allowed_exts = {".py", ".ipynb"}
    for src_file in sorted(src_dir.glob("*")):
        if src_file.is_file() and src_file.suffix in allowed_exts and not src_file.name.endswith(".pyc") and "__pycache__" not in src_file.name:
            rel_in_zip = f"code/business_entity_resolution/src/{src_file.name}"
            files_to_pack.append((src_file, rel_in_zip))

    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, arcname in files_to_pack:
            if file_path.is_file():
                zf.write(file_path, arcname=arcname)
                logger.info(f"  Added: {arcname} ({file_path.stat().st_size:,} bytes)")
            else:
                logger.warning(f"  Missing file for zip: {file_path}")

    logger.info(f"Submission zip successfully packaged: {zip_dest}")
    return zip_dest
