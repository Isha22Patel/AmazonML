"""Main CLI entrypoint for the Amazon ML Business Entity Resolution Pipeline.

Located under src/ to comply with competition structure requirements.
Supports running individual pipeline stages or the full end-to-end workflow:
  python src/main.py --all
  python src/main.py --stage [eda|blocking|evaluate|inference|validate|package]
"""

import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent
_PKG_DIR = _SRC_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import argparse
import logging
from pathlib import Path
import sys

# Configure UTF-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# Add parent of src to sys.path so imports work regardless of cwd
CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CURRENT_DIR.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from src.config import (
    CANDIDATE_PAIRS_PATH,
    DATASET_DIR,
    DEV_SAMPLE_SIZE,
    MATCHING_RESULTS_PATH,
    OPTIMAL_THRESHOLD,
    OUTPUT_DIR,
    REPORTS_DIR,
    TEST_DIR,
    TEST_SOURCE1_PATH,
    TRAIN_DIR,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
)
from src.data_loader import (
    load_ground_truth,
    load_source1,
    load_source2,
    load_source3,
)
from src.eda import run_eda
from src.pipeline import run_offline_evaluation, run_test_inference
from src.submission import create_submission_zip
from src.validation import run_submission_validation, validate_submission_deliverables


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge: Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute the complete end-to-end pipeline (EDA, training, evaluation, test inference, validation, and packaging)",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Alias for full production run (forces test_limit=None and executes full test inference)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["eda", "evaluate", "inference", "validate", "package"],
        default="all",
        help="Pipeline stage to run: eda, evaluate, inference, validate, package",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEV_SAMPLE_SIZE,
        help=f"Sample size of Source 1 for offline evaluation (default: {DEV_SAMPLE_SIZE:,})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification decision threshold (default: None, loads calibrated threshold from model artifact)",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Limit number of test records for quick sanity testing (allowed only in dev mode; disallowed with --all/--production)",
    )
    parser.add_argument(
        "--team-name",
        type=str,
        default="EntityResolution_Champions",
        help="Team name for packaging submission zip (default: EntityResolution_Champions)",
    )

    args = parser.parse_args()

    # Enforce safe separation between DEV MODE and FINAL FULL MODE
    is_full_mode = args.all or (args.production and args.stage is None)
    if (args.all or args.production) and args.test_limit is not None:
        raise ValueError(
            "Conflict: --all/--production cannot be combined with --test-limit. "
            "For full competition submission, the entire test dataset must be processed. "
            "For quick dev testing, run: python src/main.py --stage inference --test-limit <N>"
        )

    test_limit = None if (args.all or args.production) else args.test_limit

    # Stage: EDA
    if args.stage == "eda" or is_full_mode:
        print("\n" + "=" * 60)
        print("STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 60)
        if TRAIN_SOURCE1_PATH.is_file():
            t_s1 = load_source1(TRAIN_SOURCE1_PATH, nrows=50_000)
            t_s2 = load_source2(TRAIN_SOURCE2_PATH, nrows=50_000)
            t_s3 = load_source3(TRAIN_SOURCE3_PATH, nrows=50_000)
            t_gt = load_ground_truth(TRAIN_GROUND_TRUTH_PATH, nrows=50_000)
            eda_stats, eda_text = run_eda(
                train_s1=t_s1, train_s2=t_s2, train_s3=t_s3, train_gt=t_gt,
                output_path=REPORTS_DIR / "eda_report.txt",
            )
            print(eda_text)
        else:
            print(f"[NOTE] Training datasets not present at {TRAIN_DIR}. Skipping local EDA execution.")
        if not is_full_mode:
            return 0

    # Stage: Evaluate (Offline Training, Model Benchmark & Threshold Search)
    trained_model = None
    calibrated_threshold = args.threshold

    if args.stage == "evaluate" or is_full_mode:
        print("\n" + "=" * 60)
        print("STAGE 2: OFFLINE EVALUATION & MODEL BENCHMARK")
        print("=" * 60)
        if TRAIN_SOURCE1_PATH.is_file() and TRAIN_GROUND_TRUTH_PATH.is_file():
            res = run_offline_evaluation(sample_size=args.sample_size)
            trained_model = res["best_model"]
            if calibrated_threshold is None:
                calibrated_threshold = res["optimal_threshold"]
            print(f"\nEvaluation complete. Winning model '{res['best_model_name']}' selected at threshold {res['optimal_threshold']}.")
        else:
            print(f"[NOTE] Training data not present at {TRAIN_DIR}. Skipping offline training.")
        if not is_full_mode:
            return 0

    # Stage: Inference (Test Blocking + Model Scoring)
    if args.stage == "inference" or is_full_mode:
        print("\n" + "=" * 60)
        print("STAGE 3: TEST INFERENCE & DELIVERABLE GENERATION")
        print("=" * 60)
        if TEST_SOURCE1_PATH.is_file():
            m_path, c_path = run_test_inference(
                trained_model=trained_model,
                optimal_threshold=calibrated_threshold,
                test_limit=test_limit,
            )
            print(f"Deliverables generated:\n  - {m_path}\n  - {c_path}")
        else:
            print(f"[NOTE] Test source not present at {TEST_SOURCE1_PATH}. Deliverables remain as previously generated.")
        if not is_full_mode:
            return 0

    # Stage: Validate
    if args.stage == "validate" or args.all:
        print("\n" + "=" * 60)
        print("STAGE 4: SUBMISSION DELIVERABLES VALIDATION")
        print("=" * 60)
        report = validate_submission_deliverables(
            matching_file=MATCHING_RESULTS_PATH,
            candidate_file=CANDIDATE_PAIRS_PATH,
            test_s1_path=TEST_SOURCE1_PATH,
        )
        print(f"Standalone Integrity Status: {report['status']}")
        print(f"  - Matching rows:     {report['matching_rows']:,}")
        print(f"  - Candidate rows:    {report['candidate_rows']:,}")
        print(f"  - Total matches:     {report['total_matches']:,}")
        print(f"  - Empty (singleton): {report['empty_matches']:,}")
        if report["errors"]:
            print(f"  - Errors: {report['errors']}")

        rc, _, _ = run_submission_validation(
            matching_path=MATCHING_RESULTS_PATH,
            candidate_path=CANDIDATE_PAIRS_PATH,
            test_dir=TEST_DIR,
        )
        if not args.all:
            return rc

    # Stage: Package Submission Zip
    if args.stage == "package" or args.all:
        print("\n" + "=" * 60)
        print("STAGE 5: PACKAGING OFFICIAL SUBMISSION ZIP")
        print("=" * 60)
        zip_path = create_submission_zip(team_name=args.team_name)
        print(f"[SUCCESS] Official competition package created: {zip_path}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
