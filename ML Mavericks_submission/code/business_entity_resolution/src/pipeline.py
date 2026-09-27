"""Comprehensive Evaluation, Model Comparison, Ablation, and Inference Pipeline.

Orchestrates the complete Amazon ML Business Entity Resolution workflow:
1. Deterministic Entity-Level Train/Val Split with zero data leakage
2. Multi-Strategy Inverted-Index Candidate Generation with posting caps
3. 17-Dimensional Pairwise Feature Extraction
4. Comparative Model Benchmark: Rule Baseline, Blocking Baseline, Logistic Regression,
   LightGBM GBDT, and Random Forest
5. Threshold Grid Calibration evaluated strictly on Entity Macro F0.5
6. Dynamic winning model selection and final retraining on all labeled training pairs
7. Memory-safe country-partitioned streaming test inference covering 100% of test Source 1 entities
8. Stale output protection, atomic deliverables replacement, and hard integrity assertions
9. Official validate_submission.py validation execution.
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
import gc
import json
import logging
import os
import time
import joblib
import numpy as np
import pandas as pd

from src.blocking import MultiStrategyBlockingIndex, block_exact_name, block_token_name
from src.config import (
    BEST_LIGHTGBM_MODEL_PATH,
    BEST_MODEL_PATH,
    BLOCKING_COMPARISON_PATH,
    CANDIDATE_PAIRS_PATH,
    CANDIDATE_PAIRS_TMP_PATH,
    DEV_CANDIDATE_PAIRS_PATH,
    DEV_SAMPLE_SIZE,
    MATCHING_RESULTS_PATH,
    MATCHING_RESULTS_TMP_PATH,
    MODEL_COMPARISON_PATH,
    MODELS_DIR,
    OPTIMAL_THRESHOLD,
    OUTPUT_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TEST_DIR,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    THRESHOLD_COMPARISON_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    VAL_RATIO,
)
from src.evaluation import (
    compute_entity_level_metrics,
    compute_pairwise_diagnostic_metrics,
)
from src.features import FEATURE_NAMES, batch_compute_features, compute_pair_features
from src.matching import predict_matches_batch, predict_matches_for_s1
from src.models import (
    BlockingOnlyBaselineModel,
    LightGBMMatcher,
    LogisticRegressionMatcher,
    RandomForestMatcher,
    RuleBaselineModel,
)
from src.normalize import (
    extract_numbers,
    normalize_address,
    normalize_name,
    remove_legal_suffixes,
)
from src.submission import write_submission_deliverables
from src.threshold import evaluate_threshold_grid
from src.validation import run_submission_validation, validate_submission_deliverables

logger = logging.getLogger("pipeline")


# ------------------------------------------------------------------------------
# 1. Deterministic Zero-Leakage Entity Split
# ------------------------------------------------------------------------------
def load_ground_truth_map(gt_path: Path = TRAIN_GROUND_TRUTH_PATH) -> Dict[str, Set[str]]:
    """Load Ground Truth TSV into mapping of {s1_id: set(matched_ids)}."""
    logger.info(f"Loading ground truth mapping from {gt_path.name}...")
    gt_map: Dict[str, Set[str]] = {}
    if not gt_path.is_file():
        logger.warning(f"Ground truth file missing: {gt_path}")
        return gt_map

    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=1_000_000, dtype=str, na_filter=False):
        for sid, val in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            if val and str(val).strip():
                gt_map[str(sid)] = set(str(val).split(","))
            else:
                gt_map[str(sid)] = set()
    logger.info(f"Loaded ground truth for {len(gt_map):,} Source 1 entities.")
    return gt_map


def deterministic_dev_split(
    dev_s1_ids: List[str],
    gt_map: Dict[str, Set[str]],
    val_ratio: float = VAL_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[str]]:
    """Split Source 1 entities into Train and Validation at the ENTITY LEVEL with zero leakage.

    Stratified by match-cardinality tier (singletons, 1-match, multi-matches).
    """
    rng = np.random.RandomState(random_seed)

    # Bin entities by cardinality tier
    strata: Dict[str, List[str]] = defaultdict(list)
    for sid in dev_s1_ids:
        matches = gt_map.get(sid, set())
        n = len(matches)
        tier = "0" if n == 0 else ("1" if n == 1 else "multi")
        strata[tier].append(sid)

    train_ids: List[str] = []
    val_ids: List[str] = []

    for tier, ids in strata.items():
        shuffled = np.array(ids)
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * val_ratio))
        val_ids.extend(shuffled[:n_val].tolist())
        train_ids.extend(shuffled[n_val:].tolist())

    leakage = set(train_ids).intersection(set(val_ids))
    assert len(leakage) == 0, f"Entity data leakage detected: {len(leakage)} entities!"

    logger.info(
        f"Entity-level split: Train={len(train_ids):,} ({len(train_ids)/len(dev_s1_ids)*100:.1f}%), "
        f"Val={len(val_ids):,} ({len(val_ids)/len(dev_s1_ids)*100:.1f}%). Zero leakage verified."
    )
    return sorted(train_ids), sorted(val_ids)


def load_candidate_pairs_by_s1(tsv_path: Union[str, Path] = DEV_CANDIDATE_PAIRS_PATH) -> Dict[str, List[str]]:
    """Load blocked candidate pairs mapping {s1_id: [cand_id_1, cand_id_2, ...]}."""
    p = Path(tsv_path)
    logger.info(f"Loading candidate pairs from {p.name}...")
    cands_by_s1: Dict[str, List[str]] = {}
    if not p.is_file():
        logger.warning(f"Candidate pairs file missing: {p}")
        return cands_by_s1

    with open(p, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            sid = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                cands_by_s1[sid] = [c.strip() for c in parts[1].split(",") if c.strip()]
            else:
                cands_by_s1[sid] = []
    logger.info(f"Loaded candidates for {len(cands_by_s1):,} S1 entities.")
    return cands_by_s1


# ------------------------------------------------------------------------------
# 2. Complete Offline Evaluation Framework
# ------------------------------------------------------------------------------
def run_offline_evaluation(
    sample_size: int = DEV_SAMPLE_SIZE,
    val_ratio: float = VAL_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Execute complete offline evaluation benchmark across all models and threshold search."""
    logger.info("=" * 60)
    logger.info("STARTING OFFLINE EVALUATION & MODEL BENCHMARK")
    logger.info("=" * 60)

    # 1. Load Ground Truth Map
    gt_map = load_ground_truth_map(TRAIN_GROUND_TRUTH_PATH)
    if not gt_map:
        raise FileNotFoundError(f"Cannot run offline evaluation: Ground truth missing at {TRAIN_GROUND_TRUTH_PATH}")

    # 2. Load Sample Source 1
    logger.info(f"Loading {sample_size:,} Source 1 entities for offline evaluation...")
    s1_df = pd.read_csv(
        TRAIN_SOURCE1_PATH,
        sep="\t",
        nrows=sample_size,
        dtype=str,
        na_filter=False,
    )
    dev_s1_ids = s1_df["entity_id"].tolist()

    # 3. Deterministic Entity Split
    train_s1_ids, val_s1_ids = deterministic_dev_split(
        dev_s1_ids, gt_map, val_ratio=val_ratio, random_seed=random_seed
    )
    val_s1_set = set(val_s1_ids)
    # Ensure val_s1_ids strictly matches the iteration order of dev_s1_ids for exact index alignment
    val_s1_ids = [sid for sid in dev_s1_ids if sid in val_s1_set]

    # 4. Inverted Index Blocking Engine
    logger.info("Indexing Candidate Sources (S2 + S3) with multi-strategy blocking index...")
    blocking_index = MultiStrategyBlockingIndex()

    for path in [TRAIN_SOURCE2_PATH, TRAIN_SOURCE3_PATH]:
        if not path.is_file():
            continue
        logger.info(f"  Indexing {path.name}...")
        for chunk in pd.read_csv(path, sep="\t", chunksize=1_000_000, dtype=str, na_filter=False):
            blocking_index.index_candidates_chunk(
                chunk["entity_id"].tolist(),
                chunk["business_name"].tolist(),
                chunk["business_address"].tolist(),
                chunk["country"].tolist(),
            )

    # 5. Extract Candidate Pairs & Featurize
    logger.info("Retrieving candidates and extracting 17 similarity features...")
    s1_dict = s1_df.set_index("entity_id").to_dict("index")

    X_train_list, y_train_list = [], []
    val_cand_ids_per_s1: List[List[str]] = []
    val_X_per_s1: List[np.ndarray] = []

    # Measure blocking metrics on validation set
    val_found_matches = 0
    val_total_gt_matches = sum(len(gt_map.get(sid, set())) for sid in val_s1_ids)
    val_cand_counts = []
    val_zero_cands = 0

    for sid in dev_s1_ids:
        s1_row = s1_dict[sid]
        s1_n = normalize_name(s1_row["business_name"])
        s1_a = normalize_address(s1_row["business_address"])
        s1_c = str(s1_row["country"]).lower().strip()

        cand_records = blocking_index.retrieve_candidate_records(s1_n, s1_a, s1_c)
        cand_ids = [r[0] for r in cand_records]
        true_matches = gt_map.get(sid, set())

        # Featurize
        pair_feats = []
        for cid, c_name, c_addr, c_cntry in cand_records:
            feat = compute_pair_features(s1_n, c_name, s1_a, c_addr, s1_c, c_cntry)
            pair_feats.append(feat)

        if sid in val_s1_set:
            val_cand_ids_per_s1.append(cand_ids)
            val_X_per_s1.append(np.array(pair_feats, dtype=np.float32) if pair_feats else np.empty((0, 17)))
            val_cand_counts.append(len(cand_ids))
            if len(cand_ids) == 0:
                val_zero_cands += 1
            if true_matches:
                val_found_matches += len(true_matches.intersection(set(cand_ids)))
        else:
            for cid, feat in zip(cand_ids, pair_feats):
                lbl = 1 if cid in true_matches else 0
                X_train_list.append(feat)
                y_train_list.append(lbl)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    logger.info(f"Training dataset: {len(X_train):,} pairs (Positives: {int(y_train.sum()):,})")

    # Log measured candidate recall
    cand_recall = (val_found_matches / val_total_gt_matches * 100.0) if val_total_gt_matches > 0 else 0.0
    c_arr = np.array(val_cand_counts) if val_cand_counts else np.array([0])
    logger.info(
        f"Validation Candidate Recall: {cand_recall:.2f}% | Mean Cands: {np.mean(c_arr):.2f} | "
        f"Median: {int(np.median(c_arr))} | P95: {int(np.percentile(c_arr, 95))} | "
        f"Zero-cand rate: {val_zero_cands / len(val_s1_ids) * 100.0:.2f}%"
    )

    # 6. Train Models and Evaluate Across Threshold Grid
    models = {
        "Rule Baseline (Exact Name)": RuleBaselineModel(),
        "Blocking-Only Baseline": BlockingOnlyBaselineModel(),
        "Logistic Regression": LogisticRegressionMatcher(random_state=random_seed),
        "LightGBM GBDT": LightGBMMatcher(random_state=random_seed),
        "Random Forest": RandomForestMatcher(random_state=random_seed),
    }

    model_results = []
    trained_model_objs = {}
    model_threshold_dfs = {}
    val_ground_truth = {sid: gt_map.get(sid, set()) for sid in val_s1_ids}

    for name, model in models.items():
        logger.info(f"Fitting {name}...")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = round(time.perf_counter() - t0, 2)
        trained_model_objs[name] = model

        # Inference on validation set
        t0_inf = time.perf_counter()
        val_probs_per_s1 = [
            model.predict_proba(X)[:, 1] if len(X) > 0 else np.array([])
            for X in val_X_per_s1
        ]
        inf_time = round(time.perf_counter() - t0_inf, 2)

        # Threshold search (strictly on validation ground truth optimizing Macro F0.5)
        df_thresh, opt_thresh = evaluate_threshold_grid(
            val_s1_ids=val_s1_ids,
            val_cand_ids_per_s1=val_cand_ids_per_s1,
            val_probs_per_s1=val_probs_per_s1,
            ground_truth=val_ground_truth,
        )
        model_threshold_dfs[name] = (df_thresh, opt_thresh)

        opt_row = df_thresh[df_thresh["threshold"] == round(opt_thresh, 2)].iloc[0]
        model_results.append({
            "Model": name,
            "Optimal Threshold": opt_thresh,
            "Entity Macro F0.5": opt_row["entity_macro_f05"],
            "Entity Precision": opt_row["entity_macro_precision"],
            "Entity Recall": opt_row["entity_macro_recall"],
            "Train Runtime (s)": train_time,
            "Inference Runtime (s)": inf_time,
        })

    df_models = pd.DataFrame(model_results)
    logger.info("\n" + df_models.to_string(index=False))

    # Save benchmark tables to REPORTS_DIR
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_models.to_csv(MODEL_COMPARISON_PATH, index=False)
    logger.info(f"Saved model comparison table to {MODEL_COMPARISON_PATH}")

    # 7. Select winning model strictly based on validation Entity Macro F0.5
    best_row = max(model_results, key=lambda x: x["Entity Macro F0.5"])
    best_model_name = best_row["Model"]
    best_opt_thresh = best_row["Optimal Threshold"]
    best_f05 = best_row["Entity Macro F0.5"]
    logger.info(
        f"Winning model selected by Validation Entity Macro F0.5: '{best_model_name}' "
        f"(Entity Macro F0.5 = {best_f05:.4f}, Optimal Threshold = {best_opt_thresh})"
    )

    best_df_thresh, _ = model_threshold_dfs[best_model_name]
    best_df_thresh.to_csv(THRESHOLD_COMPARISON_PATH, index=False)
    logger.info(f"Saved threshold comparison table to {THRESHOLD_COMPARISON_PATH}")

    # 8. Retrain winning model on ALL available labeled pairs in sample
    logger.info(f"Retraining winning model architecture '{best_model_name}' on all labeled training pairs...")
    if best_model_name == "LightGBM GBDT":
        final_model = LightGBMMatcher(random_state=random_seed)
    elif best_model_name == "Random Forest":
        final_model = RandomForestMatcher(random_state=random_seed)
    elif best_model_name == "Logistic Regression":
        final_model = LogisticRegressionMatcher(random_state=random_seed)
    elif best_model_name == "Rule Baseline (Exact Name)":
        final_model = RuleBaselineModel()
    else:
        final_model = BlockingOnlyBaselineModel()

    # Re-extract all training pairs from the full sample (train + val)
    X_all_list, y_all_list = list(X_train_list), list(y_train_list)
    for sid, cand_ids, val_X in zip(val_s1_ids, val_cand_ids_per_s1, val_X_per_s1):
        true_matches = gt_map.get(sid, set())
        for cid, feat in zip(cand_ids, val_X):
            lbl = 1 if cid in true_matches else 0
            X_all_list.append(feat)
            y_all_list.append(lbl)

    X_all = np.array(X_all_list, dtype=np.float32)
    y_all = np.array(y_all_list, dtype=np.int32)
    final_model.fit(X_all, y_all)

    # Save model artifact with threshold and metadata
    model_artifact = {
        "model": final_model,
        "model_name": best_model_name,
        "threshold": best_opt_thresh,
        "feature_names": FEATURE_NAMES,
        "validation_macro_f05": best_f05,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, BEST_MODEL_PATH)
    joblib.dump(model_artifact, BEST_LIGHTGBM_MODEL_PATH)
    logger.info(f"Saved winning model artifact ({best_model_name}, threshold={best_opt_thresh}) to {BEST_MODEL_PATH}")

    return {
        "model_comparison": df_models,
        "best_model": final_model,
        "best_model_name": best_model_name,
        "optimal_threshold": best_opt_thresh,
        "threshold_comparison": best_df_thresh,
    }


# ------------------------------------------------------------------------------
# 3. Scalable Production Test Inference
# ------------------------------------------------------------------------------
def run_test_inference(
    trained_model: Optional[Any] = None,
    optimal_threshold: Optional[float] = None,
    test_limit: Optional[int] = None,
) -> Tuple[Path, Path]:
    """Execute complete end-to-end inference on official test data to produce deliverables.

    Guarantees:
    - 100% of test Source 1 entities are written to both deliverables
    - Empty match list for entities with no partner meeting the decision threshold
    - Every predicted match is strictly present in candidate_pairs.tsv
    - Memory-safe country-partitioned candidate indexing preventing RAM exhaustion
    - Atomic file creation via .tmp.tsv preventing corrupted or partial outputs.
    """
    if not TEST_SOURCE1_PATH.is_file():
        raise FileNotFoundError(f"Cannot run test inference: Test source missing at {TEST_SOURCE1_PATH}")

    # Mode and limit verification
    if test_limit is None:
        print("\n" + "=" * 60)
        print("FULL TEST MODE ENABLED: Processing 100% of test dataset (test_limit=None)")
        print("=" * 60)
    else:
        print(f"\n[DEV MODE] Testing on sample of {test_limit:,} Source 1 entities.")

    # 1. Resolve Model and Decision Threshold
    model = trained_model
    threshold_source = "explicit CLI override" if optimal_threshold is not None else None

    if model is None or optimal_threshold is None:
        model_candidates = [BEST_MODEL_PATH, BEST_LIGHTGBM_MODEL_PATH, OUTPUT_DIR / "best_lightgbm_model.joblib"]
        found_file = next((p for p in model_candidates if p.is_file()), None)

        if found_file:
            logger.info(f"Loading trained model artifact from {found_file}...")
            data = joblib.load(found_file)
            if isinstance(data, dict):
                if model is None:
                    model = data["model"]
                if optimal_threshold is None:
                    optimal_threshold = float(data.get("threshold", OPTIMAL_THRESHOLD))
                    threshold_source = "validation/model artifact"
            else:
                if model is None:
                    model = data
                if optimal_threshold is None:
                    optimal_threshold = OPTIMAL_THRESHOLD
                    threshold_source = "default reference"
        else:
            logger.info("Model checkpoint not found. Running offline evaluation and model selection first...")
            res = run_offline_evaluation()
            model = res["best_model"]
            optimal_threshold = res["optimal_threshold"]
            threshold_source = "validation/model selection"

    logger.info("=" * 60)
    logger.info("TEST INFERENCE START")
    logger.info(f"Using threshold: {optimal_threshold}")
    logger.info(f"Threshold source: {threshold_source}")
    logger.info("=" * 60)

    # 2. Load and Audit Test Source 1 Dataset
    t0_start = time.perf_counter()
    logger.info(f"Reading Test Source 1 from {TEST_SOURCE1_PATH.name}...")
    test_s1_df = pd.read_csv(
        TEST_SOURCE1_PATH,
        sep="\t",
        nrows=test_limit,
        dtype=str,
        na_filter=False,
    )
    total_expected = len(test_s1_df)
    unique_expected_ids = set(test_s1_df["entity_id"])
    assert len(unique_expected_ids) == total_expected, (
        f"Input integrity error: {total_expected - len(unique_expected_ids)} duplicate IDs in test Source 1!"
    )
    logger.info(f"Total expected Source-1 entities: {total_expected:,}")

    # Stale output protection: remove any pre-existing temporary files
    for p in [MATCHING_RESULTS_TMP_PATH, CANDIDATE_PAIRS_TMP_PATH]:
        if p.exists():
            p.unlink()

    # 3. Country-Partitioned Candidate Generation & Scoring
    # All true ground-truth matches occur within the same country.
    # Partitioning by country slashes peak indexing RAM from >8 GB down to ~2.5 GB.
    unique_countries = test_s1_df["country"].str.strip().str.lower().unique().tolist()
    logger.info(f"Discovered {len(unique_countries)} unique countries in test Source 1: {unique_countries}")

    predictions_map: Dict[str, List[str]] = {}
    candidates_map: Dict[str, List[str]] = {}

    total_candidates_all = 0
    total_matches_all = 0
    candidate_lengths = []
    match_lengths = []

    for cntry in unique_countries:
        s1_sub = test_s1_df[test_s1_df["country"].str.strip().str.lower() == cntry]
        n_s1_cntry = len(s1_sub)
        logger.info(f"\n--- Processing Country: '{cntry.upper()}' ({n_s1_cntry:,} Source 1 entities) ---")

        # Build Candidate Index strictly for this country
        t0_idx = time.perf_counter()
        country_index = MultiStrategyBlockingIndex()

        for path in [TEST_SOURCE2_PATH, TEST_SOURCE3_PATH]:
            if not path.is_file():
                continue
            for chunk in pd.read_csv(
                path,
                sep="\t",
                chunksize=1_000_000,
                usecols=["entity_id", "business_name", "business_address", "country"],
                dtype=str,
                na_filter=False,
            ):
                sub = chunk[chunk["country"].str.lower().str.strip() == cntry]
                if not sub.empty:
                    country_index.index_candidates_chunk(
                        sub["entity_id"].tolist(),
                        sub["business_name"].tolist(),
                        sub["business_address"].tolist(),
                        sub["country"].tolist(),
                    )

        logger.info(
            f"  Indexed {len(country_index.candidate_ids):,} candidate records for '{cntry.upper()}' "
            f"in {time.perf_counter()-t0_idx:.1f}s."
        )

        # Batch Inference for this country
        BATCH_SIZE = 5_000
        sub_ids = s1_sub["entity_id"].tolist()
        sub_names = s1_sub["business_name"].tolist()
        sub_addrs = s1_sub["business_address"].tolist()
        cntry_s1_records = [
            (
                sub_ids[i],
                normalize_name(sub_names[i]),
                normalize_address(sub_addrs[i]),
                cntry,
            )
            for i in range(len(sub_ids))
        ]

        t0_cntry = time.perf_counter()
        processed_in_cntry = 0
        matches_in_cntry = 0

        for b_start in range(0, n_s1_cntry, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, n_s1_cntry)
            b_records = cntry_s1_records[b_start:b_end]

            # Candidate retrieval for batch
            b_cand_records = [
                country_index.retrieve_candidate_records(rec[1], rec[2], rec[3])
                for rec in b_records
            ]

            # Vectorized feature generation and model scoring
            b_matched_ids = predict_matches_batch(
                trained_model=model,
                batch_s1_records=b_records,
                batch_candidates_per_s1=b_cand_records,
                threshold=optimal_threshold,
            )

            # Store predictions and verify subset consistency
            for i, rec in enumerate(b_records):
                sid = rec[0]
                c_ids = [r[0] for r in b_cand_records[i]]
                m_ids = b_matched_ids[i]

                # Strict subset assertion
                missing = set(m_ids) - set(c_ids)
                if missing:
                    raise RuntimeError(
                        f"Integrity violation for {sid}: matches {missing} not in candidate set!"
                    )

                predictions_map[sid] = m_ids
                candidates_map[sid] = c_ids
                candidate_lengths.append(len(c_ids))
                match_lengths.append(len(m_ids))
                total_candidates_all += len(c_ids)
                total_matches_all += len(m_ids)
                matches_in_cntry += len(m_ids)

            processed_in_cntry += len(b_records)
            if processed_in_cntry % 50_000 == 0 or processed_in_cntry == n_s1_cntry:
                logger.info(
                    f"  Country '{cntry.upper()}': {processed_in_cntry:,}/{n_s1_cntry:,} entities "
                    f"({time.perf_counter()-t0_cntry:.1f}s) | Matches: {matches_in_cntry:,}"
                )

        # Release country index memory immediately
        country_index.clear()
        del country_index
        gc.collect()

    # 4. Handle Entities with Unseen or Empty Countries (Preserve All S1)
    for sid in test_s1_df["entity_id"]:
        if sid not in candidates_map:
            candidates_map[sid] = []
            predictions_map[sid] = []
            candidate_lengths.append(0)
            match_lengths.append(0)

    # Pre-write verification
    assert len(candidates_map) == total_expected, (
        f"Coverage violation: {len(candidates_map)} != {total_expected}"
    )

    # 5. Write Temporary Deliverables in Exact Test Source 1 Order
    logger.info("Writing temporary deliverables to disk...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MATCHING_RESULTS_TMP_PATH, "w", encoding="utf-8") as f_match, \
         open(CANDIDATE_PAIRS_TMP_PATH, "w", encoding="utf-8") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for sid in test_s1_df["entity_id"]:
            m_list = predictions_map.get(sid, [])
            c_list = candidates_map.get(sid, [])

            if m_list:
                f_match.write(f"{sid}\t{','.join(m_list)}\n")
            else:
                f_match.write(f"{sid}\t\n")

            if c_list:
                f_cand.write(f"{sid}\t{','.join(c_list)}\n")
            else:
                f_cand.write(f"{sid}\t\n")

    # Flush file system buffers
    logger.info("Deliverables written to temporary files. Starting independent integrity verification...")

    # 6. Hard Output Integrity Assertions (Independently Re-Read from Disk)
    m_re_rows, m_re_ids, m_empty, m_self = 0, set(), 0, 0
    with open(MATCHING_RESULTS_TMP_PATH, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        assert header == ["source1_entity_id", "matched_entity_ids"], f"Bad matching header: {header}"
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            m_re_rows += 1
            sid = parts[0].strip()
            m_re_ids.add(sid)
            rest = parts[1].strip() if len(parts) > 1 else ""
            if not rest:
                m_empty += 1
            else:
                ids = rest.split(",")
                for tid in ids:
                    if tid.startswith("S1-"):
                        m_self += 1

    c_re_rows, c_re_ids = 0, set()
    with open(CANDIDATE_PAIRS_TMP_PATH, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        assert header == ["source1_entity_id", "candidate_entity_ids"], f"Bad candidate header: {header}"
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            c_re_rows += 1
            sid = parts[0].strip()
            c_re_ids.add(sid)

    # Strict Output Integrity Checks
    missing_in_m = unique_expected_ids - m_re_ids
    extra_in_m = m_re_ids - unique_expected_ids
    missing_in_c = unique_expected_ids - c_re_ids
    extra_in_c = c_re_ids - unique_expected_ids

    if missing_in_m:
        raise RuntimeError(f"CRITICAL: matching_results.tsv is missing {len(missing_in_m):,} test S1 entities!")
    if extra_in_m:
        raise RuntimeError(f"CRITICAL: matching_results.tsv has {len(extra_in_m):,} unexpected S1 entities!")
    if missing_in_c:
        raise RuntimeError(f"CRITICAL: candidate_pairs.tsv is missing {len(missing_in_c):,} test S1 entities!")
    if extra_in_c:
        raise RuntimeError(f"CRITICAL: candidate_pairs.tsv has {len(extra_in_c):,} unexpected S1 entities!")
    if m_re_rows != total_expected:
        raise RuntimeError(f"Row count mismatch in matching: {m_re_rows:,} != {total_expected:,}")
    if c_re_rows != total_expected:
        raise RuntimeError(f"Row count mismatch in candidates: {c_re_rows:,} != {total_expected:,}")
    if m_self > 0:
        raise RuntimeError(f"CRITICAL: {m_self} self-matches (S1-* IDs) found in matching_results.tsv!")

    # 7. Atomic Rename: Replace final files only after 100% verification passes
    m_final = Path(MATCHING_RESULTS_PATH)
    c_final = Path(CANDIDATE_PAIRS_PATH)
    if m_final.exists():
        m_final.unlink()
    Path(MATCHING_RESULTS_TMP_PATH).replace(m_final)
    if c_final.exists():
        c_final.unlink()
    Path(CANDIDATE_PAIRS_TMP_PATH).replace(c_final)

    # 8. Print Official Final Inference Summary
    c_np = np.array(candidate_lengths)
    m_np = np.array(match_lengths)
    zero_cands = int((c_np == 0).sum())
    zero_matches = int((m_np == 0).sum())

    print("\n" + "=" * 60)
    print("FINAL INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Expected S1 entities:   {total_expected:,}")
    print(f"Processed S1 entities:  {total_expected:,}")
    print(f"Output S1 entities:     {m_re_rows:,}")
    print(f"Missing S1 entities:    0")
    print(f"Extra S1 entities:      0")
    print(f"Duplicate S1 entities:  0")
    print("\nCandidate statistics:")
    print(f"  Mean candidates:      {float(np.mean(c_np)):.2f}")
    print(f"  Median candidates:    {int(np.median(c_np))}")
    print(f"  P95 candidates:       {int(np.percentile(c_np, 95))}")
    print(f"  P99 candidates:       {int(np.percentile(c_np, 99))}")
    print(f"  Zero-candidate rate:  {zero_cands / total_expected * 100.0:.2f}%")
    print("\nPrediction statistics:")
    print(f"  Mean matches:         {float(np.mean(m_np)):.2f}")
    print(f"  Median matches:       {int(np.median(m_np))}")
    print(f"  Zero-match rate:      {zero_matches / total_expected * 100.0:.2f}%")
    print(f"  Total matches:        {total_matches_all:,}")
    print(f"Total Inference Time:   {time.perf_counter() - t0_start:.1f}s")
    print("=" * 60 + "\n")

    # 9. Run Official Submission Validator (strictly on full production run)
    if test_limit is None:
        logger.info("Executing official validate_submission.py validator...")
        rc, _, _ = run_submission_validation(
            matching_path=m_final,
            candidate_path=c_final,
            test_dir=TEST_DIR,
        )
        if rc != 0:
            raise RuntimeError("Official validate_submission.py validator failed! Stopping pipeline.")
    else:
        logger.info(f"[DEV MODE] Skipping official validate_submission.py because test_limit={test_limit:,} was active.")

    return m_final, c_final
