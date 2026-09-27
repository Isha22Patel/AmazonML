"""Standalone blocking strategy benchmark and evaluation script.

Evaluates 7 distinct blocking strategies and their cumulative union
on 20,000 Source-1 validation entities against the full 10.3M candidate population
from Source 2 and Source 3.

Computes:
- Candidate Recall (%)
- Mean Candidates per S1
- Median Candidates per S1
- P95 / P99 / Max Candidates
- Zero-Candidate Rate (%)
- Search Space Reduction Ratio (%)
- Runtime (s)
"""

import sys
import time
import os
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np

# Configure UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent
CODE_DIR = SRC_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.config import (
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    OUTPUT_DIR,
)
from src.normalize import normalize_name, normalize_address, remove_legal_suffixes, extract_numbers
from src.blocking import (
    MultiStrategyBlockingIndex,
    block_exact_name,
    block_no_suffix_name,
    block_token_name,
    block_compact_name,
    block_name_token_address_number,
    block_address_street,
    block_prefix,
)


def main():
    print("=" * 60)
    print("AUTHORITATIVE BLOCKING EVALUATION BENCHMARK")
    print("=" * 60)

    # 1. Load 20,000 validation S1 entities
    print(f"Loading 20,000 validation S1 entities from {TRAIN_SOURCE1_PATH.name}...")
    s1_df = pd.read_csv(TRAIN_SOURCE1_PATH, sep="\t", nrows=20000, dtype=str, na_filter=False)
    s1_set = set(s1_df["entity_id"])

    # Load Ground Truth strictly for these 20,000 S1 entities
    print(f"Loading ground truth mapping from {TRAIN_GROUND_TRUTH_PATH.name}...")
    gt_map = {}
    for chunk in pd.read_csv(TRAIN_GROUND_TRUTH_PATH, sep="\t", chunksize=500000, dtype=str, na_filter=False):
        sub = chunk[chunk["source1_entity_id"].isin(s1_set)]
        for _, r in sub.iterrows():
            sid = r["source1_entity_id"]
            m = r["matched_entity_ids"].strip()
            gt_map[sid] = set(m.split(",")) if m else set()
        if len(gt_map) == len(s1_set):
            break

    # Guard singletons (entities with 0 matches)
    for sid in s1_set:
        if sid not in gt_map:
            gt_map[sid] = set()

    total_true_matches = sum(len(v) for v in gt_map.values())
    val_s1_records = []
    for _, r in s1_df.iterrows():
        sid = r["entity_id"]
        n = normalize_name(r["business_name"])
        a = normalize_address(r["business_address"])
        c = str(r["country"]).lower().strip()
        val_s1_records.append((sid, n, a, c))

    print(f"Loaded {len(val_s1_records):,} S1 entities ({total_true_matches:,} true matches in GT)")

    # 2. Build Inverted Index over Source 2 and Source 3
    print("\nBuilding multi-strategy candidate inverted index over S2 + S3 (10.3M records)...")
    blocking_index = MultiStrategyBlockingIndex()

    strategy_names = ["exact", "nosuff", "sorted_toks", "compact", "name_addr", "addr_street", "prefix6"]
    standalone_indexes = {st: defaultdict(list) for st in strategy_names}

    CAPS = {
        "exact": 40,
        "nosuff": 40,
        "sorted_toks": 35,
        "compact": 35,
        "name_addr": 30,
        "addr_street": 30,
        "prefix6": 20,
    }

    t0_idx = time.time()
    total_indexed = 0

    for path in [TRAIN_SOURCE2_PATH, TRAIN_SOURCE3_PATH]:
        if not path.is_file():
            print(f"[ERROR] Candidate source missing: {path}")
            continue
        print(f"  Indexing {path.name}...")
        for chunk in pd.read_csv(path, sep="\t", chunksize=500000, usecols=["entity_id", "business_name", "business_address", "country"], dtype=str, na_filter=False):
            c_ids = chunk["entity_id"].tolist()
            c_names = chunk["business_name"].tolist()
            c_addrs = chunk["business_address"].tolist()
            c_countries = chunk["country"].tolist()
            base_idx = len(blocking_index.candidate_ids)
            blocking_index.candidate_ids.extend(c_ids)

            for i in range(len(c_ids)):
                rec_idx = base_idx + i
                n_norm = normalize_name(c_names[i])
                a_norm = normalize_address(c_addrs[i])
                cntry = str(c_countries[i]).lower().strip()

                # Add to standalone eval indexes
                k_ex = block_exact_name(cntry, n_norm)
                if k_ex and len(standalone_indexes["exact"][k_ex]) < CAPS["exact"]:
                    standalone_indexes["exact"][k_ex].append(rec_idx)
                    if len(blocking_index.exact_idx[k_ex]) < CAPS["exact"]:
                        blocking_index.exact_idx[k_ex].append(rec_idx)

                k_ns = block_no_suffix_name(cntry, n_norm)
                if k_ns and len(standalone_indexes["nosuff"][k_ns]) < CAPS["nosuff"]:
                    standalone_indexes["nosuff"][k_ns].append(rec_idx)
                    if len(blocking_index.no_suffix_idx[k_ns]) < CAPS["nosuff"]:
                        blocking_index.no_suffix_idx[k_ns].append(rec_idx)

                k_st = block_token_name(cntry, n_norm)
                if k_st and len(standalone_indexes["sorted_toks"][k_st]) < CAPS["sorted_toks"]:
                    standalone_indexes["sorted_toks"][k_st].append(rec_idx)
                    if len(blocking_index.tokens_idx[k_st]) < CAPS["sorted_toks"]:
                        blocking_index.tokens_idx[k_st].append(rec_idx)

                k_cp = block_compact_name(cntry, n_norm)
                if k_cp and len(standalone_indexes["compact"][k_cp]) < CAPS["compact"]:
                    standalone_indexes["compact"][k_cp].append(rec_idx)
                    if len(blocking_index.compact_idx[k_cp]) < CAPS["compact"]:
                        blocking_index.compact_idx[k_cp].append(rec_idx)

                k_na = block_name_token_address_number(cntry, n_norm, a_norm)
                if k_na and len(standalone_indexes["name_addr"][k_na]) < CAPS["name_addr"]:
                    standalone_indexes["name_addr"][k_na].append(rec_idx)
                    if len(blocking_index.name_addr_idx[k_na]) < CAPS["name_addr"]:
                        blocking_index.name_addr_idx[k_na].append(rec_idx)

                k_as = block_address_street(cntry, a_norm)
                if k_as and len(standalone_indexes["addr_street"][k_as]) < CAPS["addr_street"]:
                    standalone_indexes["addr_street"][k_as].append(rec_idx)
                    if len(blocking_index.addr_street_idx[k_as]) < CAPS["addr_street"]:
                        blocking_index.addr_street_idx[k_as].append(rec_idx)

                k_pf = block_prefix(cntry, n_norm, length=6)
                if k_pf and len(standalone_indexes["prefix6"][k_pf]) < CAPS["prefix6"]:
                    standalone_indexes["prefix6"][k_pf].append(rec_idx)
                    if len(blocking_index.prefix_idx[k_pf]) < CAPS["prefix6"]:
                        blocking_index.prefix_idx[k_pf].append(rec_idx)

            total_indexed += len(c_ids)
            print(f"    Indexed {total_indexed:,} records ({time.time()-t0_idx:.1f}s)...", flush=True)

    print(f"Completed indexing {len(blocking_index.candidate_ids):,} candidate records in {time.time()-t0_idx:.1f}s.")

    # 3. Evaluate Individual Strategies
    TOTAL_CANDIDATE_POPULATION = len(blocking_index.candidate_ids)
    results = []

    print("\nEvaluating individual blocking passes on 20,000 S1 entities...")
    strategy_funcs = {
        "exact": lambda n, a, c: block_exact_name(c, n),
        "nosuff": lambda n, a, c: block_no_suffix_name(c, n),
        "sorted_toks": lambda n, a, c: block_token_name(c, n),
        "compact": lambda n, a, c: block_compact_name(c, n),
        "name_addr": lambda n, a, c: block_name_token_address_number(c, n, a),
        "addr_street": lambda n, a, c: block_address_street(c, a),
        "prefix6": lambda n, a, c: block_prefix(c, n, length=6),
    }

    strategy_labels = {
        "exact": "1. Exact Name + Country",
        "nosuff": "2. Suffix-Stripped Name + Country",
        "sorted_toks": "3. Sorted Name Tokens + Country",
        "compact": "4. Compact Name + Country",
        "name_addr": "5. Name Token + Address Number",
        "addr_street": "6. Address Number + Street Word",
        "prefix6": "7. Controlled Prefix-6 + Country",
    }

    for st in strategy_names:
        t0 = time.time()
        cands_per_s1 = []
        found_matches = 0
        zero_cands = 0
        fn = strategy_funcs[st]

        for sid, n, a, c in val_s1_records:
            k = fn(n, a, c)
            if k:
                c_set = {blocking_index.candidate_ids[idx] for idx in standalone_indexes[st].get(k, [])}
            else:
                c_set = set()

            n_c = len(c_set)
            cands_per_s1.append(n_c)
            if n_c == 0:
                zero_cands += 1

            true_set = gt_map.get(sid, set())
            if true_set:
                found_matches += len(true_set.intersection(c_set))

        dt = time.time() - t0
        cands_arr = np.array(cands_per_s1)
        recall = found_matches / total_true_matches * 100.0
        mean_c = float(np.mean(cands_arr))
        reduction = (1.0 - (mean_c / TOTAL_CANDIDATE_POPULATION)) * 100.0

        results.append({
            "Blocking Strategy": strategy_labels[st],
            "Candidate Recall (%)": round(recall, 2),
            "Mean Cands": round(mean_c, 2),
            "Median Cands": int(np.median(cands_arr)),
            "P95 Cands": int(np.percentile(cands_arr, 95)),
            "P99 Cands": int(np.percentile(cands_arr, 99)),
            "Max Cands": int(np.max(cands_arr)),
            "Zero-Candidate Rate (%)": round(zero_cands / len(val_s1_records) * 100.0, 2),
            "Reduction Ratio (%)": f"{reduction:.6f}%",
            "Runtime (s)": round(dt, 2),
        })

    # Evaluate the Final MultiStrategyBlockingIndex (Union with Relevance Ranking)
    print("\nEvaluating Production MultiStrategyBlockingIndex (Relevance-Ranked Multi-Pass)...")
    t0_prod = time.time()
    cands_per_s1_prod = []
    found_matches_prod = 0
    zero_cands_prod = 0

    for sid, n, a, c in val_s1_records:
        cand_ids = blocking_index.retrieve_candidates(n, a, c, max_candidates=60)
        c_set = set(cand_ids)
        n_c = len(c_set)
        cands_per_s1_prod.append(n_c)
        if n_c == 0:
            zero_cands_prod += 1

        true_set = gt_map.get(sid, set())
        if true_set:
            found_matches_prod += len(true_set.intersection(c_set))

    dt_prod = time.time() - t0_prod
    cands_arr_prod = np.array(cands_per_s1_prod)
    recall_prod = found_matches_prod / total_true_matches * 100.0
    mean_c_prod = float(np.mean(cands_arr_prod))
    reduction_prod = (1.0 - (mean_c_prod / TOTAL_CANDIDATE_POPULATION)) * 100.0

    results.append({
        "Blocking Strategy": "★ Production Multi-Pass (7 Passes + Relevance Ranking)",
        "Candidate Recall (%)": round(recall_prod, 2),
        "Mean Cands": round(mean_c_prod, 2),
        "Median Cands": int(np.median(cands_arr_prod)),
        "P95 Cands": int(np.percentile(cands_arr_prod, 95)),
        "P99 Cands": int(np.percentile(cands_arr_prod, 99)),
        "Max Cands": int(np.max(cands_arr_prod)),
        "Zero-Candidate Rate (%)": round(zero_cands_prod / len(val_s1_records) * 100.0, 2),
        "Reduction Ratio (%)": f"{reduction_prod:.6f}%",
        "Runtime (s)": round(dt_prod, 2),
    })

    df_res = pd.DataFrame(results)
    print("\n" + df_res.to_string(index=False))

    out_csv = OUTPUT_DIR / "blocking_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved official blocking comparison to: {out_csv}")


if __name__ == "__main__":
    main()
