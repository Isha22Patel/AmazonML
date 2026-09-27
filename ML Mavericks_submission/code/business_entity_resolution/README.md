# Amazon ML Challenge 2026: Business Entity Resolution

> **Team:** ML Mavericks | **Team Members:** Isha Patel, Ujjwal Sahu, Anushika Chauhan  
> **Official Metric:** Entity-Level Macro $F_{0.5}$ (Precision-Weighted)  
> **Architecture:** Two-Stage Entity Resolution (7-Pass Multi-Strategy Inverted-Index Blocking + 17-Feature LightGBM GBDT Classifier + Calibrated Decision Boundary $\tau = 0.95$)  
> **Validation Status:** `PASS — no blocking issues found. Safe to submit.` (Verified with ID checking on 1,732,544 entities)

---

## 1. Executive Summary & Challenge Objectives

The **Amazon ML Challenge 2026: Business Entity Resolution** task requires linking real-world business entities across three heterogeneous, unlinked sources:
- **Source 1 ($S_1$):** Deduplicated reference anchor records. Every Source 1 test entity must appear exactly once in `matching_results.tsv`.
- **Source 2 ($S_2$):** Web crawls and commercial business directories.
- **Source 3 ($S_3$):** Official commercial registries and corporate databases.

### Core Properties & Architectural Demands
1. **Combinatorial Search Space**:
   Pairwise comparison across $|S_1| \times (|S_2| + |S_3|)$ represents over $2.2 \times 10^{13}$ possible pairs ($1.73 \times 10^6 \times 9.97 \times 10^6$). Unbounded indexing exhausts system memory; the pipeline requires sub-linear candidate generation with strict posting caps.
2. **Cardinality Skew & Singleton Penalty**:
   - **Singletons (0 true matches)**: Comprise ~5.58% of reference entities. Under the competition's Macro $F_{0.5}$, correctly predicting an empty match list awards a score of 1.0; a false positive merge reduces that entity's score to 0.0.
   - **Multi-Matches**: Over 89% of matching entities have multi-source links across $S_2$ and $S_3$.
3. **Macro $F_{0.5}$ Metric**:
   $$\text{Macro } F_{0.5} = \frac{1}{N_{S1}} \sum_{i=1}^{N_{S1}} \frac{1.25 \times P_i \times R_i}{0.25 \times P_i + R_i}$$
   Precision is weighted roughly 2× over recall ($F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R}$), demanding conservative decision boundaries to prevent singleton contamination.
4. **Open-Set Geographic Distribution**:
   Training data spans `US` and `India`, whereas the test set introduces `France`. All features and blocking logic treat country codes as open-set string invariants.

---

## 2. Directory Structure

```text
AmazonML_Final/   (or <team_name>_submission/)
├── output/
│   ├── matching_results.tsv             # Official test predictions (scored on leaderboard)
│   └── candidate_pairs.tsv              # Final blocked candidate set preceding model scoring
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py              # Clean package public exports
│       │   ├── config.py                # Paths, schemas, hyperparameters, and posting caps
│       │   ├── normalize.py             # C-speed Unicode NFKD normalization & legal suffix removal
│       │   ├── blocking.py              # Multi-strategy inverted-index blocking engine (7 passes)
│       │   ├── features.py              # 17 pairwise lexical, phonetic, and structural features
│       │   ├── models.py                # Model implementations (Rule, Baseline, LR, LightGBM, RF)
│       │   ├── threshold.py             # Exhaustive threshold sweep optimizing Entity Macro F0.5
│       │   ├── evaluation.py            # Official metric implementation with singleton support
│       │   ├── error_analysis.py        # Diagnostic categorization of FP and FN error modes
│       │   ├── data_loader.py           # Memory-safe chunked TSV streaming & schema checks
│       │   ├── validation.py            # Integration with official validate_submission.py
│       │   ├── pipeline.py              # End-to-end training, evaluation, and streaming inference
│       │   ├── submission.py            # TSV file formatting and official zip packaging
│       │   ├── main.py                  # CLI entry point for individual stages or full execution
│       │   ├── test_blocking_eval.py    # Standalone blocking benchmark harness
│       │   └── EntityRelationDeduplication.ipynb # Synchronized research notebook
│       ├── README.md                    # This reproducibility and architecture document
│       └── requirements.txt             # Pinned production dependencies
│
├── .gitignore                           # Excludes datasets, caches, and intermediate artifacts
└── Documentation_template.md            # Official methodology and experimental report
```

> **Note**: Development-only folders such as `models/` and `reports/` contain local checkpoints and evaluation logs; only `output/`, `code/`, and `Documentation_template.md` form the required submission package.

---

## 3. Environment & Prerequisites

- **Python Version**: `3.10` to `3.13` (tested on Python 3.13.3).
- **Hardware Requirements**:
  - Minimum RAM: 8 GB.
  - Recommended RAM: 16 GB+ for full multi-million row test streaming.
  - Disk Space: 15 GB free for full uncompressed challenge dataset.

### Installation
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

Pinned dependencies in `requirements.txt`:
```text
pandas>=2.0.0,<=3.1.0
numpy>=1.26.0,<=2.6.0
scikit-learn>=1.4.0,<=2.0.0
lightgbm>=4.0.0,<=5.0.0
rapidfuzz>=3.0.0,<=4.0.0
joblib>=1.3.0,<=2.0.0
```

---

## 4. Dataset Setup

The official challenge dataset should be placed in `dataset/`:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

> **Note**: All TSV files must be read with tab separation (`sep="\t"`).

---

## 5. How to Run the Pipeline

The primary entry point is located at `code/business_entity_resolution/src/main.py`:

### Full End-to-End Pipeline
```bash
python code/business_entity_resolution/src/main.py --all
```

### Running Individual Stages
```bash
# 1. Exploratory Data Analysis
python code/business_entity_resolution/src/main.py --stage eda

# 2. Offline Training, Model Benchmarking & Threshold Sweep (10k Sample)
python code/business_entity_resolution/src/main.py --stage evaluate --sample-size 10000

# 3. Full Production Test Streaming Inference (all 1,732,544 test entities)
python code/business_entity_resolution/src/main.py --stage inference --production

# 4. Deliverables Validation (Checks matching_results.tsv and candidate_pairs.tsv)
python code/business_entity_resolution/src/main.py --stage validate

# 5. Package Competition Submission Zip
python code/business_entity_resolution/src/main.py --stage package --team-name ML_Maverics
```

---

## 6. Pipeline Methodology & Validated Results

### 1. Multi-Pass Inverted-Index Blocking (`src/blocking.py`)
To prevent memory saturation and candidate explosion, we enforce dynamic posting-list caps per bucket and rank candidates by relevance before truncation:
1. **Exact Normalized Name + Country**: Clean alphanumeric string partitioned by country.
2. **Legal Suffix Stripped + Country**: Removes 40+ international corporate forms (*LLC, Ltd, GmbH, Inc, S.A.*).
3. **Sorted Name Tokens + Country**: Permutation-invariant token keys (e.g. `"solutions apex"` for `"Apex Solutions"`).
4. **Compact Name + Country**: Whitespace- and punctuation-stripped alphanumeric representation.
5. **Name Token + Address Street Number**: Disambiguates common names sharing street numbers.
6. **Address Street Number + Street Word**: Captures co-located or rebranded entities sharing an address.
7. **Controlled Prefix-6 + Country**: First 6 characters of normalized name (controlled typo fallback).

**Blocking Benchmark Results (20,000 Validation Entities)**:
- Search space reduction ratio: **99.999429%** (>99.999%).
- Average candidates per $S_1$: **28.77** (Median: 24, Max cap: 60).
- Zero-candidate rate: **0.34%** (99.66% of entities retrieve candidates).
- Measured candidate recall on validation benchmark: **37.47%** (single passes range from 10.93% to 24.68%).
- **Candidate Set Semantics**: `candidate_pairs.tsv` represents the **final candidate set immediately preceding model scoring**, not merely an intermediate blocking artifact. Every match in `matching_results.tsv` is strictly a subset of `candidate_pairs.tsv`.

### 2. 17 Engineered Similarity Features (`src/features.py`)
- **Name Features (8)**: `name_exact`, `name_jaccard`, `name_common_tokens`, `name_lev`, `name_norm_lev`, `name_len_ratio`, `name_prefix_sim`, `name_token_containment`.
- **Address Features (7)**: `address_exact`, `address_jaccard`, `address_lev`, `address_norm_lev`, `address_len_ratio`, `address_number_match`, `address_token_overlap`.
- **Geographic & Interaction Features (2)**: `country_exact` (open-set equality check) and `name_address_agreement` ($\text{sim}_{\text{name}} \times \text{sim}_{\text{address}}$ cross-attribute interaction).

### 3. Model Benchmark on Validation Split (2,000 S1 Entities, 10,000 Dev Sample)
Evaluated on a deterministic 8,000 / 2,000 entity-stratified split:

| Model Architecture | Optimal Threshold ($\tau$) | Entity Macro $F_{0.5}$ | Entity Precision | Entity Recall | Train Runtime (s) | Inference Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 0: Rule Baseline (Exact Name)** | 0.10 | 0.36738 | 46.22% | 27.78% | 0.00 | 0.04 |
| **Model 1: Blocking-Only Baseline** | 0.10 | 0.14323 | 12.46% | 67.88% | 0.00 | 0.01 |
| **Model 2: Logistic Regression** | 0.95 | 0.72913 | 80.73% | 61.24% | 0.73 | 0.52 |
| **Model 3: LightGBM GBDT (Selected Model)** | **0.95** | **0.78402** | **86.48%** | **65.04%** | **2.18** | **6.08** |
| **Model 4: Random Forest (150 Trees)** | 0.85 | 0.77732 | 84.88% | 66.16% | 28.56 | 71.56 |

*Model Selection*: LightGBM GBDT achieved the highest Entity Macro $F_{0.5}$ (0.78402) and highest Entity Precision (86.48%) while maintaining fast training (2.18s) and efficient streaming inference. It complies with competition constraints (<8B parameters, Apache 2.0 license).

### 4. Decision Threshold Calibration ($\tau = 0.95$)
- At standard threshold ($\tau = 0.50$): $F_{0.5} = 0.70540$ with 66 singleton false positive merges (58.93% singleton error rate).
- At calibrated threshold ($\mathbf{\tau = 0.95}$): False positive singleton merges fell by **78.8%** (down to 14, 12.50% error rate), precision increased to **86.48%**, achieving the global peak **Entity Macro $F_{0.5} = 0.78402$**.

---

## 7. Submission Deliverables & Validator Compliance

The pipeline generates exactly two TSV deliverables in `output/`:
1. `output/matching_results.tsv`: Final predicted matches per test Source 1 entity.
2. `output/candidate_pairs.tsv`: Final blocked candidate pool passed to the matcher.

### Official Submission Validator Verification
The final outputs were verified using the official submission validator with complete ID checking:

```powershell
python .\utils\validate_submission.py `
  --matching .\AmazonML_Final\output\matching_results.tsv `
  --candidate .\AmazonML_Final\output\candidate_pairs.tsv `
  --test-dir .\dataset\test `
  --check-ids
```

**Validator Output**:
```text
ML Challenge 2026 — submission validator
required S1 entities: 1732544
valid S2/S3 match IDs: 9969589
matching_results.tsv: 1732544 rows
candidate_pairs.tsv: 1732544 rows

PASS — no blocking issues found. Safe to submit.
```

*(Note: The validator verifies submission integrity, structural formatting, and candidate-ID validity; it does not compute a leaderboard F0.5 score).*

### Verified Output Statistics
- **`matching_results.tsv`**:
  - Total rows: **1,732,544** (100.0% coverage of required Source 1 test entities)
  - Singletons / Empty predictions: **260,823**
  - Populated match predictions: **1,471,721**
- **`candidate_pairs.tsv`**:
  - Total rows: **1,732,544**
  - Empty candidate lists: **376**
  - Populated candidate lists: **1,732,168**
- **Catalog Statistics**:
  - Valid candidate IDs in test set: **9,969,589**
  - Duplicate S1 rows: **0**
  - Self-matches: **0**
  - Subset rule: **100% verified** (every predicted match is strictly present in its candidate list)
