# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** ML Maverics  
**Team Members:** Isha  
**Submission Date:** September 2026

---

## 1. Executive Summary

Our solution implements a high-throughput, two-stage Business Entity Resolution pipeline engineered to resolve reference business records (Source 1) against over 9.96 million candidate records from noisy directories and official registries (Sources 2 and 3). The architecture pairs a sub-linear 7-pass multi-strategy inverted-index blocking engine—enforcing frequency-aware posting caps and relevance-based ranking—with a 17-feature LightGBM Gradient Boosted Decision Tree (GBDT) pairwise matcher. By directly calibrating our classification decision boundary ($\tau = 0.95$) to the competition's precision-prioritizing Macro $F_{0.5}$ metric and singleton structure, we reduced false positive singleton merges by 78.8% and achieved an Entity Macro $F_{0.5}$ score of **0.78402** (86.48% precision, 65.04% recall) on our validation split, while guaranteeing 100% test entity coverage and passing official submission validation with zero errors.

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory Data Analysis across the training and test splits revealed four core structural challenges:

1. **Combinatorial Scale & Search Space**: Matching $1.73 \times 10^6$ test Source 1 entities against $9.97 \times 10^6$ candidate records across Sources 2 and 3 creates an unconstrained pairwise comparison space of over $2.2 \times 10^{13}$ pairs ($|S1| \times (|S2| + |S3|)$). Unbounded indexing causes out-of-memory crashes on commodity machines, necessitating sub-linear inverted indexing with strict posting caps.
2. **Cardinality Skew & Singleton Penalty**: Singletons (entities with 0 true matches in Sources 2 and 3) constitute ~5.58% of entities in the labeled data, while 89.02% of matching entities have multi-source links. Under the competition's Macro $F_{0.5}$ formulation, singletons receive a score of 1.0 when correctly left empty, but drop to 0.0 upon any false positive merge.
3. **Asymmetric Precision Weighting**: The competition metric is entity-level Macro $F_{0.5}$:
   $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   Precision is weighted roughly twice as heavily as recall (penalizing false merges twice as much as missed links), requiring conservative decision boundaries to avoid contaminating singleton and clean matches.
4. **Noise Patterns**:
   - *Business Names*: Pervasive legal suffix variations (*Inc*, *Corp*, *Pvt Ltd*, *LLC*, *GmbH*, *S.A.*), word order inversions (*"Apex Solutions"* vs *"Solutions Apex"*), punctuation/symbol discrepancies (*"&"* vs *"and"*), abbreviations, and whitespace variations.
   - *Addresses*: Inconsistent formatting between postal standards and landmark-based descriptions (*"Near SBI ATM, Main Road"*), missing PIN/postal codes, street type abbreviations (*St*, *Rd*, *Ave*, *Blvd*), and component permutations.
   - *Open-Set Country Distribution*: While training data spans `US` and `India`, the test set introduces `France`. The pipeline must treat country labels as open-set strings without hard-coding one-hot categorical features.

### 2.2 Solution Strategy

We adopted a decoupled, two-stage entity resolution pipeline:

1. **Stage 1 — Multi-Pass Inverted-Index Blocking**:
   Transforms the $>2.2 \times 10^{13}$ comparison space into a manageable candidate pool (~28.8 candidates per entity) using 7 complementary phonetic, structural, and cross-attribute indexing keys partitioned by country, guarded by dynamic posting caps and heuristic relevance ranking.
2. **Stage 2 — Pairwise Feature Engineering & GBDT Classification**:
   Extracts 17 lexical, token-based, structural, and cross-attribute interaction features for each candidate pair using C-accelerated RapidFuzz routines, followed by probability scoring using a LightGBM GBDT classifier.
3. **Metric-Aligned Decision Thresholding**:
   Performs an exhaustive grid sweep directly optimizing entity-level Macro $F_{0.5}$ to identify the optimal operating point ($\tau = 0.95$), heavily suppressing false merges on singletons.
4. **Memory-Safe Streaming Test Inference**:
   Streams test Source 1 entities in chunked batches against an indexed candidate database, applying atomic two-phase disk writing to guarantee 100% entity coverage without memory saturation.

**Approach Type:** Blocking + Classifier (Two-Stage Pipeline with GBDT and Calibrated Boundary)  
**Core Innovation:** A joint precision-optimization framework combining frequency-capped multi-pass inverted indexing with a high-threshold ($\tau = 0.95$) decision rule that targets the competition's singleton penalty and Macro $F_{0.5}$ metric, paired with memory-safe chunked streaming that guarantees 100% test Source 1 entity coverage and zero pipeline truncation.

---

## 3. Candidate Generation (Blocking)

To reduce the comparison space while minimizing false negatives, we implemented a multi-pass inverted indexing engine in `src/blocking.py` that partitions candidates by country and extracts multiple complementary keys per entity:

- **Blocking keys used:**
  1. *Exact Normalized Name + Country*: `country__exact__<normalized_name>` (exact string match after Unicode NFKD normalization, lowercasing, and punctuation stripping).
  2. *Legal Suffix-Stripped Name + Country*: `country__nosuff__<suffix_free_name>` (strips 40+ international corporate designators such as *LLC, Ltd, Pvt, Inc, Corp, GmbH, SA* to match across legal re-incorporations).
  3. *Sorted Name Tokens + Country*: `country__sortedtoks__<tokens>` (permutation-invariant token string filtering business stopwords to handle word transpositions).
  4. *Compact Name + Country*: `country__compact__<alphanumeric_chars>` (strips all internal whitespace and punctuation to bridge spacing and domain variations).
  5. *Composite Name Token + Address Number*: `country__namedigit__<token>_<number>` (disambiguates common business names sharing street numbers).
  6. *Address Number + Street Word*: `country__addrstreet__<number>_<street_token>` (captures co-located or rebranded entities sharing an address).
  7. *Controlled Prefix-6 + Country*: `country__prefix6__<prefix>` (first 6 characters of normalized name, strictly frequency-capped to handle minor typos).

- **Candidate pairs generated:**
  - On the full test set ($1,732,544$ Source 1 entities against $9,969,589$ candidate records in Sources 2 and 3), `candidate_pairs.tsv` contains exactly **1,732,544 rows** (1,732,168 non-empty candidate lists, 376 empty lists).
  - Search space reduction ratio exceeds **99.999%** (99.999429% measured on validation benchmark), narrowing the candidate pool to an average of ~28.8 candidates per entity, with a strict safety cap of 60 candidates per entity.
  - *Note*: `candidate_pairs.tsv` represents the **final candidate set immediately preceding model scoring** and is strictly the superset of all predicted matches in `matching_results.tsv`.

- **How you ensured true matches were not lost:**
  - *Multi-Angle Representation*: Combining 7 diverse indexing strategies ensures that an entity missed by exact naming (e.g., due to a suffix difference or word swap) is captured through suffix-stripped, sorted-token, or address-number passes.
  - *Frequency-Aware Posting Caps*: High-frequency generic keys (e.g., *"Enterprises"*) are capped to prevent uninformative candidates from crowding out informative ones.
  - *Heuristic Relevance Ranking*: Rather than truncating candidate lists based on arbitrary file ordering, candidate pairs accumulate relevance weights based on matched key types (Exact: 10.0, No-Suffix: 8.0, Sorted Tokens: 6.0, Compact: 5.0, Cross-attribute: 4.0/3.0, Prefix: 2.0). The top $K \le 60$ highest-scoring candidates are retained.
  - *Symmetrical Key Extraction*: Normalization and key extraction logic are identically applied to both reference ($S_1$) and candidate ($S_2, S_3$) records.
  - *Benchmark Validation*: On our 20,000-entity validation benchmark, multi-pass candidate recall reached 37.47% (compared to single-pass recalls between 10.93% and 24.68%), while maintaining a 99.66% non-zero candidate generation rate.

---

## 4. Matching Model

**Features used:**
We extract 17 fine-grained similarity features per candidate pair in `src/features.py`:
- Name features:
  - `name_exact`: Binary indicator of exact normalized name equality.
  - `name_jaccard`: Word token Jaccard similarity.
  - `name_common_tokens`: Count of shared distinct name tokens.
  - `name_lev`: Levenshtein edit distance computed via RapidFuzz C-extensions.
  - `name_norm_lev`: Normalized Levenshtein similarity ($1 - \text{dist} / \text{max\_len}$).
  - `name_len_ratio`: Ratio of shorter to longer character length.
  - `name_prefix_sim`: Normalized length of shared character prefix.
  - `name_token_containment`: Token containment ratio ($\text{intersection} / \min(\text{tokens}_1, \text{tokens}_2)$).
- Address features:
  - `address_exact`: Binary indicator of exact address string equality.
  - `address_jaccard`: Word token Jaccard similarity across address fields.
  - `address_lev`: RapidFuzz Levenshtein edit distance on addresses.
  - `address_norm_lev`: Normalized Levenshtein address similarity.
  - `address_len_ratio`: Ratio of shorter to longer address character length.
  - `address_number_match`: Set-intersection indicator for building/street numeric digits.
  - `address_token_overlap`: Token overlap ratio normalized by minimum token count.
- Other:
  - `country_exact`: Binary indicator of exact country code match (handles open-set countries such as France without one-hot encoding).
  - `name_address_agreement`: Multiplicative cross-attribute interaction term ($\text{name\_norm\_lev} \times \text{address\_norm\_lev}$) reinforcing pairs with simultaneous name and address consistency.

**Model type:** LightGBM Gradient Boosted Decision Tree (GBDT)
- Selected after systematic benchmark against 4 alternative architectures: Rule Baseline (Exact Name), Blocking-Only Baseline, Scaled Logistic Regression, and Random Forest (150 trees).
- LightGBM achieved the highest Entity Macro $F_{0.5}$ (0.78402) and highest Entity Precision (86.48%) while maintaining fast training (2.18s) and sub-linear inference (6.08s) on the validation benchmark.
- Hyperparameters: 31 max leaves, balanced class weighting, 100 estimators, learning rate 0.05. Complies with competition rules (<8B parameters, Apache 2.0 license).

**Threshold selection method:** Exhaustive threshold grid search on validation split optimizing entity-level Macro F_0.5
- We swept candidate decision thresholds $\tau \in [0.10, 0.95]$ in increments of 0.05 on the validation split, evaluating the official Entity Macro $F_{0.5}$ metric (including singletons) at each step.
- At the standard classification threshold ($\tau = 0.50$), validation Macro $F_{0.5}$ was 0.70540, suffering 66 singleton false positive merges (58.93% error rate on singletons).
- Setting $\tau = 0.95$ suppressed singleton false positives down to 14 (78.8% reduction), increased Entity Precision to 86.48%, and achieved the global peak Entity Macro $F_{0.5} = \mathbf{0.78402}$.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.78402** (Measured on a deterministic validation split of 2,000 Source 1 entities from a 10,000 dev sample; Entity Precision: **86.48%**, Entity Recall: **65.04%** at calibrated threshold $\tau = 0.95$).
  - *Official Validator Confirmation*: The final submission files were verified using the competition script:
    ```powershell
    python .\utils\validate_submission.py `
      --matching .\AmazonML_Final\output\matching_results.tsv `
      --candidate .\AmazonML_Final\output\candidate_pairs.tsv `
      --test-dir .\dataset\test `
      --check-ids
    ```
    The validator confirmed all 1,732,544 required Source 1 entities and 9,969,589 valid match IDs, returning: `PASS — no blocking issues found. Safe to submit.` (Note: The submission validator confirms file format, entity completeness, and ID validity; it does not calculate the leaderboard score).

- **Common false positives (wrong merges):**
  - *Chain Stores & Franchises*: Entities sharing identical brand names in the same country but situated in different locations (e.g., branches of national banks or retail chains) when address fields are brief or lack street numbers.
  - *Shared Generic Corporate Tokens*: Companies with similar commercial titles (e.g., *"Apex Enterprises"* vs *"Apex Logistics"*) where normalized Levenshtein similarity is moderately high and address data is sparse.

- **Common false negatives (missed matches):**
  - *Address Formatting Discrepancies*: Matches where one source provides a formal postal address with PIN code while another provides descriptive landmark references (*"Opposite Civil Hospital, Station Road"*), resulting in low lexical token overlap.
  - *Substantial Name Truncation / Acronyms*: Entities referred to by acronyms or colloquial trade names without shared n-grams (e.g., *"TCS"* vs *"Tata Consultancy Services"*).
  - *Multiple Severe Corruptions*: Rare instances where both name and address were simultaneously perturbed beyond the top-60 candidate ranking threshold.

---

## 6. Conclusion

We developed an explosion-proof, two-stage entity resolution pipeline that pairs a 7-pass multi-strategy inverted-index blocking engine with a 17-feature LightGBM GBDT classifier, achieving an Entity Macro $F_{0.5}$ score of 0.78402 (86.48% precision) on our validation split. By specifically tuning the decision threshold to $\tau = 0.95$, we aligned model behavior with the competition's precision-heavy $F_{0.5}$ metric and singleton structure, eliminating 78.8% of singleton false-positive merges. The complete pipeline processes over 1.73 million test Source 1 entities and 9.96 million candidate records with memory-safe streaming, successfully passing official submission validation with zero errors.

---

## Appendix

### A. Code Artefacts

The complete, runnable pipeline is organized under `code/business_entity_resolution/`:
- `src/config.py`: Centralized configuration, directory paths, schemas, hyperparameters, and posting caps.
- `src/normalize.py`: C-speed Unicode NFKD normalization, punctuation handling, and legal suffix removal.
- `src/blocking.py`: 7-pass inverted-index candidate generation with posting caps and relevance ranking.
- `src/features.py`: 17-feature pairwise similarity extraction powered by RapidFuzz C-extensions.
- `src/models.py`: Implementations for Rule Baseline, Blocking-Only, Logistic Regression, LightGBM, and Random Forest.
- `src/matching.py`: Candidate pair scoring and threshold filtering.
- `src/threshold.py`: Exhaustive threshold sweep optimizing Entity Macro $F_{0.5}$.
- `src/evaluation.py`: Authoritative evaluation metrics including singleton handling.
- `src/error_analysis.py`: Diagnostic categorization of false positive and false negative error modes.
- `src/data_loader.py`: Memory-safe TSV streaming and schema verification.
- `src/validation.py`: Standalone integrity checks and integration with `utils/validate_submission.py`.
- `src/pipeline.py`: End-to-end training, cross-validation, and streaming test inference.
- `src/submission.py`: TSV deliverable generation and official zip packaging.
- `src/main.py`: Unified CLI entry point for executing individual stages or the full pipeline.
- `src/EntityRelationDeduplication.ipynb`: Research notebook documenting exploratory analysis and modeling experiments.
- `README.md`: Step-by-step reproduction and architecture documentation.
- `requirements.txt`: Pinned production dependencies (`pandas`, `numpy`, `scikit-learn`, `lightgbm`, `rapidfuzz`, `joblib`).

**Reproduction Entry Points:**
```bash
# 1. Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# 2. Offline evaluation, model comparison, and threshold calibration
python code/business_entity_resolution/src/main.py --stage evaluate --sample-size 10000

# 3. Full streaming test inference (generates matching_results.tsv and candidate_pairs.tsv)
python code/business_entity_resolution/src/main.py --stage inference --production

# 4. Submission validation
python code/business_entity_resolution/src/main.py --stage validate

# Full pipeline execution end-to-end:
python code/business_entity_resolution/src/main.py --all
```

### B. Additional Results

#### 1. Model Architecture Benchmark Comparison
Evaluated on a deterministic 8,000 / 2,000 entity-stratified validation split (from `reports/model_comparison.csv`):

| Model Architecture | Optimal Threshold ($\tau$) | Entity Macro $F_{0.5}$ | Entity Precision | Entity Recall | Train Runtime (s) | Inference Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 0: Rule Baseline (Exact Name)** | 0.10 | 0.36738 | 46.22% | 27.78% | 0.00 | 0.04 |
| **Model 1: Blocking-Only Baseline** | 0.10 | 0.14323 | 12.46% | 67.88% | 0.00 | 0.01 |
| **Model 2: Logistic Regression** | 0.95 | 0.72913 | 80.73% | 61.24% | 0.73 | 0.52 |
| **Model 3: LightGBM GBDT (Selected Model)** | **0.95** | **0.78402** | **86.48%** | **65.04%** | **2.18** | **6.08** |
| **Model 4: Random Forest (150 Trees)** | 0.85 | 0.77732 | 84.88% | 66.16% | 28.56 | 71.56 |

#### 2. Decision Threshold Grid Sweep (LightGBM GBDT)
Progression of Entity Macro $F_{0.5}$ across threshold values on the validation split (from `reports/threshold_comparison.csv`):

| Threshold ($\tau$) | Entity Macro $F_{0.5}$ | Entity Precision | Entity Recall | Singleton FPs | Singleton FP Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.10 | 0.63562 | 64.58% | 69.29% | 82 | 73.21% |
| 0.20 | 0.65832 | 67.32% | 69.31% | 80 | 71.43% |
| 0.30 | 0.67751 | 69.69% | 69.52% | 74 | 66.07% |
| 0.40 | 0.69429 | 71.78% | 69.59% | 70 | 62.50% |
| 0.50 | 0.70540 | 73.23% | 69.56% | 66 | 58.93% |
| 0.60 | 0.72142 | 75.21% | 69.74% | 55 | 49.11% |
| 0.70 | 0.74787 | 78.89% | 69.57% | 39 | 34.82% |
| 0.80 | 0.77426 | 83.32% | 68.26% | 29 | 25.89% |
| 0.90 | 0.77959 | 85.05% | 66.26% | 20 | 17.86% |
| **0.95** | **0.78402** | **86.48%** | **65.04%** | **14** | **12.50%** |

#### 3. Blocking Strategy Benchmark
Comparison of individual blocking passes and production multi-pass union on 20,000 validation entities (from `reports/blocking_comparison.csv`):

| Blocking Strategy | Candidate Recall (%) | Mean Candidates | Median Candidates | Max Candidates | Zero-Candidate Rate (%) | Search Space Reduction (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Exact Name + Country | 11.79% | 3.81 | 1 | 40 | 43.82% | 99.999924% |
| 2. Suffix-Stripped Name + Country | 21.86% | 8.15 | 2 | 40 | 20.83% | 99.999838% |
| 3. Sorted Name Tokens + Country | 23.26% | 8.08 | 2 | 35 | 18.02% | 99.999839% |
| 4. Compact Name + Country | 21.94% | 7.73 | 2 | 35 | 20.48% | 99.999846% |
| 5. Name Token + Address Number | 22.00% | 4.76 | 1 | 30 | 30.15% | 99.999905% |
| 6. Address Number + Street Word | 24.68% | 6.14 | 2 | 30 | 26.77% | 99.999878% |
| 7. Controlled Prefix-6 + Country | 10.93% | 17.21 | 20 | 20 | 1.93% | 99.999658% |
| **Production Multi-Pass (7 Passes + Ranking)** | **37.47%** | **28.77** | **24** | **60** | **0.34%** | **99.999429%** |

#### 4. Validated Output Deliverable Statistics
- **Submission Output Files**:
  - `output/matching_results.tsv`: 1,732,544 rows (260,823 singletons/empty predictions, 1,471,721 non-empty predictions).
  - `output/candidate_pairs.tsv`: 1,732,544 rows (376 empty candidate lists, 1,732,168 populated candidate lists).
- **Integrity Verification**:
  - Test Source 1 Entity Coverage: 100.0% (1,732,544 of 1,732,544 required entities present, exactly one row each).
  - Valid Test Match IDs in Catalog: 9,969,589.
  - Candidate Set Containment: Every predicted match in `matching_results.tsv` is strictly a subset of its corresponding entry in `candidate_pairs.tsv`.
  - Duplicate & Self-Match Rate: 0.00% (zero duplicates, zero self-matches).
  - Official Validator Result: `PASS — no blocking issues found. Safe to submit.`

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
