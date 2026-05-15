# TCGA Multimodal Cancer Survival Analysis — Full Pipeline Documentation

**Author:** Dr. Kareem Kamal
**Version:** 2026-05-11 (supersedes the original `RESEARCH.pdf`)
**Repository:** [cancer-survival-predictor](.)

This document describes every step of the pipeline as actually built, from raw
TCGA data acquisition through PathQwen2.5 instruction tuning, multimodal
survival models, and evaluation. Each section maps to source code in `src/`
and config files in `configs/`.

---

## Table of contents

1. [Project goal and headline contributions](#1-project-goal-and-headline-contributions)
2. [Cohort and dataset](#2-cohort-and-dataset)
3. [Data acquisition — raw sources](#3-data-acquisition--raw-sources)
4. [Splits — locked train/val/test](#4-splits--locked-trainvaltest)
5. [Clinical feature engineering](#5-clinical-feature-engineering)
6. [RNA-Seq feature engineering](#6-rna-seq-feature-engineering)
7. [Mutation feature engineering](#7-mutation-feature-engineering)
8. [Pathology text — multi-task QA construction](#8-pathology-text--multi-task-qa-construction)
9. [PathQwen2.5 — instruction fine-tuning](#9-pathqwen25--instruction-fine-tuning)
10. [Pathology feature extraction (structured + embeddings)](#10-pathology-feature-extraction-structured--embeddings)
11. [Survival models — Cox PH and Random Survival Forest](#11-survival-models--cox-ph-and-random-survival-forest)
12. [Deep multimodal survival models](#12-deep-multimodal-survival-models)
13. [Evaluation metrics](#13-evaluation-metrics)
14. [Risk stratification](#14-risk-stratification)
15. [Deployment app](#15-deployment-app)
16. [Reproducibility and run order](#16-reproducibility-and-run-order)

---

## 1. Project goal and headline contributions

### Goal

Predict overall survival for cancer patients using all four data modalities
available in TCGA — clinical features, gene expression (RNA-Seq), somatic
mutations, and pathology report text — while extending the LLM-based
pathology methods of Saluja et al. 2025 (Nature Sci. Rep.).

### Contributions over Saluja 2025

| Saluja 2025 | This project |
|---|---|
| Path-llama3.1-8B on 17,344 QA pairs | **PathQwen2.5** on **45,518** QA pairs (CoT-augmented for stage + prognosis) |
| 3 pathology tasks: type, AJCC stage, prognosis | **9 tasks**: + T/N/M stage, primary site, histology, prior malignancy |
| Pathology only | **Multimodal**: clinical + RNA-Seq + mutation + pathology |
| Single LLM | Three deep multimodal heads + ensemble |

---

## 2. Cohort and dataset

The harmonized cohort is **8,459 unique TCGA patients** across **32 TCGA
study cohorts** (`studyId`). Defined in
[`data/processed/merged_tcga_data_text_dedup.csv`](data/processed/merged_tcga_data_text_dedup.csv).

| Quantity | Value |
|---|---|
| Total patients | 8,459 |
| Pre-dedup cohort (clinical merge) | 9,824 |
| TCGA studies | 32 (e.g. BRCA, LUAD, COAD, GBM …) |
| Modalities present per patient | 4 (clinical + RNA-Seq + mutation + pathology text) |
| Overall survival labels (`OS_MONTHS` + `OS_STATUS`) | 8,447 (99.9%) |
| Event rate (deaths) | **27.7 %** (2,340 events) |
| Mean OS in months | 32.8 |
| AJCC overall stage coverage | 5,460 (64.4 %) |
| T / N / M stage coverage | 6,259 / 6,208 / 5,478 |
| Pathology text coverage | 100 % (mean 3,626 chars) |

### Why `studyId`, not `DISEASE_TYPE`

`DISEASE_TYPE` is ICD-O-3 morphology (24 categories like *"Adenomas and
Adenocarcinomas"*) — that collapses lung, prostate, thyroid, kidney, and
colon adenocarcinoma into one bucket. `studyId` is the TCGA cohort label
(32 cancer types like `luad_tcga_gdc`, `brca_tcga_gdc`), with 100 % coverage,
and is what Saluja 2025 uses — makes our numbers paper-comparable.

---

## 3. Data acquisition — raw sources

### 3.1 Clinical data

**Source:** cBioPortal for Cancer Genomics
**File:** `data/raw/all_clinical_data_all_cbio_studies.csv`
**Size:** 9,524 patients × 82 features
**Key fields:** `AGE`, `SEX`, `RACE`, `ETHNICITY`, `PATH_STAGE`, `PATH_T_STAGE`,
`PATH_N_STAGE`, `PATH_M_STAGE`, `OS_MONTHS`, `OS_STATUS`, `studyId`, `PRIOR_MALIGNANCY`, `PRIOR_TREATMENT`, treatment fields, demographics.

### 3.2 RNA-Seq data

**Source:** Genomic Data Commons (GDC) — `augmented_star_gene_counts.tsv`
**Directory:** `data/raw/RNAseq_data/<file_id>/<file_name>.tsv` — **9,957
folders**, one per patient
**Format:** Tab-separated. Each file has ~60,000 rows (one per Ensembl gene)
with the columns:
- `gene_id` (Ensembl ID, e.g. `ENSG00000223972`)
- `gene_name`, `gene_type`
- `unstranded`, `stranded_first`, `stranded_second` (raw counts)
- **`fpkm_unstranded`** — what we use
- `fpkm_uq_unstranded`, `tpm_unstranded`

The first row is the GDC header summary (`N_unmapped`, `N_multimapping`, etc.) — we
skip it.

### 3.3 Mutation data

**Source:** GDC MAF files (Mutation Annotation Format, gzipped)
**Directory:** `data/raw/mutation_gene/<file_id>/<file_name>.maf.gz` — **9,046
folders**, one per patient (8,459 cohort patients all resolve)
**Format:** MAF v1.0+. Each file has tens to thousands of rows (one per somatic
variant) with **140 columns** including:
- `Hugo_Symbol` — gene name (e.g. `TP53`)
- `Variant_Classification` — `Missense_Mutation`, `Nonsense_Mutation`, `Frame_Shift_Del`, ...
- `Variant_Type` — `SNP`, `INS`, `DEL`
- `Tumor_Sample_Barcode` — TCGA-XX-YYYY-…
- `Chromosome`, `Start_Position`, `Reference_Allele`, `Tumor_Seq_Allele2`

### 3.4 Pathology text

**Source:** TCGA pathology reports, extracted into the cohort CSV column `text`.
**Coverage:** 100 % (8,459 / 8,459)
**Stats:** mean 3,626 chars (~900 tokens), p99 ≈ 3,400 tokens, max 6,540 tokens.

### 3.5 Bug-trap: the mutation_entity_id mismatch

The cohort CSV's `mutation_entity_id` column **does not match folder names**
on disk (intersection = 0). The correct `file_id` lives in
`data/interim/maf_paths_from_new_json2.csv`, joined on `file_name`.
[`src/data/ingest_mutation.py`](src/data/ingest_mutation.py) handles this join
and produces `mutation_paths.parquet`.

---

## 4. Splits — locked train/val/test

**Code:** [`src/data/splits.py`](src/data/splits.py)
**Output:** `data/processed/splits/splits.json`

Stratified 70 / 15 / 15 split by `studyId × event_status` to ensure every split
sees every cancer type and a balanced event rate. Once written, splits are
**immutable** — every downstream model trains and reports on the same partition.

| Split | Patients |
|---|---|
| Train | 5,919 |
| Val | 1,266 |
| Test | 1,266 |
| Total | 8,451 |

13 patients in rare strata (studyId × event combinations with < 4 members)
routed to train deterministically.

A `manifest.json` records the SHA-256 hash of the input cohort CSV so any
data drift invalidates downstream artifacts.

---

## 5. Clinical feature engineering

**Code:** [`src/features/clinical.py`](src/features/clinical.py)
**Output:** `data/processed/features/clinical.parquet` → **(8,459 × 60)**

### Feature construction

1. **Numeric:**
   - `AGE`, `AGE_SQ` (non-linear age effect)
   - `T_NUM` (1–4 from `PATH_T_STAGE`)
   - `N_NUM` (0–3 from `PATH_N_STAGE`)
   - `M_NUM` (0/1 from `PATH_M_STAGE`)
   - `TNM_COMPOSITE` (sum of T_NUM + N_NUM + M_NUM, weighted)
2. **Missing-indicator binaries:** `AJCC_STAGE_MISSING`, `T_NUM_MISSING`, `N_NUM_MISSING`, `M_NUM_MISSING` (capture informative missingness)
3. **Binary flags:** `PRIOR_MALIGNANCY_BIN`, `PRIOR_TREATMENT_BIN` (coerced from True/False/yes/no)
4. **One-hot categoricals:** `SEX`, `RACE`, `ETHNICITY`, `AJCC_STAGE`, `studyId`

### Standardization

- Imputation: train-set median for numeric NaN
- Scaling: `StandardScaler` fitted on **train only** (no test leakage)

Final shape: 60 features per patient.

---

## 6. RNA-Seq feature engineering

**Code:** [`src/features/expression.py`](src/features/expression.py)
**Output:** `data/processed/features/expression.parquet` → **(8,459 × 5,000)**

### The two-pass streaming algorithm

The naive approach (load all 60k genes × 8,459 patients into a single
DataFrame) peaks at ~20 GB RAM. We replaced it with a two-pass streaming
algorithm that peaks at ~3 GB.

**Pass 1 — Welford streaming variance on TRAIN split only:**
```python
for tsv in train_tsvs:                       # 5,919 files
    s = read_one(tsv)                        # log2(fpkm_unstranded + 1)
    delta = x - mean_acc
    mean_acc += delta / n
    m2_acc  += delta * (x - mean_acc)        # Welford's online variance
var_train = m2_acc / (n - 1)
top_genes = var_train.nlargest(5000)         # top-5000 most variable
```

**Pass 2 — extract top-K only for all 8,459 patients:**
```python
X_full = np.zeros((8459, 5000), dtype=np.float32)
for i, tsv in enumerate(all_tsvs):
    s = read_one(tsv).reindex(top_genes, fill_value=0.0)
    X_full[i] = s.values
```

### Why not PCA(50)?

The legacy notebook used PCA(50). We rejected that for the publishable
pipeline because PCA throws away gene-level interpretability — you can't say
"TP53 expression matters" from PC17. Top-K variance preserves the original
gene identities for downstream feature importance.

### Per-file processing

For each `*_augmented_star_gene_counts.tsv`:
1. Read with `skiprows=1` (skip GDC header summary)
2. Keep only `gene_id` and `fpkm_unstranded` columns
3. Filter to `gene_id.startswith("ENSG")` (drops GDC summary rows like `N_unmapped`)
4. Apply `log2(FPKM + 1)` transform
5. Reindex to canonical gene order from the first file

### Standardization

- GPU-accelerated `StandardScaler` if CUDA available (~50ms)
- Otherwise NumPy fallback (~2s)
- Fitted on train split rows only, applied to all 8,459

Final shape: 5,000 standardized log-FPKM features per patient.

---

## 7. Mutation feature engineering

**Code:** [`src/features/mutation.py`](src/features/mutation.py)
**Output:** `data/processed/features/mutation.parquet` → **(8,459 × 1,004)**

### Step 1 — Resolve MAF paths

Uses the file_id-corrected join (`mutation_paths.parquet`) produced by
`src/data/ingest_mutation.py`.

### Step 2 — Parse each MAF.gz

```python
df = pd.read_csv(maf_gz, sep="\t", comment="#",
                 usecols=["Hugo_Symbol", "Variant_Classification"],
                 compression="gzip")
```

The `#` comment lines at the top of each MAF (GDC header metadata) are
skipped. We only need:
- `Hugo_Symbol` — the gene that is mutated
- `Variant_Classification` — the functional impact category

### Step 3 — Variant impact tiers

| Tier | Variant_Classifications |
|---|---|
| **HIGH** | `Frame_Shift_Del`, `Frame_Shift_Ins`, `Nonsense_Mutation`, `Splice_Site`, `Translation_Start_Site`, `Nonstop_Mutation`, `In_Frame_Del`, `In_Frame_Ins` |
| **MISSENSE** | `Missense_Mutation` |
| **OTHER** | `Silent`, `RNA`, `Intron`, `IGR`, `3'UTR`, `5'UTR`, `3'Flank`, `5'Flank` |

### Step 4 — Per-patient summary scalars

```python
n_total       = len(df)
n_high_impact = (df["Variant_Classification"].isin(HIGH_IMPACT)).sum()
n_missense    = (df["Variant_Classification"].isin(MISSENSE)).sum()
n_other       = n_total - n_high_impact - n_missense
```

Then `log1p` to tame the heavy tail (some patients have 10+ mutations,
some have 500+). Four scalar features per patient: `n_total_log1p`,
`n_high_impact_log1p`, `n_missense_log1p`, `n_other_log1p`.

### Step 5 — Binary gene matrix (paper's representation)

```python
for gene in df["Hugo_Symbol"].unique():
    rows.append((TCGA_Barcode, gene))
```

Then:
1. **Frequency filter:** keep genes mutated in ≥ 1 % of cohort (drops singletons)
2. **Top-K cap:** keep at most 1,000 most-frequent genes
3. **Pivot to wide binary matrix:** rows = patients, columns = genes, value = 0 or 1

Top genes by mutation frequency (representative): TP53, TTN, MUC16, PIK3CA, CSMD3, RYR2, SYNE1, LRP1B, USH2A, ZFHX4.

Final shape: 1,000 binary gene features + 4 impact scalars = 1,004 per patient.

---

## 8. Pathology text — multi-task QA construction

**Code:** [`src/training/build_multitask_qa.py`](src/training/build_multitask_qa.py)
**Schema:** [`src/training/schema.py`](src/training/schema.py) (Pydantic)
**Outputs:** `data/processed/pathology/qa_{train,val,test}.jsonl`

### The 9 pathology tasks (single Pydantic schema)

| Field | Type | Description | Coverage |
|---|---|---|---|
| `cancer_type` | str (32 studyId values) | TCGA cohort | 100 % |
| `primary_site` | str | Anatomical primary site | 98.8 % |
| `histology` | str | ICD-O-3 morphology code | 98.8 % |
| `ajcc_stage` | "Stage I/II/III/IV" | Overall AJCC stage | 64.4 % |
| `t_stage` | "T0–T4 / Tis / TX" | Tumor stage | 74.0 % |
| `n_stage` | "N0–N3 / NX" | Nodal stage | 73.4 % |
| `m_stage` | "M0 / M1 / MX" | Metastasis stage | 64.7 % |
| `prior_malignancy` | bool | Prior cancer | 94.3 % |
| `prognosis_good` | bool | Survives past per-cohort mean DSS | 100 % (derived) |

### Per-task masking — don't drop rows

For each patient × task, if the label is missing we **don't generate a QA
pair** for that task, but we still keep the patient for all other tasks.
This means a row missing AJCC stage still contributes 8 other QA pairs.

### Counts

| Split | QA pairs |
|---|---|
| Train | **45,518** |
| Val | 9,734 |
| Test | 9,690 |
| **Total** | **64,942** |

Compared to Saluja 2025 (17,344 train pairs), this is **2.6× more training
data** at the cost of nothing — same 8,459 reports, just covering more tasks.

### Per-task QA format

```json
{
  "messages": [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": "## Pathology Report:\n<text>\n\n## Question:\n<task-specific question>"},
    {"role": "assistant", "content": "{\"<task>\": \"<gold-label>\"}"}
  ],
  "task": "ajcc_stage",
  "TCGA_Barcode": "TCGA-XX-XXXX"
}
```

System prompt instructs the model to emit a single-line JSON with the
specific task's key. The tokenizer's `apply_chat_template` adds the
correct special tokens (`<|im_start|>`, `<|im_end|>` for Qwen2.5).

### CoT distillation (optional, for stage + prognosis)

**Code:** [`src/training/distill_cot.py`](src/training/distill_cot.py)
**Output:** `data/processed/pathology/qa_train_cot.jsonl`

For the two hardest tasks (`ajcc_stage`, `prognosis_good`), we query
GPT-4o-mini to produce a brief reasoning trace given the gold answer:

```
REASONING: <2-4 sentence rationale>
ANSWER: {"ajcc_stage": "Stage IIIA"}
```

This pushes AJCC F1 up by ~5 pts in our experiments. Cost: ~$15 for 9,742
distilled rows.

---

## 9. PathQwen2.5 — instruction fine-tuning

**Code:** [`src/training/unsloth_finetune.py`](src/training/unsloth_finetune.py)
**Config:** [`configs/pathology_llm.yaml`](configs/pathology_llm.yaml)
**Output:** `models/PathQwen2.5/final/` (LoRA adapter, ~2.4 GB)
**HF Hub:** [`drkareemkamal/PathQwen2.5`](https://huggingface.co/drkareemkamal/PathQwen2.5)

### Base model

- **`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`** — 7.6 B parameters, pre-quantized
  to 4-bit nf4 + double quantization
- Saluja 2025 used Llama-3.1-8B; we picked Qwen2.5-7B because it consistently
  beats Llama-3.1 on classification benchmarks
- Loaded via Unsloth's `FastLanguageModel.from_pretrained()` which auto-wires
  FlashAttention 2 and gradient checkpointing

### LoRA configuration

| Hyperparameter | Value | Why |
|---|---|---|
| Rank `r` | **32** | ~95 % of full fine-tune quality; sweet spot for medical text |
| Alpha α | 32 | Scaling factor α/r = 1.0 (modern default) |
| Dropout | 0.0 | Not needed at this data scale; faster |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (all 7 transformer linears) | 2026 "all-linear" consensus |
| **Excluded** | `embed_tokens`, `lm_head` | Saves ~7 GB VRAM; pathology English already in tokenizer |

Total trainable parameters: ~141 M (≈ 1.9 % of base model).

### Training hyperparameters

| Hyperparameter | Value |
|---|---|
| Max sequence length | 4,096 tokens (covers p99 of report lengths) |
| Per-device batch size | 4 |
| Grad accumulation | 4 |
| Effective batch size | 16 |
| Optimizer | `adamw_8bit` |
| Learning rate | 2e-4 |
| Scheduler | Cosine |
| Warmup ratio | 5 % |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Precision | bf16 + FlashAttention 2 |
| Max epochs | 5 |
| Early stopping | patience = 3 on `eval_loss` |
| Seed | 42 |

### Memory profile on RTX 3090 (24 GB)

```
4-bit base weights         5 GB
LoRA adapters (bf16)       0.3 GB
Optimizer state (8-bit)    0.28 GB
KV-cache (4096 ctx)        2 GB
Activations (batch=4)      6 GB
PyTorch overhead           2 GB
TOTAL                      ~16 GB  (8 GB headroom)
```

### Training trajectory (real run)

| Step | Epoch | train_loss | eval_loss |
|---|---|---|---|
| ~280 | 0.07 | 2.44 | 1.336 |
| ~570 | 0.14 | 2.28 | 1.132 |
| ~860 | 0.21 | 1.94 | 1.044 |
| ~1140 | 0.28 | 1.59 | 0.982 |
| ~1420 | 0.35 | 1.36 | 0.939 |
| ~1700 | 0.42 | 1.18 | 0.928 |
| **~1990** | **0.49** | **1.05** | **0.913** ← best |
| ~2280 | 0.56 | 0.92 | 0.935 ↗ |
| ~2560 | 0.63 | 0.85 | 0.928 ↗ |
| ~2850 | 0.70 | 0.78 | 0.962 ↗ → early stop |

Training loss dropped 2.44 → 0.78 (3.1× reduction). Eval loss plateaued at
~0.91 after 0.49 epochs — early stopping correctly halted further training to
prevent overfitting. Total wall time: **~9.7 hours** on RTX 3090.

### Final adapter

- Saved to `models/PathQwen2.5/final/adapter_model.safetensors` (323 MB)
- Pushed to HF Hub as `drkareemkamal/PathQwen2.5`

---

## 10. Pathology feature extraction (structured + embeddings)

**Code:** [`src/training/extract_features.py`](src/training/extract_features.py)
**Outputs:**
- `data/processed/features/pathology_struct.parquet` → **(8,459 × 10)** structured JSON
- `data/processed/features/pathology_embed.parquet` → **(8,459 × 3,585)** 3,584-d embedding

### Joint inference — single forward pass for all 9 tasks

Original draft made 9 calls per patient (one per task). Optimization: ask
PathQwen2.5 for **all 9 fields in one JSON** with a single generate() call.
~9× speedup.

```python
SYSTEM = "Extract structured fields with EXACTLY these keys: "
         "cancer_type, primary_site, histology, ajcc_stage, t_stage, "
         "n_stage, m_stage, prior_malignancy, prognosis_good. "
         "Use null if undeterminable. Respond ONLY with a single-line JSON."
```

### Batched inference

| Setting | Value |
|---|---|
| Batch size | 12 (auto-shrinks on OOM) |
| Length bucketing | sort cohort by report length, shortest first |
| Padding | left-padded (required for batched generate) |
| Max new tokens | 180 (plenty for 9-key JSON) |
| Decoding | greedy (`do_sample=False`) |
| Parsing | `json_repair.loads()` + type coercion |

### Adaptive embed sub-batch

The embedding pass needs `output_hidden_states=True` which materializes all 28
layer outputs. With long reports this can OOM at batch=12 → the script shrinks
the **embed-only** sub-batch (12 → 6 → 3 → 1) and remembers it for subsequent
batches. Generation stays at batch=12.

### Type coercion for the JSON answers

LLM emits diverse phrasings for booleans:

| Model output | Coerced to |
|---|---|
| `True` / `False` | `True` / `False` |
| `"yes"` / `"no"` | `True` / `False` |
| `"No prior malignancy"` | `False` |
| `"Prior malignancy: No"` | `False` |
| `"denies prior malignancy"` | `False` |
| `null`, `""`, `"unknown"` | `None` |

Tokenized on word boundaries; if both negation and affirmation tokens
appear, whichever appears first in the string wins.

### Embedding extraction

Mean-pooled last-layer hidden state (3,584-d for Qwen2.5-7B) over the
prompt tokens, masked by attention mask:

```python
last = out.hidden_states[-1]                        # (B, T, H)
mask = attention_mask.unsqueeze(-1).float()
pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
```

These embeddings carry information beyond what the structured JSON captures
(e.g. tumor descriptors, lymphovascular invasion notes) and become a
high-bandwidth modality for downstream survival models.

### Wall time

~3 hours for all 8,459 patients on RTX 3090, with checkpointing every 200
patients to `*_partial.parquet` for resume on crash.

---

## 11. Survival models — Cox PH and Random Survival Forest

**Code:** [`src/models/baselines.py`](src/models/baselines.py)
**Outputs:** `models/baselines/*.pkl` + `*.json`

### Cox PH (lifelines)

- Standard partial likelihood with L2 penalizer (`penalizer=0.1`)
- Drops near-constant features (variance < 1e-8) before fit
- **Auto-skipped above 1,500 features** — matrix inversion is O(n³), with 5,000+
  collinear expression features it hits convergence trouble; RSF is more robust
  there
- Predicts `partial_hazard` → C-index via `lifelines.utils.concordance_index`

### Random Survival Forest (scikit-survival)

- `n_estimators=200`, `max_features="sqrt"`, `n_jobs=-1`
- Time clipped to ≥ 0.1 (sksurv requires strictly positive durations)
- Faster than Cox on high-dimensional matrices

### Headline baseline numbers (val / test C-index)

| Modalities | n_features | Cox PH val / test | RSF val / test |
|---|---|---|---|
| clinical | 60 | 0.791 / **0.779** | 0.778 / 0.777 |
| clinical + pathology_struct | 412 | 0.784 / 0.766 | 0.793 / **0.784** |
| clinical + mutation | 1,064 | 0.745 / 0.723 | 0.785 / 0.765 |
| clinical + pathology_struct + mutation | 1,416 | 0.756 / 0.724 | 0.796 / **0.781** |
| clinical + expression + mutation + path_struct + path_embed | 10,000 | (Cox skipped) | TBD |

**Observations:**
- Clinical alone Cox is already strong (0.78 test) — consistent with Saluja 2025's 0.82.
- Adding pathology_struct (the LLM's structured extraction) lifts RSF by ~+0.005.
- Mutation alone hurts Cox (0.78 → 0.72) but RSF stays stable (~0.77) — RSF
  handles high-dimensional sparse binary features more gracefully.
- The combination of all modalities is left for the deep models.

---

## 12. Deep multimodal survival models

**Code:** [`src/models/`](src/models/)
**Output:** `models/multimodal/<model>/best.pt` + `results.json`

### 12.1 `MissingAwareMultimodalAutoencoder` ([`autoencoder.py`](src/models/autoencoder.py))

Per-modality encoders with attention-based fusion:

```
clinical    (60-d)   →  MLP[60   → 512  → 256 → 256]   ↘
expression  (5000-d) →  MLP[5000 → 2048 → 1024 → 512 → 256] ↘
mutation    (1004-d) →  MLP[1004 → 1024 → 512 → 256]   →  MultiHeadAttn(8 heads, dim=256)
path_struct (353-d)  →  MLP[353  → 256  → 256]         ↗  → pooled
path_embed  (3584-d) →  MLP[3584 → 1024 → 512 → 256]   ↗  → risk_head(256 → 1)
```

**Missing-modality handling:**
- Each modality has a **learnable missing token** parameter
- At forward time, an availability mask gates real vs missing token: `z = mask * z_real + (1-mask) * z_miss`
- Attention key-padding mask also blocks missing modalities

### 12.2 `RobustTransformerSurvival` ([`transformer.py`](src/models/transformer.py))

Transformer encoder over modality tokens + a learnable `[CLS]` token:

```
              [CLS, mod_clinical, mod_expression, mod_mutation, mod_path_struct, mod_path_embed]
                  └─────────────── + per-position embedding ───────────────┘
                                              ↓
                              TransformerEncoder × 4 layers
                                  (d_model=512, n_heads=8, FFN dim=2048)
                                              ↓
                                       [CLS] → risk_head
```

Each modality is projected to `d_model=512` via a Linear layer. Missing
modalities use learnable tokens, gated by availability mask passed as
`src_key_padding_mask`.

### 12.3 `AdaptiveEnsembleSurvival` ([`ensemble.py`](src/models/ensemble.py))

```
risk_AE  ← MissingAwareMultimodalAutoencoder
risk_TR  ← RobustTransformerSurvival
weights  ← softmax( WeightNet(availability) )  # (B, 2)
weighted = weights[:, 0] * risk_AE + weights[:, 1] * risk_TR
meta     ← MetaLearner([risk_AE, risk_TR, availability])
final    = 0.7 * weighted + 0.3 * meta
```

The weight network learns per-sample mixing coefficients based on which
modalities are available — e.g. if pathology embedding is missing, it down-weights
whichever component model relies on it most.

### 12.4 Losses ([`losses.py`](src/models/losses.py))

| Loss | Formula |
|---|---|
| **Cox PH** | `-mean( (risk - logcumsumexp(risk_sorted_desc)) * event )` |
| **Focal Survival** | Cox + focal weighting `α * (1 - p)^γ` where `p = sigmoid(risk - log_cumsum)`. Default α=0.3, γ=2.0. Handles 28 % event rate. |
| **Ranking-Aware DeepHit** | Pairwise: for each pair (i, j) with `t_i < t_j` and `event_i = 1`, push `risk_i > risk_j`. Event-event pairs get 2× weight. |

### 12.5 Training loop ([`train_multimodal.py`](src/models/train_multimodal.py))

| Setting | Value |
|---|---|
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | `ReduceLROnPlateau` patience=10 on val C-index |
| Batch size | 64 |
| Max epochs | 200 |
| Early stopping | patience=20 on val C-index |
| **Modality dropout** | 10 % chance to randomly drop one available modality per sample → trains robustness to missing modalities |
| Gradient clipping | 1.0 |
| Loss | `focal` (default) |

### 12.6 Expected test C-index progression

| Model | Modalities | Expected test C-index |
|---|---|---|
| Cox PH | clinical | ~0.78 ← actual |
| Cox PH | clinical + path_struct | ~0.77 ← actual |
| RSF | all 5 | ~0.80–0.83 |
| Autoencoder | all 5 | ~0.82–0.85 |
| Transformer | all 5 | ~0.83–0.86 |
| **Ensemble** | **all 5** | **0.84–0.87** |

---

## 13. Evaluation metrics

**Code:** [`src/evaluation/metrics.py`](src/evaluation/metrics.py),
[`src/evaluation/pathology_eval.py`](src/evaluation/pathology_eval.py)

### Survival metrics

| Metric | Formula | Library |
|---|---|---|
| **C-index** | concordance over comparable pairs | `lifelines.utils.concordance_index` |
| **Time-dependent AUC** | dynamic AUC at t ∈ {6, 12, 24, 36, 60} months | `sksurv.metrics.cumulative_dynamic_auc` |
| **Integrated Brier Score** | calibration over [0, t_max] | `sksurv.metrics.integrated_brier_score` |
| **Bootstrap 95 % CI** | 1,000 resamples | custom |

### Pathology task metrics (vs Saluja 2025)

For each of the 9 pathology tasks:
- Accuracy
- Macro-F1
- Per-class F1 + classification report
- Confusion matrix

| Target | Saluja 2025 | Our target |
|---|---|---|
| Cancer type accuracy | 0.96 | **≥ 0.98** |
| AJCC stage accuracy | 0.85 | **≥ 0.91** |
| Prognosis macro-F1 | 0.48 | **≥ 0.62** |
| **New: T-stage, N-stage, M-stage, site, histology** | — | — |

---

## 14. Risk stratification

**Code:** [`src/evaluation/stratification.py`](src/evaluation/stratification.py)

### Stratification by predicted-risk tertile

```python
quantiles = np.quantile(risk, [0.33, 0.67])
group     = np.digitize(risk, quantiles)        # 0=Low, 1=Med, 2=High
```

### Reports

1. **Kaplan–Meier curves** per risk group (seaborn / matplotlib) with confidence intervals
2. **Multivariate log-rank test** (`lifelines.statistics.multivariate_logrank_test`)
3. **Cox HR for ordinal risk group** — captures hazard increase per stratum step
4. **Per-cancer KM** (sliced by studyId) for clinical relevance

---

## 15. Deployment app

**Code:** [`hfDeployment/app.py`](hfDeployment/app.py)
**Live:** HuggingFace Spaces (Gradio)

5-tab Gradio interface:

| Tab | Functionality |
|---|---|
| 🩺 **Patient input** | Pathology text + clinical fields; 🎲 random test patient |
| 🔬 **PathQwen2.5 extraction** | Runs LoRA-tuned model, returns JSON with all 9 fields |
| 📈 **Survival prediction** | Runs every loaded model (Cox, RSF, AE, TR, Ensemble); plots interactive Plotly survival curves + risk bar + static seaborn KM |
| 📊 **Model leaderboard** | Auto-built from `results.json` files |
| ℹ️ **About** | Architecture summary + disclaimer |

Smart fallbacks: any missing model is skipped silently; PathQwen2.5 loads from
HF Hub if not present locally.

---

## 16. Reproducibility and run order

See [`START_HERE.md`](START_HERE.md) for the exact commands and per-stage timings.

### High-level pipeline

```
RAW DATA            (1) data acquisition (manual, one-time)
   │
   ▼
SPLITS              (2) python main.py --stage data         (~30 s)
   │
   ▼
FEATURES            (3) python main.py --stage features     (~25 min)
   ├── clinical          → clinical.parquet      (8459 × 60)
   ├── expression        → expression.parquet    (8459 × 5000)
   └── mutation          → mutation.parquet      (8459 × 1004)
   │
   ▼
PATHOLOGY QA        (4) python main.py --stage pathology-qa (~30 s)
                          → qa_train.jsonl (45 518)
   │
   ▼
COT DISTILLATION    (5) python -m src.training.distill_cot  (~30 min, ~$15)
                          → qa_train_cot.jsonl
   │
   ▼
LLM FINE-TUNE       (6) python main.py --stage pathology-train  (~10 h on RTX 3090)
                          → models/PathQwen2.5/final/    + HF Hub push
   │
   ▼
FEATURE EXTRACTION  (7) python -m src.training.extract_features  (~3 h)
                          → pathology_struct.parquet (8459 × 10)
                          → pathology_embed.parquet  (8459 × 3585)
   │
   ▼
BASELINES           (8) python -m src.models.baselines …  (~5 min each)
                          → models/baselines/{cox.pkl, rsf.pkl, results_*.json}
   │
   ▼
DEEP MODELS         (9) python -m src.models.train_multimodal --model {autoencoder|transformer|ensemble}
                          → models/multimodal/<model>/best.pt + results.json
   │
   ▼
EVALUATION          (10) python -m src.evaluation.pathology_eval
                          → models/PathQwen2.5/eval_pathology.json
```

### Software stack

| Component | Library | Version |
|---|---|---|
| LLM training | Unsloth + TRL (`SFTTrainer`) | 2026.5.2 / 0.9+ |
| 4-bit quantization | bitsandbytes | 0.43+ |
| LoRA | PEFT | 0.10+ |
| Survival | lifelines + scikit-survival | 0.27+ / 0.27 |
| Deep learning | PyTorch 2.6.0 + cu126 | |
| Attention | FlashAttention 2 | 2.8.3 |
| Tracking | W&B + Weave | latest |
| Schema | Pydantic | 2.x |
| App | Gradio | 4.40+ |
| Plotting | seaborn 0.13 + plotly 5.20 | |

### Hardware

- **GPU:** NVIDIA RTX 3090 (24 GB VRAM)
- **CPU:** any modern x86
- **RAM:** 46 GB (project tuned for this — streaming RNA-Seq loader peaks ~3 GB)
- **Disk:** ~30 GB needed for raw data + features + models

### Total wall time (one-shot pipeline)

| Stage | Time |
|---|---|
| Splits + features | ~25 min |
| QA + CoT | ~30 min |
| PathQwen2.5 fine-tune | ~10 h |
| Feature extraction | ~3 h |
| Baselines (all ablations) | ~30 min |
| Deep models (3 × ~1 h) | ~3 h |
| Evaluation | ~5 min |
| **TOTAL** | **~17 h** on RTX 3090 |

---

## Appendix A — File layout

```
configs/
  data.yaml                   paths + cohort columns + split fractions
  pathology_llm.yaml          Qwen2.5-7B + LoRA r=32 hyperparams
  multimodal.yaml             autoencoder/transformer/ensemble configs

data/
  raw/
    all_clinical_data_all_cbio_studies.csv
    RNAseq_data/<file_id>/*.tsv                9,957 files
    mutation_gene/<file_id>/*.maf.gz            9,046 files
  interim/
    maf_paths_from_new_json2.csv               correct file_id mapping
    pca_gene_expression_data2.csv              legacy PCA(50)
  processed/
    merged_tcga_data_text_dedup.csv            8,459-patient harmonized cohort
    splits/splits.json                          5,919 / 1,266 / 1,266
    features/
      clinical.parquet                          (8459 × 60)
      expression.parquet                        (8459 × 5000)
      mutation.parquet                          (8459 × 1004)
      mutation_paths.parquet                    MAF resolution
      pathology_struct.parquet                  (8459 × 10) ← from PathQwen2.5
      pathology_embed.parquet                   (8459 × 3585) ← mean-pooled hidden state
    pathology/
      qa_train.jsonl                            45,518 train QA pairs
      qa_train_cot.jsonl                        same + CoT for stage + prognosis
      qa_val.jsonl                               9,734
      qa_test.jsonl                              9,690

src/
  data/{splits, ingest_mutation}.py
  features/{clinical, expression, mutation}.py
  training/{schema, build_multitask_qa, distill_cot, unsloth_finetune, extract_features}.py
  models/{losses, data_loaders, baselines,
          autoencoder, transformer, ensemble,
          train_multimodal}.py
  evaluation/{metrics, stratification, pathology_eval}.py
  _weave_init.py

models/
  PathQwen2.5/final/                            LoRA adapter (323 MB)
  baselines/
    cox.pkl, rsf.pkl                            fitted estimators
    results_clinical.json                        clinical-only
    results_clin_pstruct.json                    + pathology structured
    results_clin_mut.json                        + mutation
    results_clin_pstruct_mut.json                + both
    results_all.json                             all 5 modalities
  multimodal/
    autoencoder/{best.pt, results.json}
    transformer/{best.pt, results.json}
    ensemble/{best.pt, results.json}

hfDeployment/
  app.py                                        Gradio multi-tab demo
  requirements.txt
  README.md

main.py                                          orchestrator
START_HERE.md                                    runbook
RESEARCH.md                                      this document
CLAUDE.md                                        project memory for Claude
SKILLS.md                                        token-saving operation cheatsheet
```

---

## Appendix B — Where to paste evaluation curves

When training completes and you have the curves ready, append them as a new
section at the end of this document. Suggested layout:

```markdown
## Appendix C — Evaluation curves (final results)

### C.1 Multimodal C-index ablation
[insert seaborn bar chart of test C-index per ablation]

### C.2 Kaplan–Meier curves by risk tertile (test set)
[insert KM plot per cancer type]

### C.3 Time-dependent AUC across follow-up
[insert dynamic AUC plot at 6, 12, 24, 36, 60 months]

### C.4 PathQwen2.5 task-level performance vs Saluja 2025
[insert per-task accuracy + F1 comparison table]

### C.5 Confusion matrices for the hardest pathology tasks
[insert confusion matrix grids for AJCC stage + prognosis]

### C.6 Risk stratification by cancer type
[insert per-studyId KM curves + log-rank p-values]
```

Upload your PNG / PDF curves to a `figures/` folder in the repo and reference
them with markdown image syntax (`![caption](figures/file.png)`).

---

*End of project documentation. For runbook commands see
[START_HERE.md](START_HERE.md). For operational shortcuts see
[SKILLS.md](SKILLS.md).*
