# 🧬 Cancer Survival Prediction from Pathological Text Reports

## Multimodal Deep Learning for Oncological Survival Analysis using TCGA Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-orange.svg)](https://huggingface.co/)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Clinical Significance](#-clinical-significance)
- [Data Acquisition Pipeline](#-data-acquisition-pipeline)
- [Data Integration & Preprocessing](#-data-integration--preprocessing)
- [Feature Engineering](#-feature-engineering)
- [Model Architecture](#-model-architecture)
- [Fine-Tuning Strategies](#-fine-tuning-strategies)
- [Training](#-training)
- [Evaluation & Benchmarks](#-evaluation--benchmarks)
- [Deployment Guide](#-deployment-guide)
- [Clinical Applications](#-clinical-applications)
- [Limitations & Ethics](#-limitations--ethics)
- [Project Structure](#-project-structure)
- [Citation](#-citation)

---

## 🔬 Project Overview

This project implements a comprehensive multimodal deep learning approach for cancer survival prediction using data from **The Cancer Genome Atlas (TCGA)**. The system integrates **clinical, genomic, mutation, and pathological text data** to predict patient survival outcomes using state-of-the-art language models.

By fine-tuning biomedical language models on pathological text reports, we:

1. **Extract survival-relevant features** from unstructured pathological text
2. **Predict patient risk scores** correlating with survival probability
3. **Stratify patients** into high-risk and low-risk groups for treatment planning
4. **Generate embeddings** for multi-modal survival pipelines

### Models Used

| Model | Parameters | Quantization | VRAM Usage |
|---|---|---|---|
| **Bio_ClinicalBERT** | ~110M | None (FP32) | ~3.8 GB |
| **OpenBioLLM-8B** | ~8B | 4-bit NF4 | ~6-8 GB |

### Key Innovation

Unlike traditional survival analysis using hand-crafted features (stage, grade, tumor size), our approach **directly learns from raw pathological text**, capturing subtle linguistic patterns — such as pathologist phrasing correlating with aggressiveness, specific morphological descriptions, or diagnostic uncertainty language.

---

## 🏥 Clinical Significance

### Why This Matters

**1. Unstructured Text is Underutilized**: Over 80% of clinical data exists as unstructured text in EHRs. Pathological reports contain rich, expert-curated descriptions rarely used in computational models.

**2. Personalized Risk Assessment**: Traditional staging systems (TNM, AJCC) group patients into broad categories. Our model provides a **continuous risk score** capturing individual nuances.

**3. Decision Support at Diagnosis**: When a pathologist writes a report, this model provides an **immediate risk estimate** — within seconds — supporting tumor board discussions.

**4. Multi-Cancer Applicability**: Trained on 32 TCGA cancer cohorts spanning 24 disease types.

### Survival Analysis Method

We use the **Cox Proportional Hazards (Cox PH)** loss function:
- Handles **censored data** (patients still alive at last follow-up)
- Models the **relative hazard** rather than absolute survival time
- Evaluated using the **Concordance Index (C-index)**: 0.5 = random, >0.7 = strong

---

## 📥 Data Acquisition Pipeline

### 1.1 Clinical Data from cBioPortal

| Property | Value |
|---|---|
| **Source** | [cBioPortal for Cancer Genomics](https://www.cbioportal.org/) |
| **File** | `data/raw/all_clinical_data_all_cbio_studies.csv` |
| **Samples** | 9,524 patient samples |
| **Features** | 82 clinical features |
| **Key Features** | Age, Sex, Race, TNM staging, Pathological stages, Treatment history |
| **Survival Endpoints** | Overall Survival (`OS_MONTHS`, `OS_STATUS`) |

**How to obtain**: Download from cBioPortal by querying all TCGA studies and exporting the clinical data as CSV.

### 1.2 Gene Expression Data from GDC Repository

| Property | Value |
|---|---|
| **Source** | [Genomic Data Commons (GDC)](https://portal.gdc.cancer.gov/) |
| **Data Type** | RNA-Seq expression profiles (STAR gene counts) |
| **File Format** | TSV files per sample (`.rna_seq.augmented_star_gene_counts.tsv`) |
| **Raw Size** | ~10 GB compressed (`data/raw/RNAseq_data.zip`) |
| **Initial Genes** | ~60,000 genes across 8,459 samples |
| **Final Gene Set** | ~5,000 protein-binding genes (filtered by variance) |
| **Expression Values** | FPKM-normalized, log2-transformed: `log2(FPKM + 1)` |

**Processing Pipeline** (from `notebooks/srvival-analysis.ipynb`):

```python
# 1. Load metadata with TCGA barcodes and file paths
merged_df = pd.read_csv("final_clinical_survival_dataset2.csv")
valid_df = merged_df[merged_df["tsv_path"].notna()]
path_to_barcode = dict(zip(valid_df["tsv_path"], valid_df["TCGA_Barcode"]))

# 2. For each RNA-Seq file:
for path in tqdm(path_to_barcode):
    df = pd.read_csv(path, sep="\t", skiprows=1,
                     usecols=["gene_id", "fpkm_unstranded"])
    df['expression'] = np.log2(df['expression'] + 1)  # Log2 transform
    df['sample_id'] = path_to_barcode[path]

# 3. Pivot → gene × sample matrix
expression_matrix = combined_expression.pivot(
    index="gene_id", columns="sample_id", values="expression"
)

# 4. Filter low-variance genes (threshold > 0.5)
high_var_genes = expression_matrix.var(axis=1)[lambda x: x > 0.5].index

# 5. PCA to 50 components
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

# Output: pca_gene_expression_data_final.csv
```

**Memory optimization**: Processed in chunks of 50 files to handle low-memory environments (186 chunks for the full dataset).

### 1.3 Mutation Data from GDC Repository

| Property | Value |
|---|---|
| **Source** | GDC Repository mutation annotation files |
| **File** | `data/raw/mutation_gene.zip` (~871 MB) |
| **Data Type** | Somatic mutation calls (MAF format) |
| **Processing** | Binary mutation matrix (1 = mutated, 0 = not mutated) |
| **Final Format** | 8,459 samples × ~1,000 mutated genes |

### 1.4 Pathological Text Reports

| Property | Value |
|---|---|
| **Source** | Clinical pathology reports from TCGA |
| **File** | `TCGA_Reports.csv` (9,523 samples × 2 columns: `patient_filename`, `text`) |
| **Embedding Model** | Bio_ClinicalBERT (Alsentzer et al., 2019) |
| **Embedding Dimension** | 768-dimensional per sample |

---

## 🔗 Data Integration & Preprocessing

### Sample Alignment

The core challenge: **unequal sample sizes across modalities**.

| Modality | Samples |
|---|---|
| Clinical data | 9,824 |
| Gene expression (RNA-Seq) | 8,459 |
| Mutation data | 8,459 |
| Text embeddings | 9,523 |
| **Common samples** | **8,459** |

**Solution** (from `env/TCGA_final.ipynb`):

```python
# Step 1: Load clinical + pathological reports
clinical_data = pd.read_csv('all_clinical_data_all_cbio_studies.csv')  # (9523, 83)
pathological_reports = pd.read_csv('TCGA_Reports.csv')                # (9523, 2)

# Step 2: Extract patient ID and merge
pathological_reports['patientId'] = pathological_reports['patient_filename'].str.split('.').str[0]
merged_data = clinical_data.merge(
    pathological_reports[['patientId', 'text']], on='patientId', how='left'
)
# Result: (9523, 84)

# Step 3: Prepare survival data
merged_data['event'] = merged_data['OS_STATUS'].map({'0:LIVING': 0, '1:DECEASED': 1})
merged_data['time'] = merged_data['OS_MONTHS']
survival_data = merged_data.dropna(subset=['event', 'time'])
survival_data = survival_data[survival_data['time'] > 0]
# Result: 9,396 patients with complete survival data
```

### Survival Outcome Distribution

From the TCGA_final notebook analysis:
- **Living**: 6,640 patients (70.7%)
- **Deceased**: 2,756 patients (29.3%)

### Handling Missing Values

| Feature Type | Strategy | Details |
|---|---|---|
| **Priority numeric** (Age) | KNN imputation (k=5) | Preserves relationships |
| **Other numeric** | Median imputation | Robust to outliers |
| **Staging variables** | Fill with "Unknown" | Preserves information about missingness |
| **Other categorical** | Mode imputation | Most frequent value |

### Column Cleaning

Dropped 53 columns with >60% missing values, including:
`HPV_STATUS`, `BRESLOW_DEPTH`, `CLARK_LEVEL_AT_DIAGNOSIS`, `SMOKING_PACK_YEARS`, `CLINICAL_STAGE`, `YEAR_OF_DEATH`, etc.

**Final cleaned dataset**: (9,396 samples, 27 features)

---

## 🛠️ Feature Engineering

### Clinical Feature Categories (from TCGA_final.ipynb)

| Category | Available/Total | Examples |
|---|---|---|
| **Demographic** | 4/4 | AGE, SEX, RACE, ETHNICITY |
| **TNM Staging** | 5/10 | PATH_T/N/M_STAGE, PATH_STAGE, AJCC_STAGING_EDITION |
| **Tumor Characteristics** | 4/9 | PRIMARY_DIAGNOSIS, MORPHOLOGY, DISEASE_TYPE |
| **Treatment** | 2/7 | PRIOR_TREATMENT, PRIOR_MALIGNANCY |
| **Biomarkers** | 0/3 | None available |
| **Lifestyle** | 0/6 | None available |

### Engineered Features

```python
# Age groups (proven survival predictor)
survival_data['AGE_GROUP'] = pd.cut(survival_data['AGE'],
    bins=[0, 40, 50, 60, 70, 80, 120],
    labels=['<40', '40-50', '50-60', '60-70', '70-80', '80+'])

# TNM composite score
survival_data['TNM_COMBINED'] = (
    PATH_T_STAGE + '_' + PATH_N_STAGE + '_' + PATH_M_STAGE
)
```

### Univariate Cox Regression Feature Selection

From the notebook, the top significant features (p < 0.05):

| Feature | Hazard Ratio | p-value | Significance |
|---|---|---|---|
| VITAL_STATUS | 12.59 | ~0 | ★★★ |
| PRIOR_MALIGNANCY | 1.76 | 3.35e-111 | ★★★ |
| AGE | 1.03 | 9.20e-91 | ★★★ |
| DISEASE_TYPE | 1.03 | 1.78e-42 | ★★★ |
| ICD_10 | 0.99 | 1.78e-35 | ★★★ |
| PATH_STAGE | 1.11 | 6.74e-20 | ★★★ |
| PATH_N_STAGE | 1.17 | 4.82e-18 | ★★★ |
| SEX | 1.27 | 2.42e-10 | ★★★ |

**Final selected features**: 17 (after statistical filtering)

### Cancer Type Distribution (for fine-tuning)

| Disease Type | Samples | Deaths | Event Rate | Stage 2 Viable? |
|---|---|---|---|---|
| Adenomas and Adenocarcinomas | 8,977 | 1,944 | 21.7% | ✅ |
| Squamous Cell Neoplasms | 2,764 | 1,166 | 42.2% | ✅ |
| Ductal and Lobular Neoplasms | 2,362 | 498 | 21.1% | ✅ |
| Gliomas | 1,654 | 794 | 48.0% | ✅ |
| Cystic, Mucinous and Serous | 1,078 | 382 | 35.4% | ✅ |
| Transitional Cell Papillomas | 816 | 386 | 47.3% | ✅ |
| Paragangliomas and Glomus | 354 | 10 | 2.8% | ❌ |
| Mesothelial Neoplasms | 148 | 124 | 83.8% | ❌ |
| Others (<200 samples) | — | — | — | ❌ |

---

## 🏗️ Model Architecture

### Text-to-Survival Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                   Raw Pathological Text                       │
│  "Invasive ductal carcinoma, Nottingham grade 3/3,           │
│   ER negative, PR negative, HER2 positive (3+)..."           │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│               Tokenization (512 tokens max)                   │
│  [CLS] invasive ductal carcinoma , nottingham ...  [SEP]     │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│          Pre-trained LM (Frozen + LoRA Adapters)              │
│  Bio_ClinicalBERT (110M) or OpenBioLLM-8B (4-bit NF4)       │
│  LoRA: r=8/16, α=32, targets=[query,value] / [q,k,v,o_proj] │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
              [CLS] Token Embedding (768 / 4096-dim)
                        │
                ┌───────┴───────┐
                ▼               ▼
          Risk Head        Embeddings
       (Linear → 1)    (768/4096-dim vector)
                │               │
                ▼               ▼
         Cox PH Loss     Downstream Tasks
```

### LoRA Configuration

| Parameter | ClinicalBERT | OpenBioLLM-8B |
|---|---|---|
| Rank (r) | 8 | 16 |
| Alpha (α) | 32 | 32 |
| Target modules | query, value | q_proj, k_proj, v_proj, o_proj |
| Dropout | 0.1 | 0.05 |
| Trainable params | ~0.5% | ~0.3% |

### Multimodal Architecture (from TCGA.pdf — Future Work)

The project also defines three advanced architectures for multimodal integration:

1. **MissingAwareMultimodalAutoencoder**: Modality-specific encoders (Clinical→512→256, Expression→2048→1024→512, Mutation→1024→512, Text→768→512→256) with 8-head attention fusion
2. **RobustTransformerSurvival**: 4-layer transformer with learnable missing-modality tokens
3. **AdaptiveEnsembleSurvival**: Weighted ensemble with meta-learner (0.7 × ensemble + 0.3 × meta)

---

## 📊 Fine-Tuning Strategies

### Strategy 1: Baseline (Pan-Cancer)
**Files**: `text_finetune.py`, `llama_finetune.py`

Single model on all 19,611 samples. Maximum data, simplest approach.

### Strategy 2: Cancer-Type Conditioning Token
**Files**: `text_finetune_conditioned.py`, `llama_finetune_conditioned.py`

Prepends a cancer-type tag to each text:
```
Before: "Invasive ductal carcinoma, Nottingham grade 3..."
After:  "[DUCTAL AND LOBULAR NEOPLASMS] Invasive ductal carcinoma..."
```

### Strategy 3: Hierarchical Two-Stage
**Files**: `text_finetune_hierarchical.py`, `llama_finetune_hierarchical.py`

```
Stage 1: Train on ALL 19,611 samples (pan-cancer foundation)
    ↓ Save checkpoint
Stage 2: Fine-tune per viable cancer type (500+ samples, 5%+ event rate):
    → Adenomas (8,977)  → Squamous (2,764)  → Ductal (2,362)
    → Gliomas (1,654)   → Cystic (1,078)    → Transitional (816)
```

### Training Features (All Scripts)

- ✅ Early stopping (patience=3, max 20 epochs)
- ✅ Best model checkpointing
- ✅ Train/Val split (85% / 15%)
- ✅ Weights & Biases experiment tracking
- ✅ HuggingFace Hub automatic model upload
- ✅ Training loss plots (PNG) + per-epoch CSV

---

## 🚀 Training

### Installation

```bash
git clone https://github.com/drkareemkamal/cancer-survival-analysis.git
cd cancer-survival-analysis
uv venv && source .venv/bin/activate
uv pip install torch transformers peft bitsandbytes accelerate
uv pip install pandas matplotlib lifelines scikit-learn scipy
uv pip install python-dotenv wandb tqdm safetensors
```

### Configure `.env`

```env
HF_TOKEN="hf_your_token_here"
HF_REPO_ID="username/repo-name"
WANDB_API_KEY="your_wandb_key"
WANDB_PROJECT="cancer-survival-analysis"
```

### Run All Strategies

```bash
# Baseline
python src/training/text_finetune.py
python src/training/llama_finetune.py

# Conditioned
python src/training/text_finetune_conditioned.py
python src/training/llama_finetune_conditioned.py

# Hierarchical
python src/training/text_finetune_hierarchical.py
python src/training/llama_finetune_hierarchical.py

# Evaluate all
python src/training/evaluate_all_models.py
```

---

## 📈 Evaluation & Benchmarks

### Metrics

| Metric | Description |
|---|---|
| **C-index** | Primary metric — how well the model ranks patients by survival |
| **Kaplan-Meier Curves** | Visual High/Low risk group separation |
| **Risk Score Distribution** | Separation between Alive vs Deceased |
| **Per-Cancer-Type C-index** | Performance heatmap by disease type |
| **t-SNE Visualization** | Embedding quality assessment |

### Evaluation Output Files (`data/processed/evaluation/`)

| File | Description |
|---|---|
| `overall_cindex_comparison.png` | Bar chart across all 6 models |
| `per_cancer_cindex_heatmap.png` | C-index per cancer type × model |
| `kaplan_meier_comparison.png` | KM curves per model |
| `risk_distributions.png` | Alive vs Deceased histograms |
| `tsne_*.png` | t-SNE per model |
| `full_comparison_matrix.csv` | Complete numerical results |

---

## 🌐 Deployment Guide

### Option 1: HuggingFace Inference API (Easiest)

Models are auto-pushed to HF Hub during training:

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("drkareemkamal/finetunePathologicalTextUsingBioBERT")
model = AutoModel.from_pretrained("drkareemkamal/finetunePathologicalTextUsingBioBERT")
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
embedding = model(**inputs).last_hidden_state[:, 0, :]
```

### Option 2: Gradio Web Interface

```python
import gradio as gr
demo = gr.Interface(
    fn=predictor.predict,
    inputs=gr.Textbox(label="Pathological Report Text", lines=10),
    outputs=gr.Textbox(label="Survival Risk Prediction"),
    title="🧬 Cancer Survival Risk Predictor"
)
demo.launch()
```

### Option 3: FastAPI REST Service

```bash
uvicorn deploy_api:app --host 0.0.0.0 --port 8000
# POST /predict {"pathological_text": "...", "cancer_type": "optional"}
```

### Option 4: Docker Container

```dockerfile
FROM python:3.10-slim
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "deploy_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🎯 Clinical Applications

1. **Tumor Board Decision Support** — Immediate risk estimates from pathology reports
2. **Treatment Intensification/De-escalation** — High-risk → aggressive treatment; Low-risk → surveillance
3. **Clinical Trial Stratification** — Continuous risk score for balanced randomization
4. **Prognostic Biomarker Discovery** — Attention weights reveal novel indicators in pathologist language
5. **Multi-Modal Survival** — Combine text embeddings with genomic, imaging, and clinical data

---

## ⚠️ Limitations & Ethics

### Limitations
- **Not a diagnostic tool** — Predicts survival risk only
- **Dataset bias** — TCGA represents US academic medical centers
- **Text quality dependency** — Performance depends on report completeness
- **No external validation** — Requires validation on independent cohorts

### Ethics
- **Human oversight required** for all predictions
- **Regulatory approval** needed before clinical deployment
- **Patient privacy** — All texts must be de-identified
- **Bias monitoring** — Regular audits across demographics

---

## 📂 Project Structure

```
cancer-survival-analysis/
├── .env                                    # API keys
├── README.md                               # Quick-start README
├── README_full.md                          # This comprehensive document
├── TCGA.pdf                                # Project documentation
├── data/
│   ├── raw/
│   │   ├── all_clinical_data_all_cbio_studies.csv   # cBioPortal clinical data
│   │   ├── RNAseq_data.zip                          # GDC RNA-Seq (~10GB)
│   │   ├── mutation_gene.zip                         # GDC mutations (~871MB)
│   │   └── TCGA.pdf                                  # Documentation
│   └── processed/
│       ├── merged_tcga_data_final.csv               # Final merged dataset
│       ├── final_clinical_survival_dataset2.csv      # Clinical + paths
│       ├── finetuned_text_embeddings.csv             # Baseline ClinicalBERT
│       ├── finetuned_text_conditioned_embeddings.csv
│       ├── finetuned_text_hierarchical_embeddings.csv
│       ├── finetuned_llama_embeddings.csv            # Baseline Llama
│       ├── finetuned_llama_conditioned_embeddings.csv
│       ├── finetuned_llama_hierarchical_embeddings.csv
│       └── evaluation/                              # Comparison outputs
├── env/
│   ├── TCGA_final.ipynb                   # Data integration notebook
│   └── srvival-analysis.ipynb             # Gene expression processing
├── notebooks/
│   ├── srvival-analysis.ipynb             # RNA-Seq PCA pipeline
│   └── finetunePathText.ipynb             # Interactive fine-tuning
└── src/
    └── training/
        ├── text_finetune.py               # ClinicalBERT baseline
        ├── text_finetune_conditioned.py   # ClinicalBERT + conditioning
        ├── text_finetune_hierarchical.py  # ClinicalBERT hierarchical
        ├── llama_finetune.py              # Llama-3 baseline
        ├── llama_finetune_conditioned.py  # Llama-3 + conditioning
        ├── llama_finetune_hierarchical.py # Llama-3 hierarchical
        └── evaluate_all_models.py         # Comprehensive evaluation
```

---

## 📖 Citation

```bibtex
@software{kamal2026cancer_survival_text,
  title={Cancer Survival Prediction from Pathological Text Reports using Fine-Tuned LLMs},
  author={Kareem Kamal},
  year={2026},
  url={https://github.com/drkareemkamal/cancer-survival-analysis}
}
```

### References

- [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) — Alsentzer et al., 2019
- [OpenBioLLM-8B](https://huggingface.co/aaditya/Llama3-OpenBioLLM-8B) — Saama AI Labs, 2024 (based on Meta-Llama-3-8B)
- [TCGA](https://portal.gdc.cancer.gov/) — The Cancer Genome Atlas Program
- [cBioPortal](https://www.cbioportal.org/) — cBioPortal for Cancer Genomics
- [Cox PH Model](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x) — Cox, 1972

---

## 📧 Contact

- **Dr. Kareem Kamal** — [@drkareemkamal](https://github.com/drkareemkamal)

*This project is for research purposes. Always consult qualified medical professionals for clinical decisions.*
