# 🧬 Cancer Survival Prediction from Pathological Text Reports

## Fine-Tuning Large Language Models for Oncological Survival Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-orange.svg)](https://huggingface.co/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Clinical Significance](#-clinical-significance)
- [Architecture](#-architecture)
- [Fine-Tuning Strategies](#-fine-tuning-strategies)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Training](#-training)
- [Evaluation & Benchmarks](#-evaluation--benchmarks)
- [Deployment Guide](#-deployment-guide)
- [Results](#-results)
- [Clinical Applications](#-clinical-applications)
- [Limitations & Ethics](#-limitations--ethics)
- [Citation](#-citation)

---

## 🔬 Overview

This project fine-tunes state-of-the-art biomedical language models on **TCGA (The Cancer Genome Atlas)** pathological text reports to predict **patient survival outcomes**. By learning the relationship between pathological descriptions and overall survival, these models can:

1. **Extract survival-relevant features** from unstructured pathological text
2. **Predict patient risk scores** that correlate with survival probability
3. **Stratify patients** into high-risk and low-risk groups for treatment planning
4. **Generate embeddings** that can be integrated into multi-modal survival pipelines

### Models

| Model | Parameters | Quantization | VRAM Usage |
|---|---|---|---|
| **Bio_ClinicalBERT** | ~110M | None (FP32) | ~3.8 GB |
| **OpenBioLLM-8B** | ~8B | 4-bit NF4 | ~6-8 GB |

### Key Innovation

Unlike traditional survival analysis that relies on hand-crafted features (stage, grade, tumor size), our approach **directly learns from raw pathological text**, capturing subtle linguistic patterns that human-engineered features miss — such as pathologist phrasing that correlates with disease aggressiveness, specific morphological descriptions, or nuanced diagnostic uncertainty language.

---

## 🏥 Clinical Significance

### Why This Matters for Medicine

**1. Unstructured Text is Underutilized**
Over 80% of clinical data exists as unstructured text in Electronic Health Records (EHRs). Pathological reports contain rich, expert-curated descriptions that are rarely used in computational survival models.

**2. Personalized Risk Assessment**
Traditional staging systems (TNM, AJCC) group patients into broad categories. Our model provides a **continuous risk score** that captures individual patient nuances, enabling more personalized treatment decisions.

**3. Decision Support at Diagnosis**
When a pathologist writes a report, this model can immediately provide a survival risk estimate — within seconds of report completion — supporting multidisciplinary tumor board discussions.

**4. Multi-Cancer Applicability**
Trained on 32 TCGA cancer cohorts spanning 24 disease types, the model generalizes across cancer types while also offering cancer-specific models for the most common malignancies.

### Survival Analysis Method

We use the **Cox Proportional Hazards (Cox PH)** loss function — the gold standard in clinical survival analysis. This semi-parametric approach:

- Handles **censored data** (patients still alive at last follow-up)
- Models the **relative hazard** rather than absolute survival time
- Produces a **risk score** that ranks patients by survival probability
- Is evaluated using the **Concordance Index (C-index)**, where:
  - C-index = 0.5 → Random prediction (useless)
  - C-index = 0.6-0.65 → Moderate discrimination
  - C-index = 0.65-0.7 → Good discrimination
  - C-index > 0.7 → Strong discrimination

---

## 🏗️ Architecture

### Model Pipeline


![My Image](./pipeline.png)


### LoRA (Low-Rank Adaptation)

Instead of fine-tuning all model parameters (expensive, prone to catastrophic forgetting), we use **LoRA** which:
- Adds small trainable matrices to attention layers
- Trains only **0.5-2%** of total parameters
- Preserves the pre-trained medical language understanding
- Enables efficient multi-strategy training on a single GPU

---

## 📊 Fine-Tuning Strategies

We implement and compare **3 strategies × 2 models = 6 configurations**:

### Strategy 1: Baseline (Pan-Cancer)

**Files:** `text_finetune.py`, `llama_finetune.py`

Trains a single model on all 19,611 samples across all cancer types. The simplest approach — uses maximum data but doesn't distinguish between different pathological vocabularies.

### Strategy 2: Cancer-Type Conditioning Token

**Files:** `text_finetune_conditioned.py`, `llama_finetune_conditioned.py`

Prepends a cancer-type tag to each pathological text:
```
Before: "Invasive ductal carcinoma, Nottingham grade 3..."
After:  "[DUCTAL AND LOBULAR NEOPLASMS] Invasive ductal carcinoma, Nottingham grade 3..."
```

This teaches the model to produce **cancer-type-aware** embeddings and risk scores within a single model — no extra parameters, no separate training runs.

### Strategy 3: Hierarchical Two-Stage

**Files:** `text_finetune_hierarchical.py`, `llama_finetune_hierarchical.py`

```
Stage 1: Train on ALL 19,611 samples (pan-cancer foundation)
    ↓ Save checkpoint
Stage 2: Fine-tune on each viable cancer type (500+ samples):
    → Adenomas (8,977 samples)
    → Squamous Cell (2,764 samples)
    → Ductal/Lobular (2,362 samples)
    → Gliomas (1,654 samples)
    → Cystic/Mucinous (1,078 samples)
    → Transitional Cell (816 samples)
```

The most sophisticated approach: cancer types with enough data get specialized models; rare cancer types use the pan-cancer model.

---

## 📁 Dataset

### TCGA (The Cancer Genome Atlas)

| Metric | Value |
|---|---|
| Total patients | 19,611 |
| Cancer types | 24 disease types, 32 study cohorts |
| Text source | Pathological reports |
| Survival endpoint | Overall Survival (OS) |
| Overall event rate | ~27% |
| Follow-up | OS_MONTHS (continuous) |

### Top Cancer Types

| Disease Type | Samples | Deaths | Event Rate |
|---|---|---|---|
| Adenomas and Adenocarcinomas | 8,977 | 1,944 | 21.7% |
| Squamous Cell Neoplasms | 2,764 | 1,166 | 42.2% |
| Ductal and Lobular Neoplasms | 2,362 | 498 | 21.1% |
| Gliomas | 1,654 | 794 | 48.0% |
| Cystic, Mucinous and Serous | 1,078 | 382 | 35.4% |
| Transitional Cell Papillomas | 816 | 386 | 47.3% |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU with 8+ GB VRAM (24 GB recommended for Llama-3)
- NVIDIA CUDA 11.8 or 12.x

### Setup

```bash
# Clone the repository
git clone https://github.com/drkareemkamal/cancer-survival-predictor.git
cd cancer-survival-predictor

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install torch transformers peft bitsandbytes accelerate
uv pip install pandas matplotlib lifelines scikit-learn scipy
uv pip install python-dotenv wandb tqdm safetensors

# Configure credentials
cp .env.example .env
# Edit .env with your API keys:
#   HF_TOKEN="hf_your_token_here"
#   HF_REPO_ID="username/repo-name"
#   WANDB_API_KEY="your_wandb_key"
#   WANDB_PROJECT="cancer-survival-analysis"
```

---

## 🚀 Training

### Run All Strategies

```bash
# Strategy 1: Baseline
python src/training/text_finetune.py           # ClinicalBERT baseline
python src/training/llama_finetune.py          # Llama-3 baseline

# Strategy 2: Cancer-Type Conditioning
python src/training/text_finetune_conditioned.py   # ClinicalBERT conditioned
python src/training/llama_finetune_conditioned.py  # Llama-3 conditioned

# Strategy 3: Hierarchical Two-Stage
python src/training/text_finetune_hierarchical.py  # ClinicalBERT hierarchical
python src/training/llama_finetune_hierarchical.py # Llama-3 hierarchical
```

### Evaluate & Compare All Models

```bash
python src/training/evaluate_all_models.py
```

### Training Features

All scripts include:
- ✅ **Early stopping** with validation monitoring (patience=3)
- ✅ **Best model checkpointing** (saves the optimal epoch)
- ✅ **Train/Val split** (85% / 15%)
- ✅ **Weights & Biases** experiment tracking
- ✅ **HuggingFace Hub** automatic model upload
- ✅ **Training loss plots** (saved as PNG)
- ✅ **Training results CSV** (per-epoch metrics)

---

## 📈 Evaluation & Benchmarks

### Metrics

| Metric | Description |
|---|---|
| **C-index** (Concordance Index) | Primary metric. Measures how well the model ranks patients by survival time. |
| **Kaplan-Meier Curves** | Visual comparison of survival between model-identified High/Low risk groups. |
| **Risk Score Distribution** | How well the model separates Alive vs Deceased patients by risk score. |
| **Per-Cancer-Type C-index** | Performance breakdown by disease type (heatmap). |
| **t-SNE Visualization** | 2D projection of learned embeddings to verify cancer-type clustering. |

### Output Files

After running `evaluate_all_models.py`, find results in `data/processed/evaluation/`:

| File | Description |
|---|---|
| `overall_cindex_comparison.png` | Bar chart comparing all models |
| `per_cancer_cindex_heatmap.png` | C-index heatmap (model × cancer type) |
| `kaplan_meier_comparison.png` | KM curves for each model |
| `risk_distributions.png` | Risk score histograms by survival status |
| `tsne_*.png` | t-SNE embeddings per model |
| `full_comparison_matrix.csv` | Complete numerical results |

---

## 🌐 Deployment 

### Option 1: HuggingFace Inference API (Easiest)

All trained models are automatically pushed to your HuggingFace Hub. You can immediately use them via the Inference API:

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load your fine-tuned model from the Hub
tokenizer = AutoTokenizer.from_pretrained("drkareemkamal/finetunePathologicalTextUsingBioBERT")
model = AutoModel.from_pretrained("drkareemkamal/finetunePathologicalTextUsingBioBERT")

# Process new pathological text
text = "Invasive ductal carcinoma, grade 3, ER-negative, HER2-positive"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
outputs = model(**inputs)
embedding = outputs.last_hidden_state[:, 0, :]  # CLS embedding
```

### Option 2: Gradio Web Interface

you can visit [https://huggingface.co/spaces/drkareemkamal/CancerSurvivalPredictor]


## 🎯 Clinical Applications

### 1. Tumor Board Decision Support
When pathology results are presented at multidisciplinary tumor boards, the model provides an **immediate risk estimate** based on the pathological report text, complementing staging information.

### 2. Treatment Intensification/De-escalation
- **High-risk patients** → Consider more aggressive treatment, adjuvant chemotherapy, clinical trial enrollment
- **Low-risk patients** → Potential for surveillance-only approaches, avoiding overtreatment

### 3. Clinical Trial Stratification
Use the continuous risk score for **stratified randomization** in clinical trials, ensuring balanced risk groups across treatment arms.

### 4. Prognostic Biomarker Discovery
Analyze which text features (via attention weights) drive high/low risk predictions. This can reveal **novel prognostic indicators** embedded in pathologist language.

### 5. Multi-Modal Survival Models
Combine text embeddings with:
- **Genomic data** (mutation profiles, gene expression)
- **Imaging data** (histopathology slides, radiology)
- **Clinical features** (age, stage, treatment)

For a comprehensive multi-modal survival prediction pipeline.

---

## ⚠️ Limitations & Ethics

### Limitations

1. **Not a diagnostic tool**: This model predicts survival risk based on pathological text patterns. It does not diagnose cancer or recommend treatment.
2. **Dataset bias**: TCGA predominantly represents US academic medical centers, which may limit generalizability to other populations.
3. **Text quality dependency**: Model performance depends on the quality and completeness of pathological reports.
4. **Censoring bias**: Patients with shorter follow-up contribute less to the Cox PH loss, potentially biasing predictions.
5. **No external validation**: Results should be validated on independent, external cohorts before clinical deployment.

### Ethical Considerations

- **Human oversight required**: All predictions must be reviewed by qualified medical professionals.
- **Regulatory approval**: Clinical deployment requires regulatory approval (FDA, CE marking) depending on jurisdiction.
- **Patient privacy**: Ensure all pathological texts are properly de-identified before processing.
- **Bias monitoring**: Regularly audit model performance across demographic groups (race, ethnicity, age, sex).

---

## 📂 Project Structure

```
cancer-survival-analysis/
├── .env                              # API keys (HF_TOKEN, WANDB_API_KEY)
├── README.md                         # This file
├── data/
│   └── processed/
│       ├── merged_tcga_data_final.csv        # Source dataset
│       ├── finetuned_text_embeddings.csv      # Baseline ClinicalBERT output
│       ├── finetuned_text_conditioned_embeddings.csv
│       ├── finetuned_text_hierarchical_embeddings.csv
│       ├── finetuned_llama_embeddings.csv     # Baseline Llama output
│       ├── finetuned_llama_conditioned_embeddings.csv
│       ├── finetuned_llama_hierarchical_embeddings.csv
│       ├── *_training_loss.png               # Training curves
│       ├── *_training_results.csv            # Per-epoch metrics
│       └── evaluation/                       # Comparison outputs
│           ├── overall_cindex_comparison.png
│           ├── per_cancer_cindex_heatmap.png
│           ├── kaplan_meier_comparison.png
│           ├── risk_distributions.png
│           ├── tsne_*.png
│           └── full_comparison_matrix.csv
├── src/
│   └── training/
│       ├── text_finetune.py                  # ClinicalBERT baseline
│       ├── text_finetune_conditioned.py      # ClinicalBERT + conditioning
│       ├── text_finetune_hierarchical.py     # ClinicalBERT hierarchical
│       ├── llama_finetune.py                 # Llama-3 baseline
│       ├── llama_finetune_conditioned.py     # Llama-3 + conditioning
│       ├── llama_finetune_hierarchical.py    # Llama-3 hierarchical
│       └── evaluate_all_models.py            # Comprehensive evaluation
└── notebooks/
    └── finetunePathText.ipynb                # Interactive notebook
```

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{kamal2026cancer_survival_text,
  title={Cancer Survival Prediction from Pathological Text Reports using Fine-Tuned LLMs},
  author={Kareem Kamal},
  year={2026},
  url={https://github.com/drkareemkamal/cancer-survival-analysis}
}
```

### Related Work

- [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) — Alsentzer et al., 2019
- [OpenBioLLM-8B](https://huggingface.co/aaditya/Llama3-OpenBioLLM-8B) — Saama AI Labs, 2024 (based on Meta-Llama-3-8B)
- [TCGA](https://portal.gdc.cancer.gov/) — The Cancer Genome Atlas Program
- [Cox PH Model](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x) — Cox, 1972

---

## 📧 Contact

- **Dr. Kareem Kamal**
- GitHub: [@drkareemkamal](https://github.com/drkareemkamal)
- HuggingFace: [drkareemkamal](https://huggingface.co/drkareemkamal)

---

*This project is for research purposes. Always consult qualified medical professionals for clinical decisions.*
uv pip install --upgrade "torch>=2.6" --index-url https://download.pytorch.org/whl/cu124
