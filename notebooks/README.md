---
library_name: transformers
license: mit
language:
  - en
tags:
  - medical
  - clinical-nlp
  - biobert
  - bio-clinicalbert
  - cancer
  - survival-analysis
  - oncology
  - pathology
  - tcga
  - lora
  - peft
  - cox-regression
  - risk-prediction
  - text-classification
  - feature-extraction
  - pytorch
datasets:
  - custom
base_model: emilyalsentzer/Bio_ClinicalBERT
pipeline_tag: feature-extraction
model-index:
  - name: finetunePathologicalTextUsingBioBERT
    results:
      - task:
          type: feature-extraction
          name: Survival Risk Prediction
        metrics:
          - type: loss
            name: Cox PH Validation Loss
            value: 0.5290
          - type: loss
            name: Cox PH Training Loss
            value: 0.4003
---

# 🧬 Fine-Tuned Bio_ClinicalBERT for Cancer Survival Prediction from Pathological Text

> **A domain-adapted biomedical language model fine-tuned on 19,637 TCGA pathological text reports for cancer survival risk prediction using Cox Proportional Hazards loss with LoRA adapters — trained on NVIDIA RTX 3090 (24 GB VRAM).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+cu124-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-5.7.0-orange.svg)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-0.19.1-green.svg)](https://huggingface.co/docs/peft)
[![GPU](https://img.shields.io/badge/GPU-RTX_3090_24GB-76B900.svg)](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/)

---

## Model Details

### Model Description

This model is a **fine-tuned version of [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)** (Alsentzer et al., 2019) adapted for **cancer survival risk prediction** directly from unstructured pathological text reports. The model was trained on data from **The Cancer Genome Atlas (TCGA)** spanning **24 cancer types** across **32 cohorts**.

Instead of traditional hand-crafted features (stage, grade, tumor size), this model **learns survival-relevant patterns directly from raw pathological text** — capturing subtle linguistic cues such as pathologist phrasing correlating with tumor aggressiveness, specific morphological descriptions, and diagnostic uncertainty language.

The model outputs:
1. **A continuous risk score** — higher values indicate higher mortality risk (used with Cox Proportional Hazards framework)
2. **768-dimensional embeddings** — from the `[CLS]` token, suitable for downstream multimodal survival pipelines

- **Developed by:** [Dr. Kareem Kamal](https://github.com/drkareemkamal)
- **Model type:** BERT-based encoder with LoRA adapters + linear survival risk head
- **Language(s):** English (clinical/biomedical)
- **License:** MIT
- **Fine-tuned from:** [emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- **Base architecture:** BERT-Base (cased, 12-layer, 768-hidden, 12-attention-heads, ~110M parameters)

### Model Sources

- **Repository:** [github.com/drkareemkamal/cancer-survival-analysis](https://github.com/drkareemkamal/cancer-survival-analysis)
- **Base model paper:** [Publicly Available Clinical BERT Embeddings (Alsentzer et al., NAACL 2019)](https://arxiv.org/abs/1904.03323)
- **BioBERT paper:** [BioBERT: a pre-trained biomedical language representation model (Lee et al., 2020)](https://arxiv.org/abs/1901.08746)

---

## About Bio_ClinicalBERT (Base Model)

Bio_ClinicalBERT has a unique **three-stage pre-training lineage** that makes it ideal for clinical text understanding:

| Stage | Training Data | Details |
|-------|--------------|---------|
| **1. BERT-Base** | Wikipedia + BookCorpus | General English language understanding |
| **2. BioBERT v1.0** | PubMed abstracts (200K) + PMC full-text (270K) | Biomedical scientific literature |
| **3. Bio_ClinicalBERT** | MIMIC-III clinical notes (~880M words) | Real electronic health records (EHR) |

**Key specifications of the base model:**
- **Architecture:** `cased_L-12_H-768_A-12` (12 layers, 768 hidden dim, 12 attention heads)
- **Parameters:** ~110 million
- **Vocabulary:** 28,996 WordPiece tokens (domain-adapted)
- **Max sequence length:** 128 tokens (original); extended to **512 tokens** in our fine-tuning
- **Original training:** 150,000 steps on GeForce GTX TITAN X (12 GB), batch size 32, LR 5e-5

This lineage means the model understands:
- ✅ General English grammar and semantics (BERT)
- ✅ Biomedical terminology and relationships (BioBERT)
- ✅ Clinical shorthand, abbreviations, and report structure (MIMIC-III)

---

## Uses

### Direct Use

Load the fine-tuned model to extract survival-relevant embeddings or risk scores from pathological text:

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("drkareemkamal/finetunePathologicalTextUsingBioBERT")
model = AutoModel.from_pretrained("drkareemkamal/finetunePathologicalTextUsingBioBERT")
model.eval()

# Example pathological report text
text = """Invasive ductal carcinoma, Nottingham grade 3/3. 
Tumor size: 2.8 cm. ER negative, PR negative, HER2 positive (3+). 
Lymphovascular invasion present. 2 of 14 sentinel lymph nodes positive 
for metastatic carcinoma. Margins: negative, closest margin 0.3 cm."""

# Tokenize
inputs = tokenizer(
    text,
    return_tensors="pt",
    max_length=512,
    truncation=True,
    padding=True
)

# Extract [CLS] embedding (768-dim)
with torch.no_grad():
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (1, 768)

print(f"Embedding shape: {cls_embedding.shape}")  # torch.Size([1, 768])
```

### Downstream Use

**Survival Risk Scoring** — Use with the custom risk head for direct risk prediction:

```python
import torch.nn as nn

# Reconstruct the risk head (trained alongside the model)
risk_head = nn.Linear(768, 1)
# Load risk head weights from checkpoint if available

risk_score = risk_head(cls_embedding)
print(f"Risk score: {risk_score.item():.4f}")
# Higher score → higher predicted mortality risk
```

**Multimodal Fusion** — Combine text embeddings with clinical, genomic, and mutation data:

```python
# Text embedding: 768-dim from this model
# Gene expression: 50-dim from PCA of RNA-Seq FPKM values
# Mutation features: binary mutation matrix
# Clinical features: age, stage, grade, etc.

combined = torch.cat([text_emb, gene_emb, mutation_emb, clinical_emb], dim=-1)
# Feed into downstream survival model (e.g., DeepSurv, Cox-nnet)
```

### Out-of-Scope Use

- ❌ **Not a diagnostic tool** — This model predicts survival risk, not diagnosis
- ❌ **Not for non-cancer text** — Trained exclusively on oncological pathology reports
- ❌ **Not for clinical deployment without regulatory approval** — Research use only
- ❌ **Not for non-English text** — Trained on English pathology reports only
- ❌ **Not for individual patient decisions** — Requires human clinical oversight

---

## Training Details

### Training Data

| Property | Value |
|----------|-------|
| **Source** | [The Cancer Genome Atlas (TCGA)](https://portal.gdc.cancer.gov/) via [cBioPortal](https://www.cbioportal.org/) |
| **Dataset file** | `merged_tcga_data_final.csv` |
| **Total samples** | **19,637** pathological text reports with survival outcomes |
| **Train split** | 16,691 samples (85%) |
| **Validation split** | 2,946 samples (15%) |
| **Cancer types** | 24 disease types across 32 TCGA cohorts |
| **Text column** | `text` — raw pathological report content |
| **Survival endpoint** | Overall Survival: `OS_MONTHS` (time) + `OS_STATUS` (event: LIVING/DECEASED) |
| **Event distribution** | ~70.7% Living / ~29.3% Deceased |

**Cancer type distribution in training data:**

| Disease Type | Samples | Deaths | Event Rate |
|-------------|---------|--------|------------|
| Adenomas and Adenocarcinomas | 8,977 | 1,944 | 21.7% |
| Squamous Cell Neoplasms | 2,764 | 1,166 | 42.2% |
| Ductal and Lobular Neoplasms | 2,362 | 498 | 21.1% |
| Gliomas | 1,654 | 794 | 48.0% |
| Cystic, Mucinous and Serous | 1,078 | 382 | 35.4% |
| Transitional Cell Papillomas | 816 | 386 | 47.3% |
| Others (18 types) | ~1,986 | varies | varies |

### Training Procedure

#### Preprocessing

1. **Text cleaning:** Rows with missing `text`, `OS_MONTHS`, or `OS_STATUS` dropped
2. **Survival labels:** `OS_STATUS` mapped to binary events (`1:DECEASED` → 1.0, `0:LIVING` → 0.0)
3. **Tokenization:** WordPiece tokenizer from Bio_ClinicalBERT, `max_length=512`, right-truncation, `max_length` padding
4. **No text augmentation** — raw pathological reports used as-is to preserve clinical accuracy

#### Fine-Tuning Method: LoRA (Low-Rank Adaptation)

Instead of updating all 110M parameters, we use **LoRA adapters** via the [PEFT library](https://github.com/huggingface/peft) to efficiently fine-tune only ~0.5% of parameters:

| LoRA Parameter | Value |
|---------------|-------|
| **Rank (r)** | 8 |
| **Alpha (α)** | 32 |
| **Target modules** | `query`, `value` (attention layers) |
| **Dropout** | 0.1 |
| **Task type** | `FEATURE_EXTRACTION` |
| **Trainable parameters** | ~590K (~0.5% of total) |

#### Loss Function: Cox Proportional Hazards (Cox PH)

The model is trained with the **negative partial log-likelihood of the Cox PH model**, which:
- Handles **right-censored data** (patients still alive at last follow-up)
- Models **relative hazard** — ranking patients by risk, not predicting absolute survival time
- Is the gold standard for survival analysis in clinical research

```
L(β) = -Σ [log(h_i) - log(Σ exp(h_j))] × event_i
        i                j∈R(t_i)
```

Where `h_i` is the predicted log-hazard for patient `i`, and `R(t_i)` is the risk set at time `t_i`.

#### Training Hyperparameters

| Hyperparameter | Value |
|---------------|-------|
| **Optimizer** | AdamW |
| **Learning rate** | 1e-4 |
| **Batch size** | 8 |
| **Max epochs** | 20 |
| **Early stopping patience** | 3 epochs |
| **Validation split** | 15% (random, seed=42) |
| **Precision** | FP32 (full precision) |
| **Gradient clipping** | None |
| **Scheduler** | None (constant LR) |
| **Weight decay** | AdamW default (0.01) |

#### Training Results

📈 **Weights & Biases Dashboard:** [View Full Training Run & Loss Curves](https://wandb.ai/dr-kareem-kamal/cancer-survival-analysis/runs/bd7qqvhj)

The model was trained for **all 20 epochs** (early stopping was not triggered, indicating continuous improvement):

| Epoch | Train Loss | Val Loss | Best? |
|-------|-----------|----------|-------|
| 1 | 1.1658 | 0.9934 | |
| 2 | 1.0408 | 0.9006 | |
| 3 | 0.9440 | 0.8677 | |
| 4 | 0.8720 | 0.8249 | |
| 5 | 0.8122 | 0.7941 | |
| 6 | 0.7347 | 0.7653 | |
| 7 | 0.7011 | 0.7099 | |
| 8 | 0.6649 | 0.7331 | |
| 9 | 0.6167 | 0.6881 | |
| 10 | 0.5849 | 0.6672 | |
| 11 | 0.5562 | 0.6481 | |
| 12 | 0.5424 | 0.6050 | |
| 13 | 0.5150 | 0.6253 | |
| 14 | 0.4998 | 0.6108 | |
| 15 | 0.4705 | 0.5765 | |
| 16 | 0.4630 | 0.6028 | |
| 17 | 0.4347 | 0.5442 | |
| 18 | 0.4230 | 0.5298 | |
| 19 | 0.4104 | 0.5605 | |
| **20** | **0.4003** | **0.5290** | **✅** |

**Key observations:**
- Consistent downward trend in both train and validation loss over 20 epochs
- Best validation loss: **0.5290** at epoch 20
- Final training loss: **0.4003**
- No signs of catastrophic overfitting — the gap between train/val loss remains reasonable
- Model checkpoint saved at epoch 20 (~415 MB)

#### Speeds, Sizes, Times

| Property | Value |
|----------|-------|
| **Total training time** | ~4.5 hours (20 epochs on RTX 3090) |
| **VRAM usage** | ~3.8 GB (FP32, batch_size=8) |
| **Checkpoint size** | 415 MB (full state dict with LoRA adapters + risk head) |
| **Embeddings output** | 162 MB CSV (19,637 samples × 768 dimensions + risk scores) |
| **Throughput** | ~120 samples/second (inference) |

---

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **Cox PH Loss** | Primary training objective — negative partial log-likelihood |
| **C-index (Concordance Index)** | How well the model ranks patients by survival (0.5 = random, >0.7 = strong) |
| **Kaplan-Meier Curves** | Visual separation between predicted high-risk and low-risk groups |
| **Risk Score Distribution** | Separation of scores between alive vs deceased patients |

### Results

| Metric | Value |
|--------|-------|
| **Best Validation Cox PH Loss** | 0.5290 |
| **Final Training Cox PH Loss** | 0.4003 |
| **Total epochs trained** | 20 / 20 |
| **Embedding dimension** | 768 |

### Evaluation Outputs

The following evaluation artifacts are generated during training:

| File | Description |
|------|-------------|
| `clinicalbert_training_loss.png` | Train vs Validation loss curves with best epoch marked |
| `clinicalbert_training_results.csv` | Per-epoch numerical loss values |
| `finetuned_text_embeddings.csv` | 768-dim embeddings + risk scores for all 19,637 samples |

---

## Technical Specifications

### Model Architecture and Objective

```
Input: Raw pathological text (up to 512 tokens)
  │
  ▼
┌─────────────────────────────────────────────┐
│     Bio_ClinicalBERT (Frozen backbone)      │
│     12 Transformer layers, 768 hidden dim   │
│     + LoRA adapters on query/value (r=8)    │
│     ~110M total params, ~590K trainable     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
          [CLS] Token Embedding (768-dim)
                   │
            ┌──────┴──────┐
            ▼             ▼
       Risk Head     Embeddings
    (Linear 768→1)  (768-dim vector)
            │             │
            ▼             ▼
     Cox PH Loss    Downstream Tasks
```

### Compute Infrastructure

#### Hardware

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA GeForce RTX 3090 |
| **GPU Memory** | 24,576 MiB (24 GB GDDR6X) |
| **CUDA Compute Capability** | 8.6 (Ampere architecture) |
| **NVIDIA Driver** | 580.126.09 |
| **CUDA Version** | 12.4 (PyTorch) / 13.0 (driver) |

#### Software

| Package | Version |
|---------|---------|
| **Python** | 3.10+ |
| **PyTorch** | 2.6.0+cu124 |
| **Transformers** | 5.7.0 |
| **PEFT** | 0.19.1 |
| **CUDA Toolkit** | 12.4 |
| **OS** | Linux (Ubuntu) |
| **Package Manager** | [uv](https://github.com/astral-sh/uv) |
| **Experiment Tracking** | [Weights & Biases](https://wandb.ai/) |

### How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/drkareemkamal/cancer-survival-analysis.git
cd cancer-survival-analysis

# 2. Set up environment with uv
uv venv && source .venv/bin/activate
uv sync

# 3. Configure API keys in .env
cat > .env << 'EOF'
HF_TOKEN="hf_your_huggingface_token"
HF_REPO_ID="your-username/your-repo-name"
WANDB_API_KEY="your_wandb_api_key"
WANDB_PROJECT="cancer-survival-analysis"
EOF

# 4. Run fine-tuning (baseline strategy)
python src/training/text_finetune.py

# Model will automatically push to HuggingFace Hub on completion
```

---

## Fine-Tuning Strategies Available

This repository implements **three fine-tuning strategies**, each with both Bio_ClinicalBERT and OpenBioLLM-8B variants:

### Strategy 1: Pan-Cancer Baseline (This Model)
Single model trained on all 19,637 samples. Maximum data, simplest approach.
```bash
python src/training/text_finetune.py
```

### Strategy 2: Cancer-Type Conditioning Token
Prepends a cancer-type tag to each text to enable cancer-aware representations:
```
Before: "Invasive ductal carcinoma, Nottingham grade 3..."
After:  "[DUCTAL AND LOBULAR NEOPLASMS] Invasive ductal carcinoma..."
```
```bash
python src/training/text_finetune_conditioned.py
```

### Strategy 3: Hierarchical Two-Stage
Stage 1 trains on all cancers, Stage 2 fine-tunes per cancer type (500+ samples):
```bash
python src/training/text_finetune_hierarchical.py
```

---

## Bias, Risks, and Limitations

### Dataset Bias
- **Geographic bias:** TCGA data originates from US academic medical centers, which may not represent global patient populations
- **Demographic bias:** The cohort reflects the demographics of TCGA participants and may underrepresent certain racial/ethnic groups
- **Institutional bias:** Pathology report styles vary by institution; model performance may degrade on reports with different formatting conventions

### Clinical Limitations
- **Not a diagnostic tool** — predicts survival risk only, not disease diagnosis
- **Text quality dependency** — performance is directly tied to report completeness and detail
- **No external validation** — requires independent cohort validation before any clinical consideration
- **Censoring assumptions** — Cox PH model assumes non-informative censoring, which may not always hold

### Technical Limitations
- **Max 512 tokens** — longer reports are truncated from the right, potentially losing relevant information
- **Single-modality** — text-only; does not incorporate imaging, genomics, or structured clinical variables (see multimodal pipeline in repository)
- **FP32 only** — not optimized for mixed-precision inference

### Recommendations

- **Always pair with clinical judgment** — this model is a decision-support tool, not a replacement for clinical expertise
- **Validate on your institution's data** before use — report styles differ across institutions
- **Monitor for bias** — regularly audit predictions across demographics, cancer types, and institutions
- **Regulatory compliance** — any clinical deployment requires appropriate regulatory approval (e.g., FDA, CE marking)

---

## Citation

**BibTeX:**

```bibtex
@software{kamal2026cancer_survival_biobert,
  title={Cancer Survival Prediction from Pathological Text Reports using Fine-Tuned Bio_ClinicalBERT},
  author={Kareem Kamal},
  year={2026},
  url={https://huggingface.co/drkareemkamal/finetunePathologicalTextUsingBioBERT},
  note={Fine-tuned on TCGA pathological reports with Cox PH loss and LoRA adapters, trained on NVIDIA RTX 3090}
}
```

**APA:**

Kamal, K. (2026). *Cancer Survival Prediction from Pathological Text Reports using Fine-Tuned Bio_ClinicalBERT* [Computer software]. Hugging Face. https://huggingface.co/drkareemkamal/finetunePathologicalTextUsingBioBERT

---

## References

1. **Bio_ClinicalBERT:** Alsentzer, E., et al. (2019). *Publicly Available Clinical BERT Embeddings.* NAACL Clinical NLP Workshop. [HuggingFace](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) | [Paper](https://arxiv.org/abs/1904.03323)
2. **BioBERT:** Lee, J., et al. (2020). *BioBERT: a pre-trained biomedical language representation model for biomedical text mining.* Bioinformatics, 36(4), 1234–1240. [Paper](https://arxiv.org/abs/1901.08746)
3. **TCGA:** The Cancer Genome Atlas Research Network. [GDC Data Portal](https://portal.gdc.cancer.gov/)
4. **cBioPortal:** Cerami, E., et al. (2012). *The cBio Cancer Genomics Portal.* Cancer Discovery, 2(5), 401–404. [Website](https://www.cbioportal.org/)
5. **Cox PH Model:** Cox, D.R. (1972). *Regression Models and Life-Tables.* Journal of the Royal Statistical Society, Series B, 34(2), 187–220.
6. **LoRA:** Hu, E., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. [Paper](https://arxiv.org/abs/2106.09685)
7. **PEFT:** HuggingFace. *Parameter-Efficient Fine-Tuning.* [GitHub](https://github.com/huggingface/peft)

---

## Model Card Authors

- **Dr. Kareem Kamal** — [@drkareemkamal](https://github.com/drkareemkamal)

## Model Card Contact

- **GitHub:** [github.com/drkareemkamal](https://github.com/drkareemkamal)
- **HuggingFace:** [huggingface.co/drkareemkamal](https://huggingface.co/drkareemkamal)

---

*This model is for research purposes only. Always consult qualified medical professionals for clinical decisions. Not approved for clinical use.*