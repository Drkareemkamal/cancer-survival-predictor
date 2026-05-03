# Fine-Tuning Strategy Recommendations for Cancer Survival Text Analysis

## Your Data at a Glance

| Metric | Value |
|---|---|
| Total samples with text + survival | **19,611** |
| Unique disease types | **24** |
| Unique study IDs (cancer cohorts) | **32** |
| Overall event (death) rate | ~27% |

### Key Observations
- **Highly imbalanced**: Top 6 disease types hold **~90%** of the data. Bottom 10 types have fewer than 100 samples each.
- **Varying event rates**: Mesothelial Neoplasms have 84% death rate vs. Germ Cell Neoplasms at 2.5%. A single model treats these very differently.
- **Pathological text differs fundamentally** by cancer type: breast cancer reports mention HER2, ER/PR status; glioma reports mention IDH mutations, WHO grade; lung cancer mentions EGFR, ALK, etc.

---

## Strategy Comparison

### Strategy 1: Pan-Cancer (Current Approach) ✅ Baseline
**What**: Train one model on all 19,611 samples together.

| Pros | Cons |
|---|---|
| Maximum data → strongest signal | Mixes unrelated pathological vocabularies |
| Simplest to implement | May learn "average" patterns that don't apply to rare cancers |
| Good for downstream multi-cancer analysis | Cannot capture cancer-specific survival drivers |

> **Verdict**: Good baseline. Already implemented.

---

### Strategy 2: Per-Cancer-Type Fine-Tuning ⚠️ Risky
**What**: Train a separate model for each of the 24 disease types.

| Pros | Cons |
|---|---|
| Model learns cancer-specific pathological features | **14 cancer types have < 200 samples** — too few to fine-tune a BERT model |
| Can capture unique survival drivers per cancer | Needs 24 separate training runs |
| | Small types with low event rates will have near-zero signal |

> **Verdict**: Only viable for the top 5-6 cancer types (>1000 samples). For the rest, the model will overfit badly. **Not recommended as-is.**

---

### Strategy 3: Hierarchical / Two-Stage Fine-Tuning ⭐ Recommended
**What**: First fine-tune on ALL cancers (pan-cancer), then do a short second fine-tune on each large cancer type.

```
Stage 1: Pan-cancer pre-training (all 19,611 samples, ~10 epochs)
           ↓ save checkpoint
Stage 2a: Fine-tune on Adenocarcinomas (8,977 samples, ~5 epochs)
Stage 2b: Fine-tune on Squamous Cell (2,764 samples, ~5 epochs)
Stage 2c: Fine-tune on Ductal/Lobular (2,362 samples, ~5 epochs)
Stage 2d: Fine-tune on Gliomas (1,654 samples, ~5 epochs)
... (only for types with 500+ samples)
```

| Pros | Cons |
|---|---|
| Gets the best of both worlds | More complex pipeline |
| Pan-cancer stage gives general medical language understanding | Need to manage multiple checkpoints |
| Cancer-specific stage captures unique features | |
| Small cancer types still benefit from Stage 1 | |

> **Verdict**: Best approach for your dataset. The pan-cancer model gives a strong foundation, and the per-cancer fine-tuning captures specific vocabulary.

---

### Strategy 4: Cancer-Type as a Conditioning Token 🔬 Advanced
**What**: Prepend the cancer type to the pathological text as a special prefix token.

```
Input: "[GLIOMA] Histologic diagnosis: Glioblastoma multiforme, WHO Grade IV..."
Input: "[BREAST] Invasive ductal carcinoma, ER positive, PR positive, HER2 negative..."
```

| Pros | Cons |
|---|---|
| Single model learns cancer-aware representations | Slightly reduces effective text length (512 tokens) |
| No need for separate models | Needs careful prompt engineering |
| All data contributes to training | |

> **Verdict**: Excellent and simple. Highly recommended as an upgrade to Strategy 1.

---

### Strategy 5: Multi-Task Learning 🔬 Advanced
**What**: Add auxiliary prediction heads alongside the survival head.

```
Text → BERT → [CLS] embedding
                 ├── Risk Head (Cox PH Loss) ← primary task
                 ├── Cancer Type Classifier (CrossEntropy Loss) ← auxiliary
                 └── Stage Predictor (CrossEntropy Loss) ← auxiliary
```

| Pros | Cons |
|---|---|
| Forces embeddings to capture cancer-type information | More complex architecture |
| Acts as regularization, reduces overfitting | Need to tune loss weights |
| Embeddings become more informative | |

> **Verdict**: Powerful but requires more engineering. Best combined with Strategy 4.

---

## My Recommendation: Implement Strategies 3 + 4

### Phase 1 (Quick Win — Strategy 4)
Modify the current script to prepend cancer type as a conditioning token. This is a **1-line change** to your dataset class and gives immediate improvement.

### Phase 2 (Best Results — Strategy 3)
Implement the two-stage hierarchical fine-tuning for the top cancer types (those with 500+ samples and reasonable event rates):

| Cancer Type | Samples | Deaths | Event Rate | Viable? |
|---|---|---|---|---|
| Adenomas and Adenocarcinomas | 8,977 | 1,944 | 21.7% | ✅ Yes |
| Squamous Cell Neoplasms | 2,764 | 1,166 | 42.2% | ✅ Yes |
| Ductal and Lobular Neoplasms | 2,362 | 498 | 21.1% | ✅ Yes |
| Gliomas | 1,654 | 794 | 48.0% | ✅ Yes |
| Cystic, Mucinous and Serous | 1,078 | 382 | 35.4% | ✅ Yes |
| Transitional Cell Papillomas | 816 | 386 | 47.3% | ✅ Yes |
| Paragangliomas and Glomus | 354 | 10 | 2.8% | ❌ Too few events |
| Thymic Epithelial | 222 | 12 | 5.4% | ❌ Too few events |
| All others (<200 samples) | — | — | — | ❌ Use pan-cancer model |

## Open Questions

> [!IMPORTANT]
> 1. Would you like me to implement **Strategy 4** (cancer-type conditioning token) first as a quick improvement?
> 2. Or would you prefer I build the full **Strategy 3** (hierarchical two-stage pipeline)?
> 3. Or both?
