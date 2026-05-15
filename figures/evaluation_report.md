# Evaluation Report — Test Set Performance

- Test patients: **1266**
- Event rate: **27.6%**

## Survival Models — Test C-index

| Model | Val C | Test C | 95 % CI | Mean time-AUC | KM log-rank p |
|---|---|---|---|---|---|
| **Cox PH** | 0.7556 | 0.6007 | [0.5655, 0.6343] | 0.6144 | 1.94e-07 |
| **RSF** | 0.7852 | 0.7415 | [0.7176, 0.7655] | 0.7726 | 2.48e-40 |
| **Autoencoder** | 0.8075 | 0.7876 | [0.7643, 0.8104] | 0.8155 | 1.47e-68 |
| **Transformer** | 0.8169 | 0.8053 | [0.7836, 0.8267] | 0.8376 | 1.86e-71 |
| **Ensemble** | 0.8146 | 0.7950 | [0.7741, 0.8175] | 0.8251 | 3.76e-67 |

## Files generated

- `figures/ablation_table.png`
- `figures/cindex_comparison.png`
- `figures/cindex_interactive.html`
- `figures/dashboard.html`
- `figures/evaluation_report.md`
- `figures/evaluation_summary.json`
- `figures/km_by_risk_Autoencoder.png`
- `figures/km_by_risk_Cox_PH.png`
- `figures/km_by_risk_Ensemble.png`
- `figures/km_by_risk_RSF.png`
- `figures/km_by_risk_Transformer.png`
- `figures/km_interactive.html`
- `figures/pathology_tasks.png`
- `figures/pathology_tasks2.png`
- `figures/results_pathology_tasks.csv`
- `figures/risk_score_distribution.png`
- `figures/time_dependent_auc.png`

## PathQwen2.5 — Per-task accuracy

| Task | N | Accuracy | Macro F1 |
|---|---|---|---|
| cancer_type | 1266 | 0.9218 | 0.8711 |
| primary_site | 1251 | 0.8945 | 0.3500 |
| histology | 1251 | 0.6691 | 0.1848 |
| ajcc_stage | 810 | 0.5025 | 0.3493 |
| t_stage | 930 | 0.7925 | 0.4496 |
| n_stage | 917 | 0.8233 | 0.6552 |
| m_stage | 809 | 0.6329 | 0.3872 |
| prior_malignancy | 1190 | 0.8916 | 0.3200 |
| prognosis_good | 1266 | 0.4344 | 0.2809 |