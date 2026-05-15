"""Single Pydantic schema covering all 8 pathology tasks.

Used both at training time (label structure) and inference time
(constrained decoding via outlines so JSON is guaranteed valid).
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# 32 TCGA studies — paper-comparable label space
TCGA_STUDIES = [
    "acc_tcga_gdc", "blca_tcga_gdc", "brca_tcga_gdc", "ccrcc_tcga_gdc",
    "cesc_tcga_gdc", "chol_tcga_gdc", "chrcc_tcga_gdc", "coad_tcga_gdc",
    "difg_tcga_gdc", "dlbclnos_tcga_gdc", "esca_tcga_gdc", "gbm_tcga_gdc",
    "hcc_tcga_gdc", "hgsoc_tcga_gdc", "hnsc_tcga_gdc", "luad_tcga_gdc",
    "lusc_tcga_gdc", "mnet_tcga_gdc", "nsgct_tcga_gdc", "paad_tcga_gdc",
    "plmeso_tcga_gdc", "prad_tcga_gdc", "prcc_tcga_gdc", "read_tcga_gdc",
    "skcm_tcga", "soft_tissue_tcga_gdc", "stad_tcga_gdc", "thpa_tcga_gdc",
    "thym_tcga_gdc", "ucec_tcga_gdc", "ucs_tcga_gdc", "um_tcga_gdc",
]

AJCC_STAGES = ["Stage I", "Stage II", "Stage III", "Stage IV"]

T_STAGES = ["T0", "T1", "T2", "T3", "T4", "Tis", "TX"]
N_STAGES = ["N0", "N1", "N2", "N3", "NX"]
M_STAGES = ["M0", "M1", "MX"]


class PathologyExtraction(BaseModel):
    """Output of the fine-tuned pathology LLM. Per-row None means label was missing."""

    cancer_type: Optional[str] = Field(
        None, description=f"TCGA study label, one of: {TCGA_STUDIES}"
    )
    primary_site: Optional[str] = Field(
        None, description="Anatomical primary site"
    )
    histology: Optional[str] = Field(
        None, description="ICD-O-3 morphology"
    )
    ajcc_stage: Optional[str] = Field(
        None, description=f"AJCC overall stage, one of: {AJCC_STAGES}"
    )
    t_stage: Optional[str] = Field(
        None, description=f"T stage (substages collapsed), one of: {T_STAGES}"
    )
    n_stage: Optional[str] = Field(
        None, description=f"N stage (substages collapsed), one of: {N_STAGES}"
    )
    m_stage: Optional[str] = Field(
        None, description=f"M stage, one of: {M_STAGES}"
    )
    prior_malignancy: Optional[bool] = Field(
        None, description="Patient had a prior cancer"
    )
    prognosis_good: Optional[bool] = Field(
        None,
        description="True if patient is predicted to survive beyond cancer-type mean DSS",
    )


# 8 task names matching the schema fields
TASKS = [
    "cancer_type", "primary_site", "histology",
    "ajcc_stage", "t_stage", "n_stage", "m_stage",
    "prior_malignancy", "prognosis_good",
]
