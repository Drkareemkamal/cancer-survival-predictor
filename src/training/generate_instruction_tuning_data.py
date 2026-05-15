"""
Generate instruction-tuning QA pairs from pathological reports.

Converts raw pathological text into 3 task-specific QA pairs per sample:
1. Cancer Type Identification
2. AJCC Stage Determination
3. Prognosis Assessment (binary survival classification)

Output: JSONL format compatible with LLM fine-tuning
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Survival time thresholds by cancer type (from paper insights)
SURVIVAL_THRESHOLDS = {
    'Adenomas and Adenocarcinomas': 4.0,
    'Squamous Cell Neoplasms': 3.0,
    'Ductal and Lobular Neoplasms': 5.0,
    'Gliomas': 1.5,
    'Cystic, Mucinous and Serous': 3.5,
    'Transitional Cell Papillomas': 2.0,
}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load merged TCGA data with deduplication."""
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded {len(df)} total samples from CSV")

    # Keep only samples with essential fields (text + survival + cancer type)
    # More lenient: don't require AJCC_PATHOLOGIC_TUMOR_STAGE as it has many missing values
    essential_cols = ['text', 'DISEASE_TYPE', 'OS_MONTHS', 'OS_STATUS']
    df = df.dropna(subset=essential_cols)
    logger.info(f"After removing null essentials: {len(df)} samples")

    # Remove empty text
    df = df[df['text'].str.len() > 50]
    logger.info(f"After removing short text: {len(df)} samples")

    # Filter out rows with invalid OS_MONTHS (must be > 0)
    df = df[df['OS_MONTHS'] > 0]
    logger.info(f"After removing invalid OS_MONTHS: {len(df)} samples")

    logger.info(f"✓ Loaded {len(df)} samples with usable data")
    return df


def extract_tnm_components(row):
    """Extract TNM components safely."""
    tnm = {}

    # T stage
    t_stage = str(row.get('AJCC_TUMOR_PATHOLOGIC_PT', '')).strip()
    if pd.isna(row.get('AJCC_TUMOR_PATHOLOGIC_PT')) or t_stage == 'nan' or not t_stage:
        t_stage = "Unknown"

    # N stage
    n_stage = str(row.get('AJCC_NODES_PATHOLOGIC_PN', '')).strip()
    if pd.isna(row.get('AJCC_NODES_PATHOLOGIC_PN')) or n_stage == 'nan' or not n_stage:
        n_stage = "Unknown"

    # M stage
    m_stage = str(row.get('AJCC_METASTASIS_PATHOLOGIC_PM', '')).strip()
    if pd.isna(row.get('AJCC_METASTASIS_PATHOLOGIC_PM')) or m_stage == 'nan' or not m_stage:
        m_stage = "Unknown"

    return t_stage, n_stage, m_stage


def normalize_disease_type(disease_type: str) -> str:
    """Map disease type to standardized name."""
    if pd.isna(disease_type):
        return "Unknown Cancer Type"

    disease_type = str(disease_type).lower().strip()

    # Map variations to standard names
    mappings = {
        'adenoma': 'Adenomas and Adenocarcinomas',
        'squamous': 'Squamous Cell Neoplasms',
        'ductal': 'Ductal and Lobular Neoplasms',
        'lobular': 'Ductal and Lobular Neoplasms',
        'glioma': 'Gliomas',
        'cystic': 'Cystic, Mucinous and Serous',
        'mucinous': 'Cystic, Mucinous and Serous',
        'serous': 'Cystic, Mucinous and Serous',
        'transitional': 'Transitional Cell Papillomas',
    }

    for key, value in mappings.items():
        if key in disease_type:
            return value

    return disease_type.title()


def get_survival_label(row) -> tuple:
    """
    Determine survival label based on disease type and survival time.

    Returns:
        (survival_true_false, mean_survival_years, reasoning)
    """
    disease_type = normalize_disease_type(row['DISEASE_TYPE'])
    os_months = float(row.get('OS_MONTHS', 0)) if pd.notna(row.get('OS_MONTHS')) else 0
    os_status = str(row.get('OS_STATUS', 'Unknown')).strip()

    # Get survival threshold for this cancer type
    threshold_years = SURVIVAL_THRESHOLDS.get(disease_type, 3.0)
    threshold_months = threshold_years * 12

    # Survival is "True" if patient lived beyond threshold OR is still living
    if 'living' in os_status.lower() or 'alive' in os_status.lower():
        survived = True
        reason = f"Patient is alive (censored data). Mean survival for {disease_type} is ~{threshold_years:.1f} years."
    elif os_months >= threshold_months:
        survived = True
        reason = f"Patient survived {os_months:.0f} months ({os_months/12:.1f} years), exceeding {threshold_years:.1f}-year threshold for {disease_type}."
    else:
        survived = False
        reason = f"Patient deceased after {os_months:.0f} months ({os_months/12:.1f} years), below {threshold_years:.1f}-year threshold for {disease_type}."

    return ("True" if survived else "False"), threshold_years, reason


def create_cancer_type_qa(sample_id: str, text: str, disease_type: str) -> dict:
    """Create QA pair for cancer type identification."""
    system_prompt = (
        "You are an expert medical AI assistant specializing in pathology report analysis. "
        "Your task is to identify the cancer type from pathological text. "
        "Respond ONLY with the cancer type name and a brief reasoning in JSON format."
    )

    user_prompt = f"""Identify the cancer type from this pathological report:

\"\"\"{text[:2000]}\"\"\"

Respond in JSON format:
{{"cancer_type": "<cancer type>", "reasoning": "<brief reasoning>"}}"""

    cancer_type = normalize_disease_type(disease_type)

    response = {
        "cancer_type": cancer_type,
        "reasoning": f"Identified as {cancer_type} based on histological features and pathologist annotations in the report."
    }

    return {
        "sample_id": sample_id,
        "task": "cancer_type_identification",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(response)}
        ]
    }


def create_ajcc_stage_qa(sample_id: str, text: str, row) -> dict:
    """Create QA pair for AJCC stage determination."""
    t_stage, n_stage, m_stage = extract_tnm_components(row)
    ajcc_stage = row.get('AJCC_PATHOLOGIC_TUMOR_STAGE', 'Unknown')

    if pd.isna(ajcc_stage) or str(ajcc_stage).lower() == 'nan' or str(ajcc_stage).strip() == '':
        ajcc_stage = 'Unknown'

    ajcc_stage = str(ajcc_stage).strip().upper()

    system_prompt = (
        "You are an expert medical AI assistant specializing in cancer staging. "
        "Your task is to determine the AJCC TNM stage from pathological text. "
        "Reason through the T (tumor), N (node), and M (metastasis) components, then provide the overall stage."
    )

    user_prompt = f"""Determine the AJCC stage from this pathological report:

\"\"\"{text[:2000]}\"\"\"

Consider: tumor size and extent (T), lymph node involvement (N), and metastasis (M).

Respond in JSON format:
{{"t_stage": "<T0-T4 or Unknown>", "n_stage": "<N0-N3 or Unknown>", "m_stage": "<M0-M1 or Unknown>", "ajcc_stage": "<Stage I/II/III/IV or Unknown>", "reasoning": "<explanation of TNM reasoning>"}}"""

    response = {
        "t_stage": t_stage,
        "n_stage": n_stage,
        "m_stage": m_stage,
        "ajcc_stage": ajcc_stage,
        "reasoning": f"Based on pathological findings: T={t_stage}, N={n_stage}, M={m_stage}, yielding AJCC stage {ajcc_stage}."
    }

    return {
        "sample_id": sample_id,
        "task": "ajcc_stage_identification",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(response)}
        ]
    }


def create_prognosis_qa(sample_id: str, text: str, row) -> dict:
    """Create QA pair for survival/prognosis prediction."""
    disease_type = normalize_disease_type(row['DISEASE_TYPE'])
    survived, threshold_years, reason = get_survival_label(row)

    system_prompt = (
        "You are an expert medical AI assistant specializing in cancer prognosis. "
        "Your task is to assess whether a patient will likely survive beyond a disease-specific threshold. "
        "Consider pathological features, staging information, and disease biology."
    )

    user_prompt = f"""Based on this pathological report, will the patient with {disease_type} likely survive beyond {threshold_years:.1f} years?

\"\"\"{text[:2000]}\"\"\"

Respond in JSON format:
{{"survival_prediction": "<True/False>", "mean_survival_years": {threshold_years}, "reasoning": "<explanation of prognostic factors>"}}"""

    response = {
        "survival_prediction": survived,
        "mean_survival_years": threshold_years,
        "reasoning": reason
    }

    return {
        "sample_id": sample_id,
        "task": "prognosis_assessment",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(response)}
        ]
    }


def generate_qa_pairs(csv_path: str, output_path: str = None):
    """Generate all QA pairs and save to JSONL."""
    if output_path is None:
        output_path = "data/processed/instruction_tuning_data.jsonl"

    df = load_data(csv_path)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    qa_pairs = []

    logger.info(f"Generating QA pairs for {len(df)} samples (3 tasks per sample)...")

    with open(output_file, 'w') as f:
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
            sample_id = f"SAMPLE_{idx:06d}"
            text = str(row['text']).strip()

            # Skip if text is too short
            if len(text) < 50:
                continue

            try:
                # Task 1: Cancer Type Identification
                qa1 = create_cancer_type_qa(sample_id, text, row['DISEASE_TYPE'])
                f.write(json.dumps(qa1) + '\n')
                qa_pairs.append(qa1)

                # Task 2: AJCC Stage Determination
                qa2 = create_ajcc_stage_qa(sample_id, text, row)
                f.write(json.dumps(qa2) + '\n')
                qa_pairs.append(qa2)

                # Task 3: Prognosis Assessment
                qa3 = create_prognosis_qa(sample_id, text, row)
                f.write(json.dumps(qa3) + '\n')
                qa_pairs.append(qa3)

            except Exception as e:
                logger.warning(f"Error processing sample {sample_id}: {e}")
                continue

    logger.info(f"Generated {len(qa_pairs)} QA pairs in {output_file}")
    logger.info(f"  - Cancer Type Identification: {len([q for q in qa_pairs if q['task'] == 'cancer_type_identification'])} pairs")
    logger.info(f"  - AJCC Stage Determination: {len([q for q in qa_pairs if q['task'] == 'ajcc_stage_identification'])} pairs")
    logger.info(f"  - Prognosis Assessment: {len([q for q in qa_pairs if q['task'] == 'prognosis_assessment'])} pairs")

    # Create summary statistics
    task_stats = df.groupby('DISEASE_TYPE').size()
    logger.info(f"\nData distribution by cancer type:")
    for disease, count in task_stats.items():
        logger.info(f"  - {disease}: {count} samples → {count * 3} QA pairs")

    return output_file


if __name__ == "__main__":
    import sys

    csv_path = "data/processed/merged_tcga_data_text_dedup.csv"
    output_path = "data/processed/instruction_tuning_data.jsonl"

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    generate_qa_pairs(csv_path, output_path)
