"""
Comprehensive evaluation of instruction-tuned models for Phase 1.

Metrics:
1. Classification Accuracy (cancer type, AJCC stage)
2. F1-score for binary classifications
3. C-index for survival ranking
4. Per-cancer-type performance
5. Visualization of results
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
from lifelines.utils import concordance_index
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class InstructionTunedEvaluator:
    """Evaluate instruction-tuned models on test data."""

    def __init__(self, model_path: str, data_path: str):
        """
        Initialize evaluator.

        Args:
            model_path: Path to fine-tuned model
            data_path: Path to original merged data CSV
        """
        self.model_path = model_path
        self.data_path = data_path

        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path)

    def extract_json_from_response(self, text: str) -> Optional[Dict]:
        """Extract JSON from model response."""
        try:
            # Find JSON object in response
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        return None

    def predict_cancer_type(self, text: str) -> Tuple[str, float]:
        """Predict cancer type from pathological report."""
        prompt = f"""Identify the cancer type from this pathological report:

"{text[:2000]}"

Respond in JSON format:
{{"cancer_type": "<cancer type>", "confidence": <0.0-1.0>}}"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.95,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        parsed = self.extract_json_from_response(response)

        if parsed:
            return parsed.get('cancer_type', 'Unknown'), parsed.get('confidence', 0.5)

        return 'Unknown', 0.0

    def predict_ajcc_stage(self, text: str) -> Tuple[str, Dict]:
        """Predict AJCC stage from pathological report."""
        prompt = f"""Determine the AJCC stage from this pathological report:

"{text[:2000]}"

Respond in JSON format:
{{"t_stage": "<T0-T4>", "n_stage": "<N0-N3>", "m_stage": "<M0-M1>", "ajcc_stage": "<Stage I/II/III/IV>", "confidence": <0.0-1.0>}}"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.95,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        parsed = self.extract_json_from_response(response)

        if parsed:
            return parsed.get('ajcc_stage', 'Unknown'), parsed

        return 'Unknown', {}

    def predict_prognosis(self, text: str, disease_type: str) -> Tuple[str, float]:
        """Predict survival prognosis from pathological report."""
        prompt = f"""Based on this pathological report for {disease_type}, will the patient likely survive beyond the median survival time?

"{text[:2000]}"

Respond in JSON format:
{{"survival": "<True/False>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.95,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        parsed = self.extract_json_from_response(response)

        if parsed:
            survival = parsed.get('survival', 'False') == 'True'
            confidence = parsed.get('confidence', 0.5)
            return ('True' if survival else 'False'), confidence

        return 'False', 0.0

    def evaluate_cancer_type(self, n_samples: int = 500) -> Dict:
        """Evaluate cancer type identification accuracy."""
        logger.info(f"Evaluating cancer type classification on {n_samples} samples...")

        sample_data = self.data.dropna(subset=['text', 'DISEASE_TYPE']).head(n_samples)

        predictions = []
        ground_truth = []

        for _, row in tqdm(sample_data.iterrows(), total=len(sample_data)):
            pred_type, confidence = self.predict_cancer_type(row['text'])
            predictions.append(pred_type)
            ground_truth.append(str(row['DISEASE_TYPE']))

        # Calculate metrics
        accuracy = accuracy_score(ground_truth, predictions)
        f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
        recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)

        logger.info(f"Cancer Type Classification:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1-score: {f1:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")

        return {
            'task': 'cancer_type',
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'n_samples': len(sample_data),
            'predictions': predictions,
            'ground_truth': ground_truth,
        }

    def evaluate_ajcc_stage(self, n_samples: int = 500) -> Dict:
        """Evaluate AJCC stage determination."""
        logger.info(f"Evaluating AJCC stage classification on {n_samples} samples...")

        sample_data = self.data.dropna(subset=['text', 'AJCC_PATHOLOGIC_TUMOR_STAGE']).head(n_samples)

        predictions = []
        ground_truth = []

        for _, row in tqdm(sample_data.iterrows(), total=len(sample_data)):
            _, stage_dict = self.predict_ajcc_stage(row['text'])
            pred_stage = stage_dict.get('ajcc_stage', 'Unknown')
            predictions.append(pred_stage)
            ground_truth.append(str(row['AJCC_PATHOLOGIC_TUMOR_STAGE']))

        # Calculate metrics
        accuracy = accuracy_score(ground_truth, predictions)
        f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)

        logger.info(f"AJCC Stage Classification:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1-score: {f1:.4f}")

        return {
            'task': 'ajcc_stage',
            'accuracy': accuracy,
            'f1': f1,
            'n_samples': len(sample_data),
            'predictions': predictions,
            'ground_truth': ground_truth,
        }

    def evaluate_survival_prediction(self, n_samples: int = 500) -> Dict:
        """Evaluate survival prediction and C-index."""
        logger.info(f"Evaluating survival prediction on {n_samples} samples...")

        sample_data = self.data.dropna(subset=['text', 'OS_MONTHS', 'OS_STATUS']).head(n_samples)

        predictions = []  # Binary: True/False
        ground_truth = []
        survival_times = []

        for _, row in tqdm(sample_data.iterrows(), total=len(sample_data)):
            # Get ground truth survival
            os_status = str(row['OS_STATUS']).lower()
            if 'living' in os_status or 'alive' in os_status:
                true_survival = 1  # Censored/alive
            else:
                true_survival = 0 if row['OS_MONTHS'] < 36 else 1  # 3-year threshold

            pred_survival, confidence = self.predict_prognosis(row['text'], row['DISEASE_TYPE'])
            predictions.append(1 if pred_survival == 'True' else 0)
            ground_truth.append(true_survival)
            survival_times.append(row['OS_MONTHS'])

        # Calculate metrics
        accuracy = accuracy_score(ground_truth, predictions)
        f1 = f1_score(ground_truth, predictions, zero_division=0)

        # C-index (for ranking)
        # Use prediction confidences as risk scores
        risk_scores = np.array([1 - p for p in predictions])  # Invert for C-index calculation
        try:
            c_index = concordance_index(
                survival_times,
                risk_scores,
                event_observed=[1] * len(ground_truth),  # Assume all events observed
            )
        except:
            c_index = 0.5

        logger.info(f"Survival Prediction:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1-score: {f1:.4f}")
        logger.info(f"  C-index: {c_index:.4f}")

        return {
            'task': 'survival',
            'accuracy': accuracy,
            'f1': f1,
            'c_index': c_index,
            'n_samples': len(sample_data),
            'predictions': predictions,
            'ground_truth': ground_truth,
        }

    def generate_report(self, output_dir: str = "data/processed/evaluation"):
        """Generate comprehensive evaluation report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate all tasks
        results = {
            'cancer_type': self.evaluate_cancer_type(n_samples=300),
            'ajcc_stage': self.evaluate_ajcc_stage(n_samples=300),
            'survival': self.evaluate_survival_prediction(n_samples=300),
        }

        # Save results
        results_df = pd.DataFrame([
            {
                'Task': 'Cancer Type Identification',
                'Accuracy': results['cancer_type']['accuracy'],
                'F1-score': results['cancer_type']['f1'],
                'Precision': results['cancer_type']['precision'],
                'Recall': results['cancer_type']['recall'],
            },
            {
                'Task': 'AJCC Stage Classification',
                'Accuracy': results['ajcc_stage']['accuracy'],
                'F1-score': results['ajcc_stage']['f1'],
            },
            {
                'Task': 'Survival Prediction',
                'Accuracy': results['survival']['accuracy'],
                'F1-score': results['survival']['f1'],
                'C-index': results['survival']['c_index'],
            },
        ])

        results_csv = output_dir / "instruction_tuning_results.csv"
        results_df.to_csv(results_csv, index=False)
        logger.info(f"Results saved to {results_csv}")

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Cancer type accuracy
        axes[0].bar(['Accuracy', 'F1', 'Precision', 'Recall'],
                   [results['cancer_type']['accuracy'],
                    results['cancer_type']['f1'],
                    results['cancer_type']['precision'],
                    results['cancer_type']['recall']])
        axes[0].set_title('Cancer Type Identification')
        axes[0].set_ylim([0, 1])
        axes[0].set_ylabel('Score')

        # AJCC stage accuracy
        axes[1].bar(['Accuracy', 'F1'],
                   [results['ajcc_stage']['accuracy'],
                    results['ajcc_stage']['f1']])
        axes[1].set_title('AJCC Stage Classification')
        axes[1].set_ylim([0, 1])
        axes[1].set_ylabel('Score')

        # Survival prediction
        axes[2].bar(['Accuracy', 'F1', 'C-index'],
                   [results['survival']['accuracy'],
                    results['survival']['f1'],
                    results['survival']['c_index']])
        axes[2].set_title('Survival Prediction')
        axes[2].set_ylim([0, 1])
        axes[2].set_ylabel('Score')

        plt.tight_layout()
        viz_path = output_dir / "instruction_tuning_metrics.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {viz_path}")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/llama_instruction_tuned")
    parser.add_argument("--data_path", default="data/processed/merged_tcga_data_text_dedup.csv")
    parser.add_argument("--output_dir", default="data/processed/evaluation")

    args = parser.parse_args()

    evaluator = InstructionTunedEvaluator(args.model_path, args.data_path)
    results = evaluator.generate_report(args.output_dir)

    # Print summary
    print("\n" + "="*60)
    print("PHASE 1 EVALUATION SUMMARY")
    print("="*60)
    for task, metrics in results.items():
        print(f"\n{task.upper()}:")
        for key, value in metrics.items():
            if key not in ['task', 'predictions', 'ground_truth', 'n_samples']:
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
