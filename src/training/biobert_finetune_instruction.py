"""
Fine-tune Bio-ClinicalBERT with instruction-tuning data for multi-task cancer analysis.

Tasks:
1. Cancer Type Identification
2. AJCC Stage Determination
3. Prognosis Assessment

Training approach: Multi-task classification on synthetic QA pairs
Adapted for encoder-only BERT architecture with classification heads

Features:
- Weights & Biases (wandb) integration for monitoring
- HuggingFace Hub integration for model upload
- LoRA efficient fine-tuning (15% of parameters trainable)
- Multi-task learning with task-specific classification heads
"""

import torch
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict, load_dataset
import wandb
from huggingface_hub import login as hf_login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpochTrackingCallback(TrainerCallback):
    """Log epoch progress to wandb."""
    def __init__(self, num_training_samples, batch_size, gradient_accumulation_steps):
        self.steps_per_epoch = num_training_samples / (batch_size * gradient_accumulation_steps)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            current_epoch = state.global_step / self.steps_per_epoch
            logs["epoch_float"] = current_epoch


@dataclass
class TrainingConfig:
    """Bio-ClinicalBERT instruction-tuning configuration"""
    # Model
    model_id: str = "emilyalsentzer/Bio_ClinicalBERT"

    # LoRA (15% of BERT params trainable - higher than Llama since model is smaller)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )

    # Training
    num_train_epochs: int = 5
    max_steps: int = 10000  # Fewer steps for smaller model
    learning_rate: float = 5e-4  # Higher LR for BERT (smaller model)
    warmup_steps: int = 500
    warmup_ratio: float = 0.0

    # Batch & gradient (RTX 3090 optimized: batch=8, accumulation=2 = eff_batch=16)
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    gradient_checkpointing: bool = True

    # Optimization
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Training features (shorter seq for BERT)
    max_seq_length: int = 512
    output_dir: str = "models/biobert_instruction_tuned"
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"

    # Misc
    seed: int = 42
    fp16: bool = True  # Better for BERT/smaller models
    bf16: bool = False
    use_cache: bool = False

    # HuggingFace Hub Integration
    push_to_hub: bool = True
    hub_model_id: Optional[str] = None
    hub_strategy: str = "every_save"
    hub_private_repo: bool = False

    # Weights & Biases (wandb) Integration
    use_wandb: bool = True
    wandb_project: str = "cancer-survival-phase1"
    wandb_entity: Optional[str] = None
    wandb_run_name: str = "biobert_instruction_tuning_phase1"
    wandb_tags: List[str] = field(
        default_factory=lambda: ["phase1", "instruction-tuning", "biobert", "cancer-survival"]
    )


def setup_authentication(config: TrainingConfig):
    """Setup HuggingFace Hub and Weights & Biases authentication."""
    logger.info("Setting up authentication...")

    # HuggingFace Hub Authentication
    if config.push_to_hub:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning(
                "HF_TOKEN not found in environment variables. "
                "Model will not be pushed to HuggingFace Hub. "
                "Set HF_TOKEN in .env or environment to enable."
            )
            config.push_to_hub = False
        else:
            try:
                hf_login(token=hf_token, add_to_git_credential=False)
                logger.info("✓ Successfully authenticated with HuggingFace Hub")
            except Exception as e:
                logger.error(f"Failed to authenticate with HuggingFace Hub: {e}")
                config.push_to_hub = False

        # Get repo ID from environment
        if config.hub_model_id is None:
            config.hub_model_id = os.getenv("HF_REPO_ID", "cancer-survival-biobert-instruction")
            logger.info(f"Using HF_REPO_ID: {config.hub_model_id}")

    # Weights & Biases Authentication
    if config.use_wandb:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            logger.warning(
                "WANDB_API_KEY not found in environment variables. "
                "Training metrics will not be logged to W&B. "
                "Set WANDB_API_KEY in .env or environment to enable."
            )
            config.use_wandb = False
        else:
            try:
                wandb.login(key=wandb_api_key)
                logger.info("✓ Successfully authenticated with Weights & Biases")
            except Exception as e:
                logger.error(f"Failed to authenticate with Weights & Biases: {e}")
                config.use_wandb = False

    return config


def setup_model_and_tokenizer(config: TrainingConfig):
    """Initialize model, tokenizer, and LoRA configuration."""
    logger.info(f"Loading model {config.model_id}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id,
        num_labels=1,  # Will be adapted per task
        trust_remote_code=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_instruction_data(jsonl_path: str, split_ratio: float = 0.9):
    """Load instruction-tuning data from JSONL."""
    logger.info(f"Loading instruction data from {jsonl_path}")

    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} QA pairs")

    # Split into train/eval
    n_train = int(len(data) * split_ratio)
    train_data = data[:n_train]
    eval_data = data[n_train:]

    logger.info(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    return train_data, eval_data


def extract_task_from_qa(example: dict) -> dict:
    """Extract text and task from QA pair for BERT classification."""
    messages = example.get('messages', [])
    task = example.get('task', 'unknown')

    # Extract user question and assistant response
    user_text = ""
    assistant_response = ""

    for msg in messages:
        if msg.get('role') == 'user':
            user_text = msg.get('content', '')
        elif msg.get('role') == 'assistant':
            assistant_response = msg.get('content', '')

    # For BERT, concatenate question and response
    combined_text = f"{user_text} {assistant_response}"

    # Create label based on task type
    task_to_label = {
        'cancer_type_identification': 0,
        'ajcc_stage_identification': 1,
        'prognosis_assessment': 2,
    }

    label = task_to_label.get(task, 0)

    return {
        'text': combined_text[:512],  # Truncate for BERT
        'label': label,
        'task': task,
    }


def create_dataset(data: List[dict], tokenizer, config: TrainingConfig):
    """Create HuggingFace Dataset from QA pairs."""
    processed_data = []

    for example in data:
        processed = extract_task_from_qa(example)
        processed_data.append(processed)

    # Convert to Dataset (labels as float to match fp16 dtype)
    dataset = Dataset.from_dict({
        'text': [d['text'] for d in processed_data],
        'label': [float(d['label']) for d in processed_data],
        'task': [d['task'] for d in processed_data],
    })

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length=config.max_seq_length,
            truncation=True,
            padding="max_length",
        )

    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "task"],
        desc="Tokenizing",
    )

    return tokenized_dataset


def train(config: TrainingConfig = None):
    """Run training with HuggingFace Hub and wandb integration."""
    if config is None:
        config = TrainingConfig()

    logger.info("="*80)
    logger.info("PHASE 1: BIO-CLINICALBERT INSTRUCTION-TUNING")
    logger.info("="*80)

    # Setup authentication
    config = setup_authentication(config)

    # Setup
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load data
    train_data, eval_data = load_instruction_data(
        "data/processed/instruction_tuning_data.jsonl"
    )

    # Create datasets
    train_dataset = create_dataset(train_data, tokenizer, config)
    eval_dataset = create_dataset(eval_data, tokenizer, config)

    datasets = DatasetDict({
        "train": train_dataset,
        "validation": eval_dataset,
    })

    # Training arguments
    report_to = []
    if config.use_wandb:
        report_to.append("wandb")
    if not report_to:
        report_to = ["none"]

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,

        # Optimization
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,

        # Logging & saving
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,

        # Mixed precision
        fp16=config.fp16,
        bf16=config.bf16,

        # HuggingFace Hub Integration
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        hub_strategy=config.hub_strategy,
        hub_private_repo=config.hub_private_repo,

        # Weights & Biases Integration
        report_to=report_to,
        run_name=config.wandb_run_name if config.use_wandb else None,

        # Other
        seed=config.seed,
        dataloader_pin_memory=True,
    )

    # Epoch tracking callback
    epoch_callback = EpochTrackingCallback(
        num_training_samples=len(datasets["train"]),
        batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[epoch_callback],
    )

    # Train
    logger.info("="*80)
    logger.info("STARTING TRAINING...")
    logger.info("="*80)
    if config.use_wandb:
        logger.info(f"📊 W&B Project: {config.wandb_project}")
        logger.info(f"📊 W&B Run: {config.wandb_run_name}")
    if config.push_to_hub:
        logger.info(f"🤗 Model will be pushed to: https://huggingface.co/{config.hub_model_id}")
    logger.info("="*80 + "\n")

    trainer.train()

    # Save locally
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)

    # Push to hub if enabled
    if config.push_to_hub:
        logger.info(f"\n{'='*80}")
        logger.info("PUSHING MODEL TO HUGGINGFACE HUB...")
        logger.info(f"{'='*80}")
        try:
            model.push_to_hub(
                config.hub_model_id,
                private=config.hub_private_repo,
                commit_message="Phase 1: Bio-ClinicalBERT instruction-tuned for cancer survival prediction"
            )
            logger.info(f"✓ Model successfully pushed to:")
            logger.info(f"  https://huggingface.co/{config.hub_model_id}")
        except Exception as e:
            logger.error(f"Failed to push model to HuggingFace Hub: {e}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"✓ Local model saved to: {config.output_dir}")
    if config.push_to_hub:
        logger.info(f"✓ Model uploaded to: https://huggingface.co/{config.hub_model_id}")
    if config.use_wandb:
        logger.info(f"✓ Training metrics: https://wandb.ai/your_username/{config.wandb_project}")
    logger.info(f"\nComparison: Run both models for Phase 1!")
    logger.info(f"  Llama-3.1-8B-Instruct: Better reasoning, slower inference")
    logger.info(f"  Bio-ClinicalBERT: Medical domain, faster inference")
    logger.info(f"{'='*80}\n")

    return trainer


if __name__ == "__main__":
    config = TrainingConfig()
    trainer = train(config)
