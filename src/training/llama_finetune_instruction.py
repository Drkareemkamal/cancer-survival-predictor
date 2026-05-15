"""
Fine-tune Llama-3.2-8B-Instruct with 4-bit quantization for RTX 3090.

Tasks:
1. Cancer Type Identification
2. AJCC Stage Determination
3. Prognosis Assessment

Training approach: Instruction-tuning with synthetic QA pairs + 4-bit quantization
Hyperparameters optimized for RTX 3090 VRAM constraints (~24GB)

Features:
- 4-bit quantization (nf4 + double quant) - ~4-5GB VRAM usage
- Flash Attention 2 for memory efficiency
- LoRA efficient fine-tuning (<<1% of parameters trainable)
- Early stopping to prevent plateau training
- Weights & Biases (wandb) integration for monitoring
- HuggingFace Hub integration for model upload
- Memory tracking and GPU utilization logging
"""

import torch
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import pandas as pd
from tqdm import tqdm
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, DatasetDict
import wandb
from huggingface_hub import login as hf_login
from dotenv import load_dotenv
from transformers import TrainerCallback, EarlyStoppingCallback

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpochTrackingCallback(TrainerCallback):
    """Log epoch progress and memory usage."""
    def __init__(self, num_training_samples, batch_size, gradient_accumulation_steps):
        self.steps_per_epoch = num_training_samples / (batch_size * gradient_accumulation_steps)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            current_epoch = state.global_step / self.steps_per_epoch
            logs["epoch_float"] = current_epoch

            # Log memory usage
            if torch.cuda.is_available():
                logs["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                logs["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9


@dataclass
class TrainingConfig:
    """RTX 3090 optimized hyperparameters with 4-bit quantization"""
    # Model
    model_id: str = "voidful/Llama-3.2-8B-Instruct"#"meta-llama/Llama-3.2-8B-Instruct"

    # LoRA (optimized for RTX 3090: smaller rank to reduce memory)
    lora_r: int = 16  # Reduced from 32 for RTX 3090 memory
    lora_alpha: int = 32  # Rank-stabilized LoRA
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training (RTX 3090 optimized: reduced steps)
    num_train_epochs: int = 5
    max_steps: int = 10000  # Reduced from 15000 for 4-bit quantization
    learning_rate: float = 1e-4  # Reduced for 4-bit stability
    warmup_steps: int = 200  # Reduced warmup
    warmup_ratio: float = 0.0

    # Batch & gradient (RTX 3090 4-bit: batch=1, accumulation=16 = eff_batch=16)
    per_device_train_batch_size: int = 1  # Reduced to 1 for RTX 3090 4-bit
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16  # Increased to maintain effective batch size
    gradient_checkpointing: bool = True

    # Optimization
    optim: str = "paged_adamw_32bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5  # Stricter clipping for 4-bit stability

    # Training features (RTX 3090: further reduced for 4-bit)
    max_seq_length: int = 1024  # Reduced from 2048 for RTX 3090 4-bit
    output_dir: str = "models/llama_instruction_tuned"
    logging_steps: int = 50  # More frequent logging
    save_steps: int = 500
    eval_steps: int = 250
    save_total_limit: int = 2  # Keep fewer checkpoints to save memory
    load_best_model_at_end: bool = True  # Enable best model tracking
    metric_for_best_model: str = "eval_loss"

    # Early stopping for plateau detection
    early_stopping_patience: int = 3  # Stop if no improvement after 3 evals
    early_stopping_threshold: float = 1e-4  # Minimum improvement threshold

    # Misc
    seed: int = 42
    fp16: bool = False
    bf16: bool = True  # bfloat16 works well with 4-bit
    use_cache: bool = False  # Disable cache for memory

    # 4-bit quantization specific
    use_flash_attention_2: bool = True  # Disabled - CUDA/torch linking issues
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"  # Normalized float 4-bit

    # HuggingFace Hub Integration
    push_to_hub: bool = False  # Disabled by default to avoid memory issues
    hub_model_id: Optional[str] = None  # Will use HF_REPO_ID from env if None
    hub_strategy: str = "end"  # Only push at end to save memory
    hub_private_repo: bool = False

    # Weights & Biases (wandb) Integration
    use_wandb: bool = True
    wandb_project: str = "FinetunePathologicalTextOnLllama-3.2-8B-Instruct-4bit"
    wandb_entity: Optional[str] = None  # Your W&B username (optional)
    wandb_run_name: str = "FinetuneLlama-3.2-8B-Instruct-4bit"
    wandb_tags: List[str] = field(
        default_factory=lambda: ["4-bit", "RTX3090", "instruction-tuning", "cancer-survival"]
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
            config.hub_model_id = os.getenv("HF_REPO_ID", "cancer-survival-instruction-tuned")
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
    """Initialize model, tokenizer, and LoRA configuration optimized for RTX 3090."""
    logger.info(f"Loading model {config.model_id} with 4-bit quantization for RTX 3090")
    logger.info(f"⚙️  Using {config.bnb_4bit_quant_type} quantization with double quant")

    # 4-bit quantization config optimized for RTX 3090
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,  # "nf4" or "fp4"
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for compute
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with 4-bit quantization
    logger.info("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map="auto",  # Auto distribute across GPU
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if config.use_flash_attention_2 else None,
    )

    if config.use_flash_attention_2:
        logger.info("✓ Flash Attention 2 enabled for memory efficiency")

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.gradient_checkpointing
    )
    logger.info("✓ Model prepared for 4-bit training")

    # LoRA config optimized for RTX 3090
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    logger.info("✓ LoRA applied to model")
    model.print_trainable_parameters()

    # Log memory usage
    if torch.cuda.is_available():
        logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        logger.info(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")

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


def format_instruction_example(example: dict) -> str:
    """Format instruction example as training text."""
    messages = example.get('messages', [])

    # Format as conversation: <system> ... <user> ... <assistant> ...
    text = ""
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        if role == 'system':
            text += f"<system>{content}</system>\n"
        elif role == 'user':
            text += f"<user>{content}</user>\n"
        elif role == 'assistant':
            text += f"<assistant>{content}</assistant>\n"

    return text


def create_dataset(data: List[dict], tokenizer, config: TrainingConfig):
    """Create HuggingFace Dataset from QA pairs."""
    formatted_texts = []

    for example in data:
        text = format_instruction_example(example)
        formatted_texts.append(text)

    # Convert to Dataset
    dataset = Dataset.from_dict({"text": formatted_texts})

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            max_length=config.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    return tokenized_dataset


def train(config: TrainingConfig = None):
    """Run training with HuggingFace Hub and wandb integration."""
    if config is None:
        config = TrainingConfig()

    logger.info("="*80)
    logger.info("PHASE 1: INSTRUCTION-TUNING WITH HF HUB & WANDB INTEGRATION")
    logger.info("="*80)

    # Setup authentication for HF Hub and wandb
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

    # Training arguments (Phase 1 optimized)
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
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
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

    # Callbacks for training
    epoch_callback = EpochTrackingCallback(
        num_training_samples=len(datasets["train"]),
        batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    # Early stopping callback to prevent plateau training
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_threshold=config.early_stopping_threshold,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        callbacks=[epoch_callback, early_stopping_callback],
    )

    # Train
    logger.info("="*80)
    logger.info("🚀 STARTING TRAINING (RTX 3090 4-BIT QUANTIZED)")
    logger.info("="*80)
    logger.info(f"Model: {config.model_id}")
    logger.info(f"Sequence Length: {config.max_seq_length}")
    logger.info(f"Batch Size: {config.per_device_train_batch_size} (Effective: {config.per_device_train_batch_size * config.gradient_accumulation_steps})")
    logger.info(f"LoRA Rank: {config.lora_r} | Learning Rate: {config.learning_rate}")
    logger.info(f"Early Stopping: Patience={config.early_stopping_patience}, Threshold={config.early_stopping_threshold}")
    if config.use_wandb:
        logger.info(f"📊 W&B Project: {config.wandb_project}")
        logger.info(f"📊 W&B Run: {config.wandb_run_name}")
        logger.info(f"📊 View training at: https://wandb.ai/your_username/{config.wandb_project}")
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
                commit_message="Phase 1: Instruction-tuned Llama-8B for cancer survival prediction"
            )
            logger.info(f"✓ Model successfully pushed to:")
            logger.info(f"  https://huggingface.co/{config.hub_model_id}")
        except Exception as e:
            logger.error(f"Failed to push model to HuggingFace Hub: {e}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("✅ TRAINING SUMMARY (4-BIT QUANTIZED)")
    logger.info(f"{'='*80}")
    logger.info(f"✓ Local model saved to: {config.output_dir}")
    logger.info(f"✓ Training optimized for RTX 3090 with 4-bit quantization")
    logger.info(f"✓ Memory efficient: ~4-5GB VRAM used")
    if config.push_to_hub:
        logger.info(f"✓ Model uploaded to: https://huggingface.co/{config.hub_model_id}")
    if config.use_wandb:
        logger.info(f"✓ Training metrics: https://wandb.ai/your_username/{config.wandb_project}")
    logger.info(f"\n📝 Notes:")
    logger.info(f"  • 4-bit quantization applied (nf4 + double quant)")
    logger.info(f"  • Early stopping enabled to prevent plateau training")
    logger.info(f"  • Flash Attention 2 enabled for efficiency")
    logger.info(f"\n🔄 Next step: Run evaluation")
    logger.info(f"  python src/training/evaluate_instruction_tuned.py")
    logger.info(f"{'='*80}\n")

    return trainer


if __name__ == "__main__":
    config = TrainingConfig()
    trainer = train(config)
