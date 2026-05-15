#!/bin/bash

# Phase 1: Instruction-Tuning Execution Script
# Implements hybrid strategy: instruction-tuning + multi-task learning + optimized hyperparameters

set -e

echo "=================================================="
echo "Phase 1: Hybrid Strategy Implementation"
echo "=================================================="
echo ""

# Configuration
DATA_CSV="data/processed/merged_tcga_data_text_dedup.csv"
QA_PAIRS_JSONL="data/processed/instruction_tuning_data.jsonl"
MODEL_DIR="models/llama_instruction_tuned"
EVAL_DIR="data/processed/evaluation"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Generate Instruction-Tuning Data
echo -e "${BLUE}Step 1: Generating Instruction-Tuning Data${NC}"
echo "Converting 8,459 pathological reports → 25,377 QA pairs"
echo "Tasks: Cancer Type, AJCC Stage, Prognosis Assessment"
echo ""

python src/training/generate_instruction_tuning_data.py \
    "$DATA_CSV" \
    "$QA_PAIRS_JSONL"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Instruction-tuning data generated successfully${NC}"
    echo "  Output: $QA_PAIRS_JSONL"
    echo ""
else
    echo -e "${YELLOW}✗ Error generating instruction-tuning data${NC}"
    exit 1
fi

# Step 2: Fine-tune Model
echo -e "${BLUE}Step 2: Fine-tuning Llama-3.1-8B with Instruction-Tuning${NC}"
echo "Hyperparameters:"
echo "  - LoRA Rank: 32 (↑ from 16)"
echo "  - Learning Rate: 2e-4 (↑ from default)"
echo "  - Training Steps: 15,000"
echo "  - Max Tokens: 4,096 (↑ from 2,048)"
echo "  - Batch Size: 16 (effective)"
echo ""

python src/training/llama_finetune_instruction.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Model fine-tuning completed${NC}"
    echo "  Output: $MODEL_DIR"
    echo ""
else
    echo -e "${YELLOW}✗ Error during fine-tuning${NC}"
    exit 1
fi

# Step 3: Evaluate Results
echo -e "${BLUE}Step 3: Evaluating with Phase 1 Metrics${NC}"
echo "Metrics:"
echo "  - Classification Accuracy (cancer type, AJCC stage)"
echo "  - F1-score and Precision/Recall"
echo "  - C-index for survival ranking"
echo "  - Per-cancer-type performance"
echo ""

python src/training/evaluate_instruction_tuned.py \
    --model_path "$MODEL_DIR" \
    --data_path "$DATA_CSV" \
    --output_dir "$EVAL_DIR"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Evaluation completed${NC}"
    echo "  Results: $EVAL_DIR"
    echo ""
else
    echo -e "${YELLOW}✗ Error during evaluation${NC}"
    exit 1
fi

# Summary
echo "=================================================="
echo -e "${GREEN}Phase 1 Complete!${NC}"
echo "=================================================="
echo ""
echo "Generated Artifacts:"
echo "  1. Instruction-Tuning Data: $QA_PAIRS_JSONL"
echo "  2. Fine-tuned Model: $MODEL_DIR"
echo "  3. Evaluation Results: $EVAL_DIR"
echo ""
echo "Next Steps (Phase 2):"
echo "  - Try Llama-3.1-70B for better performance"
echo "  - Implement hierarchical specialization per cancer type"
echo "  - Add chain-of-thought reasoning"
echo ""
echo "Expected Improvements:"
echo "  ✓ +2-5% C-index improvement"
echo "  ✓ Better interpretability with classification metrics"
echo "  ✓ Enhanced reasoning capability"
echo ""
