import sys
import os

# Add src to path so we can import the training module
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import pandas as pd
from transformers import AutoTokenizer
from src.training.text_finetune import PathologySurvivalDataset, SurvivalTextModel, cox_ph_loss
from torch.utils.data import DataLoader

def run_sanity_check():
    print("Running sanity check for pathology text fine-tuning...")
    data_path = 'data/processed/merged_tcga_data_text_dedup.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    print("Loading 50 samples...")
    df = pd.read_csv(data_path, nrows=50)
    
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    dataset = PathologySurvivalDataset(df, tokenizer, max_length=128) # Smaller length for quick check
    
    if len(dataset) == 0:
        print("Error: No valid samples found after filtering missing data.")
        return
        
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print("Initializing model...")
    model = SurvivalTextModel(use_lora=True)
    
    batch = next(iter(dataloader))
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    durations = batch['duration']
    events = batch['event']
    
    print(f"Batch loaded. input_ids shape: {input_ids.shape}")
    
    print("Running forward pass...")
    risk_scores, embeddings = model(input_ids, attention_mask)
    
    print(f"Risk scores shape: {risk_scores.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("Computing Cox loss...")
    loss = cox_ph_loss(risk_scores, events, durations)
    
    print(f"Loss value: {loss.item()}")
    
    if torch.isnan(loss):
        print("FAILED: Loss is NaN!")
    else:
        print("SUCCESS: Forward pass and loss computation succeeded!")

if __name__ == "__main__":
    run_sanity_check()
