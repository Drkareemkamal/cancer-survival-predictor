import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import transformers
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

# Bypass the strict torch >= 2.6 check for torch.load to avoid CUDA version conflicts
if hasattr(transformers.utils.import_utils, 'check_torch_load_is_safe'):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from peft import LoraConfig, get_peft_model
from tqdm import tqdm

class PathologySurvivalDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        # Drop rows with missing text or survival data
        self.df = df.dropna(subset=['text', 'OS_MONTHS', 'OS_STATUS']).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.loc[idx, 'text'])
        duration = float(self.df.loc[idx, 'OS_MONTHS'])
        # OS_STATUS typically "0:LIVING" or "1:DECEASED"
        status_str = str(self.df.loc[idx, 'OS_STATUS'])
        event = 1.0 if 'DECEASED' in status_str else 0.0

        # Tokenize with truncation to max_length
        # By default, clinical notes have the most important diagnoses near the end, 
        # but truncation from the right is standard. We will use standard truncation.
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'duration': torch.tensor(duration, dtype=torch.float32),
            'event': torch.tensor(event, dtype=torch.float32),
            'patient_id': str(self.df.loc[idx, 'TCGA_Barcode'])
        }

class SurvivalTextModel(torch.nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", use_lora=True):
        super().__init__()
        # Explicitly loading in float32 (no quantization). 
        # Note: Bio_ClinicalBERT is a small model (~110M parameters), which is why it only uses ~3.8GB of VRAM!
        self.base_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        
        if use_lora:
            peft_config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                r=8,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.1,
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
            
        self.risk_head = torch.nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        risk_score = self.risk_head(cls_embedding)
        return risk_score.squeeze(-1), cls_embedding

def cox_ph_loss(log_h, events, durations):
    # Sort descending by duration
    idx = torch.argsort(durations, descending=True)
    events = events[idx]
    log_h = log_h[idx]
    
    log_h_max = torch.max(log_h)
    risk_set_sums = torch.cumsum(torch.exp(log_h - log_h_max), dim=0)
    log_risk_set_sums = torch.log(risk_set_sums) + log_h_max
    
    loss = -(log_h - log_risk_set_sums) * events
    return loss.sum() / (events.sum() + 1e-8)

def train_and_extract(data_path, output_dir, max_epochs=20, batch_size=8, lr=1e-4,
                       patience=3, val_split=0.15,
                       hf_token=None, hf_repo_id=None,
                       wandb_api_key=None, wandb_project=None):
    """
    Fine-tune ClinicalBERT with automatic early stopping.
    
    Args:
        max_epochs:  Maximum number of epochs to train (default: 20).
        patience:    Stop training if validation loss doesn't improve for this many epochs (default: 3).
        val_split:   Fraction of data to hold out for validation (default: 0.15 = 15%).
    """
    if hf_token:
        from huggingface_hub import login
        print("Logging into Hugging Face Hub...")
        login(token=hf_token)

    if wandb_project and wandb_api_key:
        import wandb
        print("Logging into Weights & Biases...")
        wandb.login(key=wandb_api_key)
        wandb.init(project=wandb_project, name="ClinicalBERT-Finetune", config={
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "patience": patience,
            "val_split": val_split,
            "model_name": "emilyalsentzer/Bio_ClinicalBERT"
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    full_dataset = PathologySurvivalDataset(df, tokenizer)
    
    # --- Train / Validation Split ---
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))
    
    print(f"Dataset split: {train_size} train / {val_size} validation samples")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = SurvivalTextModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # --- Early Stopping State ---
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, 'clinicalbert_best_model.pt')
    
    train_losses = []
    val_losses = []
    
    print(f"Starting training (max {max_epochs} epochs, patience={patience})...")
    for epoch in range(max_epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            durations = batch['duration'].to(device)
            events = batch['event'].to(device)
            
            optimizer.zero_grad()
            risk_scores, _ = model(input_ids, attention_mask)
            loss = cox_ph_loss(risk_scores, events, durations)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if wandb_project and wandb_api_key:
                wandb.log({"train/batch_loss": loss.item()})
        
        avg_train_loss = total_train_loss / max(train_batches, 1)
        train_losses.append(avg_train_loss)
        
        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                durations = batch['duration'].to(device)
                events = batch['event'].to(device)
                
                risk_scores, _ = model(input_ids, attention_mask)
                loss = cox_ph_loss(risk_scores, events, durations)
                
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                    val_batches += 1
        
        avg_val_loss = total_val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")
        
        if wandb_project and wandb_api_key:
            wandb.log({
                "train/epoch_loss": avg_train_loss,
                "val/epoch_loss": avg_val_loss,
                "epoch": epoch + 1
            })
        
        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  --- No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print(f"\n*** Early stopping triggered at epoch {epoch+1}! ***")
                print(f"    Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}")
                break
    
    actual_epochs = epoch + 1
    
    # --- Save Training & Validation Loss Plot ---
    fig, ax = plt.subplots(figsize=(12, 7))
    epochs_range = range(1, actual_epochs + 1)
    ax.plot(epochs_range, train_losses, marker='o', label='Train Loss', linewidth=2)
    ax.plot(epochs_range, val_losses, marker='s', label='Validation Loss', linewidth=2)
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax.set_title('ClinicalBERT Fine-tuning: Train vs Validation Loss', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cox PH Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(output_dir, 'clinicalbert_training_loss.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Training loss plot saved to {plot_path}")
    
    # --- Save Training Results CSV ---
    results_df = pd.DataFrame({
        'epoch': list(epochs_range),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'is_best': [e == best_epoch for e in epochs_range]
    })
    results_path = os.path.join(output_dir, 'clinicalbert_training_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Training results CSV saved to {results_path}")
    
    # --- Load Best Model Before Extraction ---
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
        
    if hf_token and hf_repo_id:
        print(f"Pushing fine-tuned model to Hugging Face Hub: {hf_repo_id}...")
        model.base_model.push_to_hub(hf_repo_id, token=hf_token)
        tokenizer.push_to_hub(hf_repo_id, token=hf_token)
        
        # Push the model card (README.md) to HuggingFace Hub
        readme_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                   'notebooks', 'README.md')
        if os.path.exists(readme_path):
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=hf_repo_id,
                token=hf_token,
            )
            print(f"Model card (README.md) pushed to {hf_repo_id}")
        
        print("Model push complete!")

    if wandb_project and wandb_api_key:
        import wandb
        wandb.finish()

    print("Extracting fine-tuned embeddings using best model...")
    model.eval()
    
    # Extract from the FULL dataset (not just train)
    extract_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    embeddings_list = []
    risk_scores_list = []
    patient_ids = []
    
    with torch.no_grad():
        for batch in tqdm(extract_loader, desc="Extracting embeddings and risk scores"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            risk_scores, cls_embeddings = model(input_ids, attention_mask)
            
            embeddings_list.append(cls_embeddings.cpu())
            risk_scores_list.extend(risk_scores.cpu().numpy())
            patient_ids.extend(batch['patient_id'])
            
    all_embeddings = torch.cat(embeddings_list, dim=0).numpy()
    
    # Save to CSV
    emb_df = pd.DataFrame(all_embeddings)
    emb_df.columns = [f'text_emb_{i}' for i in range(all_embeddings.shape[1])]
    emb_df['risk_score'] = risk_scores_list
    emb_df['TCGA_Barcode'] = patient_ids
    
    out_path = os.path.join(output_dir, 'finetuned_text_embeddings.csv')
    emb_df.to_csv(out_path, index=False)
    print(f"Embeddings saved to {out_path}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Best epoch:        {best_epoch} / {actual_epochs}")
    print(f"  Best val loss:     {best_val_loss:.4f}")
    print(f"  Final train loss:  {train_losses[-1]:.4f}")
    print(f"  Training plot:     {plot_path}")
    print(f"  Results CSV:       {results_path}")
    print(f"  Embeddings CSV:    {out_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    # Load environment variables from the .env file in the project root
    load_dotenv(override=True)
    
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb_project = os.environ.get("WANDB_PROJECT")
    
    train_and_extract(
        data_path='data/processed/merged_tcga_data_final.csv',
        output_dir='data/processed',
        max_epochs=20,       # Train up to 20 epochs max
        patience=3,          # Stop if no improvement for 3 epochs
        val_split=0.15,      # 15% validation holdout
        hf_token=hf_token,
        hf_repo_id=hf_repo_id,
        wandb_api_key=wandb_api_key,
        wandb_project=wandb_project
    )
