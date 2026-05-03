"""
Strategy 4: Cancer-Type Conditioning Token Fine-Tuning (OpenBioLLM-8B, 4-bit)
==========================================================================
Prepends the cancer type (e.g., "[GLIOMAS]") to each pathological text
before tokenization. Uses 4-bit quantization to fit the 8B model in 24GB VRAM.
"""
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt


class ConditionedLlamaDataset(Dataset):
    """Dataset that prepends cancer type as a conditioning token to the text."""
    
    def __init__(self, df, tokenizer, max_length=512, cancer_type_col='DISEASE_TYPE'):
        self.df = df.dropna(subset=['text', 'OS_MONTHS', 'OS_STATUS']).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cancer_type_col = cancer_type_col
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.loc[idx, 'text'])
        duration = float(self.df.loc[idx, 'OS_MONTHS'])
        status_str = str(self.df.loc[idx, 'OS_STATUS'])
        event = 1.0 if 'DECEASED' in status_str else 0.0

        # --- KEY CHANGE: Prepend cancer type ---
        cancer_type = str(self.df.loc[idx, self.cancer_type_col]).strip()
        cancer_tag = f"[{cancer_type.upper()}]"
        conditioned_text = f"{cancer_tag} {text}"

        encoding = self.tokenizer(
            conditioned_text, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'duration': torch.tensor(duration, dtype=torch.float32),
            'event': torch.tensor(event, dtype=torch.float32),
            'patient_id': str(self.df.loc[idx, 'TCGA_Barcode']),
            'cancer_type': cancer_type
        }


class SurvivalLlamaModel(torch.nn.Module):
    def __init__(self, model_name="aaditya/Llama3-OpenBioLLM-8B", use_lora=True):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        self.base_model = AutoModel.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        if use_lora:
            self.base_model = prepare_model_for_kbit_training(self.base_model)
            peft_config = LoraConfig(
                task_type="FEATURE_EXTRACTION", r=16, lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
        self.risk_head = torch.nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        sequence_lengths = torch.clamp(sequence_lengths, min=0, max=input_ids.shape[1] - 1)
        cls_embedding = outputs.last_hidden_state[
            torch.arange(batch_size, device=input_ids.device), sequence_lengths
        ]
        cls_embedding = cls_embedding.to(self.risk_head.weight.dtype)
        risk_score = self.risk_head(cls_embedding)
        return risk_score.squeeze(-1), cls_embedding


def cox_ph_loss(log_h, events, durations):
    idx = torch.argsort(durations, descending=True)
    events = events[idx]; log_h = log_h[idx]
    log_h_max = torch.max(log_h)
    risk_set_sums = torch.cumsum(torch.exp(log_h - log_h_max), dim=0)
    log_risk_set_sums = torch.log(risk_set_sums) + log_h_max
    loss = -(log_h - log_risk_set_sums) * events
    return loss.sum() / (events.sum() + 1e-8)


def train_and_extract(data_path, output_dir, max_epochs=20, batch_size=4, lr=2e-5,
                       patience=3, val_split=0.15,
                       hf_token=None, hf_repo_id=None,
                       wandb_api_key=None, wandb_project=None):
    """Fine-tune OpenBioLLM-8B with cancer-type conditioning tokens and early stopping."""
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    use_wandb = wandb_project and wandb_api_key
    if use_wandb:
        import wandb
        wandb.login(key=wandb_api_key)
        wandb.init(project=wandb_project, name="Llama3-Conditioned", config={
            "strategy": "cancer_type_conditioning_token",
            "max_epochs": max_epochs, "batch_size": batch_size,
            "learning_rate": lr, "model_name": "aaditya/Llama3-OpenBioLLM-8B"
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(data_path, low_memory=False)
    model_name = "aaditya/Llama3-OpenBioLLM-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    full_dataset = ConditionedLlamaDataset(df, tokenizer)

    sample = full_dataset[0]
    decoded = tokenizer.decode(sample['input_ids'][:30], skip_special_tokens=True)
    print(f"Example conditioned input: {decoded}...")

    total = len(full_dataset)
    val_n = int(total * val_split); train_n = total - val_n
    train_ds, val_ds = random_split(full_dataset, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    print(f"Split: {train_n} train / {val_n} val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SurvivalLlamaModel(model_name=model_name).to(device)
    model.risk_head = model.risk_head.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float('inf'); best_epoch = 0; epochs_no_improve = 0
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, 'llama_conditioned_best.pt')
    train_losses, val_losses = [], []

    print(f"\nTraining (max {max_epochs} epochs, patience={patience})...")
    for epoch in range(max_epochs):
        model.train()
        total_train, n_train = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")
        for batch in pbar:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            dur = batch['duration'].to(device)
            ev = batch['event'].to(device)
            optimizer.zero_grad()
            rs, _ = model(ids, mask)
            loss = cox_ph_loss(rs, ev, dur)
            if torch.isnan(loss): continue
            loss.backward(); optimizer.step()
            total_train += loss.item(); n_train += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            if use_wandb: wandb.log({"train/batch_loss": loss.item()})

        avg_train = total_train / max(n_train, 1)
        train_losses.append(avg_train)

        model.eval()
        total_val, n_val = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]", leave=False):
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                dur = batch['duration'].to(device)
                ev = batch['event'].to(device)
                rs, _ = model(ids, mask)
                loss = cox_ph_loss(rs, ev, dur)
                if not torch.isnan(loss): total_val += loss.item(); n_val += 1

        avg_val = total_val / max(n_val, 1)
        val_losses.append(avg_val)
        print(f"Epoch {epoch+1}: train={avg_train:.4f} val={avg_val:.4f}")
        if use_wandb:
            wandb.log({"train/epoch_loss": avg_train, "val/epoch_loss": avg_val, "epoch": epoch+1})

        if avg_val < best_val_loss:
            best_val_loss = avg_val; best_epoch = epoch + 1; epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> Best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  --- No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"\n*** Early stopping at epoch {epoch+1}! Best was {best_epoch} ***")
                break

    actual_epochs = epoch + 1

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ep = range(1, actual_epochs + 1)
    ax.plot(ep, train_losses, marker='o', label='Train', linewidth=2)
    ax.plot(ep, val_losses, marker='s', label='Val', linewidth=2)
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best ({best_epoch})')
    ax.set_title('OpenBioLLM-8B Conditioned: Train vs Val Loss', fontsize=14)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Cox PH Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    plot_path = os.path.join(output_dir, 'llama_conditioned_loss.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight'); plt.close(fig)

    results_df = pd.DataFrame({'epoch': list(ep), 'train_loss': train_losses,
                                'val_loss': val_losses, 'is_best': [e == best_epoch for e in ep]})
    results_path = os.path.join(output_dir, 'llama_conditioned_results.csv')
    results_df.to_csv(results_path, index=False)

    # Load best & extract
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    if hf_token and hf_repo_id:
        repo = hf_repo_id + "-llama-conditioned"
        model.base_model.push_to_hub(repo, token=hf_token)
        tokenizer.push_to_hub(repo, token=hf_token)
    if use_wandb: wandb.finish()

    model.eval()
    extract_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    embs, risks, pids, cts = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(extract_loader, desc="Extracting"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            rs, emb = model(ids, mask)
            embs.append(emb.to(torch.float32).cpu())
            risks.extend(rs.cpu().numpy()); pids.extend(batch['patient_id'])
            cts.extend(batch['cancer_type'])

    all_embs = torch.cat(embs, dim=0).numpy()
    emb_df = pd.DataFrame(all_embs, columns=[f'llama_emb_{i}' for i in range(all_embs.shape[1])])
    emb_df['risk_score'] = risks; emb_df['TCGA_Barcode'] = pids; emb_df['cancer_type'] = cts
    out_path = os.path.join(output_dir, 'finetuned_llama_conditioned_embeddings.csv')
    emb_df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY — Strategy 4: Cancer-Type Conditioning (OpenBioLLM-8B)")
    print(f"{'='*60}")
    print(f"  Best epoch:     {best_epoch} / {actual_epochs}")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Embeddings:     {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    load_dotenv(override=True)
    train_and_extract(
        data_path='data/processed/merged_tcga_data_final.csv',
        output_dir='data/processed',
        max_epochs=20, patience=3, val_split=0.15,
        hf_token=os.environ.get("HF_TOKEN"),
        hf_repo_id=os.environ.get("HF_REPO_ID"),
        wandb_api_key=os.environ.get("WANDB_API_KEY"),
        wandb_project=os.environ.get("WANDB_PROJECT"),
    )
