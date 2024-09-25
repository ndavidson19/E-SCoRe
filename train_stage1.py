# src/train_stage_I.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
import yaml
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn import functional as F

class CustomDatasetStageI(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        prompt = entry['prompt']
        y1 = entry['y1']
        y2 = entry['y2']
        # Encode y2 as the target, given y1
        input_text = f"{prompt}\nSolution: {y1}\nCorrected Solution:"
        input_ids = self.tokenizer.encode(input_text, truncation=True, max_length=self.max_length, return_tensors='pt').squeeze()
        target_ids = self.tokenizer.encode(y2, truncation=True, max_length=self.max_length, return_tensors='pt').squeeze()
        x1 = self.tokenizer.encode(f"{prompt}\nSolution:", truncation=True, max_length=self.max_length, return_tensors='pt').squeeze()
        return {
            'input_ids': input_ids,
            'labels': target_ids,
            'x1': x1
        }

def compute_kl_divergence(current_logits, ref_logits):
    """
    Compute the KL divergence between current and reference logits.
    """
    current_probs = F.log_softmax(current_logits, dim=-1)
    ref_probs = F.softmax(ref_logits, dim=-1)
    kl_div = F.kl_div(current_probs, ref_probs, reduction='batchmean')
    return kl_div

def main(config):
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['name'])
    model = GPT2LMHeadModel.from_pretrained(config['model']['name']).to(config['device'])
    
    # Load training dataset
    with open(config['data']['training_data_path'], 'r') as f:
        training_data = [json.loads(line) for line in f]
    
    # Initialize dataset and dataloader
    dataset = CustomDatasetStageI(training_data, tokenizer, config['training']['max_length'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate_stage_I'])
    
    # Initialize reference model for KL divergence (base model)
    ref_model = GPT2LMHeadModel.from_pretrained(config['model']['name']).to(config['device'])
    ref_model.eval()
    
    model.train()
    for epoch in range(config['training']['stage_I_epochs']):
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(config['device'])
            labels = batch['labels'].to(config['device'])
            x1 = batch['x1'].to(config['device'])

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Compute KL divergence between current and reference model for y1
            with torch.no_grad():
                ref_outputs = ref_model(input_ids=x1)
                ref_logits = ref_outputs.logits

            current_outputs = model(input_ids=x1)
            current_logits = current_outputs.logits

            kl_div = compute_kl_divergence(current_logits, ref_logits)

            # Total loss
            total_batch_loss = loss + config['training']['beta_2'] * kl_div

            # Backward pass and optimization
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Stage I Epoch {epoch+1}/{config['training']['stage_I_epochs']} - Avg Loss: {avg_loss:.4f}")
    
    # Save the initialized model
    torch.save(model.state_dict(), config['model']['save_path_stage_I'])
    print("Stage I Training Completed and Model Saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_stage_I.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config)
