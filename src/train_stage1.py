# src/train_score_multiturn.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
import yaml
import argparse
from reward_function import RewardFunction
from scheduler import InverseScheduler
from multiturn_environment import MultiTurnSelfCorrectionEnvWithScheduler
from ppo_agent import PPOAgent  # Replace with your actual PPO implementation
from torch.utils.data import Dataset, DataLoader

def main(config):
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['name'])
    model = GPT2LMHeadModel.from_pretrained(config['model']['name']).to(config['device'])

    # Load training dataset
    with open(config['data']['training_data_path'], 'r') as f:
        training_data = [json.loads(line) for line in f]

    # Initialize scheduler parameters
    scheduler_params = config['scheduler']

    # Initialize reward function parameters
    reward_function_params = config['reward_function']

    # Initialize the environment
    env = MultiTurnSelfCorrectionEnvWithScheduler(
        model=model,
        tokenizer=tokenizer,
        dataset=training_data,
        reward_function=reward_function_params,
        alpha=config['entropy']['alpha'],
        entropy_weight=config['entropy']['entropy_weight'],
        max_length=config['training']['max_length'],
        n_max_turns=config['training']['n_max_turns'],
        scheduler_params=scheduler_params,
        stop_threshold=config['training']['stop_threshold']
    )

    # Initialize PPO agent
    agent = PPOAgent(model=model, tokenizer=tokenizer, env=env, config=config)

    # Stage I: Model Initialization to Prevent Collapse
    print("Starting Stage I: Model Initialization")
    initialize_model_stage_I(model, tokenizer, training_data, config)

    # Stage II: Multi-Turn RL Training with Reward Shaping
    print("Starting Stage II: Reinforcement Learning with PPO")
    for epoch in range(config['training']['rl_epochs']):
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward, done)
        
        # Update PPO agent after each episode
        agent.update()
        
        # Logging
        metrics = env.get_metrics()
        print(f"RL Epoch {epoch+1}/{config['training']['rl_epochs']} - Metrics: {metrics}")

    # Save the trained model
    torch.save(model.state_dict(), config['model']['save_path'])

def initialize_model_stage_I(model, tokenizer, training_data, config):
    """
    Stage I: Fine-tune the model to produce high-reward second attempts while keeping first attempts close to the base model.
    """
    # Implement Stage I specific fine-tuning here
    # Example: Iterate over DataLoader and optimize model to produce y2 given y1 with KL constraints
    dataset = CustomDatasetStageI(training_data, tokenizer, config['training']['max_length'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate_stage_I'])

    model.train()
    for epoch in range(config['training']['stage_I_epochs']):
        for batch in dataloader:
            inputs = batch['input_ids'].to(config['device'])
            labels = batch['labels'].to(config['device'])
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            # Compute KL divergence between pi_theta and pi_ref for first attempt
            kl_loss = compute_kl_divergence(model, tokenizer, batch['x1'].to(config['device']))
            total_loss = loss + config['stage_I']['beta_2'] * kl_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        print(f"Stage I Epoch {epoch+1}/{config['training']['stage_I_epochs']} - Loss: {total_loss.item()}")
    model.eval()

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

def compute_kl_divergence(model, tokenizer, x1):
    """
    Compute the KL divergence between pi_theta and pi_ref for the first attempt.
    """
    # Placeholder: Implement KL divergence computation between current model and reference model
    # This requires access to the reference model's outputs
    # For simplicity, assume pi_ref is the base model before any fine-tuning
    # You need to have a reference model loaded separately
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_score_multiturn.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config)
