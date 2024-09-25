# src/train_stage_II.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
import yaml
import argparse
from utils import RewardFunction
from scheduler import InverseScheduler
from multiturn_environment import MultiTurnSelfCorrectionEnvWithScheduler
from ppo_agent import PPOAgent  # Ensure this is your PPO implementation
from torch.utils.data import Dataset, DataLoader

def main(config):
    # Load tokenizer and model (Stage I initialized model)
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['name'])
    model = GPT2LMHeadModel.from_pretrained(config['model']['name']).to(config['device'])
    model.load_state_dict(torch.load(config['model']['save_path_stage_I']))
    model.train()

    # Load training dataset
    with open(config['data']['training_data_path'], 'r') as f:
        training_data = [json.loads(line) for line in f]
    
    # Initialize reward function
    reward_fn = RewardFunction(
        lambda_correct=config['reward_function']['lambda_correct'],
        lambda_incorrect=config['reward_function']['lambda_incorrect'],
        lambda_efficiency=config['reward_function']['lambda_efficiency'],
        lambda_entropy=config['reward_function']['lambda_entropy'],
        lambda_stability=config['reward_function']['lambda_stability'],
        model=model,
        tokenizer=tokenizer
    )
    
    # Initialize scheduler parameters
    scheduler_params = config['scheduler']
    
    # Initialize the environment
    env = MultiTurnSelfCorrectionEnvWithScheduler(
        model=model,
        tokenizer=tokenizer,
        dataset=training_data,
        reward_function_params=config['reward_function'],
        alpha=config['entropy']['alpha'],
        entropy_weight=config['entropy']['entropy_weight'],
        max_length=config['training']['max_length'],
        max_turns=config['training']['max_turns'],
        scheduler_params=scheduler_params
    )
    
    # Initialize PPO agent
    agent = PPOAgent(model=model, tokenizer=tokenizer, env=env, config=config)
    
    # Stage II: Reinforcement Learning with PPO and Reward Shaping
    print("Starting Stage II: Reinforcement Learning with PPO")
    for epoch in range(config['training']['stage_II_epochs']):
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward, done)
        
        # Update PPO agent after each episode
        agent.update()
        
        # Logging (implement get_metrics appropriately)
        metrics = env.get_metrics()
        print(f"Stage II Epoch {epoch+1}/{config['training']['stage_II_epochs']} - Metrics: {metrics}")
    
    # Save the fine-tuned model
    torch.save(model.state_dict(), config['model']['save_path_stage_II'])
    print("Stage II Training Completed and Model Saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_stage_II.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config)
