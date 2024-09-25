import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, model, tokenizer, env, config):
        self.model = model
        self.tokenizer = tokenizer
        self.env = env
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate_stage_II'])
        self.gamma = config['training']['gamma']
        self.epsilon = config['training']['epsilon']
        self.k_epochs = config['training']['k_epochs']
        self.memory = []
        self.device = config['model']['device']
        self.model.to(self.device)

    def select_action(self, state):
        state = torch.tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(state)
            logits = outputs.logits[:, -1, :]  # Take logits of the last token
            probs = torch.softmax(logits, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return action.item()

    def store_transition(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns

    def update(self):
        if not self.memory:
            return  # No updates if memory is empty

        # Separate states, actions, rewards, dones
        states, actions, rewards, dones = zip(*self.memory)
        states = torch.tensor(states, dtype=torch.long).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        # Compute discounted returns
        returns = self.compute_returns(rewards, dones)

        # Compute old policy logits
        with torch.no_grad():
            old_logits = []
            for state in states:
                state = state.unsqueeze(0)
                outputs = self.model(state)
                logits = outputs.logits[:, -1, :]  # Last token
                old_logits.append(logits)
            old_logits = torch.stack(old_logits).squeeze(1)  # Shape: (batch_size, vocab_size)
            old_probs = torch.softmax(old_logits, dim=-1)
            old_distributions = Categorical(old_probs)
            old_log_probs = old_distributions.log_prob(actions)

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            logits = []
            for state in states:
                state = state.unsqueeze(0)
                outputs = self.model(state)
                logit = outputs.logits[:, -1, :]  # Last token
                logits.append(logit)
            logits = torch.stack(logits).squeeze(1)  # Shape: (batch_size, vocab_size)
            probs = torch.softmax(logits, dim=-1)
            distributions = Categorical(probs)
            log_probs = distributions.log_prob(actions)

            # Calculate ratios
            ratios = torch.exp(log_probs - old_log_probs)

            # Calculate surrogate losses
            surr1 = ratios * returns
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * returns

            # Calculate loss
            loss = -torch.min(surr1, surr2).mean()

            # Backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear memory after update
        self.memory = []
