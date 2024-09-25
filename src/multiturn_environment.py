import gym
from gym import spaces
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import RewardFunction
from scheduler import InverseScheduler
import numpy as np
import logging
import re

class MultiTurnSelfCorrectionEnvWithScheduler(gym.Env):
    def __init__(self, 
                 model, 
                 tokenizer, 
                 dataset, 
                 reward_function_params, 
                 alpha=2.0, 
                 entropy_weight=0.01, 
                 max_length=512, 
                 max_turns=3,
                 scheduler_params=None):
        super(MultiTurnSelfCorrectionEnvWithScheduler, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.reward_function = RewardFunction(
            lambda_correct=reward_function_params.get('lambda_correct', 10.0),
            lambda_incorrect=reward_function_params.get('lambda_incorrect', -5.0),
            lambda_efficiency=reward_function_params.get('lambda_efficiency', 2.0),
            lambda_entropy=reward_function_params.get('lambda_entropy', entropy_weight),  
            lambda_stability=reward_function_params.get('lambda_stability', -1.0),
            model=self.model,
            tokenizer=self.tokenizer
        )
        self.entropy_weight = entropy_weight  # Used to scale lambda_entropy
        self.max_length = max_length
        self.max_turns = max_turns
        self.current_turn = 0
        self.current_index = 0
        self.done = False
        self.episode_rewards = []
        self.episode_success = []
        self.turns_per_episode = []
        self.entropy_scores = []
        
        # Initialize the scheduler
        if scheduler_params is None:
            scheduler_params = {}
        self.scheduler = InverseScheduler(**scheduler_params)

        # Define action and observation space
        self.action_space = spaces.Discrete(self.tokenizer.vocab_size)
        self.observation_space = spaces.Box(low=0, high=self.tokenizer.vocab_size, 
                                            shape=(max_length,), dtype=int)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def reset(self):
        self.current_turn = 0
        self.done = False
        self.current_index = np.random.randint(0, len(self.dataset))
        prompt = self.dataset[self.current_index]["prompt"]
        self.ground_truth = self.dataset[self.current_index]["y2"]  # Ground truth is y2
        self.history = [prompt]
        obs = self.tokenizer.encode(prompt, return_tensors='pt').squeeze().tolist()
        if len(obs) < self.max_length:
            obs += [self.tokenizer.pad_token_id] * (self.max_length - len(obs))
        else:
            obs = obs[:self.max_length]
        return np.array(obs)

    def step(self, action):
        if self.done:
            raise ValueError("Episode has ended. Call reset() to start a new episode.")

        # Decode action to token and append to history
        generated_token = self.tokenizer.decode([action]).strip()
        current_prompt = self.history[-1] + generated_token
        self.history.append(generated_token)

        # Generate response with CoT
        cot_prompt = f"{current_prompt}\nSolution:"
        inputs = self.tokenizer.encode(cot_prompt, return_tensors='pt').to(self.model.device)
        outputs = self.model.generate(
            inputs,
            max_length=self.max_length,
            do_sample=True,
            temperature=self.scheduler.temperature,  # Dynamic temperature
            top_k=50,
            top_p=0.95,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True,  # To retrieve logits
            return_dict_in_generate=True
        )
        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip()
        logits = outputs.scores[-1]  # Logits of the last generated token
        self.history.append(response)

        # Extract the final numerical answer from the response
        final_answer = self.extract_final_answer(response)

        # Compute rewards
        turn_number = self.current_turn + 1  # 1-based indexing
        reward, reward_details = self.reward_function.get_reward(
            current_response=final_answer, 
            y_star=self.ground_truth, 
            logits_current=logits,  # Pass logits of the last token
            turn_number=turn_number
        )

        # Determine success of self-correction
        success = 1.0 if reward_details["R_correct"] > 0 else 0.0
        if success:
            self.scheduler.decrease_temperature()
        else:
            self.scheduler.increase_temperature()

        # Log the temperature update and rewards
        self.logger.info(f"Turn {turn_number}: Success={success}, Total Reward={reward}, New Temperature={self.scheduler.temperature}")
        self.logger.debug(f"Reward Details: {reward_details}")

        # Update observation
        if success or (self.current_turn + 1 >= self.max_turns):
            self.done = True
            obs = np.zeros(self.max_length, dtype=int)
        else:
            self.current_turn += 1
            # Reset to the original prompt for the next turn
            obs_text = self.history[0]  # Original prompt
            obs = self.tokenizer.encode(obs_text, return_tensors='pt').squeeze().tolist()
            if len(obs) < self.max_length:
                obs += [self.tokenizer.pad_token_id] * (self.max_length - len(obs))
            else:
                obs = obs[:self.max_length]
            obs = np.array(obs)

        # After computing reward and before returning
        self.episode_rewards.append(reward)
        self.turns_per_episode.append(self.current_turn + 1)
        self.entropy_scores.append(reward_details["R_entropy"])
        if success:
            self.episode_success.append(1)
        else:
            self.episode_success.append(0)

        info = {
            "response": response,
            "final_answer": final_answer,
            "reward": reward,
            "reward_details": reward_details,
            "updated_temperature": self.scheduler.temperature,
            "success": success
        }

        return obs, reward, self.done, info

    def extract_final_answer(self, response):
        """
        Extracts the final numerical answer from the response using regex.

        Args:
            response (str): The generated response with CoT.

        Returns:
            str: The extracted numerical answer.
        """
        # Attempt to find all numbers in the response
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        return matches[-1] if matches else ""

    def render(self, mode='human'):
        # Optional: Implement rendering if needed
        pass

    def get_metrics(self):
        """
        Aggregates and returns metrics for the current episode.
        """
        accuracy = sum(self.episode_success) / len(self.episode_success) if self.episode_success else 0.0
        average_turns = sum(self.turns_per_episode) / len(self.turns_per_episode) if self.turns_per_episode else 0.0
        average_entropy = sum(self.entropy_scores) / len(self.entropy_scores) if self.entropy_scores else 0.0
        
        # Reset metrics after retrieval
        self.episode_rewards = []
        self.episode_success = []
        self.turns_per_episode = []
        self.entropy_scores = []
        
        return {
            "accuracy": accuracy,
            "average_turns": average_turns,
            "average_entropy": average_entropy
        }
