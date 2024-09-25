import torch.nn.functional as F
import torch

class RewardFunction:
    def __init__(self, 
                 lambda_correct=10.0, 
                 lambda_incorrect=-5.0, 
                 lambda_efficiency=2.0, 
                 lambda_entropy=0.1, 
                 lambda_stability=-1.0,
                 model=None, 
                 tokenizer=None):
        """
        Initializes the RewardFunction.

        Args:
            lambda_correct (float): Reward for correct answer.
            lambda_incorrect (float): Penalty for incorrect answer.
            lambda_efficiency (float): Reward for reaching correct answer in fewer turns.
            lambda_entropy (float): Base weight for entropy bonus.
            lambda_stability (float): Penalty for unnecessary changes after correct answer.
            model (transformers.PreTrainedModel): The LLM model.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        """
        self.lambda_correct = lambda_correct
        self.lambda_incorrect = lambda_incorrect
        self.lambda_efficiency = lambda_efficiency
        self.base_lambda_entropy = lambda_entropy
        self.lambda_stability = lambda_stability
        self.model = model
        self.tokenizer = tokenizer

    def compute_exact_match(self, y, y_star):
        """
        Checks if the generated answer matches the ground truth exactly.

        Args:
            y (str): Generated answer.
            y_star (str): Ground truth answer.

        Returns:
            float: 1.0 if exact match, 0.0 otherwise.
        """
        return 1.0 if y.strip() == y_star.strip() else 0.0

    def get_embedding(self, text):
        """
        Generates the embedding for a given text using the LLM's hidden states.

        Args:
            text (str): The text to embed.

        Returns:
            torch.Tensor: The embedding vector.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer.encode(text, return_tensors='pt').to(self.model.device)
            outputs = self.model(inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of hidden states from all layers
            # Choose the last hidden state from the final layer
            last_hidden_state = hidden_states[-1]  # Shape: (1, sequence_length, hidden_size)
            # Aggregate token embeddings (e.g., mean pooling)
            embedding = torch.mean(last_hidden_state, dim=1).squeeze()  # Shape: (hidden_size,)
        return embedding.cpu()

    def compute_entropy(self, logits):
        """
        Computes entropy from model's logits.

        Args:
            logits (torch.Tensor): Logits from the model.

        Returns:
            float: Entropy value.
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        return entropy.mean().item()  # Average entropy over the batch

    def get_reward(self, current_response, y_star, logits_current, turn_number):
        """
        Computes the reward based on the current response.

        Args:
            current_response (str): Current generated answer.
            y_star (str): Ground truth answer.
            logits_current (torch.Tensor): Logits from the current response generation.
            turn_number (int): Current turn number.

        Returns:
            float: Total reward.
            dict: Breakdown of reward components.
        """
        em_current = self.compute_exact_match(current_response, y_star)

        # Dynamically adjust lambda_entropy based on turn_number
        # For example, linearly increase entropy weight with each turn
        dynamic_lambda_entropy = self.base_lambda_entropy * turn_number

        entropy_bonus = dynamic_lambda_entropy * self.compute_entropy(logits_current)

        if em_current:
            # Reward for correct answer
            R_correct = self.lambda_correct
            # Additional reward for efficiency (fewer turns)
            R_efficiency = self.lambda_efficiency / turn_number
            # Stability reward to prevent changes after correct answer
            R_stability = self.lambda_stability
            # Entropy bonus
            R_entropy = entropy_bonus
            # Total reward
            R_total = R_correct + R_efficiency + R_stability + R_entropy
            # No penalty
            R_incorrect = 0.0
        else:
            # Penalty for incorrect answer
            R_correct = 0.0
            R_efficiency = 0.0
            R_stability = 0.0
            R_incorrect = self.lambda_incorrect
            # Entropy bonus to encourage exploration
            R_entropy = entropy_bonus
            # Total reward
            R_total = R_incorrect + R_entropy

        return R_total, {
            "R_correct": R_correct,
            "R_incorrect": R_incorrect,
            "R_efficiency": R_efficiency,
            "R_stability": R_stability,
            "R_entropy": R_entropy
        }
