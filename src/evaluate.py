# src/evaluate.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import reward_function, compute_entropy
import yaml
import argparse
import pandas as pd

def evaluate_model_multiturn(model, tokenizer, prompts, ground_truths, max_turns=3, scheduler=None):
    model.eval()
    correct_first_attempt = 0
    correct_second_attempt = 0
    correct_third_attempt = 0
    entropy_scores = []
    temperature_history = []

    for prompt, gt in zip(prompts, ground_truths):
        history = [prompt]
        current_temperature = scheduler.temperature if scheduler else 1.0

        # Turn 1
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        outputs1 = model.generate(
            inputs,
            max_length=50,
            do_sample=True,
            temperature=current_temperature,
            top_k=50,
            top_p=0.95
        )
        y1 = tokenizer.decode(outputs1[0][len(inputs[0]):], skip_special_tokens=True)
        history.append(y1)
        if y1.strip().lower() == gt.strip().lower():
            correct_first_attempt += 1

        # Turn 2: Self-Correction
        correction_instruction = " There might be an error in the solution above. Please correct it."
        combined_prompt = prompt + correction_instruction
        inputs2 = tokenizer.encode(combined_prompt, return_tensors='pt').to(model.device)
        outputs2 = model.generate(
            inputs2,
            max_length=50,
            do_sample=True,
            temperature=current_temperature,
            top_k=50,
            top_p=0.95
        )
        y2 = tokenizer.decode(outputs2[0][len(inputs2[0]):], skip_special_tokens=True)
        history.append(y2)
        if y2.strip().lower() == gt.strip().lower():
            correct_second_attempt += 1

        # Compute entropy and update temperature
        entropy_y2 = compute_entropy(outputs2, tokenizer)
        entropy_scores.append(entropy_y2)

        if scheduler:
            success = 1.0 if y2.strip().lower() == gt.strip().lower() else 0.0
            scheduler.update_temperature(success)
            current_temperature = scheduler.temperature
            temperature_history.append(current_temperature)

        # Turn 3: Additional Self-Correction (if applicable)
        if max_turns >= 3:
            combined_prompt3 = y2 + " Please review your previous answer for any possible errors."
            inputs3 = tokenizer.encode(combined_prompt3, return_tensors='pt').to(model.device)
            outputs3 = model.generate(
                inputs3,
                max_length=50,
                do_sample=True,
                temperature=current_temperature,
                top_k=50,
                top_p=0.95
            )
            y3 = tokenizer.decode(outputs3[0][len(inputs3[0]):], skip_special_tokens=True)
            history.append(y3)
            if y3.strip().lower() == gt.strip().lower():
                correct_third_attempt += 1

            # Compute entropy for y3
            entropy_y3 = compute_entropy(outputs3, tokenizer)
            entropy_scores.append(entropy_y3)

            if scheduler:
                success = 1.0 if y3.strip().lower() == gt.strip().lower() else 0.0
                scheduler.update_temperature(success)
                current_temperature = scheduler.temperature
                temperature_history.append(current_temperature)

    accuracy_t1 = correct_first_attempt / len(prompts) * 100
    accuracy_t2 = correct_second_attempt / len(prompts) * 100
    if max_turns >=3:
        accuracy_t3 = correct_third_attempt / len(prompts) * 100
    delta_t1_t2 = accuracy_t2 - accuracy_t1
    if max_turns >=3:
        delta_t2_t3 = accuracy_t3 - accuracy_t2
    avg_entropy = sum(entropy_scores) / len(entropy_scores)
    avg_temperature = sum(temperature_history) / len(temperature_history) if temperature_history else 1.0

    print(f"Accuracy@t1: {accuracy_t1:.2f}%")
    print(f"Accuracy@t2: {accuracy_t2:.2f}%")
    if max_turns >=3:
        print(f"Accuracy@t3: {accuracy_t3:.2f}%")
    print(f"Delta(t1, t2): {delta_t1_t2:.2f}%")
    if max_turns >=3:
        print(f"Delta(t2, t3): {delta_t2_t3:.2f}%")
    print(f"Average Entropy: {avg_entropy:.4f}")
    print(f"Average Temperature: {avg_temperature:.2f}")

def main(config):
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['name'])
    model = GPT2LMHeadModel.from_pretrained(config['model']['name']).to(config['device'])
    model.load_state_dict(torch.load(config['evaluation']['model_path'], map_location=config['device']))
    model.eval()

    # Load evaluation dataset
    eval_df = pd.read_csv(config['data']['eval_path'])
    prompts = eval_df['prompt'].tolist()
    ground_truths = eval_df['ground_truth'].tolist()

    # Initialize scheduler if needed
    from scheduler import InverseScheduler
    scheduler_params = {
        "initial_temperature": config['scheduler']['initial_temperature'],
        "min_temperature": config['scheduler']['min_temperature'],
        "max_temperature": config['scheduler']['max_temperature'],
        "adjustment_factor": config['scheduler']['adjustment_factor']
    }
    scheduler = InverseScheduler(**scheduler_params)

    # Evaluate
    evaluate_model_multiturn(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        ground_truths=ground_truths,
        max_turns=config['evaluation']['max_turns'],
        scheduler=scheduler
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_evaluate.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config)
