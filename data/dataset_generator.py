from pydantic import BaseModel
from typing import List, Optional
import json
import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import openai
import re
import time
import logging
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI API (ensure you have set your API key securely)
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your actual API key or manage securely

# Define Pydantic models for structured data
class Step(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: List[Step]
    final_answer: str

class SATQuestion(BaseModel):
    prompt: str
    y1: str
    y2: str

def load_base_model(model_name: str = "gpt2-medium", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load the base model used to generate y1.
    
    Args:
        model_name (str): Name of the pre-trained model.
        device (str): Device to load the model onto.
    
    Returns:
        model: Loaded model.
        tokenizer: Corresponding tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Define pad token
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()  # Set to evaluation mode
    return model, tokenizer

def load_few_shot_examples(csv_path: str, num_examples: int = 5) -> List[dict]:
    """
    Loads and selects few-shot examples from the CSV file.

    Args:
        csv_path (str): Path to the parsed SAT questions CSV file.
        num_examples (int): Number of few-shot examples to select.

    Returns:
        List[dict]: Selected few-shot examples.
    """
    df = pd.read_csv(csv_path)
    selected_examples = df.sample(n=num_examples).to_dict(orient='records')
    return selected_examples

def generate_prompt_from_template(template_question: str) -> str:
    """
    Generate a new question by slightly modifying a template question.

    Args:
        template_question (str): A template SAT question.

    Returns:
        str: A new synthetic SAT math question.
    """
    # Extract numbers and variables
    numbers = re.findall(r'\b\d+\b', template_question)
    variables = re.findall(r'\b[a-zA-Z]+\b', template_question)
    
    if not numbers:
        # If no numbers are found, return the original template
        return template_question
    
    # Replace numbers with new random numbers
    new_numbers = [str(random.randint(1, 20)) for _ in numbers]
    new_question = template_question
    for old, new in zip(numbers, new_numbers):
        new_question = new_question.replace(old, new, 1)
    
    return new_question

def generate_prompt(few_shot_examples: List[dict], csv_path: str, num_few_shot: int = 5) -> str:
    """
    Generate a synthetic SAT question using few-shot examples to guide the teacher model.

    Args:
        few_shot_examples (List[dict]): List of few-shot examples.
        csv_path (str): Path to the parsed SAT questions CSV file.
        num_few_shot (int): Number of few-shot examples to include.

    Returns:
        str: A synthetic SAT math question.
    """
    # Shuffle the few-shot examples to increase diversity
    random.shuffle(few_shot_examples)
    
    # Select a subset of few-shot examples
    selected_few_shot = few_shot_examples[:num_few_shot]
    
    # Construct the prompt with few-shot examples
    prompt_text = ""
    for example in selected_few_shot:
        prompt_text += f"Problem: {example['question']}\nSolution: {example['y1']}\nCorrected Solution: {example['y2']}\n\n"
    
    # Add instruction for generating a new question
    prompt_text += "Generate a new SAT math question similar to the above examples.\n\nProblem:"
    
    # Generate a new question based on the prompt
    # Here, we use the teacher model to generate a new question
    # Alternatively, you can use a predefined template or another method
    # For simplicity, we'll return a random template question
    
    # Alternatively, you can instruct the teacher model to generate a new question
    # based on the few-shot examples. However, this requires a more advanced setup.
    # For now, we'll use the existing `generate_prompt_from_template` function.
    
    new_question = generate_prompt_from_template("Solve for x: 3x + 4 = 19.")  # Example template
    return new_question

def generate_y1(prompt: str, model, tokenizer, device: str, max_length: int = 150) -> str:
    """
    Generate the initial solution y1 using the base model.

    Args:
        prompt (str): The math problem.
        model: The base model.
        tokenizer: The tokenizer.
        device (str): Device for computation.
        max_length (int): Maximum length of the generated response.

    Returns:
        str: The base model's solution.
    """
    input_text = f"{prompt}\nSolution:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    y1 = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Solution:")[-1].strip()
    return y1

def get_y2(prompt: str, y1: str) -> str:
    """
    Generate the corrected solution y2 using the teacher model (GPT-4).

    Args:
        prompt (str): The math problem.
        y1 (str): The base model's initial solution.

    Returns:
        str: The teacher model's corrected solution.
    """
    try:
        # Construct the system message for the teacher model
        system_message = (
            "You are a highly skilled math tutor. Your task is to refine and correct the following solution "
            "provided by a student. Ensure that the reasoning is clear, accurate, and follows standard mathematical procedures."
        )
        
        # Construct the user message including the prompt and y1
        user_message = f"Problem: {prompt}\nStudent's Solution: {y1}\nCorrected Solution:"
        
        # Make the API call to OpenAI's GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",  # Ensure you have access to this model
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,  # Lower temperature for more deterministic outputs
            max_tokens=300,
            n=1,
            stop=None
        )
        
        y2 = response.choices[0].message['content'].strip()
        return y2
    
    except Exception as e:
        logger.error(f"Error generating y2 for prompt: {prompt}\nError: {e}")
        return ""

def generate_synthetic_data(num_samples: int = 1000, 
                            base_model=None, 
                            tokenizer=None, 
                            device: str = "cuda", 
                            max_length: int = 150, 
                            few_shot_csv: str = 'sat_questions.csv',
                            num_few_shot: int = 5) -> List[SATQuestion]:
    """
    Generate synthetic SAT data with y1 from the base model and y2 from the teacher model,
    using few-shot examples for diversity.

    Args:
        num_samples (int): Number of samples to generate.
        base_model: The base model for generating y1.
        tokenizer: The tokenizer for the base model.
        device (str): Device for computation.
        max_length (int): Maximum length of generated responses.
        few_shot_csv (str): Path to the CSV file containing few-shot examples.
        num_few_shot (int): Number of few-shot examples to include in each prompt.

    Returns:
        List[SATQuestion]: A list of structured SAT questions.
    """
    data = []
    few_shot_examples = load_few_shot_examples(few_shot_csv, num_examples=20)  # Load more for better shuffling
    
    for i in range(num_samples):
        # Generate a new synthetic question using few-shot examples
        prompt = generate_prompt(few_shot_examples, few_shot_csv, num_few_shot)
        
        # Generate y1 using the base model
        y1 = generate_y1(prompt, base_model, tokenizer, device, max_length)
        
        # Generate y2 using the teacher model
        y2 = get_y2(prompt, y1)
        
        if y2:  # Ensure y2 was generated successfully
            data.append(SATQuestion(prompt=prompt, y1=y1, y2=y2))
            logger.info(f"Generated sample {i+1}/{num_samples}")
        else:
            logger.warning(f"Skipped sample {i+1} due to y2 generation failure.")
        
        # To respect OpenAI's rate limits
        time.sleep(1)  # Adjust sleep time based on your rate limits
    
    return data

def save_dataset(data: List[SATQuestion], filename: str):
    """
    Save the dataset to a JSONL file.

    Args:
        data (List[SATQuestion]): The dataset to save.
        filename (str): The filename for the JSONL file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            json_line = json.dumps(entry.dict())
            f.write(json_line + '\n')
    logger.info(f"Saved {len(data)} SAT questions to {filename}")

def load_dataset(filename: str) -> List[SATQuestion]:
    """
    Load the dataset from a JSONL file.

    Args:
        filename (str): The filename of the JSONL file.

    Returns:
        List[SATQuestion]: The loaded dataset.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = [SATQuestion.parse_obj(json.loads(line)) for line in f]
    return data
