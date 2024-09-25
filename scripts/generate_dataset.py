from data.dataset_generator import generate_synthetic_data, save_dataset, load_base_model
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic SAT Dataset with y1 and y2")
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--output_file', type=str, default='synthetic_sat_dataset.jsonl', help='Output JSONL file name')
    parser.add_argument('--model_name', type=str, default='gpt2-medium', help='Base model name')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--max_length', type=int, default=150, help='Maximum length for generated responses')
    
    args = parser.parse_args()
    
    # Load the base model
    model, tokenizer = load_base_model(model_name=args.model_name, device=args.device)
    
    # Generate synthetic data
    data = generate_synthetic_data(
        num_samples=args.num_samples,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_length=args.max_length
    )
    
    # Save the dataset
    save_dataset(data, args.output_file)
    
    print(f"Generated and saved {len(data)} synthetic SAT data samples to {args.output_file}.")

if __name__ == "__main__":
    main()
