import pandas as pd

def validate_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    print("Dataset Overview:")
    print(df.head())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSample Correctness Check:")
    for idx, row in df.iterrows():
        if not row['y2']:
            print(f"Missing y2 for question ID: {row['question_id']}")
    print("\nValidation Completed.")

if __name__ == "__main__":
    validate_dataset('synthetic_sat_dataset.jsonl')  # Adjust path as needed
