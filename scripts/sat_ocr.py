# File: entropy_score/scripts/parse_sat_pdf.py

import pytesseract
from pdf2image import convert_from_path
import re
import pandas as pd
import argparse
import logging
from typing import List

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from the given PDF file using OCR.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text.
    """
    pages = convert_from_path(pdf_path, 300)
    text = ""
    for i, page in enumerate(pages):
        logging.info(f"Extracting text from page {i + 1} / {len(pages)}")
        logging.debug(f"Extracted text: {pytesseract.image_to_string(page)}")
        text += pytesseract.image_to_string(page)
    return text

def parse_sat_questions(text: str) -> List[dict]:
    """
    Parses SAT questions, answers, rationale, and difficulty from the extracted text.

    Args:
        text (str): Extracted text from the PDF.

    Returns:
        List[dict]: List of parsed SAT questions.
    """
    blocks = re.split(r"Question ID [0-9a-z]+", text, flags=re.IGNORECASE)
    blocks = [block.strip() for block in blocks if block.strip()]

    sat_questions = []

    for block in blocks:
        try:
            # Extract question ID
            question_id_match = re.search(r"ID:\s*([a-z0-9]+)", block, re.IGNORECASE)
            question_id = question_id_match.group(1) if question_id_match else None

            # Extract question (everything up to the first choice or "Correct Answer:")
            question_match = re.search(r"ID:.*?\n(.*?)(?=A\.|Correct Answer:)", block, re.DOTALL | re.IGNORECASE)
            question = question_match.group(1).strip() if question_match else None

            # Extract answer choices with their values
            choices_match = re.findall(r"([A-D]\.)(.*?)(?=[A-D]\.|Correct Answer:|$)", block, re.DOTALL)
            choices = [f"{letter.strip()} {value.strip()}" for letter, value in choices_match]

            # Remove newline and question ID from the last choice if present
            if choices and '\n' in choices[-1]:
                choices[-1] = choices[-1].split('\n')[0]

            # Extract correct answer(s)
            correct_answer_match = re.search(r"Correct Answer:\s*([^\n]+)", block, re.IGNORECASE)
            correct_answer = correct_answer_match.group(1).strip() if correct_answer_match else None
            correct_answers = [ans.strip() for ans in correct_answer.split(',')] if correct_answer else []

            # Extract rationale (everything between "Rationale" and "Question Difficulty")
            rationale_match = re.search(r"Rationale(.*?)(?=Question Difficulty:)", block, re.DOTALL | re.IGNORECASE)
            rationale = rationale_match.group(1).strip() if rationale_match else None

            # Extract difficulty
            difficulty_match = re.search(r"Question Difficulty:\s*(\w+)", block, re.IGNORECASE)
            difficulty = difficulty_match.group(1) if difficulty_match else None

            if question_id and question:
                sat_questions.append({
                    "question_id": question_id,
                    "question": question,
                    "choices": choices,
                    "correct_answers": correct_answers,
                    "rationale": rationale,
                    "difficulty": difficulty
                })
            else:
                logger.warning(f"Incomplete data for a question block. Skipping...")
        except Exception as e:
            logger.error(f"Error parsing a question block: {e}")
            continue

    return sat_questions

def save_to_csv(data: List[dict], csv_path: str):
    """
    Saves the parsed SAT questions to a CSV file.

    Args:
        data (List[dict]): List of parsed SAT questions.
        csv_path (str): Path to save the CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} SAT questions to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Parse SAT Questions from PDF to CSV using OCR")
    parser.add_argument('--pdf_path', type=str, required=True, help='Path to the SAT questions PDF file')
    parser.add_argument('--csv_path', type=str, default='parsed_sat_questions.csv', help='Output CSV file path')
    args = parser.parse_args()

    # Extract text from PDF using OCR
    logger.info(f"Extracting text from PDF using OCR: {args.pdf_path}")
    text = extract_text_from_pdf(args.pdf_path)

    # Parse SAT questions
    logger.info("Parsing SAT questions from extracted text")
    sat_questions = parse_sat_questions(text)

    # Save to CSV
    logger.info(f"Saving parsed data to CSV: {args.csv_path}")
    save_to_csv(sat_questions, args.csv_path)

    logger.info("Parsing and saving completed successfully.")

if __name__ == "__main__":
    main()