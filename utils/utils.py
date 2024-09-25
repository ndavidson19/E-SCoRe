import re

def extract_final_answer(self, response):
    """
    Extracts the final numerical answer from the response using regex.

    Args:
        response (str): The generated response with CoT.

    Returns:
        str: The extracted numerical answer.
    """
    # Attempt to find the last number in the response
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", response)
    return matches[-1] if matches else ""
