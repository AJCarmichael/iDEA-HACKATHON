import requests
import json
import os
import pandas as pd
from datetime import datetime

# Load Gemini API Key (replace with your actual key or use environment variable)
GEMINI_API_KEY = 'AIzaSyAf0sZIQap3SCpyGHIWmbJ5LG5dfbskw3U'  # Replace with your valid API key

# Metadata folder
METADATA_FOLDER = "metadata"

def ensure_metadata_folder():
    """Creates the metadata folder if it doesn't exist."""
    if not os.path.exists(METADATA_FOLDER):
        os.makedirs(METADATA_FOLDER)

def load_csv(file_path):
    """
    Loads a CSV file and returns its contents as a string and Customer ID.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        tuple: (CSV data as string, Customer ID)
    """
    try:
        df = pd.read_csv(file_path)
        # Extract Customer ID (assuming it's consistent across rows)
        customer_id = df["Customer ID"].iloc[0]
        # Convert DataFrame to string (first 10 rows for brevity, adjust as needed)
        csv_string = df.head(10).to_csv(index=False)
        return csv_string, customer_id
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None, None

def get_gemini_insights(csv_data):
    """
    Sends CSV data to Gemini API and returns the AI-generated insights for mule characteristics.
    
    Args:
        csv_data (str): Transaction data from the CSV file as a string.
    
    Returns:
        dict: Structured response with mule characteristics analysis.
    """
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY in environment variables.")

    # API URL (assumed model: gemini-1.5-flash, adjust as per actual model name)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    # Prompt to detect mule characteristics in money laundering
    prompt = (
        "You are an expert in financial fraud detection. Analyze the following transaction data from a CSV file "
        "to identify potential mule characteristics in a money laundering scenario. Mules often exhibit patterns "
        "such as high cash usage, frequent large transactions to unrelated accounts, rapid movement of funds, "
        "High Cash Usage: Frequent cash transactions, especially for large amounts, which are unusual for a legitimate business."
        "Rapid Fund Movement: Quick transfers of large sums to multiple unrelated accounts, often shortly after receiving funds."
        "Inconsistent Patterns: Transactions that dont align with typical business operations (e.g., personal expenses mixed with business, irregular high-value payments)."
        "Frequent Large Transactions: Unexplained large debits and credits to various banks, suggesting fund layering or pass-through activity."
        "Suspicious Descriptions: Vague or misleading transaction descriptions (e.g., Consulting fee for large cash withdrawals)."
        "or inconsistent behavior compared to typical profiles (e.g., student or businessman). Provide your analysis "
        "in this exact JSON format:\n"
        "{\n"
        "  \"mule_characteristics_detected\": \"Yes/No\",\n"
        "  \"details\": \"Where and how mule characteristics are showcased, if any\",\n"
        "  \"severity\": \"Low/Medium/High (if applicable)\",\n"
        "  \"send_to_compliance_team\": \"Yes/No\"\n"
        "}\n\n"
        "Transaction Data:\n" + csv_data
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        response_data = response.json()
        insight_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No insights returned")
        
        try:
            result = json.loads(insight_text)
            return result
        except json.JSONDecodeError:
            print("Warning: Gemini response not in JSON format, simulating parsing.")
            return {
                "mule_characteristics_detected": "No",
                "details": insight_text,
                "severity": "N/A",
                "send_to_compliance_team": "No"
            }

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Gemini API: {e}")
        print(f"Response: {response.text if 'response' in locals() else 'No response'}")
        return {
            "mule_characteristics_detected": "Error",
            "details": f"API error: {str(e)}",
            "severity": "N/A",
            "send_to_compliance_team": "No"
        }

def update_existing_metadata(customer_id, inference):
    """
    Updates the existing metadata file with the new inference.
    
    Args:
        customer_id (str): Customer ID (e.g., 'C001').
        inference (dict): AI-generated inference.
    """
    metadata_file = os.path.join(METADATA_FOLDER, f"{customer_id}_metadata.json")
    
    # Load existing metadata if it exists
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    else:
        # If file doesn't exist, create a basic structure
        metadata = {
            "Customer ID": customer_id,
            "Mule Analysis": {}
        }

    # Append inference with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if "Mule Analysis" not in metadata:
        metadata["Mule Analysis"] = {}
    metadata["Mule Analysis"][timestamp] = inference

    # Save updated metadata back to file
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Updated metadata saved to {metadata_file}")

def analyze_csv_for_mule_characteristics(file_path):
    """
    Main function to analyze a CSV file for mule characteristics and update existing metadata.
    
    Args:
        file_path (str): Path to the CSV file.
    """
    # Ensure metadata folder exists
    ensure_metadata_folder()

    # Load CSV data and get Customer ID
    csv_data, customer_id = load_csv(file_path)
    if not csv_data or not customer_id:
        return

    # Get inference from Gemini
    inference = get_gemini_insights(csv_data)
    print("\nGemini Inference:")
    print(json.dumps(inference, indent=2))

    # Update existing metadata file
    update_existing_metadata(customer_id, inference)

# Example usage
if __name__ == "__main__":
    # Replace with the path to one of your CSV files (e.g., student1_transactions_50_updated.csv)
    csv_file_path = "businessMan3.csv"
    
    if os.path.exists(csv_file_path):
        analyze_csv_for_mule_characteristics(csv_file_path)
    else:
        print(f"CSV file {csv_file_path} not found. Please provide a valid file path.")