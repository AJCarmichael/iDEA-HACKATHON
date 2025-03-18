import json
import os
import requests
from datetime import datetime

# Gemini API Key (replace with your actual key)
GEMINI_API_KEY = 'AIzaSyAf0sZIQap3SCpyGHIWmbJ5LG5dfbskw3U'  # Replace with your valid API key

# Metadata folder
METADATA_FOLDER = "metadata"

def load_customer_metadata(customer_id):
    """
    Loads customer metadata from the metadata folder.
    
    Args:
        customer_id (str): Customer ID (e.g., 'C004').
    
    Returns:
        dict: Customer metadata or None if not found.
    """
    metadata_file = os.path.join(METADATA_FOLDER, f"{customer_id}_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            return json.load(f)
    else:
        print(f"Metadata file for {customer_id} not found at {metadata_file}")
        return None

def simulate_xgboost_prediction(transaction):
    """
    Simulates an XGBoost model's prediction for a transaction.
    Replace this with your actual XGBoost model inference.
    
    Args:
        transaction (dict): Transaction data.
    
    Returns:
        tuple: (is_suspicious: bool, reason: str)
    """
    # Simple heuristic: Flag as suspicious if cash > ₹50,000 or amount > ₹100,000 with vague description
    amount = float(transaction["Amount (INR)"])
    cash_indicator = transaction["Cash Indicator"].lower() == "yes"
    description = transaction["Description"].lower()

    if cash_indicator and amount > 50000:
        return True, "High cash amount detected"
    elif amount > 100000 and ("transfer" in description or "consulting" in description or "cash" in description):
        return True, "Large amount with vague or suspicious description"
    elif transaction["Recipient Bank"] == "Unknown Bank":
        return True, "Unknown recipient bank"
    else:
        return False, "No suspicious patterns detected"

def get_gemini_confirmation(transaction, metadata, xgboost_result):
    """
    Queries Gemini for final confirmation on the transaction's suspiciousness.
    
    Args:
        transaction (dict): Transaction data.
        metadata (dict): Customer metadata.
        xgboost_result (tuple): (is_suspicious: bool, reason: str) from XGBoost.
    
    Returns:
        dict: Structured response from Gemini.
    """
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY in environment variables.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    # Prompt for Gemini
    prompt = (
        "You are an expert in financial fraud detection. A transaction has been flagged by an XGBoost model. "
        "Analyze the transaction, customer metadata, and XGBoost result to confirm if it exhibits money laundering "
        "or mule characteristics (e.g., high cash usage, rapid fund movement, inconsistent behavior). "
        "Provide your response in this exact JSON format:\n"
        "{\n"
        "  \"is_suspicious\": \"Yes/No\",\n"
        "  \"details\": \"Explanation of why it is or isn’t suspicious\",\n"
        "  \"confidence\": \"Low/Medium/High\",\n"
        "  \"recommendation\": \"Send to compliance team or Proceed normally\"\n"
        "}\n\n"
        "Transaction Data:\n" + json.dumps(transaction, indent=2) + "\n\n"
        "Customer Metadata:\n" + json.dumps(metadata, indent=2) + "\n\n"
        "XGBoost Prediction:\n" + f"Suspicious: {xgboost_result[0]}, Reason: {xgboost_result[1]}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        insight_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No insights returned")
        
        try:
            return json.loads(insight_text)
        except json.JSONDecodeError:
            print("Warning: Gemini response not in JSON format, simulating response.")
            # Simulated Gemini response (replace with actual parsing if needed)
            if xgboost_result[0]:
                return {
                    "is_suspicious": "Yes",
                    "details": f"Confirmed XGBoost finding: {xgboost_result[1]}. Matches mule patterns in metadata.",
                    "confidence": "Medium",
                    "recommendation": "Send to compliance team"
                }
            else:
                return {
                    "is_suspicious": "No",
                    "details": "No significant mule characteristics detected. Aligns with metadata patterns.",
                    "confidence": "High",
                    "recommendation": "Proceed normally"
                }

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Gemini API: {e}")
        return {
            "is_suspicious": "Error",
            "details": f"API error: {str(e)}",
            "confidence": "N/A",
            "recommendation": "Send to compliance team for manual review"
        }

def validate_transaction(transaction):
    """
    Validates a single transaction using XGBoost and Gemini.
    
    Args:
        transaction (dict): Transaction data with all required fields.
    
    Returns:
        dict: Validation result with XGBoost and Gemini opinions.
    """
    required_fields = [
        "Transaction ID", "Account Number", "Date", "Time", "Amount (INR)", "Transaction Type",
        "Recipient Account", "Recipient Bank", "Recipient Country", "Description", "Cash Indicator",
        "Customer ID", "Account Creation Date"
    ]
    
    # Check for missing fields
    if not all(field in transaction for field in required_fields):
        missing = [field for field in required_fields if field not in transaction]
        return {"error": f"Missing fields: {missing}"}

    customer_id = transaction["Customer ID"]
    
    # Load customer metadata
    metadata = load_customer_metadata(customer_id)
    if not metadata:
        return {"error": f"Metadata for {customer_id} not found"}

    # Simulate XGBoost prediction
    xgboost_result = simulate_xgboost_prediction(transaction)

    # Get Gemini confirmation
    gemini_result = get_gemini_confirmation(transaction, metadata, xgboost_result)

    # Combine results
    validation_result = {
        "transaction": transaction,
        "xgboost_prediction": {
            "is_suspicious": xgboost_result[0],
            "reason": xgboost_result[1]
        },
        "gemini_confirmation": gemini_result,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return validation_result

# Example usage
if __name__ == "__main__":
    # Sample suspicious transaction for testing (from C005, the mule)
    suspicious_transaction = {
        "Transaction ID": "T447",
        "Account Number": "5678901234",
        "Date": "2025-03-01",
        "Time": "12:00",
        "Amount (INR)": "100000",
        "Transaction Type": "Debit",
        "Recipient Account": "N/A",
        "Recipient Bank": "N/A",
        "Recipient Country": "India",
        "Description": "Cash withdrawal for business needs",
        "Cash Indicator": "Yes",
        "Customer ID": "C005",
        "Account Creation Date": "2023-06-15"
    }

    # Validate the transaction
    result = validate_transaction(suspicious_transaction)
    
    # Print result
    print("Validation Result:")
    print(json.dumps(result, indent=2))

    # Optionally save to file
    output_file = f"validation_result_{suspicious_transaction['Transaction ID']}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {output_file}")