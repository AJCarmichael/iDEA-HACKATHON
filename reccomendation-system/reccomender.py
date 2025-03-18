import os
import pandas as pd
import google.generativeai as genai
from flask import Flask, request, jsonify
import re
import google.api_core.exceptions
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load Gemini API Key
GEMINI_API_KEY = "AIzaSyDVJdRye4ECAFhpd2Lib7rnv-B-tRl5BPw"
genai.configure(api_key=GEMINI_API_KEY)

# Path to the CSV containing available services.
SERVICES_CSV_PATH = "services.csv"

def load_services():
    """Load Union Bank's available services from CSV."""
    if not os.path.exists(SERVICES_CSV_PATH):
        return None, {"error": f"File '{SERVICES_CSV_PATH}' not found."}
    try:
        return pd.read_csv(SERVICES_CSV_PATH), None
    except Exception as e:
        return None, {"error": f"Error reading '{SERVICES_CSV_PATH}': {e}"}

def extract_transaction_insights(user_df):
    """Extract useful transaction details for LLM processing."""
    insights = {
        "age": int(user_df["age"].mode()[0]),  # Most common age
        "occupation": user_df["occupation"].mode()[0],  # Most common occupation
        "top_categories": list(user_df["rmt_inf_ustrd1"].value_counts().keys())[:5],  # Top 5 spending categories
        "top_merchants": list(user_df["ctpty_nm"].value_counts().keys())[:5],  # Top 5 merchants
        "average_balance": user_df["bal_aftr"].mean(),
    }
    return insights

def query_gemini(insights, services_df, custom_prompt=None):
    """Use Gemini API to generate personalized recommendations."""
    base_prompt = f"""
    You are an AI financial advisor for Union Bank. Based on the user's data below, return ONLY the service numbers that best match the user's profile.
    
    User Details:
    - Age: {insights['age']}
    - Occupation: {insights['occupation']}
    - Top Spending Categories: {', '.join(insights['top_categories'])}
    - Frequent Merchants: {', '.join(insights['top_merchants'])}
    - Average Account Balance: {insights['average_balance']:.2f} CAD
    
    Union Bank Services:
    {services_df.to_string(index=False)}

    Respond ONLY with a comma-separated list of service numbers, without any explanations. Example response format: "1, 5, 9".
    """
    if custom_prompt:
        custom_prompt = f"{base_prompt}\n\n{custom_prompt}"
    else:
        custom_prompt = base_prompt

    try:
        response = genai.GenerativeModel("gemini-2.0-flash-thinking-exp").generate_content(custom_prompt)
        if response and response.text:
            # Extract only numbers using regex.
            service_numbers = re.findall(r'\d+', response.text)
            return service_numbers
    except google.api_core.exceptions.NotFound as e:
        print(f"Model not found: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    return []

@app.route('/process', methods=['POST'])
def process_transaction():
    """Process uploaded CSV file and return only service numbers."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        # Directly convert the uploaded CSV to a Pandas DataFrame.
        user_df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error reading file: {e}"}), 400

    services_df, error = load_services()
    if error:
        return jsonify(error), 500

    insights = extract_transaction_insights(user_df)
    custom_prompt = request.form.get('custom_prompt')
    recommended_service_numbers = query_gemini(insights, services_df, custom_prompt)
    
    recommended_service_names = services_df[services_df['service_id'].isin(map(int, recommended_service_numbers))]['service_name'].tolist()
    
    return jsonify({"recommended_services": recommended_service_numbers, "service_names": recommended_service_names}), 200

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
