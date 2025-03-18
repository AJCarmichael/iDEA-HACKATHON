import requests
import json
import os

# Load Gemini API Key from environment variable
GEMINI_API_KEY = 'AIzaSyAf0sZIQap3SCpyGHIWmbJ5LG5dfbskw3U'

def get_gemini_insights(customer_metadata):
    """
    Sends customer metadata to Gemini API and returns the AI-generated insights.
    
    Args:
        customer_metadata (dict): The existing customer metadata.
    
    Returns:
        str: AI-generated customer profile summary or error message.
    """
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY in environment variables.")

    # Correct API URL with proper model name (assumed model: gemini-1.5-flash or similar)
    # Replace 'gemini-1.5-flash' with the exact model name if different
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    # Structure the payload according to Gemini API requirements
    prompt = (
        "Based on this transaction metadata, infer the customer's financial profile, behavior, "
        "and potential fraud risk:\n" + json.dumps(customer_metadata, indent=2)
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
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        
        # Parse the response (adjust based on actual API response structure)
        response_data = response.json()
        # Assuming the text is in candidates[0].content.parts[0].text
        insight_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No insights returned")
        return insight_text.strip()

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Gemini API: {e}")
        print(f"Response: {response.text if 'response' in locals() else 'No response'}")
        return "Error retrieving insights"