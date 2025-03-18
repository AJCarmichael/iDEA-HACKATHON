import json
import os
import datetime
from gemini_api import get_gemini_insights  # Import Gemini function

# Define the metadata folder
METADATA_DIR = "metadata"

# Ensure the metadata directory exists
if not os.path.exists(METADATA_DIR):
    print("Metadata folder not found. Please ensure you have customer metadata.")
    exit()

# Process each metadata file
for filename in os.listdir(METADATA_DIR):
    if filename.endswith("_metadata.json"):  # Ensure we only process metadata files
        file_path = os.path.join(METADATA_DIR, filename)

        try:
            # Load existing metadata
            with open(file_path, "r") as f:
                customer_metadata = json.load(f)

            # Get AI-generated insights
            gemini_response = get_gemini_insights(customer_metadata)

            # Append response with timestamp
            timestamp = datetime.datetime.now().isoformat()
            customer_metadata[timestamp] = {"Gemini Insights": gemini_response}

            # Save updated metadata
            with open(file_path, "w") as f:
                json.dump(customer_metadata, f, indent=4)

            print(f"Updated metadata with Gemini insights for {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
