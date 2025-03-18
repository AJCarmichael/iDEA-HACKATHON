import pandas as pd
import json

df = pd.read_csv("transactions.csv")
df.columns = ["Transaction_ID", "Account_Number", "Date_Time", "Amount_INR", "Transaction_Type", 
              "Recipient_Account", "Recipient_Bank", "Recipient_Country", "Description", 
              "Cash_Indicator", "Customer_ID", "Account_Creation_Date", "Is_Fraud"]
transactions_json = df.drop(columns=["Is_Fraud"]).to_dict(orient="records")
with open("transactions_full.json", "w") as f:
    json.dump(transactions_json, f, indent=2)