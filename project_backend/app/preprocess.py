# from turtle import pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
import pandas as pd

le_transaction_type = LabelEncoder().fit(["Withdrawal", "Transfer", "Cash Depos"])
le_recipient_bank = LabelEncoder().fit(["ICICI Bank", "Bank of America"])
le_recipient_country = LabelEncoder().fit(["India"])
le_description = LabelEncoder().fit(["Rent Payment", "Payment"])

def preprocess_data(data_dict):
    features = {
        "Amount_INR": float(data_dict["Amount_INR"]),
        "Transaction_Type": le_transaction_type.transform([data_dict["Transaction_Type"]])[0] if data_dict["Transaction_Type"] in le_transaction_type.classes_ else -1,
        "Recipient_Account": float(data_dict["Recipient_Account"]) if data_dict["Recipient_Account"] != "N/A" and data_dict["Recipient_Account"] != "" else 0.0,
        "Recipient_Bank": le_recipient_bank.transform([data_dict["Recipient_Bank"]])[0] if data_dict["Recipient_Bank"] in le_recipient_bank.classes_ else -1,
        "Recipient_Country": le_recipient_country.transform([data_dict["Recipient_Country"]])[0] if data_dict["Recipient_Country"] in le_recipient_country.classes_ else -1,
        "Description": le_description.transform([data_dict["Description"]])[0] if data_dict["Description"] in le_description.classes_ else -1,
        "Cash_Indicator": 1 if data_dict["Cash_Indicator"] == "Yes" else 0,
    }
    dt = datetime.strptime(data_dict["Date_Time"], "%Y-%m-%d %H:%M:%S")
    features["Day"] = dt.day / 31.0
    features["Hour"] = dt.hour / 24.0
    creation_dt = datetime.strptime(data_dict["Account_Creation_Date"], "%Y-%m-%d %H:%M:%S")
    days_since = max((dt - creation_dt).days, 0) / 365.0
    features["Days_Since_Creation"] = days_since

    feature_order = ["Amount_INR", "Transaction_Type", "Recipient_Account", "Recipient_Bank", "Recipient_Country", "Description", "Cash_Indicator", "Day", "Hour", "Days_Since_Creation"]
    X = np.array([[features[feat] for feat in feature_order]], dtype=np.float32)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.T).T
    return X_scaled

def preprocess_data_batch(data_dicts):
    n_samples = len(data_dicts)
    X = np.zeros((n_samples, 10), dtype=np.float32)
    
    def safe_float(x):
        try:
            return float(x) if x != "N/A" and pd.notna(x) else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    for i, data_dict in enumerate(data_dicts):
        X[i, 0] = float(data_dict["Amount_INR"])
        X[i, 1] = le_transaction_type.transform([data_dict["Transaction_Type"]])[0] if data_dict["Transaction_Type"] in le_transaction_type.classes_ else -1
        X[i, 2] = safe_float(data_dict["Recipient_Account"])
        X[i, 3] = le_recipient_bank.transform([data_dict["Recipient_Bank"]])[0] if data_dict["Recipient_Bank"] in le_recipient_bank.classes_ else -1
        X[i, 4] = le_recipient_country.transform([data_dict["Recipient_Country"]])[0] if data_dict["Recipient_Country"] in le_recipient_country.classes_ else -1
        X[i, 5] = le_description.transform([data_dict["Description"]])[0] if data_dict["Description"] in le_description.classes_ else -1
        X[i, 6] = 1 if data_dict["Cash_Indicator"] == "Yes" else 0
        dt = datetime.strptime(data_dict["Date_Time"], "%Y-%m-%d %H:%M:%S")
        X[i, 7] = dt.day / 31.0
        X[i, 8] = dt.hour / 24.0
        creation_dt = datetime.strptime(data_dict["Account_Creation_Date"], "%Y-%m-%d %H:%M:%S")
        days_since = max((dt - creation_dt).days, 0) / 365.0
        X[i, 9] = days_since

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    if np.any(np.isnan(X_scaled)):
        print("Warning: NaN values detected in preprocessed X")
    return X_scaled