import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from app.model import train_xgboost, save_model

def load_training_data(file_path="transactions.csv"):
    df = pd.read_csv(file_path)
    print("Available columns:", df.columns.tolist())
    
    # Encode categorical variables
    le_transaction_type = LabelEncoder().fit(df["Transaction Type"])
    le_recipient_bank = LabelEncoder().fit(df["Recipient Bank"])
    le_recipient_country = LabelEncoder().fit(df["Recipient Country"])
    le_description = LabelEncoder().fit(df["Description"])
    
    # Preprocess features
    X = np.zeros((len(df), 10), dtype=np.float32)
    X[:, 0] = df["Amount (INR)"].astype(float)
    X[:, 1] = le_transaction_type.transform(df["Transaction Type"])
    
    def safe_float(x):
        try:
            return float(x) if x and pd.notna(x) else 0.0
        except (ValueError, TypeError):
            print(f"Warning: Invalid Recipient Account value '{x}', defaulting to 0.0")
            return 0.0
    X[:, 2] = [safe_float(x) for x in df["Recipient Account"]]
    
    X[:, 3] = le_recipient_bank.transform(df["Recipient Bank"])
    X[:, 4] = le_recipient_country.transform(df["Recipient Country"])
    X[:, 5] = le_description.transform(df["Description"])
    X[:, 6] = df["Cash Indicator"].apply(lambda x: 1 if x == "Yes" else 0)
    
    df["Date/Time"] = pd.to_datetime(df["Date/Time"])
    df["Account Creation Date"] = pd.to_datetime(df["Account Creation Date"])
    X[:, 7] = df["Date/Time"].dt.day / 31.0
    X[:, 8] = df["Date/Time"].dt.hour / 24.0
    X[:, 9] = (df["Date/Time"] - df["Account Creation Date"]).dt.days.clip(lower=0) / 365.0
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    y = df["Is_Fraud"].astype(int)
    print(f"Preprocessed X - Shape: {X_scaled.shape}, Min: {np.min(X_scaled, axis=0)}, Max: {np.max(X_scaled, axis=0)}")
    print(f"Labels - Fraud: {sum(y)}, Normal: {len(y) - sum(y)}")
    return X_scaled, y

def main():
    file_path = "transactions.csv"
    X, y = load_training_data(file_path)
    model = train_xgboost(X, y)
    save_model(model)

if __name__ == "__main__":
    main()