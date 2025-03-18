from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
from app.model import load_model, predict_anomalies
from app.preprocess import preprocess_data_batch
import pandas as pd
import asyncio
import json
from typing import List

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "app/saved_model/xgboost_model.pkl"
model = load_model(MODEL_PATH)
if model is None:
    raise RuntimeError("Failed to initialize model")

TRANSACTIONS_DF = pd.read_csv("transactions.csv")
TRANSACTIONS_DF.columns = ["Transaction_ID", "Account_Number", "Date_Time", "Amount_INR", "Transaction_Type", 
                          "Recipient_Account", "Recipient_Bank", "Recipient_Country", "Description", 
                          "Cash_Indicator", "Customer_ID", "Account_Creation_Date", "Is_Fraud"]
TRANSACTIONS_LIST = TRANSACTIONS_DF.drop(columns=["Is_Fraud"]).to_dict(orient="records")
TRANSACTION_INDEX = 0

class Transaction(BaseModel):
    Transaction_ID: str
    Account_Number: int
    Date_Time: str
    Amount_INR: float
    Transaction_Type: str
    Recipient_Account: str
    Recipient_Bank: str
    Recipient_Country: str
    Description: str
    Cash_Indicator: str
    Customer_ID: str
    Account_Creation_Date: str

class PredictionResult(BaseModel):
    Transaction_ID: str
    anomaly_score: float
    predicted_fraud: int

@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        data_dict = transaction.dict()
        X = preprocess_data_batch([data_dict])
        anomaly_scores, predictions = predict_anomalies(model, X)
        return {"anomaly_score": float(anomaly_scores[0]), "predicted_fraud": int(predictions[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_bulk", response_model=List[PredictionResult])
async def predict_bulk(transactions: List[Transaction]):
    try:
        data_dicts = [t.dict() for t in transactions]
        X = preprocess_data_batch(data_dicts)
        anomaly_scores, predictions = predict_anomalies(model, X)
        results = [PredictionResult(Transaction_ID=data_dicts[i]["Transaction_ID"], anomaly_score=float(anomaly_scores[i]), predicted_fraud=int(predictions[i])) for i in range(len(data_dicts))]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/stream_transactions")
async def stream_transactions():
    async def event_generator():
        global TRANSACTION_INDEX
        while True:
            if TRANSACTION_INDEX >= len(TRANSACTIONS_LIST):
                TRANSACTION_INDEX = 0
            transaction = TRANSACTIONS_LIST[TRANSACTION_INDEX]
            X = preprocess_data_batch([transaction])
            anomaly_scores, predictions = predict_anomalies(model, X)
            result = {
                "Transaction_ID": transaction["Transaction_ID"],
                "anomaly_score": float(anomaly_scores[0]),
                "predicted_fraud": int(predictions[0]),
                "Amount_INR": transaction["Amount_INR"],
                "Transaction_Type": transaction["Transaction_Type"],
                "Date_Time": transaction["Date_Time"]
            }
            yield f"data: {json.dumps(result)}\n\n"
            TRANSACTION_INDEX += 1
            await asyncio.sleep(5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")