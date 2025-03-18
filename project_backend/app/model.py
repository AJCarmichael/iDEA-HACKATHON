import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def train_xgboost(X, y, scale_pos_weight=None):
    """
    Train an XGBoost classifier.
    X: Feature array (n_samples, n_features)
    y: Labels (0 = normal, 1 = fraud)
    scale_pos_weight: Weight for positive class (default: ratio of negatives to positives)
    """
    if scale_pos_weight is None:
        scale_pos_weight = (len(y) - sum(y)) / sum(y)  # Negatives / Positives
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        max_depth=6,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Evaluate on training data
    y_pred = model.predict(X)
    print(f"Training Metrics - Accuracy: {accuracy_score(y, y_pred):.4f}, "
          f"Precision: {precision_score(y, y_pred):.4f}, "
          f"Recall: {recall_score(y, y_pred):.4f}, "
          f"F1: {f1_score(y, y_pred):.4f}")
    return model

def save_model(model, path="app/saved_model/xgboost_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def load_model(path="app/saved_model/xgboost_model.pkl"):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {path}")
        dummy_X = np.ones((1, 10))  # 10 features
        probs = model.predict_proba(dummy_X)
        print(f"Dummy probs: {probs}")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def predict_anomalies(model, X):
    """
    Predict fraud using XGBoost.
    Returns anomaly scores (0-100, higher = more fraudulent) and predictions (1 = fraud, 0 = normal).
    """
    # Probability of fraud (class 1)
    probs = model.predict_proba(X)[:, 1]
    anomaly_scores = probs * 100  # Scale to 0-100
    predictions = model.predict(X)  # 0 or 1
    
    print(f"Anomaly Scores - Min: {np.min(anomaly_scores):.4f}, Max: {np.max(anomaly_scores):.4f}, "
          f"Mean: {np.mean(anomaly_scores):.4f}, Std: {np.std(anomaly_scores):.4f}")
    return anomaly_scores, predictions