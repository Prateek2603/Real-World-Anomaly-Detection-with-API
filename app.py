from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import numpy as np
import joblib

from utils.model_loader import select_model
from utils.inference import run_inference
from utils.reasoning import generate_reasoning
from utils.preprocessor import transform_raw_to_behavioral

app = FastAPI(title="Real-World Fraud API")

# Load Scaler (trained on the 4 features)
SCALER = joblib.load("artifacts/models/scaler.pkl")

# Mock database to simulate looking up customer history by cc_num
MOCK_CUSTOMER_DB = {
    "1234567890123456": {"avg_amt": 500.0, "txn_count": 2, "cat_freq": {"grocery": 10}}
}

class RawTransactionRequest(BaseModel):
    # Model selection option
    model_name: Literal["autoencoder", "isolation_forest", "or_ensemble"] = "autoencoder"
    # Raw Columns from your dataset
    cc_num: str
    amt: float
    category: str
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    unix_time: int

@app.post("/predict")
def predict(request: RawTransactionRequest):
    # 1. Look up history (using cc_num as key)
    history = MOCK_CUSTOMER_DB.get(request.cc_num, {"avg_amt": request.amt, "txn_count": 1, "cat_freq": {}})

    # 2. Preprocess: Raw -> Behavioral Features (the 4 input features)
    behavioral = transform_raw_to_behavioral(request.dict(), history)
    
    # 3. Scale the features
    feature_vector = np.array([[
        behavioral["amt_deviation"],
        behavioral["txn_count_cust"],
        behavioral["cust_category_count"],
        behavioral["distance_from_home"]
    ]])
    scaled_features = SCALER.transform(feature_vector)

    # 4. Model Selection & Inference
    m_name, _, models = select_model(request.model_name)
    result = run_inference(scaled_features, m_name, models)

    # 5. Reasoning & Response
    reasons = generate_reasoning(behavioral)
    
    return {
        "cc_num": request.cc_num,
        "fraud_probability": round(result["fraud_probability"], 3),
        "risk_level": "High" if result["fraud_probability"] > 0.6 else "Low",
        "model_used": m_name,
        "reasoning": reasons,
        "debug_behavioral_features": behavioral # Great for the technical report!
    }