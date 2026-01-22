🛡️ Fraud Detection using Anomaly Detection & FastAPI
📌 Problem Statement

Fintech payment systems process millions of transactions daily, while fraudulent activity accounts for <0.5% of traffic.
The challenge is to detect fraud in real time under:

Extreme class imbalance

High false-positive cost (10× false-negative)

Strict latency requirement (<100 ms)

Need for explainable decisions

Traditional supervised models struggle in this setting, motivating an anomaly detection–based approach.

💡 Solution Overview

This project implements a production-ready fraud detection system using unsupervised anomaly detection models, deployed as a FastAPI service.

The solution is built in two layers:

1️⃣ Offline Modeling (Analysis & Benchmarking)

Isolation Forest

One-Class SVM

Autoencoder

Models were trained on engineered transactional features to understand fraud behavior and compare performance.
Failed approaches and limitations are documented in notebooks.

2️⃣ Online Behavioral Modeling (Production)

For real-time deployment, models were retrained on 4 stable behavioral features to ensure:

Low latency

Robustness to data drift

Human interpretability

🧠 Behavioral Features Used
Feature	What it captures
amt_deviation	Unusual transaction amount
txn_count_cust	Sudden transaction bursts
cust_category_count	New or unfamiliar merchant category
distance_from_home	Geographic anomaly

Raw transaction data is first converted into these behavioral features before model inference.

🤖 Models & Final Choice

Isolation Forest → Fast, stable baseline

One-Class SVM → Rejected (poor scalability & recall)

Autoencoder → Best performance (non-linear patterns)

Ensemble (IF + AE) → Improved recall

🎯 Final Deployment Model: Behavioral Autoencoder
(Optional ensemble supported via configuration)

💰 Cost-Sensitive Decisioning

Given:

False Positive Cost = 10 × False Negative


Model thresholds and ensemble weights were optimized using a custom cost function, prioritizing business impact over raw accuracy.

🚀 API Usage
Start the API
python -m uvicorn app:app --reload

Interactive Docs
http://127.0.0.1:8000/docs

Example Request
{
  "model_name": "autoencoder",
  "amt_deviation": 8200,
  "txn_count_cust": 14,
  "cust_category_count": 0,
  "distance_from_home": 220
}

Example Response
{
  "fraud_probability": 0.78,
  "risk_level": "High",
  "reasons": [
    "Transaction amount is unusually high",
    "Transaction occurred far from home"
  ]
}

🏗️ Key Design Decisions

✔ Separate offline analysis from online inference
✔ Avoid one-hot encodings in production
✔ Use behavioral signals instead of raw fields
✔ Optimize for business cost, not accuracy
✔ Keep the API fast, explainable, and configurable

📘 Where to Find Details

📓 Notebooks → experiments, failures, iterations

📄 Technical Report → full methodology & results

🧠 utils/ → feature engineering, inference, reasoning

⚙️ artifacts/ → trained models & configs

✅ Summary

This project demonstrates a real-world ML system that balances performance, explainability, and production constraints—moving from experimentation to a deployable fraud detection API.
