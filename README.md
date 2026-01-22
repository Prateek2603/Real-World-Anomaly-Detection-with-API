# Real-World-Anomaly-Detection-with-API
# Project Overview

This project implements a real-world fraud detection system for a fintech payment gateway processing high-volume transactions under extreme class imbalance. The system uses unsupervised anomaly detection models, optimized with cost-sensitive decisioning, and is deployed as a production-ready FastAPI service.

The solution emphasizes:

Low-latency inference (<100 ms)

High interpretability

False positive minimization (10× cost vs false negatives)

Clear separation between offline analysis and online inference

# Problem Statement

Fraudulent transactions account for <0.5% of total traffic, yet impose disproportionate financial and customer experience costs. Traditional supervised models struggle due to:

Label scarcity

Evolving fraud patterns

Severe class imbalance

This project addresses these challenges using anomaly detection techniques with a carefully designed deployment pipeline.
