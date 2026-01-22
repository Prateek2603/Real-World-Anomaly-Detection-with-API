import numpy as np

def run_inference(X_scaled, model_name, models):
    """
    Maps raw anomaly scores to a 0-1 probability.
    Note: Thresholds should be tuned based on your validation set.
    """
    if model_name == "isolation_forest":
        model = models["isolation_forest"]
        # raw_score: Negative decision function (higher is more anomalous)
        raw_score = -model.decision_function(X_scaled)[0]
        # Mapping: anything above 0.2 is suspicious
        prob = min(max((raw_score + 0.1) / 0.5, 0), 1.0)
        
    elif model_name == "autoencoder":
        model = models["autoencoder"]
        recon = model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - recon, 2))
        # Mapping: MSE > 5.0 is usually a strong anomaly
        prob = min(mse / 8.0, 1.0)

    elif model_name == "or_ensemble":
        # Recursively get both and take the max probability
        iso_res = run_inference(X_scaled, "isolation_forest", models)
        ae_res = run_inference(X_scaled, "autoencoder", models)
        prob = max(iso_res["fraud_probability"], ae_res["fraud_probability"])
        raw_score = max(iso_res["raw_score"], ae_res["raw_score"])

    return {
        "fraud_probability": float(prob),
        "raw_score": float(raw_score if 'raw_score' in locals() else mse)
    }