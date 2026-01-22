import json
import joblib
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(".")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
def load_model_registry():
    config_path = ARTIFACTS_DIR / "config" / "model_registry.json"
    with open(config_path) as f:
        return json.load(f)

def load_models():
    registry = load_model_registry()
    models = {}

    for model_name, meta in registry["available_models"].items():
        model_type = meta["type"]
        model_path = PROJECT_ROOT / meta["path"]

        if model_type == "sklearn":
            models[model_name] = joblib.load(model_path)

        elif model_type == "tensorflow":
            models[model_name] = tf.keras.models.load_model(model_path)

        elif model_type == "ensemble":
            with open(model_path) as f:
                models[model_name] = json.load(f)

    return models, registry

def select_model(model_name=None):
    models, registry = load_models()

    if model_name is None:
        model_name = registry["default_model"]

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not available")

    return model_name, models[model_name], models
