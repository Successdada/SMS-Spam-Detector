# Load the saved spam pipeline and classify a single SMS message as spam or ham.
import os
import joblib

# Locate the model relative to this file, so it works no matter which folder Python is started from.
MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "spam_pipeline.joblib")
)


def load_model(model_path=MODEL_PATH):
    """Load the fitted scikit-learn pipeline (TF-IDF + LinearSVC) from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_message(message, model):
    """Classify one message. Returns {"label": "spam" or "ham", "decision_score": float}."""
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    if not message.strip():
        raise ValueError("message must not be empty or whitespace-only")

    # The original text is passed unchanged, exactly as the model was trained and evaluated.
    label = model.predict([message])[0]
    score = float(model.decision_function([message])[0])
    return {"label": str(label), "decision_score": score}