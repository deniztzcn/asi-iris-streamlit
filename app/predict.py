import joblib
import os
import numpy as np


# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
model = None


def load_model():
    """Load the trained model from disk."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    return model


def predict(features):
    """
    Predict the Iris species based on input features.

    Args:
        features (list or array): A list of 4 features in order:
            [sepal_length, sepal_width, petal_length, petal_width]

    Returns:
        str: The predicted Iris species name
    """
    # Species mapping
    species_mapping = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    # Load model if not already loaded
    clf = load_model()

    # Convert features to numpy array and reshape for prediction
    features_array = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = clf.predict(features_array)[0]

    # Get prediction probabilities
    probabilities = clf.predict_proba(features_array)[0]

    # Return species name and probabilities
    return species_mapping[prediction], probabilities
