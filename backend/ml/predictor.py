import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Resolve paths dynamically relative to this file to prevent execution directory errors.
# File is located at: backend/ml/predictor.py
# Root folder is 2 levels up from backend/ml/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_cnn.h5")

# Load model globally so it resides in memory and runs inference quickly
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    print(f"⚠️ Warning: Model weights not found at '{MODEL_PATH}'. Ensure you train the model first.")
    model = None

def predict_tumor(image_path):
    """
    Takes an image file path, preprocesses it, and uses the CNN model to detect brain tumors.
    
    Args:
        image_path (str): Absolute or relative path to the image scan.
        
    Returns:
        tuple: (result_string, confidence_percentage)
    """
    if model is None:
        return "❌ Model Not Loaded", 0.0

    # Reads the uploaded image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    # Process using exact same configurations as training (224x224, normalized pixels)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    # Run inference pipeline
    prediction = model.predict(img)[0]
    class_index = np.argmax(prediction)
    confidence = round(float(prediction[class_index]) * 100, 2)

    # Class 1 maps to 'yes' (tumor), Class 0 maps to 'no' (healthy)
    if class_index == 1:
        result = "🧠 Tumor Detected"
    else:
        result = "✅ No Tumor Detected"

    return result, confidence
