import os
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Dynamic path resolution relative to repository root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_cnn.h5")
SAMPLES_DIR = os.path.join(BASE_DIR, "frontend", "static", "samples")
UPLOAD_DIR = os.path.join(BASE_DIR, "frontend", "static", "uploads")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Curated reference MRI scans for instant 1-click clinical testing
SAMPLE_SCANS = [
    {
        "id": "sample_tumor_1.jpg",
        "title": "Scan #1 (Tumor)",
        "type": "tumor",
        "description": "Axial T1-weighted contrast MRI with hyperintense lesion",
        "filename": "sample_tumor_1.jpg",
        "image_url": "/static/samples/sample_tumor_1.jpg"
    },
    {
        "id": "sample_normal_1.jpeg",
        "title": "Scan #2 (Healthy)",
        "type": "normal",
        "description": "Axial slice demonstrating standard cranial anatomy",
        "filename": "sample_normal_1.jpeg",
        "image_url": "/static/samples/sample_normal_1.jpeg"
    },
    {
        "id": "sample_tumor_2.jpg",
        "title": "Scan #3 (Tumor)",
        "type": "tumor",
        "description": "Axial MRI showing visible intracranial mass lesion",
        "filename": "sample_tumor_2.jpg",
        "image_url": "/static/samples/sample_tumor_2.jpg"
    },
    {
        "id": "sample_normal_2.jpeg",
        "title": "Scan #4 (Healthy)",
        "type": "normal",
        "description": "Clear ventricular and cortical baseline without masses",
        "filename": "sample_normal_2.jpeg",
        "image_url": "/static/samples/sample_normal_2.jpeg"
    }
]

# Global model instance
_model = None

def get_model():
    """Lazily loads and caches the CNN model to avoid startup delays when imported."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model weights file not found at: {MODEL_PATH}. "
                "Please ensure the model file is present or run ml_pipeline/train_model.py."
            )
        print(f"[NeuroScan AI] Loading CNN model from: {MODEL_PATH}")
        _model = load_model(MODEL_PATH)
        print("[NeuroScan AI] Model loaded successfully.")
    return _model

def allowed_file(filename):
    """Validates uploaded image file extension against allowed formats."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def run_inference_detailed(image_path):
    """
    Performs CNN inference on an MRI image slice and returns comprehensive diagnostic data.
    """
    start_time = time.perf_counter()
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Unable to decode the provided image. Please upload a valid MRI scan in JPG or PNG format.")

    orig_h, orig_w = img.shape[:2]

    # Preprocessing identical to training (224x224, normalized to [0, 1])
    img_resized = cv2.resize(img, (224, 224))
    img_norm = img_resized / 255.0
    img_batch = np.reshape(img_norm, (1, 224, 224, 3))

    model = get_model()
    prediction = model.predict(img_batch, verbose=0)[0]
    normal_prob = float(prediction[0]) * 100.0
    tumor_prob = float(prediction[1]) * 100.0
    class_index = int(np.argmax(prediction))
    confidence = round(float(prediction[class_index]) * 100, 2)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    is_tumor = (class_index == 1)

    if is_tumor:
        result_title = "Tumor Detected"
        status_type = "positive"
        severity = "Abnormal / Pathological Anomaly"
        summary_text = "The deep learning network identified hyperintense feature patterns consistent with neoplastic tissue / intracranial mass."
        recommendation = "Clinical correlation with neuro-radiological consultation, contrast-enhanced volumetric MRI, and histopathological evaluation is strongly indicated."
    else:
        result_title = "No Tumor Detected"
        status_type = "negative"
        severity = "Normal / Baseline Clear"
        summary_text = "No evidence of intracranial mass, hyperintensity abnormalities, or space-occupying neoplastic lesions detected in this slice."
        recommendation = "Scan findings fall within normal limits for this slice view. Continued routine clinical monitoring recommended as clinically indicated."

    return {
        "prediction": result_title,
        "is_tumor": is_tumor,
        "status_type": status_type,
        "confidence": confidence,
        "tumor_prob": round(tumor_prob, 2),
        "normal_prob": round(normal_prob, 2),
        "dimensions": f"{orig_w} × {orig_h} px",
        "latency_ms": latency_ms,
        "severity": severity,
        "summary": summary_text,
        "recommendation": recommendation,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }

def predict_tumor(image_path):
    """
    Standard binary prediction interface returning (status_string, confidence_score).
    Maintained for backwards-compatibility.
    """
    data = run_inference_detailed(image_path)
    result = "🧠 Tumor Detected" if data["is_tumor"] else "✅ No Tumor Detected"
    return result, data["confidence"]
