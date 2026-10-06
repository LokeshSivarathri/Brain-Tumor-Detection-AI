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

# Model paths
TFLITE_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_cnn.tflite")
H5_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_cnn.h5")
MODEL_PATH = TFLITE_PATH if os.path.exists(TFLITE_PATH) else H5_PATH

# Global model / interpreter instances
_model = None
_tflite_interpreter = None
_tflite_input_idx = None
_tflite_output_idx = None

def get_model():
    """Lazily loads and caches the inference model (TFLite or Keras CNN) for high-efficiency evaluation."""
    global _model, _tflite_interpreter, _tflite_input_idx, _tflite_output_idx

    # Prefer TFLite: uses < 25 MB RAM and evaluates in ~30ms, ideal for cloud hosting
    if os.path.exists(TFLITE_PATH):
        if _tflite_interpreter is None:
            import tensorflow as tf
            print(f"[NeuroScan AI] Loading lightweight TFLite model from: {TFLITE_PATH}")
            _tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
            _tflite_interpreter.allocate_tensors()
            _tflite_input_idx = _tflite_interpreter.get_input_details()[0]["index"]
            _tflite_output_idx = _tflite_interpreter.get_output_details()[0]["index"]
            print("[NeuroScan AI] TFLite inference engine ready.")
        return _tflite_interpreter

    # Fallback to Keras HDF5 model
    if _model is None:
        if not os.path.exists(H5_PATH):
            raise FileNotFoundError(
                f"Model weights file not found at: {H5_PATH} or {TFLITE_PATH}. "
                "Please ensure the model file is present or run ml_pipeline/train_model.py."
            )
        print(f"[NeuroScan AI] Loading Keras CNN model from: {H5_PATH}")
        _model = load_model(H5_PATH)
        print("[NeuroScan AI] Keras CNN model loaded successfully.")
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
    img_norm = img_resized.astype(np.float32) / 255.0
    img_batch = np.reshape(img_norm, (1, 224, 224, 3))

    get_model()

    if _tflite_interpreter is not None:
        _tflite_interpreter.set_tensor(_tflite_input_idx, img_batch)
        _tflite_interpreter.invoke()
        prediction = _tflite_interpreter.get_tensor(_tflite_output_idx)[0]
    else:
        prediction = _model.predict(img_batch, verbose=0)[0]

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
