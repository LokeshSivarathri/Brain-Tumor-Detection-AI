# 🧠 NeuroScan AI — Brain Tumor Detection & Diagnostic Suite

An advanced deep-learning neuro-radiology diagnostic web application that detects and classifies brain tumors from cranial MRI slices using Convolutional Neural Networks (CNN) and Flask.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black.svg)
![Git LFS](https://img.shields.io/badge/Git%20LFS-Tracked-informational.svg)

---

## 🚀 Features

- **Automated MRI Lesion Detection**: High-sensitivity binary classification identifying presence of intracranial tumors/neoplastic tissue.
- **Real-Time Interactive Diagnostic HUD**: Clinical viewport featuring an animated medical laser scanning reticle and real-time processing metrics.
- **Detailed Probability & Confidence Metrics**: Displays tumor vs. healthy probability distributions, inference latency (ms), and scan image resolution.
- **Curated 1-Click Reference Scans**: Pre-loaded reference MRI scans (Tumor & Healthy) for immediate evaluation without needing to upload personal files.
- **RESTful JSON API**: Asynchronous API (`/api/predict` and `/api/samples`) enabling headless inference and third-party integrations.
- **Clinical Impression & Recommendations**: Generates structured diagnostic impressions, anomaly severity levels, and clinical next steps.

---

## 🛠 Tech Stack

- **Deep Learning / Computer Vision**: TensorFlow / Keras (Sequential CNN), OpenCV (`cv2`), NumPy, Scikit-learn
- **Backend**: Python, Flask, Werkzeug
- **Frontend**: Modern Vanilla HTML5, CSS3 (Glassmorphism, CSS Grid, Custom animations), JavaScript (Fetch API, dynamic HUD)
- **Large File Management**: Git LFS for binary model weights (`.h5`)

---

## 📂 Project Architecture

The project follows a clean, modular structure separating backend inference services, frontend presentations, and ML training pipelines:

```text
Brain-Tumor-Detection-AI/
├── backend/
│   ├── app.py                      # Flask server, routing & REST API endpoints
│   ├── ml/
│   │   ├── __init__.py             # ML package exports
│   │   └── predictor.py            # Model loading, image preprocessing & inference engine
│   └── __init__.py
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css           # Modern clinical UI stylesheet
│   │   ├── samples/                # Pre-loaded reference MRI scans
│   │   └── uploads/                # User uploaded scans (.gitkeep tracked)
│   └── templates/
│       └── index.html              # Interactive NeuroScan AI dashboard
├── ml_pipeline/
│   ├── __init__.py
│   └── train_model.py              # CNN model training and evaluation pipeline
├── dataset/
│   ├── no/                         # Normal cranial MRI scans
│   └── yes/                        # Pathological tumor MRI scans
├── model/
│   └── brain_tumor_cnn.h5          # Trained CNN weights (tracked via Git LFS)
├── .gitattributes                  # Git LFS configuration for *.h5
├── .gitignore                      # Git ignore rules
├── app.py                          # Forwarding entrypoint
├── train_model.py                  # Forwarding training entrypoint
├── requirements.txt                # Python dependencies
├── run.bat                         # Windows quick-launch script
└── README.md                       # Project documentation
```

---

## ⚡ Quick Start

### 1. Clone the Repository
Ensure [Git LFS](https://git-lfs.com) is installed before cloning to pull the model weights:
```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone the repository
git clone https://github.com/LokeshSivarathri/Brain-Tumor-Detection-AI.git
cd Brain-Tumor-Detection-AI

# Pull LFS assets (model weights)
git lfs pull
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
python backend/app.py
```
*(On Windows, you can also simply double-click `run.bat`)*

Open your browser and navigate to:
```
http://127.0.0.1:8080/
```

---

## 🔬 Model Training

To retrain the CNN model using the dataset in `dataset/`:

```bash
python ml_pipeline/train_model.py
```

The pipeline:
1. Loads MRI images from `dataset/yes/` and `dataset/no/`.
2. Resizes images to `224×224` and normalizes pixel intensities to `[0, 1]`.
3. Splits data into 80% training and 20% validation sets.
4. Trains a multi-layer Convolutional Neural Network with Dropout regularization.
5. Saves the final trained weights to `model/brain_tumor_cnn.h5`.

---

## 📡 REST API Documentation

### Predict Scan Endpoint
- **URL**: `/api/predict`
- **Method**: `POST`
- **Payload**:
  - `multipart/form-data`: `image` (File) OR `sample_id` (String: e.g., `"sample_tumor_1.jpg"`)
- **Response**:
```json
{
  "success": true,
  "data": {
    "prediction": "Tumor Detected",
    "is_tumor": true,
    "status_type": "positive",
    "confidence": 98.45,
    "tumor_prob": 98.45,
    "normal_prob": 1.55,
    "dimensions": "512 × 512 px",
    "latency_ms": 42.1,
    "severity": "Abnormal / Pathological Anomaly",
    "summary": "The deep learning network identified hyperintense feature patterns consistent with neoplastic tissue / intracranial mass.",
    "recommendation": "Clinical correlation with neuro-radiological consultation is strongly indicated.",
    "image_url": "/static/uploads/scan_abc123.jpg",
    "filename": "scan_sample.jpg",
    "timestamp": "2026-10-07 03:00:00 UTC"
  }
}
```

### Preloaded Samples Endpoint
- **URL**: `/api/samples`
- **Method**: `GET`
- **Response**: List of available reference MRI scans.

---

## 📄 License & Disclaimer

**Disclaimer**: This application is built for research, academic demonstration, and educational purposes. It is **not** intended for independent clinical diagnosis. Diagnostic assessments should always be conducted by certified neuro-radiologists.
