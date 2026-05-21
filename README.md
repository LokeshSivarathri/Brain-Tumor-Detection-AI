# 🧠 Brain Tumor Detection AI

A deep learning web application that detects brain tumors from MRI images using a CNN model and Flask.

## 🚀 Features
- Upload MRI brain image
- CNN-based tumor prediction
- Confidence score
- Clean & modern UI
- Flask backend

## 🛠 Tech Stack
- Python
- TensorFlow / Keras
- Flask
- HTML, CSS
- OpenCV

## 📂 Project Structure

```text
├── backend/
│   ├── app.py              # Flask server and routing
│   └── ml/
│       ├── __init__.py
│       └── predictor.py    # CNN inference & image preprocessing
├── frontend/
│   ├── static/             # Static assets and temporary uploads
│   └── templates/
│       └── index.html      # Glassmorphic, modern web UI
├── ml_pipeline/
│   └── train_model.py      # CNN training pipeline with Keras
├── model/
│   └── brain_tumor_cnn.h5  # Trained model weights (locally stored, git-ignored)
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 🌐 Deployment Note

This project uses a Flask backend and a trained CNN model.

⚠️ GitHub Pages hosts a **static UI demo only**.  
The full image upload and prediction functionality requires a Python backend.

To run locally:
```bash
pip install -r requirements.txt
python backend/app.py
```
