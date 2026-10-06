"""
NeuroScan AI — Brain Tumor Detection & Diagnostic Suite
Root-level application entry point. Forwards execution to backend.app.
"""
from backend.app import app

if __name__ == "__main__":
    app.run(debug=True, port=8080)
