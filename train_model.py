"""
NeuroScan AI — Brain Tumor Detection & Diagnostic Suite
Root-level model training entry point. Forwards execution to ml_pipeline.train_model.
"""
from ml_pipeline.train_model import train

if __name__ == "__main__":
    train()
