import os
import sys
import time
import uuid
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

# Ensure backend directory is in Python path for direct script execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from ml.predictor import (
    predict_tumor,
    run_inference_detailed,
    SAMPLE_SCANS,
    allowed_file,
    BASE_DIR,
    UPLOAD_DIR,
    SAMPLES_DIR,
    get_model
)

# Initialize Flask with explicit paths to frontend/templates and frontend/static
TEMPLATE_DIR = os.path.join(BASE_DIR, "frontend", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Warm up CNN model on server startup
try:
    get_model()
except Exception as err:
    print(f"⚠️ Notice: Model initialization deferred: {err}")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_data = None
    image_url = None
    error_message = None

    if request.method == "POST":
        file = request.files.get("image")
        sample_id = request.form.get("sample_id")

        try:
            if file and file.filename and allowed_file(file.filename):
                safe_name = secure_filename(file.filename)
                unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
                file.save(file_path)

                prediction_data = run_inference_detailed(file_path)
                image_url = url_for("static", filename=f"uploads/{unique_name}")

            elif sample_id:
                # Handle sample image selection
                matched_sample = next((s for s in SAMPLE_SCANS if s["id"] == sample_id), None)
                if matched_sample:
                    sample_path = os.path.join(SAMPLES_DIR, matched_sample["filename"])
                    prediction_data = run_inference_detailed(sample_path)
                    image_url = matched_sample["image_url"]
                else:
                    error_message = "Selected sample scan could not be found."
            else:
                error_message = "Please upload a valid MRI scan image (.jpg, .png, .jpeg)."
        except Exception as e:
            error_message = f"Inference error: {str(e)}"

    return render_template(
        "index.html",
        prediction_data=prediction_data,
        image_url=image_url,
        error_message=error_message,
        samples=SAMPLE_SCANS
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Asynchronous JSON API endpoint for interactive scans without full-page reloads.
    """
    try:
        sample_id = request.form.get("sample_id")
        file = request.files.get("image")

        if file and file.filename:
            if not allowed_file(file.filename):
                return jsonify({
                    "success": False,
                    "error": "Invalid file extension. Please upload JPG, PNG, or WEBP MRI scans."
                }), 400

            safe_name = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(file_path)

            data = run_inference_detailed(file_path)
            data["image_url"] = url_for("static", filename=f"uploads/{unique_name}")
            data["filename"] = safe_name
            return jsonify({"success": True, "data": data})

        elif sample_id:
            matched_sample = next((s for s in SAMPLE_SCANS if s["id"] == sample_id), None)
            if not matched_sample:
                return jsonify({"success": False, "error": "Sample scan not found."}), 404

            sample_path = os.path.join(SAMPLES_DIR, matched_sample["filename"])
            data = run_inference_detailed(sample_path)
            data["image_url"] = matched_sample["image_url"]
            data["filename"] = matched_sample["title"]
            return jsonify({"success": True, "data": data})

        else:
            return jsonify({"success": False, "error": "No image file or sample ID provided."}), 400

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/samples", methods=["GET"])
def api_samples():
    """Return available preloaded sample MRI scans."""
    return jsonify({"success": True, "samples": SAMPLE_SCANS})

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for service monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "NeuroScan AI",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    })

if __name__ == "__main__":
    app.run(debug=True, port=8080)
