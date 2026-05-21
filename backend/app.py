from flask import Flask, render_template, request
import os

# Sibling import from the online ML inference sub-module
from ml.predictor import predict_tumor

# Initialize Flask, pointing template and static directories to the new frontend folder structure.
# Paths are resolved relative to the 'backend/' directory where this file resides.
app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

# Dynamically resolve the absolute uploads location inside our separated frontend assets structure.
# This makes file writing path-independent.
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route handler for index page.
    Handles user uploaded MRI images and returns deep learning prediction results.
    """
    prediction = ""
    confidence = ""
    image_path = ""

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            # Safely join target path and save user upload
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Call our modular predictor logic
            prediction, confidence = predict_tumor(file_path)
            
            # The static URL loader expects paths relative to the static directory
            image_path = file.filename

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == "__main__":
    # Start web server on standard development port
    app.run(debug=True, port=8080)
