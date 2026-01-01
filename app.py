from flask import Flask, render_template, request
import numpy as np
import cv2
import os

# Flask → web server
# render_template → show HTML page
# request → receive uploaded image
# cv2 + numpy → image preprocessing
# load_model → load trained CNN

from tensorflow.keras.models import load_model


app = Flask(__name__)  #This creates your Flask web application.

# Loads your already-trained brain
# No retraining needed
# Predictions become instant

# Load trained CNN model
model = load_model("model/brain_tumor_cnn.h5")

# Uploaded images must be saved temporarily
# Flask cannot process files directly from browser memory
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Converts uploaded image → CNN format
# Uses same preprocessing as training
# Returns human-readable result

def predict_tumor(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    prediction = model.predict(img)[0]
    class_index = np.argmax(prediction)
    confidence = round(float(prediction[class_index]) * 100, 2)

    if class_index == 1:
        result = "🧠 Tumor Detected"
    else:
        result = "✅ No Tumor Detected"

    return result, confidence


# User opens website → GET request
# User uploads image → POST request
# Image saved → prediction made
# Result sent back to HTML

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    confidence = ""
    image_path = ""

    if request.method == "POST":
        file = request.files["image"]

        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            prediction, confidence = predict_tumor(file_path)
            image_path = file.filename

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )



if __name__ == "__main__":
    app.run(debug=True, port=8080)

