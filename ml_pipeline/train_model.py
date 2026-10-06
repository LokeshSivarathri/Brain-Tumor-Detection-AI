import os
import sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Resolve paths dynamically relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_cnn.h5")

IMG_SIZE = 224

def load_and_preprocess_data():
    data = []
    labels = []

    for category in ["yes", "no"]:
        folder_path = os.path.join(DATASET_PATH, category)
        label = 1 if category == "yes" else 0

        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Folder '{folder_path}' not found. Please ensure dataset exists.")
            continue

        valid_count = 0
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            data.append(image)
            labels.append(label)
            valid_count += 1

        print(f"Loaded {valid_count} images from category '{category}'.")

    if len(data) == 0:
        raise ValueError(f"No valid images found in dataset directory '{DATASET_PATH}'.")

    data = np.array(data, dtype="float32") / 255.0
    labels = to_categorical(np.array(labels), num_classes=2)

    return data, labels

def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train(epochs=10, batch_size=32):
    print("==================================================")
    print("  Brain Tumor Detection AI — Model Training")
    print("==================================================")
    print(f"Dataset root: {DATASET_PATH}")
    print(f"Target model save path: {MODEL_PATH}")

    data, labels = load_and_preprocess_data()
    print(f"Total dataset size: {len(data)} images.")

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    print(f"Training set: {len(X_train)} samples | Testing set: {len(X_test)} samples")

    model = build_cnn_model()
    model.summary()

    print("\nStarting model training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test)
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Model successfully saved to: {MODEL_PATH}")
    return history

if __name__ == "__main__":
    train()
