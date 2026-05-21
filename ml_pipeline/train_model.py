import os
import sys
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Resolve paths dynamically relative to this file's location to prevent execution directory mismatches.
# File is located at: ml_pipeline/train_model.py
# Root folder is 1 level up from ml_pipeline/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

IMG_SIZE = 224

# data -> images
# labels -> tumor (1) / no tumor (0)
data = []
labels = []

# Reads all MRI images
# Resizes them to a fixed size
# Assigns:
# yes -> 1
# no -> 0
for category in ["yes", "no"]:
    folder_path = os.path.join(DATASET_PATH, category)
    label = 1 if category == "yes" else 0

    if not os.path.exists(folder_path):
        print(f"⚠️  Warning: Directory '{folder_path}' not found. Please create it and add MRI images.")
        continue

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)

        # Skip unreadable images
        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        data.append(image)
        labels.append(label)

if len(data) == 0:
    print(f"❌ Error: No images found in the dataset! Please add MRI images to '{os.path.join('dataset', 'yes')}' and '{os.path.join('dataset', 'no')}'.")
    sys.exit(1)

# Normalization (/255) -> faster & stable learning
# One-hot encoding -> required for softmax output
data = np.array(data) / 255.0
labels = to_categorical(labels, 2)

# 80% training data
# 20% testing data
# Prevents overfitting
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Learns edges -> shapes -> tumor regions
# Dropout avoids memorization
# Softmax outputs probabilities
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
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

# Adam -> best default optimizer
# Categorical loss -> 2-class classification
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Shows model layers
model.summary()

# Trains CNN for 10 iterations
# Validation checks real performance
model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# Saves trained model inside root model folder
os.makedirs(MODEL_DIR, exist_ok=True)
model_save_path = os.path.join(MODEL_DIR, "brain_tumor_cnn.h5")
model.save(model_save_path)

print(f"Model training completed and saved successfully to '{model_save_path}'")
