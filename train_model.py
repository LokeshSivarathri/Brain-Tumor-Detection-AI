import os
import cv2
import numpy as np

# cv2 → read MRI images
# numpy → numerical processing
# train_test_split → evaluate model correctly
# CNN layers → feature learning

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# CNN requires same image size
# 224×224 is a standard CNN input size
# Dataset configuration
DATASET_PATH = "dataset"
IMG_SIZE = 224

# data → images
# labels → tumor (1) / no tumor (0)
data = []
labels = []

# Reads all MRI images
# Resizes them to a fixed size
# Assigns:
# yes → 1
# no → 0
for category in ["yes", "no"]:
    folder_path = os.path.join(DATASET_PATH, category)
    label = 1 if category == "yes" else 0

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        image = cv2.imread(image_path)

        # Skip unreadable images
        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        data.append(image)
        labels.append(label)

# Normalization (/255) → faster & stable learning
# One-hot encoding → required for softmax output
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

# Learns edges → shapes → tumor regions
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

# Adam → best default optimizer
# Categorical loss → 2-class classification
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Shows model layers
# Useful for viva / interview explanation
model.summary()

# Trains CNN for 10 iterations
# Validation checks real performance
model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# Saves trained model
# Flask web app will load this file
os.makedirs("model", exist_ok=True)
model.save("model/brain_tumor_cnn.h5")

print("Model training completed and saved successfully")
