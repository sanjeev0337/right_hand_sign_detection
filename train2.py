#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# ---------------- PATHS ----------------
KEYPOINT_CSV = r"C:\Users\VICTUS\Desktop\project\project\project\data\keypoints.csv"
LABELS_CSV   = r"C:\Users\VICTUS\Desktop\project\project\project\data\labels.csv"
MODEL_PATH   = r"project/project/model/sign_lstm_right.h5"

# ---------------- LOAD LABELS ----------------
with open(LABELS_CSV, "r") as f:
    LABELS = [line.strip() for line in f.readlines() if line.strip()]

print("Loaded labels:", LABELS)
num_classes = len(LABELS)

# ---------------- LOAD RAW DATA ----------------
df = pd.read_csv(KEYPOINT_CSV, header=None)

y = df.iloc[:, 0].astype(int).values          # labels
X_raw = df.iloc[:, 1:].values.astype("float32")  # raw keypoints (NOT normalized)

num_features = X_raw.shape[1]  # 42 or 84
print("Feature count:", num_features)

# ---------------- NORMALIZATION FUNCTION ----------------
def normalize_sample(sample):
    """
    Normalizes a flat array of Mediapipe points.
    Handles 21 landmarks per hand → can support 42 or 84 values.
    """

    pts = sample.reshape(-1, 2)   # (21,2) or (42,2) for both hands
    wrist = pts[0]

    # subtract wrist coordinates
    pts = pts - wrist

    # scale so largest absolute value is 1
    max_val = np.max(np.abs(pts))
    if max_val != 0:
        pts = pts / max_val

    return pts.flatten()

# ---------------- APPLY NORMALIZATION ----------------
X_norm = np.array([normalize_sample(row) for row in X_raw], dtype="float32")

print("Example normalized row:", X_norm[0][:10])

# ---------------- PREP FOR CNN ----------------
X = np.expand_dims(X_norm, axis=-1)   # shape (N, features, 1)
y = tf.keras.utils.to_categorical(y, num_classes)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True
)

# ---------------- BUILD CNN-1D MODEL ----------------
inputs = Input(shape=(num_features, 1))

x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)

x = Conv1D(128, 3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Conv1D(256, 3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)

output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- CALLBACKS ----------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy")
]

# ---------------- TRAIN ----------------
print("🚀 Training CNN model on normalized Mediapipe keypoints...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=35,
    batch_size=32,
    callbacks=callbacks
)

print("🎉 Training finished!")
print("📦 Model saved to:", MODEL_PATH)
