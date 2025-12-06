#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split

# ---------------- PATHS ----------------
KEYPOINT_CSV = r"C:\Users\VICTUS\Desktop\project\project\project\data\keypoints.csv"
LABELS_CSV   = r"C:\Users\VICTUS\Desktop\project\project\project\data\labels.csv"
MODEL_PATH   = r"project/project/model/sign_lstm_right_part2morecc.h5"

# ---------------- LOAD LABELS ----------------
with open(LABELS_CSV, "r") as f:
    LABELS = [line.strip() for line in f.readlines() if line.strip()]

print("Loaded labels:", LABELS)
num_classes = len(LABELS)

# ---------------- LOAD RAW DATA ----------------
df = pd.read_csv(KEYPOINT_CSV, header=None)

y_int = df.iloc[:, 0].astype(int).values          # integer labels (0..num_classes-1)
X_raw = df.iloc[:, 1:].values.astype("float32")   # raw keypoints (NOT normalized)

num_features = X_raw.shape[1]  # should be 42
print("Feature count:", num_features)

# ---------------- NORMALIZATION FUNCTION ----------------
def normalize_sample(sample):
    """
    Normalizes a flat array of Mediapipe points.
    Handles 21 landmarks per hand → can support 42 or 84 values.
    """

    pts = sample.reshape(-1, 2)   # (21,2) or (42,2)
    wrist = pts[0]

    # subtract wrist coordinates
    pts = pts - wrist

    # scale so largest absolute value is 1
    max_val = np.max(np.abs(pts))
    if max_val != 0:
        pts = pts / max_val

    return pts.flatten().astype("float32")

# ---------------- APPLY NORMALIZATION ----------------
X_norm = np.array([normalize_sample(row) for row in X_raw], dtype="float32")

print("Example normalized row:", X_norm[0][:10])

# ---------------- TRAIN / VAL SPLIT (STRATIFIED) ----------------
# stratify on integer labels to keep class balance in train/val
X_train_raw, X_val_raw, y_train_int, y_val_int = train_test_split(
    X_norm,
    y_int,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=y_int,
)

# expand dims AFTER split
X_train = np.expand_dims(X_train_raw, axis=-1)   # (N_train, features, 1)
X_val   = np.expand_dims(X_val_raw, axis=-1)     # (N_val, features, 1)

# one-hot encode AFTER split
y_train = tf.keras.utils.to_categorical(y_train_int, num_classes)
y_val   = tf.keras.utils.to_categorical(y_val_int, num_classes)

print("Train shape:", X_train.shape, y_train.shape)
print("Val shape:",   X_val.shape,   y_val.shape)

# ---------------- BUILD CNN-1D MODEL ----------------
inputs = Input(shape=(num_features, 1))

# Slightly larger first kernel + small L2 regularization
x = Conv1D(
    64,
    kernel_size=5,
    activation='relu',
    padding='same',
    kernel_regularizer=regularizers.l2(1e-4)
)(inputs)
x = BatchNormalization()(x)

x = Conv1D(
    128,
    kernel_size=3,
    activation='relu',
    padding='same',
    kernel_regularizer=regularizers.l2(1e-4)
)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Conv1D(
    256,
    kernel_size=3,
    activation='relu',
    padding='same',
    kernel_regularizer=regularizers.l2(1e-4)
)(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Flatten()(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = Dropout(0.4)(x)

output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, output)

# Slightly smaller LR (more stable) – LR will be reduced automatically on plateau
initial_lr = 3e-4
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- CALLBACKS ----------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,              # allow more epochs before stopping
        restore_best_weights=True
    ),
    ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True,
        monitor="val_loss"        # save based on val_loss (more stable than val_accuracy)
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,               # reduce LR by 2x
        patience=3,               # if no improvement in 3 epochs
        min_lr=1e-6,
        verbose=1
    )
]

# ---------------- TRAIN ----------------
print("🚀 Training CNN model on normalized Mediapipe keypoints...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,            # higher; EarlyStopping will usually stop earlier
    batch_size=32,
    callbacks=callbacks
)

print("🎉 Training finished!")
print("📦 Model saved to:", MODEL_PATH)
