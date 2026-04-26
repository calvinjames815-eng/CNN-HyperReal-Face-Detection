"""
Face Classification Pipeline: MobileNetV2 + Fine-Tuning + TTA + Grad-CAM
Author: Calvin James B. Demegillo

NOTE:
This project supports both Google Colab and local execution.
Adjust CONFIG section depending on environment.
"""

# ===========================
# 0. CONFIG (IMPORTANT)
# ===========================
USE_COLAB_DRIVE = False   # Set True only in Colab
DATA_DIR = "./FaceDataset"
MODEL_PATH = "./CNN_HyperRealFaces_BestModel.h5"
HISTORY_PATH = "./training_history.pkl"
GRADCAM_DIR = "./GradCAM_Results"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

import os
os.makedirs(GRADCAM_DIR, exist_ok=True)

# ===========================
# 1. OPTIONAL COLAB SUPPORT
# ===========================
if USE_COLAB_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
    except:
        print("Colab drive mount failed. Continuing locally.")

# ===========================
# 2. IMPORTS
# ===========================
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ===========================
# 3. REPRODUCIBILITY
# ===========================
np.random.seed(42)
tf.random.set_seed(42)

# ===========================
# 4. LOAD DATASET
# ===========================
class_names = sorted([
    f for f in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, f))
])

if len(class_names) == 0:
    raise FileNotFoundError("No class folders found in dataset directory.")

def load_images(img_dir, classes, target_size):
    images, labels = [], []
    for idx, cls in enumerate(classes):
        folder = os.path.join(img_dir, cls)
        if not os.path.exists(folder): continue
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = Image.open(os.path.join(folder, file)).convert("RGB")
                    img = img.resize(target_size)
                    images.append(np.array(img) / 255.0)
                    labels.append(idx)
                except: continue
    return np.array(images, dtype=np.float32), np.array(labels)

images, labels = load_images(DATA_DIR, class_names, IMG_SIZE)

# ===========================
# 5. DATA SPLIT (80% Train, 20% Val)
# ===========================
x_train, x_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.20, stratify=labels, random_state=42
)

print(f"Dataset Split Complete: {len(x_train)} Train | {len(x_val)} Val")

# ===========================
# 6. AUGMENTATION
# ===========================
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_gen = tf.keras.preprocessing.image.ImageDataGenerator()

train_flow = train_gen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
val_flow = val_gen.flow(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

# ===========================
# 7. MODEL
# ===========================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ===========================
# 8. CALLBACKS
# ===========================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
]

# ===========================
# 9. TRAINING STAGE 1
# ===========================
history1 = model.fit(train_flow, validation_data=val_flow, epochs=30, callbacks=callbacks)

# ===========================
# 10. FINE-TUNING
# ===========================
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history2 = model.fit(train_flow, validation_data=val_flow, epochs=50, callbacks=callbacks)

# ===========================
# 11. SAVE MODEL & HISTORY
# ===========================
model.save(MODEL_PATH)
with open(HISTORY_PATH, "wb") as f:
    pickle.dump((history1.history, history2.history), f)
print("Model and history saved successfully.")

# ===========================
# 12. EVALUATION
# ===========================
val_preds = model.predict(x_val)
val_labels = np.argmax(val_preds, axis=1)

print("\n=== VALIDATION RESULTS ===")
print(classification_report(y_val, val_labels, target_names=class_names))
val_acc = accuracy_score(y_val, val_labels)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# ===========================
# 13. TTA (Test-Time Augmentation)
# ===========================
def tta_predict(model, img, rounds=7):
    preds = []
    for _ in range(rounds):
        aug = tf.image.random_flip_left_right(img)
        aug = tf.image.random_brightness(aug, 0.1)
        aug = tf.expand_dims(aug, 0)
        preds.append(model.predict(aug, verbose=0)[0])
    return np.mean(preds, axis=0)

tta_preds = [np.argmax(tta_predict(model, img)) for img in x_val]
tta_acc = accuracy_score(y_val, tta_preds)
print(f"TTA Accuracy: {tta_acc*100:.2f}%")

# ===========================
# 14. GRAD-CAM
# ===========================
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if "conv" in layer.name: return layer.name
    return None

def gradcam(img, model):
    last_conv = get_last_conv_layer(model)
    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(last_conv).output, model.output])
    img_tensor = tf.expand_dims(img, 0)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        loss = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    return cam.numpy()

print("Pipeline complete.")
