"""
Face Classification Pipeline: MobileNetV2 + Fine-Tuning + TTA + Grad-CAM
Author: Calvin James B. Demegillo
"""

# --- CONFIG ---
USE_COLAB_DRIVE = False   
DATA_DIR = "./FaceDataset"
MODEL_PATH = "./CNN_HyperRealFaces_BestModel.h5"
HISTORY_PATH = "./training_history.pkl"
GRADCAM_DIR = "./GradCAM_Results"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

import os
if not os.path.exists(GRADCAM_DIR):
    os.makedirs(GRADCAM_DIR)

# --- DRIVE MOUNT ---
if USE_COLAB_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

# --- IMPORTS ---
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

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- DATA LOADING ---
class_names = sorted([f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))])

def load_images(img_dir, classes, target_size):
    images, labels = [], []
    for idx, cls in enumerate(classes):
        folder = os.path.join(img_dir, cls)
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = Image.open(os.path.join(folder, file)).convert("RGB").resize(target_size)
                    images.append(np.array(img) / 255.0)
                    labels.append(idx)
                except Exception as e:
                    print(f"Skipping broken image {file}: {e}")
    return np.array(images, dtype=np.float32), np.array(labels)

print(f"Found {len(class_names)} classes: {class_names}")
images, labels = load_images(DATA_DIR, class_names, IMG_SIZE)
print(f"Successfully loaded {len(images)} images.")

# --- SPLITTING ---
x_train, x_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.20, stratify=labels, random_state=42
)
print(f"Split results -> Train: {x_train.shape[0]} | Val: {x_val.shape[0]}")

# --- TRAINING SETUP ---
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.1, horizontal_flip=True
).flow(x_train, y_train, batch_size=BATCH_SIZE)

val_flow = tf.keras.preprocessing.image.ImageDataGenerator().flow(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

# Build Model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base.output)
out = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=base.input, outputs=out)

# Phase 1: Transfer Learning
for layer in base.layers: layer.trainable = False
model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

ckpt = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

print("\nStarting Stage 1 training...")
h1 = model.fit(train_gen, validation_data=val_flow, epochs=30, callbacks=[ckpt, es])

# Phase 2: Fine-Tuning
print("\nUnfreezing last 50 layers for fine-tuning...")
for layer in base.layers[-50:]: layer.trainable = True
model.compile(optimizer=Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
h2 = model.fit(train_gen, validation_data=val_flow, epochs=50, callbacks=[ckpt, es])

# --- EVALUATION ---
print("\n--- Final Evaluation ---")
v_preds = model.predict(x_val)
print(classification_report(y_val, np.argmax(v_preds, axis=1), target_names=class_names))

# TTA Logic
def get_tta_pred(m, img):
    p = []
    for _ in range(7):
        aug = tf.image.random_brightness(tf.image.random_flip_left_right(img), 0.1)
        p.append(m.predict(tf.expand_dims(aug, 0), verbose=0)[0])
    return np.mean(p, axis=0)

tta_res = [np.argmax(get_tta_pred(model, i)) for i in x_val]
print(f"Standard Acc: {accuracy_score(y_val, np.argmax(v_preds, axis=1)):.4f}")
print(f"TTA Accuracy: {accuracy_score(y_val, tta_res):.4f}")

# --- GRAD-CAM & DEBUG PLOTS ---
def compute_gradcam(img, m):
    # Find last conv layer automatically
    target_layer = [l.name for l in reversed(m.layers) if "conv" in l.name][0]
    gm = Model(m.inputs, [m.get_layer(target_layer).output, m.output])
    with tf.GradientTape() as tape:
        c_out, preds = gm(tf.expand_dims(img, 0))
        loss = preds[:, np.argmax(preds[0])]
    grads = tape.gradient(loss, c_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(c_out[0] * pooled, axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap

# Visualizing Grad-CAM results for model interpretability
print("\nGenerating visual debug plots...")
plt.figure(figsize=(10, 5))
for i in range(3): # Check first 3 validation images
    hm = compute_gradcam(x_val[i], model)
    plt.subplot(1, 3, i+1)
    plt.imshow(x_val[i])
    plt.imshow(hm, cmap='jet', alpha=0.5)
    plt.title(f"Pred: {class_names[np.argmax(v_preds[i])]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

print("Process finished. Check Grad-CAM directory for full results.")
