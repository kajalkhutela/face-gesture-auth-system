import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")

classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
print(f"Classes: {classes}")

class_to_idx = {cls: i for i, cls in enumerate(classes)}

data = []
labels = []

for folder in classes:
    path = os.path.join(data_dir, folder)
    images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
    print(f"{folder}: {len(images)} images")
    for img_name in images:
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img_rgb = cv2.resize(img, (96, 96))
            img_rgb = img_rgb.astype(np.float32) / 255.0
            data.append(img_rgb)
            labels.append(class_to_idx[folder])

data = np.array(data)
labels = np.array(labels)

print(f"Total dataset: {len(data)} images, {len(classes)} classes")

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(96,96,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)
]

# Fit with augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    steps_per_epoch=len(X_train) // 16,
    epochs=35,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

os.makedirs("models", exist_ok=True)
model.save("models/gesture_model.h5")
print("✅ Model trained & saved! Best val accuracy:", max(history.history['val_accuracy']))
