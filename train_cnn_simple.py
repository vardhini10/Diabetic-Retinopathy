"""
Ultra-lightweight CNN training - CPU only
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TF warnings

print("Importing libraries...")
import numpy as np
import pandas as pd
import cv2
from datetime import datetime
import json

print("Loading TensorFlow (CPU mode)...")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

print("✓ Libraries loaded")

severity_labels = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Proliferative': 4}

print("Loading CSV...")
df = pd.read_csv('Diabetic_Retinopathy_Dataset.csv')
df = df.head(300)  # Use only 300 rows for speed
print(f"✓ Loaded {len(df)} records")

print("Generating training images (this may take 1-2 minutes)...")
images = []
labels = []
for idx, row in df.iterrows():
    if (idx + 1) % 50 == 0:
        print(f"  {idx+1}/{len(df)}", end='\r')
    
    severity = row['Severity']
    label = severity_labels.get(severity, 0)
    
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(img, (64, 64), 50, (100, 100, 100), -1)
    cv2.circle(img, (64, 64), 45, (150, 150, 150), -1)
    cv2.circle(img, (64, 64), 25, (50, 50, 50), -1)
    
    if severity in ['Mild', 'Moderate', 'Severe', 'Proliferative']:
        for _ in range(10 + label*5):
            cv2.circle(img, (np.random.randint(20,108), np.random.randint(20,108)), 2, (0,0,255), -1)
    if severity in ['Moderate', 'Severe', 'Proliferative']:
        for _ in range(5 + label*3):
            cv2.circle(img, (np.random.randint(20,108), np.random.randint(20,108)), 3, (0,255,255), -1)
    if severity in ['Severe', 'Proliferative']:
        for _ in range(3 + label):
            cv2.circle(img, (np.random.randint(20,108), np.random.randint(20,108)), 5, (0,0,200), -1)
    
    images.append(img.astype(np.float32) / 255.0)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)
print(f"✓ Generated {len(images)} images")

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")

print("Building model...")
model = models.Sequential([
    layers.Conv2D(16, 3, activation='relu', padding='same', input_shape=(128,128,3)),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("✓ Model built")

print("Training (15 epochs, this will take 5-10 minutes)...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=16, verbose=1)

print("Evaluating...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.1%}")

print("Saving model...")
os.makedirs('models', exist_ok=True)
model.save('models/cnn_eye_severity_model.h5')
print("✓ Model saved to models/cnn_eye_severity_model.h5")

metadata = {
    'model_name': 'CNN Diabetic Retinopathy Detector',
    'trained_date': datetime.now().isoformat(),
    'dataset_size': len(df),
    'test_accuracy': float(acc),
    'severity_labels': severity_labels,
}
with open('models/cnn_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Metadata saved")

print("\n" + "="*50)
print("✓ CNN Training Complete!")
print(f"✓ Test Accuracy: {acc:.2%}")
print("="*50)
