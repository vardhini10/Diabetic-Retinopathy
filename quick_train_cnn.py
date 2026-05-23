"""
Quick CNN Training Script - Optimized for faster training
"""
import os
import sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image

print("Loading TensorFlow...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

print("TensorFlow loaded successfully")

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

severity_labels = {
    'Normal': 0,
    'Mild': 1,
    'Moderate': 2,
    'Severe': 3,
    'Proliferative': 4
}

def load_dataset(path, max_rows=500):
    """Load CSV file - default 500 rows to save memory"""
    print(f"Loading dataset from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    if max_rows:
        df = df.head(max_rows)
        print(f"Using first {max_rows} rows")
    print(f"Dataset shape: {df.shape}")
    return df

def generate_images(df):
    """Generate synthetic training images"""
    print(f"Generating {len(df)} synthetic images...")
    os.makedirs('training_images', exist_ok=True)
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(df)} images generated...")
        
        severity = row['Severity']
        label = severity_labels.get(severity)
        if label is None:
            continue
        
        # Create synthetic retinal image
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Optic disc (gray circle)
        cv2.circle(image, (64, 64), 50, (100, 100, 100), -1)
        cv2.circle(image, (64, 64), 45, (150, 150, 150), -1)
        cv2.circle(image, (64, 64), 25, (50, 50, 50), -1)
        
        # Add severity markers
        if severity in ['Mild', 'Moderate', 'Severe', 'Proliferative']:
            num_dots = 10 + label * 5
            for _ in range(num_dots):
                x = np.random.randint(20, 108)
                y = np.random.randint(20, 108)
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        
        if severity in ['Moderate', 'Severe', 'Proliferative']:
            num_exudates = 5 + label * 3
            for _ in range(num_exudates):
                x = np.random.randint(20, 108)
                y = np.random.randint(20, 108)
                cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
        
        if severity in ['Severe', 'Proliferative']:
            num_hemorrhages = 3 + label
            for _ in range(num_hemorrhages):
                x = np.random.randint(20, 108)
                y = np.random.randint(20, 108)
                cv2.circle(image, (x, y), 5, (0, 0, 200), -1)
        
        # Add noise
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.addWeighted(image, 0.9, noise, 0.1, 0)
        
        images.append(image)
        labels.append(label)
    
    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels)
    
    print(f"Generated {len(images)} images")
    print(f"Class distribution: {np.bincount(labels)}")
    return images, labels

def build_model():
    """Build CNN model"""
    print("Building CNN model...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == '__main__':
    try:
        print("=" * 60)
        print("CNN Training Started")
        print("=" * 60)
        
        # Load data
        df = load_dataset('Diabetic_Retinopathy_Dataset.csv', max_rows=500)
        print(f"Loaded {len(df)} records")
        
        # Generate images
        images, labels = generate_images(df)
        print(f"Generated {len(images)} images with shape {images.shape}")
        
        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Build model
        model = build_model()
        print("Model built successfully")
        
        # Train model
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=32,
            callbacks=[
                callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
            ],
            verbose=1
        )
        
        # Evaluate
        print("Evaluating model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Save model
        print("Saving model...")
        os.makedirs('models', exist_ok=True)
        model.save('models/cnn_eye_severity_model.h5')
        print("✓ Model saved to models/cnn_eye_severity_model.h5")
        
        # Save metadata
        metadata = {
            'model_name': 'CNN Diabetic Retinopathy Detector',
            'trained_date': datetime.now().isoformat(),
            'dataset_size': len(df),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'severity_labels': severity_labels,
            'input_shape': (128, 128, 3),
            'num_classes': 5
        }
        with open('models/cnn_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✓ Metadata saved to models/cnn_model_metadata.json")
        
        print("=" * 60)
        print("✓ Model training complete!")
        print(f"✓ Test Accuracy: {test_accuracy:.2%}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
