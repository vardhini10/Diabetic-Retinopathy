#!/usr/bin/env python3
"""
Test script to verify eye severity classification is working properly
"""
import sys
import os
sys.path.append('.')

from models import EyeSeverityDetector
from PIL import Image
import numpy as np
import cv2
import cv2

def create_test_image(type_name):
    """Create different types of test images to simulate various severities"""
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    if type_name == "normal":
        # Normal eye: balanced colors, some red vessels
        img[:, :, 0] = 120  # Red
        img[:, :, 1] = 140  # Green
        img[:, :, 2] = 100  # Blue
        # Add some random vessel-like patterns
        for _ in range(20):
            x, y = np.random.randint(0, 128, 2)
            cv2.circle(img, (x, y), 2, (150, 100, 100), -1)

    elif type_name == "mild":
        # Mild: some red spots (microaneurysms)
        img[:, :, 0] = 115
        img[:, :, 1] = 145
        img[:, :, 2] = 105
        # Add microaneurysms
        for _ in range(15):
            x, y = np.random.randint(0, 128, 2)
            cv2.circle(img, (x, y), 1, (180, 80, 80), -1)

    elif type_name == "moderate":
        # Moderate: more red spots and some bright areas
        img[:, :, 0] = 110
        img[:, :, 1] = 150
        img[:, :, 2] = 110
        # Add hemorrhages and exudates
        for _ in range(25):
            x, y = np.random.randint(0, 128, 2)
            if np.random.random() > 0.5:
                cv2.circle(img, (x, y), 3, (200, 60, 60), -1)  # Hemorrhage
            else:
                cv2.circle(img, (x, y), 4, (220, 220, 150), -1)  # Exudate

    elif type_name == "severe":
        # Severe: many dark red patches and bright spots
        img[:, :, 0] = 105
        img[:, :, 1] = 155
        img[:, :, 2] = 115
        # Add severe pathology
        for _ in range(40):
            x, y = np.random.randint(0, 128, 2)
            if np.random.random() > 0.6:
                cv2.circle(img, (x, y), 5, (50, 30, 30), -1)  # Dark hemorrhage
            elif np.random.random() > 0.3:
                cv2.circle(img, (x, y), 4, (230, 230, 180), -1)  # Bright exudate
            else:
                cv2.circle(img, (x, y), 2, (190, 70, 70), -1)  # Red spot

    elif type_name == "proliferative":
        # Proliferative: extensive abnormalities
        img[:, :, 0] = 100
        img[:, :, 1] = 160
        img[:, :, 2] = 120
        # Add proliferative features
        for _ in range(60):
            x, y = np.random.randint(0, 128, 2)
            size = np.random.randint(1, 8)
            if np.random.random() > 0.5:
                cv2.circle(img, (x, y), size, (40, 25, 25), -1)  # Large hemorrhages
            else:
                cv2.circle(img, (x, y), size, (240, 240, 200), -1)  # Large exudates

    return Image.fromarray(img)

def test_classification():
    """Test the classification with different image types"""
    detector = EyeSeverityDetector()

    test_types = ["normal", "mild", "moderate", "severe", "proliferative"]

    print("Testing Eye Severity Classification")
    print("=" * 40)

    for test_type in test_types:
        print(f"\nTesting {test_type.upper()} image:")
        test_img = create_test_image(test_type)
        result = detector.classify_severity(test_img)

        print(f"  Predicted: {result['severity']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Model: {result['model_type']}")
        if 'score' in result:
            print(f"  Score: {result['score']}")

if __name__ == "__main__":
    test_classification()