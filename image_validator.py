"""
Image validation to detect if uploaded image is an eye retinal image
"""
import cv2
import numpy as np
from PIL import Image
import streamlit as st


def detect_eye(image_array):
    """Return True if a clear eye is present using Haar cascade.

    This function is used for general eye detection (iris, external
    photos) and should succeed on any image containing a visible eye.
    """
    try:
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return len(eyes) > 0
    except Exception:
        return False

def is_eye_image(image_array):
    """
    Validate if image is likely a retinal/eye image
    Returns (is_valid: bool, confidence: float, message: str)
    """
    try:
        # Convert to grayscale if color
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Initialize score
        score = 0
        max_score = 0
        details = []
        
        # Check 1: Image should have moderate size (retinal images typically 512x512 or larger)
        max_score += 1
        h, w = gray.shape[:2]
        if h >= 100 and w >= 100:
            score += 1
            details.append(f"✓ Size check passed ({h}x{w})")
        else:
            details.append(f"✗ Image too small ({h}x{w})")
        
        # Check 2: Circular optic disc detection
        # Retinal images have multiple circular structures
        max_score += 1
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        circular_count = 0
        if len(contours) > 0:
            for contour in contours[:20]:  # look at a few more
                area = cv2.contourArea(contour)
                if area > 200:  # require larger circles
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if 0.6 < circularity < 1.3:
                            circular_count += 1
        if circular_count >= 3:
            score += 1
            details.append(f"✓ Circular structures detected ({circular_count})")
        else:
            details.append(f"✗ No clear circular structures found ({circular_count})")
        
        # Check 3: Central bright spot (optic disc)
        max_score += 1
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        center_brightness = np.mean(center_region)
        
        if center_brightness > 90:  # higher threshold
            score += 1
            details.append(f"✓ Central bright region detected (brightness: {center_brightness:.0f})")
        else:
            details.append(f"✗ No bright central region (brightness: {center_brightness:.0f})")
        
        # Check 4: Histogram characteristics (retinal images have specific distribution)
        max_score += 1
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # Retinal images typically have peaks in mid-range
        mid_range = np.sum(hist[80:180]) / np.sum(hist)
        
        if circular_count >= 1:
            score += 1
            details.append(f"✓ Circular structures detected ({circular_count})")
        else:
            details.append(f"✗ Histogram pattern doesn't match retinal image ({mid_range:.2f})")
        
        # Check 5: Edge detection (retinal images have specific edge patterns)
        max_score += 1
        edge_ratio = np.sum(edges > 0) / edges.size
        
        if 0.07 < edge_ratio < 0.3:  # tighten range
            score += 1
            details.append(f"✓ Edge distribution normal (ratio: {edge_ratio:.2%})")
        else:
            details.append(f"✗ Unusual edge distribution (ratio: {edge_ratio:.2%})")
        
        # Calculate confidence
        confidence = (score / max_score) * 100

        # Require both minimum size and at least one circular structure
        size_ok = h >= 100 and w >= 100
        eye_ok = circular_count >= 1

        # Fallback: if no circles found, try Haar cascade eye detector (for external eye photos)
        if not eye_ok:
            try:
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                # detect eyes in smaller image for performance
                small = cv2.resize(gray, (w//2, h//2))
                detected = eye_cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=3)
                if len(detected) > 0:
                    eye_ok = True
                    details.append(f"✓ Haar cascade detected eye ({len(detected)} regions)")
                else:
                    details.append("✗ Haar cascade did not detect eye regions")
            except Exception as e:
                details.append(f"✗ Haar cascade error: {e}")

        is_valid = size_ok and eye_ok

        return is_valid, confidence, details
        
    except Exception as e:
        return False, 0, [f"Error validating image: {str(e)}"]


def is_retinal_image(image_array):
    """Return True if the array is likely a retinal fundus image.

    This is a stricter check than :func:`is_eye_image`. It uses the
    same grayscale/circle detection but requires multiple circular
    structures and a bright central disc. It mimics the earlier
    validation logic.
    """
    try:
        # reuse grayscale conversion from is_eye_image
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        h, w = gray.shape[:2]
        if h < 100 or w < 100:
            return False

        # detect circular regions
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        circular_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:
                peri = cv2.arcLength(contour, True)
                if peri > 0:
                    circ = 4 * np.pi * area / (peri * peri)
                    if 0.6 < circ < 1.3:
                        circular_count += 1
        if circular_count < 2:
            return False

        # central brightness check
        center = gray[h//4:3*h//4, w//4:3*w//4]
        if np.mean(center) < 80:
            return False

        return True
    except Exception:
        return False

def classify_external_eye_severity(image_array):
    """Heuristic fallback severity mapping for non-fundus external eye images."""
    try:
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        if image_array.shape[-1] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        red_intensity = float(np.mean(image_array[:, :, 0]))
        green_intensity = float(np.mean(image_array[:, :, 1]))
        blue_intensity = float(np.mean(image_array[:, :, 2]))

        red_ratio = red_intensity / (green_intensity + 1)
        dark_ratio = np.sum(np.mean(image_array, axis=2) < 70) / (image_array.shape[0] * image_array.shape[1])

        edges = cv2.Canny(cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY), 80, 180)
        edge_ratio = np.sum(edges > 0) / edges.size

        # Severity indications from external photos
        pathology_score = 0
        pathology_score += 2 if red_ratio > 1.8 else 1 if red_ratio > 1.4 else 0
        pathology_score += 2 if dark_ratio > 0.15 else 1 if dark_ratio > 0.09 else 0
        pathology_score += 2 if edge_ratio > 0.18 else 1 if edge_ratio > 0.13 else 0

        if pathology_score <= 1:
            severity = 'Normal (External eye)'
            confidence = 0.60
        elif pathology_score <= 3:
            severity = 'Mild'
            confidence = 0.65
        elif pathology_score <= 4:
            severity = 'Moderate'
            confidence = 0.75
        elif pathology_score <= 5:
            severity = 'Severe'
            confidence = 0.85
        else:
            severity = 'Proliferative'
            confidence = 0.90

        return severity, confidence, {
            'red_ratio': red_ratio,
            'dark_ratio': dark_ratio,
            'edge_ratio': edge_ratio,
            'red_intensity': red_intensity,
            'green_intensity': green_intensity,
            'blue_intensity': blue_intensity,
            'pathology_score': pathology_score,
            'note': 'External eye image detected; decision is heuristic-based for this non-fundus capture.'
        }
    except Exception as e:
        return 'Normal', 0.5, {'error': str(e)}


def validate_uploaded_image(uploaded_file):
    """
    Validate uploaded image and return result with message
    Returns (is_valid: bool, image: array or None, message: str)
    """
    try:
        # Read image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Handle different image formats
        if image_array.shape[-1] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif len(image_array.shape) == 2:  # Grayscale
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

        # first, make sure an eye is present (any eye photo)
        eye_present = detect_eye(image_array)
        if not eye_present:
            message = "This image is not supported. Please upload a clear eye image."
            return False, None, message

        if is_retinal_image(image_array):
            is_valid, confidence, details = is_eye_image(image_array)
            message = f"✓ Retinal eye image detected (Confidence: {confidence:.1f}%)"
            return True, image_array, message

        message = (
            "External/non-fundus eye image detected. Using external-eye severity heuristic. "
            "This is an approximate evaluation and not formal diabetic retinopathy diagnosis."
        )
        return True, image_array, message
            
    except Exception as e:
        message = f"Error processing image: {str(e)}"
        return False, None, message

def get_validation_details(uploaded_file):
    """
    Get detailed validation information for display
    """
    try:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        if image_array.shape[-1] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        is_valid, confidence, details = is_eye_image(image_array)
        
        return {
            'is_valid': is_valid,
            'confidence': confidence,
            'details': details
        }
    except Exception as e:
        return {
            'is_valid': False,
            'confidence': 0,
            'details': [f"Error: {str(e)}"]
        } 