# Configuration settings for the Eye Severity Detection Application

import os

# Application Settings
APP_NAME = "Eye Severity Detection System"
APP_VERSION = "1.0.0"

# User Database (In-memory for demo, use SQL database in production)
USERS = {
    "admin": "admin123",
    "user": "user123",
    "doctor": "doctor123"
}

# Eye Severity Classifications
SEVERITY_LEVELS = {
    "normal": {"value": 0, "description": "No diabetic retinopathy detected", "color": "#00AA00"},
    "mild": {"value": 1, "description": "Mild non-proliferative diabetic retinopathy", "color": "#FFFF00"},
    "moderate": {"value": 2, "description": "Moderate non-proliferative diabetic retinopathy", "color": "#FFA500"},
    "severe": {"value": 3, "description": "Severe non-proliferative diabetic retinopathy", "color": "#FF6600"},
    "proliferative": {"value": 4, "description": "Proliferative diabetic retinopathy", "color": "#FF0000"}
}

# File paths
UPLOAD_DIR = "uploads"
MODEL_DIR = "models"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
