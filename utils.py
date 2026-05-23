# Utility Functions

import streamlit as st
from datetime import datetime
import json
import os
import uuid
from config import UPLOAD_DIR, SEVERITY_LEVELS

def normalize_severity_label(severity):
    """Normalize severity label to standard keys."""
    if not severity:
        return 'normal'
    sev = str(severity).strip().lower()
    if 'normal' in sev:
        return 'normal'
    if 'mild' in sev:
        return 'mild'
    if 'moderate' in sev:
        return 'moderate'
    if 'severe' in sev and 'proliferative' not in sev:
        return 'severe'
    if 'proliferative' in sev:
        return 'proliferative'
    return 'normal'


def make_serializable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_prediction(username, filename, severity, confidence, features, image_path=None, predictions=None):
    """Save prediction to a log file."""
    normalized_severity = normalize_severity_label(severity)
    
    # Ensure confidence is in 0-100 range
    conf_value = confidence
    if isinstance(conf_value, (int, float)):
        if 0 <= conf_value <= 1:
            conf_value = conf_value * 100
    conf_value = round(float(conf_value), 2)
    
    # Convert any numpy types back to standard python types for JSON serialization
    safe_features = make_serializable(features)
    safe_predictions = make_serializable(predictions or {})
    
    prediction_data = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'username': username,
        'filename': filename,
        'severity': normalized_severity,
        'confidence': conf_value,
        'features': safe_features,
        'predictions': safe_predictions
    }
    if image_path:
        prediction_data['image_path'] = image_path
    
    # Save to JSON file
    log_file = os.path.join(UPLOAD_DIR, 'predictions.json')
    
    # Ensure the uploads directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(prediction_data)
        
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        
        print(f"✓ Prediction saved for {username}: severity={normalized_severity}, confidence={conf_value}%")
        return prediction_data
    except Exception as e:
        print(f"ERROR saving prediction: {e}")
        return None

def get_user_predictions(username):
    """Get all predictions for a user"""
    log_file = os.path.join(UPLOAD_DIR, 'predictions.json')
    predictions = []
    
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            # Filter by username and normalize severity
            for p in data:
                if p.get('username') == username:
                    # Ensure severity is normalized
                    if 'severity' in p:
                        p['severity'] = normalize_severity_label(p.get('severity'))
                    predictions.append(p)
            
            print(f"✓ Retrieved {len(predictions)} predictions for {username}")
        else:
            print(f"⚠ Predictions file not found at {log_file}")
    except Exception as e:
        print(f"ERROR reading predictions: {e}")
    
    return predictions

def format_severity_display(severity):
    """Format severity level for display"""
    info = SEVERITY_LEVELS.get(severity, {})
    return {
        'name': severity.upper(),
        'description': info.get('description', ''),
        'color': info.get('color', '#000000')
    }

def get_statistics():
    """Get overall statistics from predictions"""
    log_file = os.path.join(UPLOAD_DIR, 'predictions.json')
    stats = {
        'total_predictions': 0,
        'by_severity': {},
        'average_confidence': 0
    }
    
    # Initialize all severity levels with 0
    for severity in SEVERITY_LEVELS:
        stats['by_severity'][severity] = 0
    
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            stats['total_predictions'] = len(data)
            
            # Count by severity - normalize each severity
            for p in data:
                severity = normalize_severity_label(p.get('severity', ''))
                stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
            
            # Calculate average confidence
            if data:
                confidences = [p.get('confidence', 0) for p in data]
                avg_conf = sum(confidences) / len(confidences)
                stats['average_confidence'] = round(avg_conf, 2)
            
            print(f"✓ Statistics: Total={stats['total_predictions']}, Avg Confidence={stats['average_confidence']}%")
            print(f"  Severity Distribution: {stats['by_severity']}")
        else:
            print(f"⚠ Predictions file not found for statistics")
    
    except Exception as e:
        print(f"ERROR calculating statistics: {e}")
    
    return stats

def get_severity_color(severity):
    """Get color code for severity level"""
    return SEVERITY_LEVELS.get(severity, {}).get('color', '#000000')
