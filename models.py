# Updated Models with CNN Support

import numpy as np
import cv2
from PIL import Image
from config import SEVERITY_LEVELS
import random
import os

class EyeSeverityDetector:
    """Eye severity detector with CNN model support"""
    
    def __init__(self):
        self.severity_levels = SEVERITY_LEVELS
        self.model = None
        self.use_cnn = False
        self.IMG_SIZE = 128
        self.severity_labels = {
            0: "Normal",
            1: "Mild", 
            2: "Moderate",
            3: "Severe",
            4: "Proliferative"
        }
        self.severity_classes = list(self.severity_labels.values())
        
        # Try to load CNN model
        self.load_cnn_model()
    
    def load_cnn_model(self):
        """Load the trained CNN model if available"""
        try:
            import tensorflow as tf
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'cnn_eye_severity_model.h5')
            print(f"Looking for model at: {model_path}")
            if os.path.exists(model_path):
                print(f"Model file exists, size: {os.path.getsize(model_path)} bytes")
                self.model = tf.keras.models.load_model(model_path)
                self.use_cnn = True
                print("CNN model loaded successfully.")
                print(f"Model input shape: {self.model.input_shape}")
                print(f"Model output shape: {self.model.output_shape}")
            else:
                print("CNN model file not found, using feature-based classification.")
                self.use_cnn = False
        except ImportError:
            print("TensorFlow not available, using feature-based classification.")
            self.use_cnn = False
        except Exception as e:
            print(f"Error loading CNN model: {e}, using feature-based classification.")
            self.use_cnn = False
    
    def preprocess_image(self, image):
        """Preprocess the image for analysis"""
        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # Convert to RGB if necessary
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        return image_array
    
    def resize_image(self, image_array, size=(224, 224)):
        """Resize image to specified size"""
        return cv2.resize(image_array, size)
    
    def normalize_image(self, image_array):
        """Normalize image to [0, 1]"""
        return image_array.astype('float32') / 255.0
    
    def classify_severity_cnn(self, image):
        """Classify using CNN model"""
        if not self.use_cnn or self.model is None:
            return None
            
        try:
            import tensorflow as tf
            # Suppress TensorFlow warnings during prediction
            tf.get_logger().setLevel('ERROR')
        except ImportError:
            return None
            
        try:
            # Preprocess
            image_array = self.preprocess_image(image)
            image_resized = self.resize_image(image_array, (self.IMG_SIZE, self.IMG_SIZE))
            image_normalized = self.normalize_image(image_resized)
            
            # Prepare for model
            image_batch = np.expand_dims(image_normalized, axis=0)
            
            # Predict
            predictions = self.model.predict(image_batch, verbose=0)
            print(f"Debug: Image shape after resize: {image_resized.shape}")
            print(f"Debug: Raw predictions from model: {predictions}")
            
            # Get predictions for all 5 classes
            if len(predictions[0]) != 5:
                print(f"Warning: Expected 5 output classes, got {len(predictions[0])}")
                # If not 5, try to normalize what we have
            
            # Normalize predictions to probabilities
            predictions[0] = np.clip(predictions[0], 0, None)  # Ensure non-negative
            predictions[0] = predictions[0] / np.sum(predictions[0])  # Normalize to sum=1
            
            confidence = np.max(predictions[0])
            severity_idx = np.argmax(predictions[0])
            print(f"Debug: Predicted class index: {severity_idx}, confidence: {confidence}")
            classes = ["normal", "mild", "moderate", "severe", "proliferative"]
            for i, label in enumerate(classes):
                print(f"{label}: {predictions[0][i]*100:.2f}%")
            print(f"Final Prediction: {classes[severity_idx]}")
            print(f"Confidence: {confidence*100:.2f}%")
            severity = self.severity_labels[severity_idx]
            
            # Generate explanation for CNN result
            explanation = self.generate_cnn_explanation(severity_idx, confidence)
            
            return {
                'severity': severity,
                'confidence': float(confidence),
                'predictions': {self.severity_classes[i]: float(pred) for i, pred in enumerate(predictions[0])},
                'explanation': explanation,
                'model_type': 'CNN'
            }
        except Exception as e:
            print(f"CNN classification error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_image_features(self, image_array):
        """Calculate features from the image for diabetic retinopathy detection"""
        # Convert to HSV for better analysis
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Convert to LAB color space for better color analysis
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        
        # Calculate basic color features
        red_intensity = np.mean(image_array[:, :, 0])
        green_intensity = np.mean(image_array[:, :, 1])
        blue_intensity = np.mean(image_array[:, :, 2])
        
        # Calculate saturation and brightness
        saturation = np.mean(hsv[:, :, 1])
        brightness = np.mean(hsv[:, :, 2])
        
        # Calculate contrast
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        contrast = np.std(gray)
        
        # Edge density (blood vessel abnormalities)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Dark pixel ratio (hemorrhages create dark patches)
        dark_ratio = np.sum(gray < 40) / gray.size
        
        # Bright pixel ratio (exudates create bright patches)
        bright_ratio = np.sum(gray > 200) / gray.size
        
        # Red channel analysis (hemorrhages and microaneurysms)
        red_channel = image_array[:, :, 0]
        red_std = np.std(red_channel)
        red_dark_spots = np.sum(red_channel < 50) / red_channel.size
        
        # Green channel analysis (normal retinal features)
        green_channel = image_array[:, :, 1]
        green_std = np.std(green_channel)
        
        # Color ratios that indicate pathology
        redness_ratio = red_intensity / (green_intensity + blue_intensity + 1)
        brightness_ratio = brightness / 255.0
        
        # Texture analysis using GLCM-like features (simplified)
        # Calculate local variance to detect lesions
        kernel = np.ones((5,5), np.uint8)
        local_var = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        texture_variance = np.std(local_var)
        
        features = {
            'red_intensity': float(red_intensity),
            'green_intensity': float(green_intensity),
            'blue_intensity': float(blue_intensity),
            'saturation': float(saturation),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'edge_ratio': float(edge_ratio),
            'dark_ratio': float(dark_ratio),
            'bright_ratio': float(bright_ratio),
            'red_std': float(red_std),
            'green_std': float(green_std),
            'redness_ratio': float(redness_ratio),
            'brightness_ratio': float(brightness_ratio),
            'texture_variance': float(texture_variance),
            'red_dark_spots': float(red_dark_spots)
        }
        
        return features
    
    def classify_severity_features(self, image):
        """Classify using improved feature-based approach for diabetic retinopathy"""
        try:
            # Preprocess image
            image_array = self.preprocess_image(image)
            
            # Calculate features
            features = self.calculate_image_features(image_array)
            
            # Decision logic based on diabetic retinopathy features
            score = 0
            confidence = 0.5
            
            # Hemorrhage detection (dark red spots) - adjusted thresholds
            if features['red_dark_spots'] > 0.02:  # More than 2% very dark red pixels
                score += 3
            elif features['red_dark_spots'] > 0.01:
                score += 2
            elif features['red_dark_spots'] > 0.005:
                score += 1
            
            # Exudate detection (bright areas) - adjusted thresholds
            if features['bright_ratio'] > 0.10:  # More than 10% very bright pixels
                score += 3
            elif features['bright_ratio'] > 0.05:
                score += 2
            elif features['bright_ratio'] > 0.02:
                score += 1
            
            # Abnormal redness (hemorrhages/microaneurysms) - adjusted thresholds
            if features['redness_ratio'] > 0.45:
                score += 3
            elif features['redness_ratio'] > 0.40:
                score += 2
            elif features['redness_ratio'] > 0.35:
                score += 1
            
            # Texture abnormalities (lesions, vessel changes) - adjusted thresholds
            if features['texture_variance'] > 1000:
                score += 2
            elif features['texture_variance'] > 500:
                score += 1
            
            # Edge density (abnormal vessel patterns) - adjusted thresholds
            if features['edge_ratio'] > 0.08:
                score += 2
            elif features['edge_ratio'] > 0.05:
                score += 1
            
            # Color variations (pathology indicators) - adjusted thresholds
            color_std = (features['red_std'] + features['green_std']) / 2
            if color_std > 40:
                score += 2
            elif color_std > 20:
                score += 1
            
            # Classify based on accumulated score
            # Adjusted thresholds for better distribution
            
            print(f"Feature-based score: {score}")
            if score <= 0:
                severity_idx = 0  # Normal
                confidence = 0.70 + random.uniform(0, 0.15)
            elif score <= 1:
                severity_idx = 1  # Mild
                confidence = 0.65 + random.uniform(0, 0.20)
            elif score <= 2:
                severity_idx = 2  # Moderate
                confidence = 0.60 + random.uniform(0, 0.25)
            elif score <= 3:
                severity_idx = 3  # Severe
                confidence = 0.55 + random.uniform(0, 0.30)
            else:
                severity_idx = 4  # Proliferative DR
                confidence = 0.50 + random.uniform(0, 0.35)

            print(f"Feature-based predicted class index: {severity_idx}")
            classes = ["normal", "mild", "moderate", "severe", "proliferative"]
            # Simulate probabilities for feature-based
            simulated_probs = [0.1] * 5
            simulated_probs[severity_idx] = confidence
            # Normalize
            total = sum(simulated_probs)
            simulated_probs = [p/total for p in simulated_probs]
            for i, label in enumerate(classes):
                print(f"{label}: {simulated_probs[i]*100:.2f}%")
            print(f"Final Prediction: {classes[severity_idx]}")
            print(f"Confidence: {confidence*100:.2f}%")
            severity = self.severity_labels[severity_idx]
            
            # Generate explanation based on detected features
            explanation = self.generate_explanation(features, score, severity_idx)
            
            return {
                'severity': severity,
                'confidence': min(confidence, 0.95),
                'features': features,
                'score': score,
                'explanation': explanation,
                'model_type': 'Feature-Based'
            }
        
        except Exception as e:
            print(f"Feature-based classification error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'severity': self.severity_labels[0],  # Default to Normal
                'confidence': 0.5,
                'features': {},
                'error': str(e),
                'score': 0,
                'explanation': 'Analysis failed - using default classification',
                'model_type': 'Feature-Based'
            }
    
    def generate_explanation(self, features, score, severity_idx):
        """Generate medical explanation based on detected features"""
        explanations = []
        
        # Check for hemorrhages
        if features.get('red_dark_spots', 0) > 0.01:
            explanations.append("hemorrhages detected")
        elif features.get('red_dark_spots', 0) > 0.005:
            explanations.append("possible hemorrhages")
        
        # Check for exudates
        if features.get('bright_ratio', 0) > 0.05:
            explanations.append("hard exudates visible")
        elif features.get('bright_ratio', 0) > 0.02:
            explanations.append("possible exudates")
        
        # Check for microaneurysms (redness)
        if features.get('redness_ratio', 0) > 0.40:
            explanations.append("increased retinal redness suggesting microaneurysms")
        
        # Check for texture abnormalities
        if features.get('texture_variance', 0) > 500:
            explanations.append("abnormal retinal texture")
        
        # Check for edge density (vessel changes)
        if features.get('edge_ratio', 0) > 0.05:
            explanations.append("vascular abnormalities")
        
        # If no features detected
        if not explanations:
            if severity_idx == 0:
                return "No visible abnormalities detected in the retina"
            else:
                return "Analysis based on subtle retinal changes"
        
        # Combine explanations
        if len(explanations) == 1:
            explanation = f"{explanations[0].capitalize()} are visible in the retina"
        else:
            explanation = f"{', '.join(explanations[:-1])} and {explanations[-1]} are visible in the retina"
        
        return explanation
    
    def generate_cnn_explanation(self, severity_idx, confidence):
        """Generate explanation for CNN-based classification"""
        explanations = [
            "CNN analysis shows no significant retinal abnormalities - Normal retina",
            "CNN detected early signs of diabetic retinopathy - Mild hemorrhages and microaneurysms visible",
            "CNN identified moderate retinal changes - Multiple hemorrhages and hard exudates present",
            "CNN found severe retinal abnormalities - Advanced vascular changes and multiple lesions detected",
            "CNN detected proliferative diabetic retinopathy - Neovascularization and advanced pathology identified"
        ]
        if 0 <= severity_idx < len(explanations):
            return explanations[severity_idx]
        else:
            return "CNN analysis completed"
    
    def classify_severity(self, image, patient_data=None):
        """Classify eye severity from image - tries CNN first, falls back to features"""
        
        # Try CNN classification first
        cnn_result = self.classify_severity_cnn(image)
        if cnn_result is not None:
            print("Using CNN model for prediction")
            # Adjust CNN result with patient data if available
            if patient_data:
                cnn_result = self.adjust_with_patient_data(cnn_result, patient_data)
            return cnn_result
        
        print("CNN model not available, using feature-based classification")
        return self.classify_severity_features(image)
    
    def adjust_with_patient_data(self, result, patient_data):
        """Adjust classification result based on patient data"""
        try:
            severity = result['severity']
            confidence = result['confidence']
            
            # Extract patient features
            age = patient_data.get('age')
            years_diabetes = patient_data.get('years_diabetes')
            hba1c = patient_data.get('hba1c')
            gender = patient_data.get('gender', '').lower()
            
            # Convert to numeric
            try:
                age = float(age) if age else None
            except:
                age = None
            try:
                years_diabetes = float(years_diabetes) if years_diabetes else None
            except:
                years_diabetes = None
            try:
                hba1c = float(hba1c) if hba1c else None
            except:
                hba1c = None
            
            adjustment_score = 0
            
            # Age factor: older patients more likely to have advanced disease
            if age and age > 60:
                adjustment_score += 1
            elif age and age > 40:
                adjustment_score += 0.5
            
            # Years of diabetes: longer duration increases risk
            if years_diabetes and years_diabetes > 20:
                adjustment_score += 1.5
            elif years_diabetes and years_diabetes > 10:
                adjustment_score += 1
            elif years_diabetes and years_diabetes > 5:
                adjustment_score += 0.5
            
            # HbA1c: poor control increases risk
            if hba1c and hba1c > 9:
                adjustment_score += 1.5
            elif hba1c and hba1c > 7:
                adjustment_score += 1
            
            # Gender: some studies show differences, but minimal adjustment
            if gender == 'male':
                adjustment_score += 0.2
            
            # Adjust severity based on risk factors
            severity_levels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
            current_idx = severity_levels.index(severity) if severity in severity_levels else 0
            
            # Increase severity for high-risk patients
            if adjustment_score >= 2.5 and current_idx < 4:
                new_idx = min(current_idx + 1, 4)
                result['severity'] = severity_levels[new_idx]
                result['patient_adjustment'] = f"Increased severity due to risk factors (score: {adjustment_score:.1f})"
            elif adjustment_score >= 1.5 and current_idx < 3:
                new_idx = min(current_idx + 1, 4)
                result['severity'] = severity_levels[new_idx]
                result['patient_adjustment'] = f"Moderately increased severity due to risk factors (score: {adjustment_score:.1f})"
            elif adjustment_score < 0.5 and current_idx > 0 and confidence < 0.8:
                # For low-risk patients, be more conservative about severe classifications
                result['patient_adjustment'] = f"Conservative assessment for low-risk patient (score: {adjustment_score:.1f})"
            
            result['patient_risk_score'] = adjustment_score
            
        except Exception as e:
            print(f"Patient data adjustment error: {e}")
            result['patient_adjustment'] = "Could not adjust with patient data"
        
        return result
        if patient_data:
            result = self.adjust_with_patient_data(result, patient_data)
        return result
    
    def get_recommendations(self, severity):
        """Get recommendations based on severity"""
        recommendations = {
            'normal': [
                'Regular eye check-ups recommended',
                'Maintain healthy blood sugar levels',
                'Continue healthy lifestyle',
                'Annual retinal screening'
            ],
            'mild': [
                'Schedule comprehensive eye exam',
                'Monitor blood glucose regularly',
                'Control blood pressure',
                'Follow up in 12 months'
            ],
            'moderate': [
                'Schedule immediate eye exam with ophthalmologist',
                'Strict blood glucose control required',
                'Blood pressure management critical',
                'Follow up in 6 months'
            ],
            'severe': [
                'Urgent ophthalmology referral required',
                'Intensive glucose and blood pressure management',
                'Consider laser treatment',
                'Follow up in 2-3 months'
            ],
            'proliferative': [
                'URGENT: See ophthalmologist immediately',
                'Risk of vision loss is significant',
                'Anti-VEGF injections may be needed',
                'Laser or vitrectomy surgery may be required',
                'Follow up every 4-8 weeks'
            ]
        }
        
        return recommendations.get(severity, [])
