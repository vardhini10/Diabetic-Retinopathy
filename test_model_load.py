import os
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_path = 'models/cnn_eye_severity_model.h5'
print("Looking for model at:", os.path.abspath(model_path))
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        print("Input shape:", model.input_shape)
        print("Output shape:", model.output_shape)
    except Exception as e:
        print("Error loading model:", e)
else:
    print("Model file not found")