import tensorflow as tf
import numpy as np
from PIL import Image
import logging
import os

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the TensorFlow model"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}. Using mock predictions.")
                return
            
            logger.info(f"Loading model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to mock predictions")
    
    def preprocess_image(self, image_file):
        """Preprocess uploaded image for prediction"""
        try:
            image = Image.open(image_file.stream)
            
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize to model input size
            image = image.resize((224, 224))
            
            # Convert to numpy array and normalize
            image_array = np.array(image) / 255.0
            
            # Add batch and channel dimensions
            image_array = image_array[np.newaxis, ..., np.newaxis]
            
            return image_array
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            raise
    
    def predict(self, image_file):
        """Make prediction on uploaded image"""
        try:
            # If model is not loaded, use mock prediction
            if self.model is None:
                return self._mock_prediction()
            
            # Preprocess image
            processed_image = self.preprocess_image(image_file)
            
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)[0][0]
            
            # Calculate confidence and result
            confidence = float(abs(prediction - 0.5) + 0.5)
            result = "malignant" if prediction > 0.5 else "benign"
            
            return {
                'result': result,
                'confidence': confidence,
                'prediction_score': float(prediction),
                'message': f"Prediction: {result.upper()} with {confidence:.1%} confidence"
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._mock_prediction()
    
    def _mock_prediction(self):
        """Provide mock prediction when model is not available"""
        import random
        result = random.choice(['benign', 'malignant'])
        confidence = random.uniform(0.7, 0.95)
        
        return {
            'result': result,
            'confidence': confidence,
            'prediction_score': confidence if result == 'malignant' else 1 - confidence,
            'message': f"Prediction: {result.upper()} with {confidence:.1%} confidence",
            'is_mock': True
        }