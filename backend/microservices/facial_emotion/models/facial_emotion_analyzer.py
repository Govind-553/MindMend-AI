import cv2
import numpy as np
import os
import logging
from tensorflow.keras.models import load_model
import tensorflow as tf

logger = logging.getLogger(__name__)

class FacialEmotionAnalyzer:
    def __init__(self, model_path='../../models'):
        self.model_path = model_path
        self.model = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.input_shape = (48, 48, 1)
        self.load_model()
    
    def load_model(self):
        """Load pre-trained FER2013 model"""
        try:
            model_file = os.path.join(self.model_path, 'fer2013_model.h5')
            
            if os.path.exists(model_file):
                self.model = load_model(model_file)
                logger.info("Facial emotion model loaded successfully")
            else:
                logger.warning("Pre-trained facial model not found, using mock predictions")
                self.model = None
                
        except Exception as e:
            logger.error(f"Error loading facial model: {str(e)}")
            self.model = None
    
    def preprocess_face(self, face_image):
        """Preprocess face image for emotion recognition"""
        try:
            # Convert PIL to OpenCV format if needed
            if hasattr(face_image, 'mode'):
                face_image = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
            
            # Resize to model input size
            face_resized = cv2.resize(face_image, (48, 48))
            
            # Convert to grayscale
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_resized
            
            # Normalize pixel values
            face_normalized = face_gray.astype('float32') / 255.0
            
            # Reshape for model input
            face_input = np.expand_dims(face_normalized, axis=0)
            face_input = np.expand_dims(face_input, axis=-1)
            
            return face_input
            
        except Exception as e:
            logger.error(f"Face preprocessing error: {str(e)}")
            return None
    
    def predict_emotion(self, face_image):
        """Predict emotion from face image"""
        if self.model is None:
            return self._mock_prediction()
        
        try:
            # Preprocess face
            processed_face = self.preprocess_face(face_image)
            if processed_face is None:
                return self._mock_prediction()
            
            # Predict probabilities
            probabilities = self.model.predict(processed_face, verbose=0)[0]
            predicted_idx = np.argmax(probabilities)
            
            emotion_label = self.emotion_labels[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            # Map to wellness-relevant emotions
            emotion_mapping = {
                'happy': 'happy',
                'neutral': 'neutral',
                'sad': 'sad',
                'angry': 'angry',
                'fear': 'anxious',
                'surprise': 'surprised',
                'disgust': 'frustrated'
            }
            
            mapped_emotion = emotion_mapping.get(emotion_label, emotion_label)
            
            return {
                'label': mapped_emotion,
                'confidence': confidence,
                'raw_emotion': emotion_label,
                'all_probabilities': {
                    emotion_mapping.get(label, label): float(prob) 
                    for label, prob in zip(self.emotion_labels, probabilities)
                }
            }
            
        except Exception as e:
            logger.error(f"Emotion prediction error: {str(e)}")
            return self._mock_prediction()
    
    def _mock_prediction(self):
        """Generate mock prediction for demo purposes"""
        import random
        emotions = ['happy', 'neutral', 'surprised', 'sad', 'angry']
        emotion = random.choice(emotions)
        confidence = random.uniform(0.5, 0.85)
        
        return {
            'label': emotion,
            'confidence': confidence,
            'mock': True,
            'all_probabilities': {
                e: random.uniform(0.1, 0.3) if e != emotion else confidence 
                for e in emotions
            }
        }