import numpy as np
import pickle
import os
import logging
from sklearn.preprocessing import StandardScaler
import librosa

logger = logging.getLogger(__name__)

class SpeechEmotionAnalyzer:
    def __init__(self, model_path='../../models'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model and scaler"""
        try:
            model_file = os.path.join(self.model_path, 'speech_emotion_model.pkl')
            scaler_file = os.path.join(self.model_path, 'speech_scaler.pkl')
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                logger.info("Speech emotion model loaded successfully")
            else:
                logger.warning("Pre-trained model not found, using mock predictions")
                self.model = None
                self.scaler = None
                
        except Exception as e:
            logger.error(f"Error loading speech model: {str(e)}")
            self.model = None
            self.scaler = None
    
    def extract_audio_features(self, audio_path):
        """Extract features from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050, duration=30)
            
            # Extract features
            features = []
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.append(np.mean(spectral_centroids))
            features.append(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return np.zeros(50)  # Return zeros if extraction fails
    
    def predict_emotion(self, features):
        """Predict emotion from audio features"""
        if self.model is None or self.scaler is None:
            return self._mock_prediction()
        
        try:
            # Reshape and scale features
            features = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # Predict probabilities
            probabilities = self.model.predict_proba(features_scaled)[0]
            predicted_idx = np.argmax(probabilities)
            
            emotion_label = self.emotion_labels[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            # Map to wellness-relevant emotions
            emotion_mapping = {
                'happy': 'happy',
                'neutral': 'neutral',
                'sad': 'sad',
                'angry': 'stressed',
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
            logger.error(f"Prediction error: {str(e)}")
            return self._mock_prediction()
    
    def _mock_prediction(self):
        """Generate mock prediction for demo purposes"""
        import random
        emotions = ['happy', 'neutral', 'calm', 'sad', 'stressed']
        emotion = random.choice(emotions)
        confidence = random.uniform(0.6, 0.9)
        
        return {
            'label': emotion,
            'confidence': confidence,
            'mock': True,
            'all_probabilities': {
                e: random.uniform(0.1, 0.3) if e != emotion else confidence 
                for e in emotions
            }
        }