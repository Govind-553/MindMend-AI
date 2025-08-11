# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import tempfile
import base64
import io
import json
import pickle
import random
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, Any, Optional, List

# Third-party imports
import cv2
import numpy as np
import librosa
from PIL import Image
from dotenv import load_dotenv

# Firebase imports
from firebase_admin import credentials, initialize_app, auth, firestore

load_dotenv()

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper()), # Add .upper() here
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# FIREBASE INITIALIZATION
# ============================================================

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        # Check if Firebase is already initialized
        try:
            from firebase_admin import _apps
            if _apps:
                logger.info("Firebase already initialized")
                return
        except:
            pass
        
        # Initialize with service account key or default credentials
        cred_path = os.getenv('FIREBASE_CREDENTIAL_PATH')
        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            initialize_app(cred)
            logger.info("Firebase initialized with service account")
        else:
            # Try to initialize with default credentials (for Cloud Run/GCP)
            initialize_app()
            logger.info("Firebase initialized with default credentials")
            
    except Exception as e:
        logger.warning(f"Firebase initialization failed: {str(e)}. Using mock mode.")

# ============================================================
# FIREBASE AUTHENTICATION
# ============================================================

def verify_firebase_token(id_token):
    """Verify Firebase ID token and return user ID"""
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token.get('uid')
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        return None

def require_auth(f):
    """Decorator to require Firebase authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip auth in development mode
        if os.getenv('ENVIRONMENT', 'development') == 'development':
            return f('demo_user', *args, **kwargs)
        
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Invalid authorization header'}), 401
        
        id_token = auth_header.split('Bearer ')[1]
        user_id = verify_firebase_token(id_token)
        
        if not user_id:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        return f(user_id, *args, **kwargs)
    
    return decorated_function

# ============================================================
# DATABASE CLIENT (FIRESTORE)
# ============================================================

class FirestoreClient:
    def __init__(self):
        try:
            self.db = firestore.client()
        except Exception as e:
            logger.warning(f"Firestore client initialization failed: {str(e)}")
            self.db = None
    
    def add_wellness_data(self, user_id: str, data: Dict[str, Any]) -> str:
        """Add wellness data to Firestore"""
        if not self.db:
            return "mock_doc_id"
        
        try:
            doc_data = {
                'userId': user_id,
                'timestamp': datetime.utcnow(),
                **data
            }
            doc_ref = self.db.collection('wellnessData').add(doc_data)
            return doc_ref[1].id
        except Exception as e:
            logger.error(f"Error adding wellness data: {str(e)}")
            return "error_doc_id"
    
    def get_user_wellness_history(self, user_id: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get user's wellness history"""
        if not self.db:
            # Return mock data for development
            return [
                {
                    'id': f'mock_{i}',
                    'wellnessIndex': random.uniform(40, 90),
                    'timestamp': (datetime.utcnow() - timedelta(days=i)).isoformat(),
                    'speechEmotion': {'label': random.choice(['happy', 'neutral', 'calm'])},
                    'facialEmotion': {'label': random.choice(['happy', 'neutral', 'surprised'])}
                }
                for i in range(min(days_back, 10))
            ]
        
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            query = self.db.collection('wellnessData') \
                          .where('userId', '==', user_id) \
                          .where('timestamp', '>=', start_date) \
                          .order_by('timestamp', direction=firestore.Query.DESCENDING) \
                          .limit(50)
            
            docs = query.stream()
            history = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                if 'timestamp' in data:
                    data['timestamp'] = data['timestamp'].isoformat()
                history.append(data)
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving history: {str(e)}")
            return []

# ============================================================
# SPEECH EMOTION ANALYSIS
# ============================================================

class SpeechEmotionAnalyzer:
    def __init__(self, model_path='models'):
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
                logger.warning("Speech model not found, using mock predictions")
        except Exception as e:
            logger.error(f"Error loading speech model: {str(e)}")
            self.model = None
    
    def extract_features(self, audio_path):
        """Extract features from audio file"""
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=30)
            
            features = []
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)
            
            return np.array(features)
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return np.zeros(50)
    
    def predict_emotion(self, features):
        """Predict emotion from audio features"""
        if self.model is None:
            return self._mock_prediction()
        
        try:
            features = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            probabilities = self.model.predict_proba(features_scaled)[0]
            predicted_idx = np.argmax(probabilities)
            
            emotion_label = self.emotion_labels[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
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
        """Generate mock prediction"""
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

# ============================================================
# FACIAL EMOTION ANALYSIS
# ============================================================

class FacialEmotionAnalyzer:
    def __init__(self, model_path='models'):
        self.model_path = model_path
        self.model = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.face_cascade = None
        self.load_model()
        self.load_face_detector()
    
    def load_model(self):
        """Load pre-trained FER2013 model"""
        try:
            model_file = os.path.join(self.model_path, 'fer2013_model.h5')
            if os.path.exists(model_file):
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_file)
                logger.info("Facial emotion model loaded successfully")
            else:
                logger.warning("Facial model not found, using mock predictions")
        except Exception as e:
            logger.error(f"Error loading facial model: {str(e)}")
            self.model = None
    
    def load_face_detector(self):
        """Load OpenCV face cascade"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            logger.error(f"Error loading face detector: {str(e)}")
    
    def detect_faces(self, image):
        """Detect faces in image"""
        try:
            # Convert PIL to OpenCV format
            if hasattr(image, 'mode'):
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                image_cv = image
            
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                face_images = []
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    face_images.append(face_roi)
                return face_images
            else:
                # Return the whole image as a face if detector is not available
                return [gray]
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return []
    
    def preprocess_face(self, face_image):
        """Preprocess face for emotion recognition"""
        try:
            face_resized = cv2.resize(face_image, (48, 48))
            face_normalized = face_resized.astype('float32') / 255.0
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
            processed_face = self.preprocess_face(face_image)
            if processed_face is None:
                return self._mock_prediction()
            
            probabilities = self.model.predict(processed_face, verbose=0)[0]
            predicted_idx = np.argmax(probabilities)
            
            emotion_label = self.emotion_labels[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
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
        """Generate mock prediction"""
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

# ============================================================
# KEYSTROKE ANALYSIS
# ============================================================

class KeystrokeAnalyzer:
    def __init__(self, model_path='models'):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load keystroke analysis model"""
        try:
            model_file = os.path.join(self.model_path, 'keystroke_model.pkl')
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Keystroke model loaded successfully")
            else:
                logger.warning("Keystroke model not found, using heuristic analysis")
        except Exception as e:
            logger.error(f"Error loading keystroke model: {str(e)}")
            self.model = None
    
    def analyze_keystrokes(self, keystroke_data):
        """Analyze keystroke patterns for stress detection"""
        try:
            if not keystroke_data:
                return self._mock_response()
            
            # Extract features from keystroke data
            features = self.extract_keystroke_features(keystroke_data)
            
            if self.model:
                # Use trained model if available
                stress_score = self.model.predict([features])[0]
            else:
                # Use heuristic analysis
                stress_score = self.heuristic_stress_analysis(features)
            
            return {
                'score': float(stress_score),
                'wpm': features.get('wpm', 0),
                'pauseCount': features.get('pause_count', 0),
                'avgDwellTime': features.get('avg_dwell_time', 0),
                'mock': self.model is None
            }
        except Exception as e:
            logger.error(f"Keystroke analysis error: {str(e)}")
            return self._mock_response()
    
    def extract_keystroke_features(self, keystroke_data):
        """Extract features from keystroke timing data"""
        if not keystroke_data or len(keystroke_data) < 2:
            return {'wpm': 0, 'pause_count': 0, 'avg_dwell_time': 0}
        
        # Calculate typing speed (WPM)
        total_time = max(ks.get('timestamp', 0) for ks in keystroke_data) - \
                    min(ks.get('timestamp', 0) for ks in keystroke_data)
        
        char_count = len([ks for ks in keystroke_data if ks.get('type') == 'keydown'])
        wpm = (char_count / 5) / (total_time / 60000) if total_time > 0 else 0
        
        # Calculate pauses (gaps > 500ms)
        timestamps = sorted([ks.get('timestamp', 0) for ks in keystroke_data])
        pause_count = sum(1 for i in range(1, len(timestamps)) 
                         if timestamps[i] - timestamps[i-1] > 500)
        
        # Calculate average dwell time
        dwell_times = []
        keystroke_map = {}
        
        for ks in keystroke_data:
            key = ks.get('key', '')
            timestamp = ks.get('timestamp', 0)
            event_type = ks.get('type', '')
            
            if event_type == 'keydown':
                keystroke_map[key] = timestamp
            elif event_type == 'keyup' and key in keystroke_map:
                dwell_time = timestamp - keystroke_map[key]
                dwell_times.append(dwell_time)
                del keystroke_map[key]
        
        avg_dwell_time = sum(dwell_times) / len(dwell_times) if dwell_times else 0
        
        return {
            'wpm': wpm,
            'pause_count': pause_count,
            'avg_dwell_time': avg_dwell_time,
            'char_count': char_count,
            'total_time': total_time
        }
    
    def heuristic_stress_analysis(self, features):
        """Heuristic-based stress analysis"""
        stress_score = 0
        
        # Fast typing might indicate stress
        wpm = features.get('wpm', 0)
        if wpm > 80:
            stress_score += 20
        elif wpm < 20:
            stress_score += 15
        
        # Many pauses might indicate hesitation
        pause_count = features.get('pause_count', 0)
        if pause_count > 10:
            stress_score += 25
        
        # Very short or very long dwell times
        avg_dwell_time = features.get('avg_dwell_time', 0)
        if avg_dwell_time < 50 or avg_dwell_time > 200:
            stress_score += 20
        
        return min(100, max(0, stress_score))
    
    def _mock_response(self):
        """Generate mock keystroke analysis"""
        return {
            'score': random.uniform(20, 80),
            'wpm': random.uniform(30, 80),
            'pauseCount': random.randint(2, 15),
            'avgDwellTime': random.uniform(80, 200),
            'mock': True
        }

# ============================================================
# WELLNESS CALCULATOR
# ============================================================

class WellnessCalculator:
    def __init__(self):
        self.emotion_scores = {
            'happy': 90,
            'calm': 85,
            'neutral': 70,
            'surprised': 65,
            'content': 80,
            'relaxed': 85,
            'excited': 75,
            'sad': 40,
            'angry': 30,
            'fear': 25,
            'stressed': 20,
            'anxious': 25,
            'disgust': 35,
            'frustrated': 35,
            'worried': 30
        }
        
        self.weights = {
            'speech': 0.4,
            'facial': 0.4,
            'keystroke': 0.2
        }
    
    def calculate_index(self, speech_emotion, facial_emotion, keystroke_metrics):
        """Calculate overall wellness index"""
        try:
            scores = []
            weights = []
            
            # Calculate speech score
            if speech_emotion and speech_emotion.get('label'):
                speech_score = self._calculate_emotion_score(speech_emotion)
                scores.append(speech_score)
                weights.append(self.weights['speech'])
            
            # Calculate facial score
            if facial_emotion and facial_emotion.get('label') and \
               facial_emotion.get('detectedFaces', 1) > 0:
                facial_score = self._calculate_emotion_score(facial_emotion)
                scores.append(facial_score)
                weights.append(self.weights['facial'])
            
            # Calculate keystroke score
            if keystroke_metrics and keystroke_metrics.get('score') is not None:
                keystroke_score = 100 - keystroke_metrics['score']  # Invert stress score
                scores.append(keystroke_score)
                weights.append(self.weights['keystroke'])
            
            if not scores:
                return 50.0  # Neutral default
            
            # Normalize weights and calculate weighted average
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            wellness_index = sum(score * weight for score, weight in zip(scores, normalized_weights))
            
            return max(0.0, min(100.0, round(wellness_index, 1)))
            
        except Exception as e:
            logger.error(f"Wellness calculation error: {str(e)}")
            return 50.0
    
    def _calculate_emotion_score(self, emotion_data):
        """Calculate score from emotion data"""
        emotion = emotion_data['label'].lower()
        confidence = emotion_data.get('confidence', 0.5)
        base_score = self.emotion_scores.get(emotion, 50)
        
        # Adjust for confidence
        adjusted_score = base_score * confidence + 50 * (1 - confidence)
        return adjusted_score
    
    def get_recommendation(self, wellness_index):
        """Get wellness recommendations based on index"""
        if wellness_index >= 80:
            level = "excellent"
            message = "Your wellness levels look great! Keep up the positive habits."
            suggestions = [
                "Continue your current routine",
                "Share positive energy with others",
                "Take time to appreciate your good mood"
            ]
        elif wellness_index >= 65:
            level = "good"
            message = "You're doing well overall. Small improvements could help."
            suggestions = [
                "Take short breaks throughout the day",
                "Practice mindfulness or meditation",
                "Stay hydrated and get fresh air"
            ]
        elif wellness_index >= 45:
            level = "moderate"
            message = "Your wellness could use some attention."
            suggestions = [
                "Take regular breaks from work",
                "Try some light exercise or stretching",
                "Connect with friends or family",
                "Practice deep breathing exercises"
            ]
        elif wellness_index >= 30:
            level = "concerning"
            message = "Consider focusing on self-care activities."
            suggestions = [
                "Take a longer break or walk outside",
                "Practice relaxation techniques",
                "Talk to someone you trust",
                "Consider adjusting your workload"
            ]
        else:
            level = "low"
            message = "Please consider seeking support."
            suggestions = [
                "Take immediate steps to reduce stress",
                "Speak with a counselor or therapist",
                "Reach out to support networks",
                "Focus on basic self-care: sleep, nutrition, hydration"
            ]
        
        return {
            'level': level,
            'score': wellness_index,
            'message': message,
            'suggestions': suggestions,
            'urgency': 'high' if wellness_index < 30 else 'medium' if wellness_index < 50 else 'low'
        }

# ============================================================
# FLASK APPLICATION SETUP
# ============================================================

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    app.config['CORS_ORIGINS'] = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
    
    # CORS setup
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Initialize Firebase
    initialize_firebase()
    
    # Initialize components
    db_client = FirestoreClient()
    speech_analyzer = SpeechEmotionAnalyzer()
    facial_analyzer = FacialEmotionAnalyzer()
    keystroke_analyzer = KeystrokeAnalyzer()
    wellness_calc = WellnessCalculator()
    
    # ============================================================
    # HEALTH CHECK ENDPOINTS
    # ============================================================
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0',
            'services': ['speech', 'facial', 'keystroke', 'wellness'],
            'environment': os.getenv('ENVIRONMENT', 'development')
        })
    
    @app.route('/health', methods=['GET'])
    def service_health():
        return jsonify({'status': 'healthy', 'service': 'consolidated_app'})
    
    # ============================================================
    # SPEECH EMOTION ANALYSIS ENDPOINTS
    # ============================================================
    
    @app.route('/analyze-speech', methods=['POST'])
    def analyze_speech():
        """Analyze speech emotion from audio file"""
        try:
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'}), 400
            
            audio_file = request.files['audio']
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                audio_file.save(temp_file.name)
                features = speech_analyzer.extract_features(temp_file.name)
                result = speech_analyzer.predict_emotion(features)
                os.unlink(temp_file.name)
            
            logger.info(f"Speech analysis: {result['label']} ({result['confidence']:.3f})")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Speech analysis error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    # ============================================================
    # FACIAL EMOTION ANALYSIS ENDPOINTS
    # ============================================================
    
    @app.route('/analyze-face', methods=['POST'])
    def analyze_face():
        """Analyze facial emotion from image"""
        try:
            image = None
            
            if 'image' in request.files:
                image_file = request.files['image']
                image = Image.open(image_file.stream)
            elif request.json and 'image' in request.json:
                image_data = request.json['image']
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                return jsonify({'error': 'No image provided'}), 400
            
            faces = facial_analyzer.detect_faces(image)
            
            if not faces:
                return jsonify({
                    'label': 'no_face_detected',
                    'confidence': 0.0,
                    'detectedFaces': 0,
                    'message': 'No faces detected in the image'
                })
            
            result = facial_analyzer.predict_emotion(faces[0])
            result['detectedFaces'] = len(faces)
            
            logger.info(f"Facial analysis: {result['label']} ({result['confidence']:.3f})")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Facial analysis error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    # ============================================================
    # KEYSTROKE ANALYSIS ENDPOINTS
    # ============================================================
    
    @app.route('/analyze-keystroke', methods=['POST'])
    def analyze_keystroke():
        """Analyze keystroke patterns for stress detection"""
        try:
            keystroke_data = request.json.get('keystrokes', [])
            result = keystroke_analyzer.analyze_keystrokes(keystroke_data)
            
            logger.info(f"Keystroke analysis: stress score {result['score']:.1f}")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Keystroke analysis error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    # ============================================================
    # MAIN WELLNESS ANALYSIS ENDPOINTS
    # ============================================================
    
    @app.route('/api/wellness/analyze', methods=['POST'])
    @require_auth
    def analyze_wellness(user_id):
        """Main wellness analysis endpoint"""
        try:
            # Initialize analysis results
            analysis_results = {
                'speechEmotion': None,
                'facialEmotion': None,
                'keystrokeMetrics': None,
                'wellnessIndex': 0,
                'timestamp': datetime.utcnow().isoformat(),
                'sessionId': request.form.get('sessionId', 'default')
            }
            
            # Analyze speech emotion
            if 'audio' in request.files:
                logger.info("Processing audio for speech emotion analysis")
                audio_file = request.files['audio']
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    audio_file.save(temp_file.name)
                    features = speech_analyzer.extract_features(temp_file.name)
                    speech_result = speech_analyzer.predict_emotion(features)
                    analysis_results['speechEmotion'] = speech_result
                    os.unlink(temp_file.name)
            
            # Analyze facial emotion
            if 'image' in request.files:
                logger.info("Processing image for facial emotion analysis")
                image_file = request.files['image']
                image = Image.open(image_file.stream)
                faces = facial_analyzer.detect_faces(image)
                if faces:
                    facial_result = facial_analyzer.predict_emotion(faces[0])
                    facial_result['detectedFaces'] = len(faces)
                    analysis_results['facialEmotion'] = facial_result
            elif request.json and 'imageData' in request.json:
                logger.info("Processing base64 image for facial emotion analysis")
                image_data = request.json['imageData']
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                faces = facial_analyzer.detect_faces(image)
                if faces:
                    facial_result = facial_analyzer.predict_emotion(faces[0])
                    facial_result['detectedFaces'] = len(faces)
                    analysis_results['facialEmotion'] = facial_result
            
            # Analyze keystroke patterns
            keystroke_data = None
            if request.json and 'keystrokes' in request.json:
                keystroke_data = request.json['keystrokes']
            elif request.form.get('keystrokes'):
                keystroke_data = json.loads(request.form.get('keystrokes'))
            
            if keystroke_data:
                logger.info("Processing keystroke data")
                keystroke_result = keystroke_analyzer.analyze_keystrokes(keystroke_data)
                analysis_results['keystrokeMetrics'] = keystroke_result
            
            # Calculate wellness index
            wellness_index = wellness_calc.calculate_index(
                analysis_results['speechEmotion'],
                analysis_results['facialEmotion'],
                analysis_results['keystrokeMetrics']
            )
            analysis_results['wellnessIndex'] = wellness_index
            
            # Store results in database
            doc_data = {
                'speechEmotion': analysis_results['speechEmotion'],
                'facialEmotion': analysis_results['facialEmotion'],
                'keystrokeMetrics': analysis_results['keystrokeMetrics'],
                'wellnessIndex': wellness_index,
                'sessionId': analysis_results['sessionId'],
                'metadata': {
                    'userAgent': request.headers.get('User-Agent'),
                    'ip': request.remote_addr
                }
            }
            
            doc_id = db_client.add_wellness_data(user_id, doc_data)
            analysis_results['documentId'] = doc_id
            
            logger.info(f"Wellness analysis completed for user {user_id}, wellness index: {wellness_index}")
            
            return jsonify({
                'success': True,
                'data': analysis_results
            })
            
        except Exception as e:
            logger.error(f"Wellness analysis error: {str(e)}")
            return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500
    
    @app.route('/api/wellness/quick-check', methods=['POST'])
    @require_auth
    def quick_wellness_check(user_id):
        """Quick wellness check with minimal data"""
        try:
            data = request.json or {}
            mood_rating = data.get('moodRating', 5)  # 1-10 scale
            energy_level = data.get('energyLevel', 5)  # 1-10 scale
            stress_level = data.get('stressLevel', 5)  # 1-10 scale
            
            # Calculate simple wellness index
            wellness_index = ((mood_rating + energy_level + (10 - stress_level)) / 3) * 10
            
            # Store simplified data
            doc_data = {
                'quickCheck': {
                    'moodRating': mood_rating,
                    'energyLevel': energy_level,
                    'stressLevel': stress_level
                },
                'wellnessIndex': wellness_index,
                'sessionId': data.get('sessionId', 'quick-check'),
                'type': 'quick_check'
            }
            
            db_client.add_wellness_data(user_id, doc_data)
            
            return jsonify({
                'success': True,
                'data': {
                    'wellnessIndex': wellness_index,
                    'recommendation': wellness_calc.get_recommendation(wellness_index),
                    'timestamp': datetime.utcnow().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Quick check error: {str(e)}")
            return jsonify({'error': 'Quick check failed'}), 500
    
    # ============================================================
    # HISTORY AND ANALYTICS ENDPOINTS
    # ============================================================
    
    @app.route('/api/history/wellness', methods=['GET'])
    @require_auth
    def get_wellness_history(user_id):
        """Get user's wellness history"""
        try:
            days_back = int(request.args.get('days', 7))
            history = db_client.get_user_wellness_history(user_id, days_back)
            
            return jsonify({
                'success': True,
                'data': history,
                'count': len(history),
                'period_days': days_back
            })
            
        except Exception as e:
            logger.error(f"History retrieval error: {str(e)}")
            return jsonify({'error': 'Failed to retrieve history'}), 500
    
    @app.route('/api/history/stats', methods=['GET'])
    @require_auth
    def get_wellness_stats(user_id):
        """Get user wellness statistics"""
        try:
            days_back = int(request.args.get('days', 30))
            history = db_client.get_user_wellness_history(user_id, days_back)
            
            if not history:
                return jsonify({
                    'success': True,
                    'data': {'message': 'No data available', 'stats': None}
                })
            
            # Extract wellness indices
            indices = [entry.get('wellnessIndex', 0) for entry in history if entry.get('wellnessIndex')]
            
            if not indices:
                return jsonify({
                    'success': True,
                    'data': {'message': 'No wellness index data', 'stats': None}
                })
            
            # Calculate statistics
            stats = {
                'average': sum(indices) / len(indices),
                'minimum': min(indices),
                'maximum': max(indices),
                'latest': indices[0] if indices else 0,
                'data_points': len(indices),
                'period_days': days_back
            }
            
            # Calculate trend
            if len(indices) >= 4:
                midpoint = len(indices) // 2
                recent_avg = sum(indices[:midpoint]) / midpoint
                older_avg = sum(indices[midpoint:]) / (len(indices) - midpoint)
                trend_diff = recent_avg - older_avg
                
                if abs(trend_diff) < 3:
                    stats['trend'] = 'stable'
                elif trend_diff > 0:
                    stats['trend'] = 'improving' if trend_diff < 10 else 'significantly_improving'
                else:
                    stats['trend'] = 'declining' if abs(trend_diff) < 10 else 'significantly_declining'
            else:
                stats['trend'] = 'insufficient_data'
            
            return jsonify({
                'success': True,
                'data': {'stats': stats}
            })
            
        except Exception as e:
            logger.error(f"Stats calculation error: {str(e)}")
            return jsonify({'error': 'Failed to calculate stats'}), 500
    
    # ============================================================
    # AUTHENTICATION ENDPOINTS
    # ============================================================
    
    @app.route('/api/auth/user', methods=['GET'])
    @require_auth
    def get_user_info(user_id):
        """Get current user information"""
        try:
            if os.getenv('ENVIRONMENT', 'development') == 'development':
                # Return mock user info for development
                user_info = {
                    'uid': user_id,
                    'email': 'demo@example.com',
                    'displayName': 'Demo User',
                    'emailVerified': True
                }
            else:
                user_record = auth.get_user(user_id)
                user_info = {
                    'uid': user_record.uid,
                    'email': user_record.email,
                    'displayName': user_record.display_name,
                    'photoURL': user_record.photo_url,
                    'emailVerified': user_record.email_verified,
                    'creationTime': user_record.user_metadata.creation_timestamp,
                    'lastSignInTime': user_record.user_metadata.last_sign_in_timestamp
                }
            
            return jsonify({
                'success': True,
                'data': user_info
            })
            
        except Exception as e:
            logger.error(f"Get user info error: {str(e)}")
            return jsonify({'error': 'Failed to get user info'}), 500
    
    # ============================================================
    # MOCK ENDPOINTS FOR TESTING
    # ============================================================
    
    @app.route('/analyze-speech-mock', methods=['POST'])
    def analyze_speech_mock():
        """Mock speech analysis endpoint for testing"""
        emotions = ['happy', 'neutral', 'calm', 'sad', 'stressed', 'excited']
        emotion = random.choice(emotions)
        confidence = random.uniform(0.6, 0.9)
        
        return jsonify({
            'label': emotion,
            'confidence': confidence,
            'mock': True,
            'all_probabilities': {
                e: random.uniform(0.05, 0.25) if e != emotion else confidence 
                for e in emotions
            }
        })
    
    @app.route('/analyze-face-mock', methods=['POST'])
    def analyze_face_mock():
        """Mock facial analysis endpoint for testing"""
        emotions = ['happy', 'neutral', 'surprised', 'sad', 'angry', 'calm']
        emotion = random.choice(emotions)
        confidence = random.uniform(0.5, 0.9)
        
        return jsonify({
            'label': emotion,
            'confidence': confidence,
            'detectedFaces': 1,
            'mock': True,
            'all_probabilities': {
                e: random.uniform(0.05, 0.25) if e != emotion else confidence 
                for e in emotions
            }
        })
    
    @app.route('/analyze-keystroke-mock', methods=['POST'])
    def analyze_keystroke_mock():
        """Mock keystroke analysis endpoint for testing"""
        return jsonify({
            'score': random.uniform(20, 80),
            'wpm': random.uniform(30, 80),
            'pauseCount': random.randint(2, 15),
            'avgDwellTime': random.uniform(80, 200),
            'mock': True
        })
    
    # ============================================================
    # ERROR HANDLERS
    # ============================================================
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(413)
    def file_too_large(error):
        return jsonify({'error': 'File too large'}), 413
    
    # ============================================================
    # UTILITY ENDPOINTS
    # ============================================================
    
    @app.route('/api/wellness/recommendation', methods=['POST'])
    def get_wellness_recommendation():
        """Get wellness recommendation for a given index"""
        try:
            data = request.json or {}
            wellness_index = data.get('wellnessIndex', 50)
            
            recommendation = wellness_calc.get_recommendation(wellness_index)
            
            return jsonify({
                'success': True,
                'data': recommendation
            })
            
        except Exception as e:
            logger.error(f"Recommendation error: {str(e)}")
            return jsonify({'error': 'Failed to get recommendation'}), 500
    
    @app.route('/api/config', methods=['GET'])
    def get_app_config():
        """Get application configuration for frontend"""
        return jsonify({
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'version': '1.0.0',
            'features': {
                'speech_analysis': True,
                'facial_analysis': True,
                'keystroke_analysis': True,
                'firebase_auth': True,
                'history_tracking': True
            },
            'mock_mode': os.getenv('USE_MOCK_AI', 'false').lower() == 'true'
        })
    
    return app

# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == '__main__':
    app = create_app()
    
    # Get configuration from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('ENVIRONMENT', 'development') == 'development'
    
    logger.info(f"Starting Wellness Monitoring System on {host}:{port}")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )