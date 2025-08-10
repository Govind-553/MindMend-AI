import requests
import os
import logging
from typing import Optional, Dict, Any
import io
import base64

logger = logging.getLogger(__name__)

class AIServiceClient:
    def __init__(self):
        self.speech_url = os.getenv('SPEECH_SERVICE_URL', 'http://localhost:5001')
        self.facial_url = os.getenv('FACIAL_SERVICE_URL', 'http://localhost:5002')
        self.keystroke_url = os.getenv('KEYSTROKE_SERVICE_URL', 'http://localhost:5003')
        self.timeout = int(os.getenv('MODEL_TIMEOUT', 10))
        self.use_mock = os.getenv('USE_MOCK_AI', 'false').lower() == 'true'
    
    def analyze_speech(self, audio_file) -> Optional[Dict[str, Any]]:
        """Analyze speech emotion from audio file"""
        try:
            if self.use_mock:
                return self._mock_speech_response()
            
            # Reset file pointer
            audio_file.seek(0)
            
            response = requests.post(
                f"{self.speech_url}/analyze-speech",
                files={'audio': audio_file},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Speech analysis successful: {result.get('label', 'unknown')} ({result.get('confidence', 0):.2f})")
                return result
            else:
                logger.warning(f"Speech service returned {response.status_code}, using mock")
                return self._mock_speech_response()
                
        except requests.RequestException as e:
            logger.error(f"Speech service request failed: {str(e)}, using mock")
            return self._mock_speech_response()
        except Exception as e:
            logger.error(f"Speech analysis error: {str(e)}")
            return None
    
    def analyze_facial_emotion(self, image_file) -> Optional[Dict[str, Any]]:
        """Analyze facial emotion from image file"""
        try:
            if self.use_mock:
                return self._mock_facial_response()
            
            # Reset file pointer
            image_file.seek(0)
            
            response = requests.post(
                f"{self.facial_url}/analyze-face",
                files={'image': image_file},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Facial analysis successful: {result.get('label', 'unknown')} ({result.get('confidence', 0):.2f})")
                return result
            else:
                logger.warning(f"Facial service returned {response.status_code}, using mock")
                return self._mock_facial_response()
                
        except requests.RequestException as e:
            logger.error(f"Facial service request failed: {str(e)}, using mock")
            return self._mock_facial_response()
        except Exception as e:
            logger.error(f"Facial analysis error: {str(e)}")
            return None
    
    def analyze_facial_emotion_base64(self, image_data: str) -> Optional[Dict[str, Any]]:
        """Analyze facial emotion from base64 image data"""
        try:
            if self.use_mock:
                return self._mock_facial_response()
            
            response = requests.post(
                f"{self.facial_url}/analyze-face",
                json={'image': image_data},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Facial analysis (base64) successful: {result.get('label', 'unknown')}")
                return result
            else:
                logger.warning(f"Facial service returned {response.status_code}, using mock")
                return self._mock_facial_response()
                
        except requests.RequestException as e:
            logger.error(f"Facial service request failed: {str(e)}, using mock")
            return self._mock_facial_response()
        except Exception as e:
            logger.error(f"Facial analysis error: {str(e)}")
            return None
    
    def analyze_keystrokes(self, keystroke_data: list) -> Optional[Dict[str, Any]]:
        """Analyze keystroke patterns for stress detection"""
        try:
            if self.use_mock:
                return self._mock_keystroke_response()
            
            response = requests.post(
                f"{self.keystroke_url}/analyze-keystroke",
                json={'keystrokes': keystroke_data},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Keystroke analysis successful: stress score {result.get('score', 0):.1f}")
                return result
            else:
                logger.warning(f"Keystroke service returned {response.status_code}, using mock")
                return self._mock_keystroke_response()
                
        except requests.RequestException as e:
            logger.error(f"Keystroke service request failed: {str(e)}, using mock")
            return self._mock_keystroke_response()
        except Exception as e:
            logger.error(f"Keystroke analysis error: {str(e)}")
            return None
    
    # Mock responses for demo reliability
    def _mock_speech_response(self):
        import random
        emotions = ['happy', 'neutral', 'calm', 'sad', 'stressed']
        emotion = random.choice(emotions)
        confidence = random.uniform(0.6, 0.9)
        
        return {
            'label': emotion,
            'confidence': confidence,
            'mock': True,
            'all_probabilities': {e: random.uniform(0.1, 0.3) if e != emotion else confidence for e in emotions}
        }
    
    def _mock_facial_response(self):
        import random
        emotions = ['happy', 'neutral', 'surprised', 'sad', 'angry']
        emotion = random.choice(emotions)
        confidence = random.uniform(0.5, 0.85)
        
        return {
            'label': emotion,
            'confidence': confidence,
            'detectedFaces': 1,
            'mock': True,
            'all_probabilities': {e: random.uniform(0.1, 0.3) if e != emotion else confidence for e in emotions}
        }
    
    def _mock_keystroke_response(self):
        import random
        return {
            'score': random.uniform(20, 80),
            'wpm': random.uniform(30, 80),
            'pauseCount': random.randint(2, 15),
            'avgDwellTime': random.uniform(80, 200),
            'mock': True
        }