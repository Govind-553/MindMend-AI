import logging
import os
import pickle
import random
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class KeystrokeAnalyzer:
    def __init__(self, model_path='../../models'):
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
    
    def analyze_keystrokes(self, keystroke_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze keystroke patterns for stress detection"""
        try:
            if not keystroke_data:
                return self._mock_response()
            
            features = self.extract_keystroke_features(keystroke_data)
            
            if self.model:
                # Use trained model if available
                stress_score = self.model.predict([list(features.values())])[0]
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
    
    def extract_keystroke_features(self, keystroke_data: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        }
    
    def heuristic_stress_analysis(self, features: Dict[str, Any]) -> float:
        """Heuristic-based stress analysis"""
        stress_score = 0
        
        wpm = features.get('wpm', 0)
        if wpm > 80: stress_score += 20
        elif wpm < 20: stress_score += 15
        
        pause_count = features.get('pause_count', 0)
        if pause_count > 10: stress_score += 25
        
        avg_dwell_time = features.get('avg_dwell_time', 0)
        if avg_dwell_time < 50 or avg_dwell_time > 200: stress_score += 20
        
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