import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class WellnessCalculator:
    def __init__(self):
        # Emotion to wellness score mapping (0-100 scale)
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
        
        # Weights for different components
        self.weights = {
            'speech': 0.4,
            'facial': 0.4,
            'keystroke': 0.2
        }
    
    def calculate_index(self, 
                       speech_emotion: Optional[Dict[str, Any]], 
                       facial_emotion: Optional[Dict[str, Any]], 
                       keystroke_metrics: Optional[Dict[str, Any]]) -> float:
        """Calculate overall wellness index from all available data"""
        
        try:
            # Calculate individual component scores
            speech_score = self._calculate_speech_score(speech_emotion)
            facial_score = self._calculate_facial_score(facial_emotion)
            keystroke_score = self._calculate_keystroke_score(keystroke_metrics)
            
            # Adjust weights based on available data
            available_components = []
            scores = []
            weights = []
            
            if speech_score is not None:
                available_components.append('speech')
                scores.append(speech_score)
                weights.append(self.weights['speech'])
            
            if facial_score is not None:
                available_components.append('facial')
                scores.append(facial_score)
                weights.append(self.weights['facial'])
            
            if keystroke_score is not None:
                available_components.append('keystroke')
                scores.append(keystroke_score)
                weights.append(self.weights['keystroke'])
            
            if not scores:
                logger.warning("No valid emotion data available for wellness calculation")
                return 50.0  # Neutral default
            
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Calculate weighted average
            wellness_index = sum(score * weight for score, weight in zip(scores, normalized_weights))
            
            # Apply confidence adjustments
            wellness_index = self._apply_confidence_adjustments(
                wellness_index, speech_emotion, facial_emotion, keystroke_metrics
            )
            
            # Clamp to valid range
            wellness_index = max(0.0, min(100.0, wellness_index))
            
            logger.info(f"Wellness calculation: components={available_components}, "
                       f"scores={[f'{s:.1f}' for s in scores]}, "
                       f"final_index={wellness_index:.1f}")
            
            return round(wellness_index, 1)
            
        except Exception as e:
            logger.error(f"Wellness calculation error: {str(e)}")
            return 50.0  # Safe default
    
    def _calculate_speech_score(self, speech_emotion: Optional[Dict[str, Any]]) -> Optional[float]:
        """Calculate wellness score from speech emotion"""
        if not speech_emotion or not speech_emotion.get('label'):
            return None
        
        emotion = speech_emotion['label'].lower()
        confidence = speech_emotion.get('confidence', 0.5)
        
        # Get base score for this emotion
        base_score = self.emotion_scores.get(emotion, 50)  # Default to neutral
        
        # Adjust for confidence (low confidence -> pull toward neutral)
        adjusted_score = base_score * confidence + 50 * (1 - confidence)
        
        return adjusted_score
    
    def _calculate_facial_score(self, facial_emotion: Optional[Dict[str, Any]]) -> Optional[float]:
        """Calculate wellness score from facial emotion"""
        if not facial_emotion or not facial_emotion.get('label'):
            return None
        
        # Skip if no faces detected
        if facial_emotion.get('detectedFaces', 0) == 0:
            return None
        
        emotion = facial_emotion['label'].lower()
        confidence = facial_emotion.get('confidence', 0.5)
        
        # Get base score for this emotion
        base_score = self.emotion_scores.get(emotion, 50)
        
        # Adjust for confidence
        adjusted_score = base_score * confidence + 50 * (1 - confidence)
        
        return adjusted_score
    
    def _calculate_keystroke_score(self, keystroke_metrics: Optional[Dict[str, Any]]) -> Optional[float]:
        """Calculate wellness score from keystroke analysis"""
        if not keystroke_metrics or keystroke_metrics.get('score') is None:
            return None
        
        stress_score = keystroke_metrics['score']
        
        # Convert stress score (0-100) to wellness score (100-0)
        # High stress = low wellness
        wellness_score = 100 - stress_score
        
        # Apply additional factors
        wpm = keystroke_metrics.get('wpm', 50)
        pause_count = keystroke_metrics.get('pauseCount', 5)
        
        # Very fast typing might indicate stress
        if wpm > 80:
            wellness_score *= 0.9
        
        # Too many pauses might indicate hesitation/stress
        if pause_count > 10:
            wellness_score *= 0.95
        
        return max(0, min(100, wellness_score))
    
    def _apply_confidence_adjustments(self, 
                                    wellness_index: float,
                                    speech_emotion: Optional[Dict[str, Any]], 
                                    facial_emotion: Optional[Dict[str, Any]], 
                                    keystroke_metrics: Optional[Dict[str, Any]]) -> float:
        """Apply confidence-based adjustments to final score"""
        
        # Calculate average confidence across all components
        confidences = []
        
        if speech_emotion and speech_emotion.get('confidence'):
            confidences.append(speech_emotion['confidence'])
        
        if facial_emotion and facial_emotion.get('confidence'):
            confidences.append(facial_emotion['confidence'])
        
        # Keystroke analysis doesn't have confidence, but we can infer it
        if keystroke_metrics and not keystroke_metrics.get('mock', False):
            # Assume reasonable confidence for keystroke data
            confidences.append(0.7)
        
        if confidences:
            avg_confidence = np.mean(confidences)
            
            # If overall confidence is low, pull toward neutral
            if avg_confidence < 0.6:
                adjustment_factor = 0.7 + (avg_confidence * 0.3)
                wellness_index = wellness_index * adjustment_factor + 50 * (1 - adjustment_factor)
        
        return wellness_index
    
    def get_recommendation(self, wellness_index: float) -> Dict[str, Any]:
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
            message = "Your wellness could use some attention. Consider taking some positive steps."
            suggestions = [
                "Take regular breaks from work",
                "Try some light exercise or stretching",
                "Connect with friends or family",
                "Practice deep breathing exercises"
            ]
        elif wellness_index >= 30:
            level = "concerning"
            message = "Your wellness levels suggest you may be experiencing stress. Consider self-care activities."
            suggestions = [
                "Take a longer break or walk outside",
                "Practice relaxation techniques",
                "Talk to someone you trust",
                "Consider adjusting your workload",
                "Ensure you're getting enough sleep"
            ]
        else:
            level = "low"
            message = "Your wellness indicators suggest significant stress. Please consider seeking support."
            suggestions = [
                "Take immediate steps to reduce stress",
                "Speak with a counselor or therapist",
                "Reach out to support networks",
                "Consider taking time off if possible",
                "Focus on basic self-care: sleep, nutrition, hydration"
            ]
        
        return {
            'level': level,
            'score': wellness_index,
            'message': message,
            'suggestions': suggestions,
            'urgency': 'high' if wellness_index < 30 else 'medium' if wellness_index < 50 else 'low'
        }
    
    def get_trend_analysis(self, historical_data: list) -> Dict[str, Any]:
        """Analyze wellness trends from historical data"""
        if len(historical_data) < 2:
            return {'trend': 'insufficient_data', 'message': 'Need more data points for trend analysis'}
        
        # Extract wellness indices with timestamps
        indices = []
        timestamps = []
        
        for entry in historical_data:
            if 'wellnessIndex' in entry and 'timestamp' in entry:
                indices.append(entry['wellnessIndex'])
                timestamps.append(entry['timestamp'])
        
        if len(indices) < 2:
            return {'trend': 'insufficient_data', 'message': 'Need more wellness data for trend analysis'}
        
        # Calculate trend
        recent_avg = np.mean(indices[:len(indices)//2]) if len(indices) >= 4 else indices[0]
        older_avg = np.mean(indices[len(indices)//2:]) if len(indices) >= 4 else indices[-1]
        
        trend_direction = recent_avg - older_avg
        trend_magnitude = abs(trend_direction)
        
        if trend_magnitude < 5:
            trend = 'stable'
            message = 'Your wellness levels have been relatively stable.'
        elif trend_direction > 0:
            if trend_magnitude > 15:
                trend = 'improving_significantly'
                message = 'Great! Your wellness has been improving significantly.'
            else:
                trend = 'improving'
                message = 'Your wellness has been trending upward.'
        else:
            if trend_magnitude > 15:
                trend = 'declining_significantly'
                message = 'Your wellness has been declining. Consider focusing on self-care.'
            else:
                trend = 'declining'
                message = 'Your wellness has been trending downward slightly.'
        
        return {
            'trend': trend,
            'direction': trend_direction,
            'magnitude': trend_magnitude,
            'message': message,
            'current_avg': recent_avg,
            'previous_avg': older_avg,
            'data_points': len(indices)
        }