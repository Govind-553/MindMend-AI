from flask import Blueprint, request, jsonify
import logging
from datetime import datetime

from services.ai_service_client import AIServiceClient
from services.wellness_calculator import WellnessCalculator
from services.firebase_service import db
from utils.decorators import require_auth
from utils.validators import validate_wellness_request

wellness_bp = Blueprint('wellness', __name__)
ai_client = AIServiceClient()
wellness_calc = WellnessCalculator()
logger = logging.getLogger(__name__)

@wellness_bp.route('/analyze', methods=['POST'])
@require_auth
def analyze_wellness(user_id):
    """Main wellness analysis endpoint"""
    try:
        # Validate request
        validation_result = validate_wellness_request(request)
        if not validation_result['valid']:
            return jsonify({'error': validation_result['message']}), 400
        
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
            speech_result = ai_client.analyze_speech(request.files['audio'])
            analysis_results['speechEmotion'] = speech_result
        
        # Analyze facial emotion
        if 'image' in request.files:
            logger.info("Processing image for facial emotion analysis")
            facial_result = ai_client.analyze_facial_emotion(request.files['image'])
            analysis_results['facialEmotion'] = facial_result
        elif request.json and 'imageData' in request.json:
            logger.info("Processing base64 image for facial emotion analysis")
            facial_result = ai_client.analyze_facial_emotion_base64(request.json['imageData'])
            analysis_results['facialEmotion'] = facial_result
        
        # Analyze keystroke patterns
        keystroke_data = None
        if request.json and 'keystrokes' in request.json:
            keystroke_data = request.json['keystrokes']
        elif request.form.get('keystrokes'):
            import json
            keystroke_data = json.loads(request.form.get('keystrokes'))
        
        if keystroke_data:
            logger.info("Processing keystroke data")
            keystroke_result = ai_client.analyze_keystrokes(keystroke_data)
            analysis_results['keystrokeMetrics'] = keystroke_result
        
        # Calculate wellness index
        wellness_index = wellness_calc.calculate_index(
            analysis_results['speechEmotion'],
            analysis_results['facialEmotion'],
            analysis_results['keystrokeMetrics']
        )
        analysis_results['wellnessIndex'] = wellness_index
        
        # Store results in Firestore
        doc_data = {
            'userId': user_id,
            'timestamp': datetime.utcnow(),
            'speechEmotion': analysis_results['speechEmotion'],
            'facialEmotion': analysis_results['facialEmotion'],
            'keystrokeMetrics': analysis_results['keystrokeMetrics'],
            'wellnessIndex': wellness_index,
            'sessionId': analysis_results['sessionId'],
            'metadata': {
                'userAgent': request.headers.get('User-Agent'),
                'ip': request.remote_addr,
                'processingTime': None  # Will be calculated
            }
        }
        
        # Add to database
        doc_ref = db.collection('wellnessData').add(doc_data)
        analysis_results['documentId'] = doc_ref[1].id
        
        logger.info(f"Wellness analysis completed for user {user_id}, wellness index: {wellness_index}")
        
        return jsonify({
            'success': True,
            'data': analysis_results
        })
        
    except Exception as e:
        logger.error(f"Wellness analysis error: {str(e)}")
        return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500

@wellness_bp.route('/quick-check', methods=['POST'])
@require_auth
def quick_wellness_check(user_id):
    """Quick wellness check with minimal data"""
    try:
        # Simple mood check based on user input
        data = request.json or {}
        mood_rating = data.get('moodRating', 5)  # 1-10 scale
        energy_level = data.get('energyLevel', 5)  # 1-10 scale
        stress_level = data.get('stressLevel', 5)  # 1-10 scale
        
        # Calculate simple wellness index
        wellness_index = ((mood_rating + energy_level + (10 - stress_level)) / 3) * 10
        
        # Store simplified data
        doc_data = {
            'userId': user_id,
            'timestamp': datetime.utcnow(),
            'quickCheck': {
                'moodRating': mood_rating,
                'energyLevel': energy_level,
                'stressLevel': stress_level
            },
            'wellnessIndex': wellness_index,
            'sessionId': data.get('sessionId', 'quick-check'),
            'type': 'quick_check'
        }
        
        db.collection('wellnessData').add(doc_data)
        
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