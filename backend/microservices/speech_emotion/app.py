from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import tempfile
import librosa
import numpy as np
import pickle
from models.speech_emotion_analyzer import SpeechEmotionAnalyzer
from utils.audio_processor import AudioProcessor

app = Flask(__name__)
CORS(app)

# Initialize components
analyzer = SpeechEmotionAnalyzer()
audio_processor = AudioProcessor()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'speech_emotion'})

@app.route('/analyze-speech', methods=['POST'])
def analyze_speech():
    """Analyze speech emotion from audio file"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            
            # Process audio
            features = audio_processor.extract_features(temp_file.name)
            
            # Analyze emotion
            result = analyzer.predict_emotion(features)
            
            # Clean up
            os.unlink(temp_file.name)
        
        logger.info(f"Speech analysis result: {result['label']} ({result['confidence']:.3f})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Speech analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-speech-mock', methods=['POST'])
def analyze_speech_mock():
    """Mock endpoint for demo reliability"""
    import random
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)