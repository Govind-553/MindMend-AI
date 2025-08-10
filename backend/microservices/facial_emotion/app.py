from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import logging
from models.facial_emotion_analyzer import FacialEmotionAnalyzer
from utils.image_processor import ImageProcessor

app = Flask(__name__)
CORS(app)

# Initialize components
analyzer = FacialEmotionAnalyzer()
image_processor = ImageProcessor()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'facial_emotion'})

@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    """Analyze facial emotion from image"""
    try:
        # Get image data
        image = None
        
        if 'image' in request.files:
            # Handle file upload
            image_file = request.files['image']
            image = Image.open(image_file.stream)
        elif request.json and 'image' in request.json:
            # Handle base64 image
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Process image
        faces = image_processor.detect_faces(image)
        
        if not faces:
            return jsonify({
                'label': 'no_face_detected',
                'confidence': 0.0,
                'detectedFaces': 0,
                'message': 'No faces detected in the image'
            })
        
        # Analyze emotion for the first detected face
        face_roi = faces[0]
        result = analyzer.predict_emotion(face_roi)
        result['detectedFaces'] = len(faces)
        
        logger.info(f"Facial analysis result: {result['label']} ({result['confidence']:.3f}), faces: {len(faces)}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Facial analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-face-mock', methods=['POST'])
def analyze_face_mock():
    """Mock endpoint for demo reliability"""
    import random
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)