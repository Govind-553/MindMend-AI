from marshmallow import Schema, fields, post_load
from datetime import datetime
from typing import Dict, Any

class KeystrokeMetricsSchema(Schema):
    """Schema for keystroke analysis metrics."""
    score = fields.Float(required=True)
    wpm = fields.Float()
    pauseCount = fields.Integer()
    avgDwellTime = fields.Float()
    mock = fields.Boolean(missing=False)

class EmotionAnalysisSchema(Schema):
    """Base schema for emotion analysis results."""
    label = fields.String(required=True)
    confidence = fields.Float(required=True)
    raw_emotion = fields.String()
    all_probabilities = fields.Dict(keys=fields.String(), values=fields.Float())
    mock = fields.Boolean(missing=False)

class FacialEmotionSchema(EmotionAnalysisSchema):
    """Specific schema for facial emotion analysis."""
    detectedFaces = fields.Integer()

class SpeechEmotionSchema(EmotionAnalysisSchema):
    """Specific schema for speech emotion analysis."""
    pass

class WellnessDataSchema(Schema):
    """Schema for the main wellness data entry."""
    userId = fields.String(required=True)
    timestamp = fields.DateTime(format='iso', missing=datetime.utcnow)
    wellnessIndex = fields.Float()
    sessionId = fields.String()
    type = fields.String(missing="full_analysis")
    
    # Nested schemas for analysis results
    speechEmotion = fields.Nested(SpeechEmotionSchema, allow_none=True)
    facialEmotion = fields.Nested(FacialEmotionSchema, allow_none=True)
    keystrokeMetrics = fields.Nested(KeystrokeMetricsSchema, allow_none=True)
    
    @post_load
    def make_wellness_data(self, data, **kwargs) -> Dict[str, Any]:
        """Convert loaded data back into a dictionary for use."""
        return data

# Quick check schema (for the simpler endpoint)
class QuickCheckSchema(Schema):
    moodRating = fields.Integer(required=True)
    energyLevel = fields.Integer(required=True)
    stressLevel = fields.Integer(required=True)
    
class QuickWellnessDataSchema(Schema):
    userId = fields.String(required=True)
    timestamp = fields.DateTime(format='iso', missing=datetime.utcnow)
    wellnessIndex = fields.Float()
    sessionId = fields.String(missing="quick-check")
    type = fields.String(missing="quick_check")
    quickCheck = fields.Nested(QuickCheckSchema, required=True)