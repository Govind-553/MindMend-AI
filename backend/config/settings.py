import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # General app settings
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
    
    # AI service URLs
    SPEECH_SERVICE_URL = os.getenv('SPEECH_SERVICE_URL', 'http://localhost:5001')
    FACIAL_SERVICE_URL = os.getenv('FACIAL_SERVICE_URL', 'http://localhost:5002')
    KEYSTROKE_SERVICE_URL = os.getenv('KEYSTROKE_SERVICE_URL', 'http://localhost:5003')
    MODEL_TIMEOUT = int(os.getenv('MODEL_TIMEOUT', 10))
    USE_MOCK_AI = os.getenv('USE_MOCK_AI', 'false').lower() == 'true'
    
    # Firebase settings
    FIREBASE_CREDENTIAL_PATH = os.getenv('FIREBASE_CREDENTIAL_PATH')
    
settings = Settings()