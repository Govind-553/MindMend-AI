import os
import logging
from firebase_admin import credentials, initialize_app, _apps
from .settings import settings

logger = logging.getLogger(__name__)

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        # Check if Firebase is already initialized
        if _apps:
            logger.info("Firebase already initialized")
            return
        
        # Initialize with service account key or default credentials
        cred_path = settings.FIREBASE_CREDENTIAL_PATH
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