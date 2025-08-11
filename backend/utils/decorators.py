from functools import wraps
from flask import request, jsonify
from firebase_admin import auth
import os
import logging

logger = logging.getLogger(__name__)

def verify_firebase_token(id_token):
    """Verify Firebase ID token and return user ID"""
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token.get('uid')
    except auth.InvalidIdTokenError:
        logger.error("Invalid ID token")
        return None
    except auth.ExpiredIdTokenError:
        logger.error("Expired ID token")
        return None
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