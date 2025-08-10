from firebase_admin import auth
import logging
from functools import wraps
from flask import request, jsonify

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
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'error': 'Authorization header missing'}), 401
        
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Invalid authorization header format'}), 401
        
        id_token = auth_header.split('Bearer ')[1]
        user_id = verify_firebase_token(id_token)
        
        if not user_id:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Pass user_id to the route function
        return f(user_id, *args, **kwargs)
    
    return decorated_function

def get_user_info(user_id):
    """Get user information from Firebase Auth"""
    try:
        user = auth.get_user(user_id)
        return {
            'uid': user.uid,
            'email': user.email,
            'displayName': user.display_name,
            'photoURL': user.photo_url,
            'emailVerified': user.email_verified,
            'creationTime': user.user_metadata.creation_timestamp,
            'lastSignInTime': user.user_metadata.last_sign_in_timestamp
        }
    except Exception as e:
        logger.error(f"Error getting user info: {str(e)}")
        return None