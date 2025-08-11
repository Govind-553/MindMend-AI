from flask import Blueprint, jsonify, request
import os
import logging
from utils.decorators import require_auth
from firebase_admin import auth

auth_bp = Blueprint('auth', __name__)
logger = logging.getLogger(__name__)

def get_user_info(user_id):
    """Get user information from Firebase Auth, with mock data for dev"""
    try:
        if os.getenv('ENVIRONMENT', 'development') == 'development' and user_id == 'demo_user':
            return {
                'uid': user_id,
                'email': 'demo@example.com',
                'displayName': 'Demo User',
                'emailVerified': True
            }
        else:
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
        logger.error(f"Error getting user info for {user_id}: {str(e)}")
        return None

@auth_bp.route('/user', methods=['GET'])
@require_auth
def get_user(user_id):
    """Get current user information"""
    try:
        user_info = get_user_info(user_id)
        if user_info:
            return jsonify({'success': True, 'data': user_info})
        else:
            return jsonify({'error': 'Failed to retrieve user information'}), 404
    except Exception as e:
        logger.error(f"API route error for /user: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500