import pytest
from unittest.mock import patch, MagicMock
from flask import Flask, jsonify
from auth.firebase_auth import require_auth, verify_firebase_token, get_user_info
from firebase_admin import auth

# Mock the Firebase Admin SDK for testing
@pytest.fixture
def mock_auth():
    with patch('firebase_admin.auth.verify_id_token') as mock_verify:
        yield mock_verify

@pytest.fixture
def mock_get_user():
    with patch('firebase_admin.auth.get_user') as mock_user:
        yield mock_user

@pytest.fixture
def app_with_auth():
    app = Flask(__name__)
    app.config['TESTING'] = True
    
    @app.route('/test-auth')
    @require_auth
    def protected_route(user_id):
        return jsonify({'message': f'Authenticated as {user_id}'}), 200

    return app.test_client()

def test_require_auth_valid_token(app_with_auth, mock_auth):
    """Test the require_auth decorator with a valid token"""
    mock_auth.return_value = {'uid': 'test_user_id'}
    headers = {'Authorization': 'Bearer valid-token'}
    
    rv = app_with_auth.get('/test-auth', headers=headers)
    assert rv.status_code == 200
    assert rv.get_json()['message'] == 'Authenticated as test_user_id'

def test_require_auth_invalid_token(app_with_auth, mock_auth):
    """Test the require_auth decorator with an invalid token"""
    mock_auth.side_effect = auth.InvalidIdTokenError('Invalid token')
    headers = {'Authorization': 'Bearer invalid-token'}
    
    rv = app_with_auth.get('/test-auth', headers=headers)
    assert rv.status_code == 401
    assert 'Invalid or expired token' in rv.get_json()['error']

def test_get_user_info_valid(mock_get_user):
    """Test get_user_info with a valid user ID"""
    mock_user_record = MagicMock()
    mock_user_record.uid = 'test_user'
    mock_user_record.email = 'test@example.com'
    mock_user_record.display_name = 'Test User'
    mock_get_user.return_value = mock_user_record
    
    user_info = get_user_info('test_user')
    assert user_info['uid'] == 'test_user'
    assert user_info['email'] == 'test@example.com'