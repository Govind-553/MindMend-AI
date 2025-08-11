import pytest
from unittest.mock import patch, MagicMock
from app import create_app
import json

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    app.config['CORS_ORIGINS'] = ['http://localhost:3000']
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test the /api/health endpoint"""
    rv = client.get('/api/health')
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert json_data['status'] == 'healthy'
    assert 'version' in json_data
    assert 'services' in json_data

@patch('app.db_client')
@patch('app.wellness_calc')
def test_quick_wellness_check(mock_wellness_calc, mock_db_client, client):
    """Test the /api/wellness/quick-check endpoint"""
    mock_wellness_calc.get_recommendation.return_value = {
        'level': 'good',
        'message': 'You are doing well.'
    }
    mock_db_client.add_wellness_data.return_value = 'mock_doc_id'

    headers = {'Authorization': 'Bearer mock_token'}
    data = {
        'moodRating': 8,
        'energyLevel': 7,
        'stressLevel': 3
    }
    
    rv = client.post('/api/wellness/quick-check', data=json.dumps(data), content_type='application/json', headers=headers)
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert json_data['success'] == True
    assert 'wellnessIndex' in json_data['data']
    assert json_data['data']['wellnessIndex'] > 0
    mock_db_client.add_wellness_data.assert_called_once()