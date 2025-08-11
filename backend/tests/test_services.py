import pytest
from unittest.mock import patch, MagicMock
from services.ai_service_client import AIServiceClient
import requests
import os

@pytest.fixture
def ai_client():
    return AIServiceClient()

@patch.dict(os.environ, {'USE_MOCK_AI': 'false'})
@patch('requests.post')
def test_analyze_speech_success(mock_post, ai_client):
    """Test successful speech analysis API call"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'label': 'happy', 'confidence': 0.9}
    mock_post.return_value = mock_response
    
    mock_audio_file = MagicMock()
    
    result = ai_client.analyze_speech(mock_audio_file)
    assert result['label'] == 'happy'
    assert result['confidence'] == 0.9
    mock_post.assert_called_once()

@patch.dict(os.environ, {'USE_MOCK_AI': 'true'})
def test_analyze_facial_emotion_mock(ai_client):
    """Test facial emotion analysis in mock mode"""
    mock_image_file = MagicMock()
    result = ai_client.analyze_facial_emotion(mock_image_file)
    
    assert result['mock'] == True
    assert 'label' in result
    assert 'confidence' in result

@patch.dict(os.environ, {'USE_MOCK_AI': 'false'})
@patch('requests.post')
def test_analyze_keystrokes_network_error(mock_post, ai_client):
    """Test keystroke analysis when a network error occurs"""
    mock_post.side_effect = requests.exceptions.RequestException('Network down')
    
    keystrokes_data = [{'key': 'a', 'type': 'keydown', 'timestamp': 100}]
    result = ai_client.analyze_keystrokes(keystrokes_data)
    
    assert result['mock'] == True
    assert 'score' in result 
    