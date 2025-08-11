import pytest
from services.wellness_calculator import WellnessCalculator

@pytest.fixture
def wellness_calc():
    return WellnessCalculator()

def test_calculate_index_all_inputs(wellness_calc):
    """Test index calculation with all three inputs"""
    speech_data = {'label': 'happy', 'confidence': 0.8}
    facial_data = {'label': 'happy', 'confidence': 0.9, 'detectedFaces': 1}
    keystroke_data = {'score': 20}
    
    index = wellness_calc.calculate_index(speech_data, facial_data, keystroke_data)
    assert index > 80
    assert index <= 100

def test_calculate_index_no_facial(wellness_calc):
    """Test index calculation with missing facial data"""
    speech_data = {'label': 'happy', 'confidence': 0.8}
    facial_data = None
    keystroke_data = {'score': 20}
    
    index = wellness_calc.calculate_index(speech_data, facial_data, keystroke_data)
    assert index > 80

def test_get_recommendation_high_score(wellness_calc):
    """Test recommendation for a high wellness score"""
    recommendation = wellness_calc.get_recommendation(90)
    assert recommendation['level'] == 'excellent'
    assert 'Keep up the positive habits' in recommendation['message']

def test_get_recommendation_low_score(wellness_calc):
    """Test recommendation for a low wellness score"""
    recommendation = wellness_calc.get_recommendation(25)
    assert recommendation['level'] == 'low'
    assert 'seekin' in recommendation['message']