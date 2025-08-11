import pytest
from unittest.mock import patch, MagicMock
from database.firestore_client import FirestoreClient
from datetime import datetime, timedelta

@pytest.fixture
def firestore_client():
    return FirestoreClient()

@patch('database.firestore_client.firestore.client')
def test_add_wellness_data_success(mock_client, firestore_client):
    """Test successful data addition to Firestore"""
    mock_collection = MagicMock()
    mock_client.return_value.collection.return_value = mock_collection
    mock_collection.add.return_value = (None, MagicMock(id='test_doc_id'))

    data = {'wellnessIndex': 85.0}
    doc_id = firestore_client.add_wellness_data('test_user', data)
    
    assert doc_id == 'test_doc_id'
    mock_collection.add.assert_called_once()

@patch('database.firestore_client.firestore.client')
def test_get_user_wellness_history_success(mock_client, firestore_client):
    """Test successful retrieval of user history"""
    mock_doc1 = MagicMock()
    mock_doc1.to_dict.return_value = {'wellnessIndex': 80.0, 'timestamp': datetime.utcnow()}
    mock_doc1.id = 'doc1'
    
    mock_doc2 = MagicMock()
    mock_doc2.to_dict.return_value = {'wellnessIndex': 75.0, 'timestamp': datetime.utcnow() - timedelta(days=1)}
    mock_doc2.id = 'doc2'
    
    mock_query = MagicMock()
    mock_query.stream.return_value = [mock_doc1, mock_doc2]
    mock_client.return_value.collection.return_value.where.return_value.order_by.return_value.limit.return_value = mock_query
    
    history = firestore_client.get_user_wellness_history('test_user', days_back=7)
    
    assert len(history) == 2
    assert history[0]['wellnessIndex'] == 80.0
    assert history[1]['wellnessIndex'] == 75.0

@patch('database.firestore_client.firestore.client')
def test_get_user_stats_valid(mock_client, firestore_client):
    """Test statistics calculation with valid data"""
    mock_history = [
        {'wellnessIndex': 90.0},
        {'wellnessIndex': 80.0},
        {'wellnessIndex': 70.0}
    ]
    with patch('database.firestore_client.FirestoreClient.get_user_wellness_history', return_value=mock_history):
        stats_data = firestore_client.get_user_stats('test_user', days_back=30)
        
        stats = stats_data['stats']
        assert stats['average'] == 80.0
        assert stats['minimum'] == 70.0
        assert stats['maximum'] == 90.0
        assert stats['trend'] == 'stable'