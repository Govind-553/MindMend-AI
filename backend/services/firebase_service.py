import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from firebase_admin import firestore
import random
import os

logger = logging.getLogger(__name__)

class FirestoreClient:
    def __init__(self):
        try:
            self.db = firestore.client()
        except Exception as e:
            logger.warning(f"Firestore client initialization failed: {str(e)}")
            self.db = None
    
    def add_wellness_data(self, user_id: str, data: Dict[str, Any]) -> str:
        """Add wellness data to Firestore"""
        if not self.db:
            return "mock_doc_id"
        
        try:
            doc_data = {
                'userId': user_id,
                'timestamp': datetime.utcnow(),
                **data
            }
            doc_ref = self.db.collection('wellnessData').add(doc_data)
            logger.info(f"Added wellness data for user {user_id}")
            return doc_ref[1].id
        except Exception as e:
            logger.error(f"Error adding wellness data: {str(e)}")
            return "error_doc_id"
    
    def get_user_wellness_history(self, user_id: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get user's wellness history"""
        if not self.db:
            # Return mock data for development
            return [
                {
                    'id': f'mock_{i}',
                    'wellnessIndex': random.uniform(40, 90),
                    'timestamp': (datetime.utcnow() - timedelta(days=i)).isoformat(),
                    'speechEmotion': {'label': random.choice(['happy', 'neutral', 'calm'])},
                    'facialEmotion': {'label': random.choice(['happy', 'neutral', 'surprised'])}
                }
                for i in range(min(days_back, 10))
            ]
        
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            query = self.db.collection('wellnessData') \
                          .where('userId', '==', user_id) \
                          .where('timestamp', '>=', start_date) \
                          .order_by('timestamp', direction=firestore.Query.DESCENDING) \
                          .limit(50)
            
            docs = query.stream()
            history = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                if 'timestamp' in data:
                    data['timestamp'] = data['timestamp'].isoformat()
                history.append(data)
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving history: {str(e)}")
            return []

    def get_user_stats(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        """Get user wellness statistics"""
        if not self.db:
            return {'message': 'No data available', 'stats': None}
        
        try:
            history = self.get_user_wellness_history(user_id, days_back, 1000)
            
            if not history:
                return {'message': 'No data available', 'stats': None}
            
            indices = [entry.get('wellnessIndex', 0) for entry in history if entry.get('wellnessIndex')]
            
            if not indices:
                return {'message': 'No wellness index data', 'stats': None}
            
            stats = {
                'average': sum(indices) / len(indices),
                'minimum': min(indices),
                'maximum': max(indices),
                'latest': indices[0] if indices else 0,
                'trend': self._calculate_trend(indices),
                'data_points': len(indices),
                'period_days': days_back
            }
            
            return {'stats': stats, 'message': 'Statistics calculated successfully'}
            
        except Exception as e:
            logger.error(f"Error calculating user stats: {str(e)}")
            raise e
            
    def _calculate_trend(self, indices: List[float]) -> str:
        """Calculate wellness trend"""
        if len(indices) < 2:
            return 'insufficient_data'
        
        midpoint = len(indices) // 2
        recent_avg = sum(indices[:midpoint]) / midpoint
        older_avg = sum(indices[midpoint:]) / (len(indices) - midpoint)
        
        difference = recent_avg - older_avg
        
        if abs(difference) < 3:
            return 'stable'
        elif difference > 0:
            return 'improving' if difference < 10 else 'significantly_improving'
        else:
            return 'declining' if abs(difference) < 10 else 'significantly_declining'

# `db` client reference to be imported from other modules
db = FirestoreClient()