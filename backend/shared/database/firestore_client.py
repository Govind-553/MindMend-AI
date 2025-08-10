from firebase_admin import firestore
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class FirestoreClient:
    def __init__(self):
        self.db = firestore.client()
    
    def add_wellness_data(self, user_id: str, data: Dict[str, Any]) -> str:
        """Add wellness data to Firestore"""
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
            raise e
    
    def get_user_wellness_history(self, 
                                 user_id: str, 
                                 days_back: int = 7, 
                                 limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's wellness history"""
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            # Query Firestore
            query = self.db.collection('wellnessData') \
                          .where('userId', '==', user_id) \
                          .where('timestamp', '>=', start_date) \
                          .where('timestamp', '<=', end_date) \
                          .order_by('timestamp', direction=firestore.Query.DESCENDING) \
                          .limit(limit)
            
            docs = query.stream()
            
            history = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                # Convert timestamp to ISO string
                if 'timestamp' in data:
                    data['timestamp'] = data['timestamp'].isoformat()
                history.append(data)
            
            logger.info(f"Retrieved {len(history)} wellness records for user {user_id}")
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving wellness history: {str(e)}")
            raise e
    
    def get_user_stats(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        """Get user wellness statistics"""
        try:
            history = self.get_user_wellness_history(user_id, days_back, 1000)
            
            if not history:
                return {'message': 'No data available', 'stats': None}
            
            # Extract wellness indices
            indices = [entry.get('wellnessIndex', 0) for entry in history if entry.get('wellnessIndex')]
            
            if not indices:
                return {'message': 'No wellness index data', 'stats': None}
            
            # Calculate statistics
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
        
        # Compare recent vs older data
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
    
    def create_user_profile(self, user_id: str, user_data: Dict[str, Any]) -> bool:
        """Create or update user profile"""
        try:
            doc_data = {
                'uid': user_id,
                'createdAt': datetime.utcnow(),
                'updatedAt': datetime.utcnow(),
                **user_data
            }
            
            self.db.collection('users').document(user_id).set(doc_data, merge=True)
            logger.info(f"Created/updated user profile for {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating user profile: {str(e)}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile"""
        try:
            doc = self.db.collection('users').document(user_id).get()
            
            if doc.exists:
                data = doc.to_dict()
                # Convert timestamps
                for field in ['createdAt', 'updatedAt']:
                    if field in data and data[field]:
                        data[field] = data[field].isoformat()
                return data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting user profile: {str(e)}")
            return None