from flask import Blueprint, request, jsonify
import logging
from utils.decorators import require_auth
from database.firestore_client import FirestoreClient
from services.wellness_calculator import WellnessCalculator

history_bp = Blueprint('history', __name__)
logger = logging.getLogger(__name__)

db_client = FirestoreClient()
wellness_calc = WellnessCalculator()

@history_bp.route('/wellness', methods=['GET'])
@require_auth
def get_wellness_history(user_id):
    """Get user's wellness history"""
    try:
        days_back = int(request.args.get('days', 7))
        history = db_client.get_user_wellness_history(user_id, days_back)
        
        return jsonify({
            'success': True,
            'data': history,
            'count': len(history),
            'period_days': days_back
        })
        
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve history'}), 500

@history_bp.route('/stats', methods=['GET'])
@require_auth
def get_wellness_stats(user_id):
    """Get user wellness statistics"""
    try:
        days_back = int(request.args.get('days', 30))
        stats_data = db_client.get_user_stats(user_id, days_back)
        
        return jsonify({
            'success': True,
            'data': stats_data
        })
        
    except Exception as e:
        logger.error(f"Stats calculation error: {str(e)}")
        return jsonify({'error': 'Failed to calculate stats'}), 500