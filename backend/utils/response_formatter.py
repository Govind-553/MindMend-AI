from flask import jsonify
from typing import Any, Dict, Optional

def success_response(data: Optional[Dict[str, Any]] = None, message: str = "Success", status_code: int = 200):
    """
    Creates a standardized success response.
    
    Args:
        data: The payload to return.
        message: A descriptive success message.
        status_code: The HTTP status code.
        
    Returns:
        A Flask JSON response.
    """
    response_data = {
        "success": True,
        "message": message
    }
    if data is not None:
        response_data["data"] = data
        
    return jsonify(response_data), status_code

def error_response(message: str = "An error occurred", details: Optional[str] = None, status_code: int = 500):
    """
    Creates a standardized error response.
    
    Args:
        message: A high-level error message.
        details: A more detailed explanation of the error.
        status_code: The HTTP status code.
        
    Returns:
        A Flask JSON response.
    """
    response_data = {
        "success": False,
        "error": message
    }
    if details:
        response_data["details"] = details
        
    return jsonify(response_data), status_code