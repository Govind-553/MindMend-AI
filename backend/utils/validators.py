def validate_wellness_request(request):
    """
    Validate if the incoming request contains any of the required wellness data.
    
    Returns:
        dict: A dictionary with a 'valid' boolean and an optional 'message'.
    """
    
    has_audio = 'audio' in request.files
    has_image_file = 'image' in request.files
    has_image_data = request.json and 'imageData' in request.json
    has_keystrokes_json = request.json and 'keystrokes' in request.json
    has_keystrokes_form = 'keystrokes' in request.form
    
    if not (has_audio or has_image_file or has_image_data or has_keystrokes_json or has_keystrokes_form):
        return {
            'valid': False,
            'message': 'No valid wellness data (audio, image, or keystrokes) found in request.'
        }
    
    return {'valid': True}