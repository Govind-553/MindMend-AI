import logging
import os

def setup_logger():
    """
    Configures the application-wide logger.
    """
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set a custom format for the root logger if needed
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Example: Suppress verbose logs from external libraries
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logger configured with level: {log_level}")
    
    return logger