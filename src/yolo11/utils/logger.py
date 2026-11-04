import logging
import sys
from pathlib import Path

def get_logger(name: str, log_file: str = None, log_level: str = 'INFO') -> logging.Logger:
    """Get logger with specified configuration"""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    if log_file:
        file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    if log_file:
        logger.addHandler(file_handler)
    
    return logger