import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

def setup_logging(log_dir: Optional[str], log_file: Optional[str], config: Optional[Dict] = None, default_dir: str = 'trading_logs', default_file: str = 'trading.log'):
    """
    Sets up file logging with rotation.
    
    Args:
        log_dir: Directory to store log files. Overrides config and default.
        log_file: Filename for the log. Overrides config and default.
        config: Configuration dictionary containing 'logging' section.
        default_dir: Default directory if not specified elsewhere.
        default_file: Default filename if not specified elsewhere.
    """
    # Determine log directory and file
    # Priority: CLI args > Config file > Defaults
    
    target_dir = log_dir
    if not target_dir and config:
        target_dir = config.get('logging', {}).get('directory')
    if not target_dir:
        target_dir = default_dir
        
    target_file = log_file
    if not target_file and config:
        target_file = config.get('logging', {}).get('filename')
    if not target_file:
        target_file = default_file
        
    try:
        path = Path(target_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / target_file
        
        # Rotate existing log file
        if file_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            new_path = path / new_name
            try:
                file_path.rename(new_path)
                logging.getLogger().info(f"Rotated previous log file to: {new_path}")
            except OSError as e:
                logging.getLogger().warning(f"Failed to rotate log file: {e}")
        
        # Add FileHandler to root logger
        handler = logging.FileHandler(file_path)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(handler)
        
        logging.getLogger().info(f"Logging output to: {file_path}")
    except Exception as e:
        logging.getLogger().error(f"Failed to setup file logging: {e}")