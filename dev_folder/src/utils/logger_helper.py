import logging
import sys
import colorlog
import logging
from logging.handlers import RotatingFileHandler

class ConfigurableHandler(colorlog.StreamHandler):
    def __init__(self, stream=None, overwrite_line=False):
        super().__init__(stream)
        self.overwrite_line = overwrite_line

    def emit(self, record):
        msg = self.format(record)
        if self.overwrite_line:
            self.stream.write(f'\r{msg}')
        else:
            self.stream.write(f'{msg}\n')
        self.flush()

def setup_logger(name="", level=logging.INFO, overwrite_line=False):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    handler = ConfigurableHandler(stream=sys.stdout, overwrite_line=overwrite_line)
    formatter = colorlog.ColoredFormatter(
        '{log_color}{asctime} {levelname}: {message}',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        style='{',
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def setup_file_logger(
    logger_name: str,
    log_file: str,
) -> logging.Logger:
    """
    Set up a file logger with rotation that does not log to the console.

    Args:
    logger_name (str): Name of the logger.
    log_file (str): Path to the log file.
    level (str): Logging level (default: "INFO").
    max_file_size (int): Maximum size of each log file in bytes (default: 1 MB).
    backup_count (int): Number of backup files to keep (default: 3).
    log_format (str): Format string for log messages.

    Returns:
    logging.Logger: Configured logger object.
    """
    # Create a logger
    logger = logging.getLogger(logger_name)
    
    # Set the logging level
    logger.setLevel(getattr(logging, 'INFO'))

    # Create a rotating file handler
    file_handler = RotatingFileHandler(
        log_file, 
        backupCount=1
    )

    # Create a formatter and add it to the handler
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(message)s", datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False

    return logger

def setup_logging(log_file):
    # Remove the existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'
    )
    # Add a stream handler to also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
