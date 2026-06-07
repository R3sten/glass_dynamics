import logging
import sys

def setup_logger(name="glass_dynamics", level=logging.INFO):
    """
    Sets up a custom logger for the glass analysis package.
    It formats the output to look clean and professional in the terminal.
    """
    # Create the logger object
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if the logger is called multiple times
    if not logger.handlers:
        # Create a console handler (prints to the terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Define the format of the messages
        # Example output: [INFO] 2023-10-25 15:30:00 - Starting Ridge regression...
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        # Attach the handler to the logger
        logger.addHandler(console_handler)

    return logger

# We instantiate a default logger here so it can be imported directly by other modules
logger = setup_logger()