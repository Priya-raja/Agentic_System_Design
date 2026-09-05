import logging

# Keep third-party libraries quiet.
# Our application's loggers will explicitly use DEBUG.

logging.basicConfig(
    level=logging.WARNING, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    )


def get_logger(name:str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: A logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger