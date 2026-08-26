import logging
from typing import Optional

def setup_logging(
    execution_name: str = "eval",
    log_level: int = logging.INFO,
    verbose: bool = False
) -> logging.Logger:
    """Configure basic console logging."""
    logger = logging.getLogger("reliability_eval")
    logger.setLevel(log_level)

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create and add console handler with formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Set debug level if verbose
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Starting new execution: {execution_name}")
    return logger