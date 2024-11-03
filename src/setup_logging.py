# Option 1: Add a FileHandler to the existing logger
import logging
import os

def setup_logging(log_dir="logs", logger_name="quant_logger"):
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get the logger
    logger = logging.getLogger(logger_name)
    
    # Set the logging level
    logger.setLevel(logging.INFO)
    
    # Create a file handler
    log_file = os.path.join(log_dir, "quant_reliability.log")
    file_handler = logging.FileHandler(log_file)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    # Also add a stream handler to see logs in console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return log_file

# Add this at the start of your script
log_file = setup_logging()
print(f"Logs will be saved to: {log_file}")

# To read the log file at any point:
def read_log_file(log_file):
    try:
        with open(log_file, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading log file: {str(e)}")
        return None

# Example usage:
# log_contents = read_log_file(log_file)
# print(log_contents)