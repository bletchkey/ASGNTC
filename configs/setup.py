import logging
import logging.config
import json
import os
import sys

from pathlib import Path

from configs.paths import PROJECT_NAME, APP_LOGS_DIR

def setup_base_directory():
    """
    Set the base directory to the root of the project.

    """

    dir = Path(__file__).resolve().parent.parent

    if dir.name != PROJECT_NAME:
        print(f"Error: The base directory is not set correctly. Expected: {PROJECT_NAME}, got: {dir.name}")
        sys.exit(1)

    os.chdir(dir)
    sys.path.append(str(dir))


def setup_logging(path, default_level=logging.INFO):
    """
    Get the logging configuration from .json file and set it up.

    """

    try:
        with open(path, 'rt') as file:
            logging_config = json.load(file)

        # Replace the placeholder with the actual log directory
        log_filename = logging_config['handlers']['file']['filename'].format(log_directory=APP_LOGS_DIR)

        # Ensure the log directory exists
        log_directory_path = Path(log_filename).parent
        log_directory_path.mkdir(parents=True, exist_ok=True)

        # Update the filename in the configuration
        logging_config['handlers']['file']['filename'] = log_filename

        # Apply the logging configuration
        logging.config.dictConfig(logging_config)
        logging.debug(f"Logging configuration loaded from {path}")
    except Exception as e:
        logging.error(f"Error in logging configuration (using default settings): {e}")
        logging.basicConfig(level=default_level)

