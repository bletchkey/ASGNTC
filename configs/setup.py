import logging
import logging.config
import json
import os
import sys

from pathlib import Path

from configs.paths import PROJECT_NAME

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
            config = json.load(file)
        logging.config.dictConfig(config)
        logging.debug(f"Logging configuration loaded from {path}")
    except Exception as e:
        logging.error(f"Error in logging configuration (using default settings): {e}")
        logging.basicConfig(level=default_level)

