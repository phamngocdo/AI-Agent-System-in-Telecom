import logging
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_PATH = os.path.join(os.getcwd(), "logs")

if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

log_filename = os.path.join(LOG_PATH, f"{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger()

def log_info(message: str):
    logger.info(message)

def log_warning(message: str):
    logger.warning(message)

def log_error(message: str):
    logger.error(message, exc_info=True)

def log_debug(message: str):
    logger.debug(message)