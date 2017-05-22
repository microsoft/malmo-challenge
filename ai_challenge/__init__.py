import os
import logging

logger = logging.getLogger(__name__)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
logger.log(msg='Initialized rood dir of the project to {}'.format(ROOT_DIR), level=logging.INFO)
