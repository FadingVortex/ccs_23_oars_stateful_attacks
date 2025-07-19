# logger_utils.py
import logging
import os

def get_attack_logger(attack_name: str) -> logging.Logger:
    """
    Returns an isolated logger for the given attack algorithm name.
    Log file will be written to: output/{attack_name}.txt

    :param attack_name: e.g. "hsja", "nescore"
    :return: logger object
    """
    os.makedirs('output', exist_ok=True)
    logger = logging.getLogger(f'logger_{attack_name}')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(f'output/{attack_name}.txt', mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger
