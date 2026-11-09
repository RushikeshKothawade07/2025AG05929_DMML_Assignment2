"""
Configuration loader and standard logger utility.
"""

import os
import yaml
import logging


def setup_logger(name: str = "pipeline") -> logging.Logger:
    """Configures a clean console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Loads YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
