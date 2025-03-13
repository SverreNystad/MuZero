import yaml
import os

CONFIG_PATH = os.path.dirname(__file__)


def load_config(filename: str) -> dict:
    """
    Returns:
        dict: Configuration as a dictionary.
    """
    path = os.path.join(CONFIG_PATH, filename)
    with open(path, "r") as file:
        return yaml.safe_load(file)
