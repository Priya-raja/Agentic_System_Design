from pathlib import Path

import yaml


def load_config() -> dict:
    """Load the configuration from the config.yaml file."""

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()    

