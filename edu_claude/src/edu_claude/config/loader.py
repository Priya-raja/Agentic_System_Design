from pathlib import Path

import yaml
from dotenv import load_dotenv

from .settings import Settings


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_settings() -> Settings:
    """Load environment variables and validate the application configuration."""
    load_dotenv(PROJECT_ROOT / ".env")

    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return Settings.model_validate(data)
