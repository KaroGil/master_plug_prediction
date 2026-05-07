from pathlib import Path
import yaml
from functools import lru_cache

@lru_cache
def get_config():
    """Load configuration from config.yaml file to avoid hardcoding values in the code and enable easy adjustments"""
    root = Path(__file__).resolve().parents[2]
    config_path = root / "config.yaml"

    with open(config_path) as f:
        return yaml.safe_load(f)
    
