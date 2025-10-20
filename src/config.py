"""
Configuration management module.
Loads configuration from JSON files and exposes keys as Python attributes.
"""

import json
from pathlib import Path


class Config:
    """
    Simple configuration class that loads a JSON file and exposes
    its keys as Python attributes for easy access.
    
    Example:
        >>> cfg = Config.from_json("configs/train_config.json")
        >>> print(cfg.learning_rate)
    """
    
    def __init__(self, json_path: str):
        """
        Initialize configuration from a JSON file.
        
        Args:
            json_path: Path to the JSON configuration file
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Set all JSON keys as instance attributes
        for key, value in data.items():
            setattr(self, key, value)

    @classmethod
    def from_json(cls, json_path: str):
        """
        Alternative constructor from JSON path.
        
        Args:
            json_path: Path to the JSON configuration file
            
        Returns:
            Config instance
        """
        return cls(json_path)

    def __repr__(self):
        """String representation showing available configuration keys."""
        keys = ", ".join(self.__dict__.keys())
        return f"<Config keys=[{keys}]>"