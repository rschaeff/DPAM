#!/usr/bin/env python3
"""
Configuration management for DPAM pipeline.

This module provides utilities for loading, validating, and accessing
configuration settings for the DPAM pipeline.
"""

import os
import json
import logging
import tempfile
import copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Set

# Configure logging
logger = logging.getLogger("dpam.config")

# Default configuration paths
DEFAULT_CONFIG_PATHS = [
    "./config.json",
    "~/.dpam/config.json",
    "/etc/dpam/config.json"
]

# Environment variable prefix for overriding config
ENV_PREFIX = "DPAM_"

# Default configuration values
DEFAULT_CONFIG = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "dbname": "dpam",
        "user": "dpam",
        "password": "dpam_password",
        "schema": "dpam_queue"
    },
    "data_dir": "/data/dpam",
    "batch_dir": "/data/dpam/batches",
    "grid": {
        "script_path": "/opt/dpam/bin/dpam-worker",
        "log_dir": "/var/log/dpam",
        "output_dir": "/data/dpam/output",
        "env_path": "/opt/dpam/venv/bin/activate",
        "max_runtime": "24:00:00",
        "memory": "8G",
        "threads": 4,
        "hhsearch_threads": 4,
        "foldseek_threads": 6,
        "queue_name": "normal.q"
    },
    "pipeline": {
        "min_domain_size": 30,
        "max_domains": 20,
        "dali_min_z_score": 8.0,
        "disorder_threshold": 70.0,
        "min_support_score": 0.5,
        "ecod_weight": 2.0,
        "dali_weight": 1.5,
        "foldseek_weight": 1.0,
        "pae_weight": 2.0
    },
    "binaries": {
        "dssp": "mkdssp",
        "hhsearch": "hhsearch",
        "hhblits": "hhblits",
        "foldseek": "foldseek",
        "dali": "dali.pl"
    },
    "api": {
        "host": "localhost",
        "port": 8000,
        "debug": False,
        "workers": 4,
        "timeout": 60
    },
    "logging": {
        "level": "INFO",
        "file": "/var/log/dpam/dpam.log",
        "max_bytes": 10485760,
        "backup_count": 10
    }
}

# Required configuration keys
REQUIRED_KEYS = [
    "database.host",
    "database.dbname",
    "database.user",
    "data_dir"
]

@dataclass
class ConfigManager:
    """
    Configuration manager for DPAM pipeline.
    
    Provides access to configuration values with optional environment
    variable overrides.
    """
    config: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[str] = None
    _initialized: bool = False
    
    def __post_init__(self):
        """Initialize with default config if empty"""
        if not self.config:
            self.config = copy.deepcopy(DEFAULT_CONFIG)
        self._initialized = True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.
        
        Environment variables override config file values.
        
        Args:
            key: Dot-separated path to config value (e.g., "database.host")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        # Check environment variable override
        env_key = ENV_PREFIX + key.upper().replace(".", "_")
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return self._convert_value(env_value)
            
        # Navigate config using key path
        parts = key.split(".")
        value = self.config
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-separated key path.
        
        Args:
            key: Dot-separated path to config value (e.g., "database.host")
            value: Value to set
        """
        parts = key.split(".")
        config = self.config
        
        # Navigate to the parent of the target key
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        # Set the value
        config[parts[-1]] = value
    
    def validate(self) -> List[str]:
        """
        Validate configuration against required keys.
        
        Returns:
            List of missing required keys
        """
        missing = []
        
        for required_key in REQUIRED_KEYS:
            if self.get(required_key) is None:
                missing.append(required_key)
        
        return missing
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to save configuration (defaults to loaded path)
            
        Returns:
            Path where configuration was saved
        """
        save_path = path or self.config_path
        
        if not save_path:
            save_path = os.path.join(tempfile.gettempdir(), "dpam_config.json")
            logger.warning(f"No config path specified, saving to {save_path}")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
        logger.info(f"Configuration saved to {save_path}")
        return save_path
    
    def get_db_config(self) -> Dict[str, Any]:
        """
        Get database configuration section.
        
        Returns:
            Database configuration dictionary
        """
        return {
            "host": self.get("database.host"),
            "port": self.get("database.port"),
            "dbname": self.get("database.dbname"),
            "user": self.get("database.user"),
            "password": self.get("database.password"),
            "schema": self.get("database.schema", "dpam_queue")
        }
    
    def get_grid_config(self) -> Dict[str, Any]:
        """
        Get grid configuration section.
        
        Returns:
            Grid configuration dictionary
        """
        return self.get("grid", {})
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type"""
        # Try to convert to appropriate type
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        elif value.lower() == "null" or value.lower() == "none":
            return None
            
        # Try to convert to int or float
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            # Return as string if not a number
            return value

def load_config(config_path: Optional[str] = None, 
                search_paths: Optional[List[str]] = None) -> ConfigManager:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file (optional)
        search_paths: Paths to search for configuration (optional)
        
    Returns:
        ConfigManager instance
    """
    if search_paths is None:
        search_paths = DEFAULT_CONFIG_PATHS
    
    # Use explicit path if provided
    if config_path:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return ConfigManager(config=config, config_path=config_path)
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
        else:
            logger.error(f"Configuration file not found: {config_path}")
    
    # Search for configuration file
    for path in search_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            try:
                with open(expanded_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {expanded_path}")
                return ConfigManager(config=config, config_path=expanded_path)
            except Exception as e:
                logger.warning(f"Failed to load config from {expanded_path}: {e}")
    
    # No configuration found, use defaults
    logger.warning("No configuration file found, using default configuration")
    return ConfigManager()

# Create a global configuration instance
config_manager = None

def get_config() -> ConfigManager:
    """
    Get global configuration instance.
    
    Returns:
        ConfigManager instance
    """
    global config_manager
    if config_manager is None:
        config_manager = load_config()
    return config_manager

# Initialize config on module import
config_manager = get_config()