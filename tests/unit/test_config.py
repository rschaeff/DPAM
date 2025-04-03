# tests/unit/test_config.py
import os
import json
import tempfile
import pytest
from dpam.config import ConfigManager, load_config, DEFAULT_CONFIG, REQUIRED_KEYS

class TestConfigManager:
    """Test suite for ConfigManager class"""
    
    def test_init_with_empty_config(self):
        """Test initialization with empty config uses defaults"""
        config = ConfigManager()
        assert config.config is not None
        assert config.config == DEFAULT_CONFIG
    
    def test_init_with_custom_config(self):
        """Test initialization with custom config"""
        custom_config = {
            "database": {"host": "custom_host"},
            "custom_key": "custom_value"
        }
        config = ConfigManager(custom_config)
        assert config.config["database"]["host"] == "custom_host"
        assert config.config["custom_key"] == "custom_value"
        
    def test_get_existing_key(self):
        """Test getting existing keys from config"""
        config = ConfigManager({
            "database": {"host": "testhost", "port": 5432},
            "data_dir": "/test/dir"
        })
        
        assert config.get("database.host") == "testhost"
        assert config.get("data_dir") == "/test/dir"
        
    def test_get_nested_key(self):
        """Test getting nested keys"""
        config = ConfigManager({
            "nested": {
                "level1": {
                    "level2": {
                        "value": "nested_value"
                    }
                }
            }
        })
        
        assert config.get("nested.level1.level2.value") == "nested_value"
    
    def test_get_missing_key_with_default(self):
        """Test getting missing keys with default value"""
        config = ConfigManager({})
        assert config.get("missing.key", "default_value") == "default_value"
    
    def test_get_missing_key_without_default(self):
        """Test getting missing keys without default value returns None"""
        config = ConfigManager({})
        assert config.get("missing.key") is None
    
    def test_set_value(self):
        """Test setting configuration values"""
        config = ConfigManager({})
        config.set("new.key", "new_value")
        assert config.get("new.key") == "new_value"
        
        # Test modifying existing value
        config.set("new.key", "modified_value")
        assert config.get("new.key") == "modified_value"
    
    def test_set_nested_value(self):
        """Test setting nested configuration values"""
        config = ConfigManager({})
        config.set("parent.child.grandchild", "nested_value")
        
        # Check structure was created
        assert isinstance(config.config["parent"], dict)
        assert isinstance(config.config["parent"]["child"], dict)
        assert config.config["parent"]["child"]["grandchild"] == "nested_value"
    
    def test_validate_complete_config(self):
        """Test validating a complete configuration"""
        # Create config with all required keys
        test_config = {}
        for key in REQUIRED_KEYS:
            parts = key.split(".")
            current = test_config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = "test_value"
            
        config = ConfigManager(test_config)
        missing = config.validate()
        assert len(missing) == 0
    
    def test_validate_incomplete_config(self):
        """Test validating an incomplete configuration"""
        config = ConfigManager({
            "database": {
                "host": "testhost",
                # Missing other required keys
            }
        })
        
        missing = config.validate()
        assert len(missing) > 0  # Some keys should be missing
        
        # Check if "database.dbname" is among missing keys
        if "database.dbname" in REQUIRED_KEYS:
            assert "database.dbname" in missing
    
    def test_get_db_config(self):
        """Test getting database configuration"""
        config = ConfigManager({
            "database": {
                "host": "testhost",
                "port": 5432,
                "dbname": "testdb",
                "user": "testuser",
                "password": "testpass",
                "schema": "testschema"
            }
        })
        
        db_config = config.get_db_config()
        assert db_config["host"] == "testhost"
        assert db_config["port"] == 5432
        assert db_config["dbname"] == "testdb"
        assert db_config["user"] == "testuser"
        assert db_config["password"] == "testpass"
        assert db_config["schema"] == "testschema"


class TestConfigFileLoading:
    """Test suite for configuration file loading"""
    
    def test_load_config_from_file(self):
        """Test loading configuration from a file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            # Create a test config file
            test_config = {
                "database": {"host": "filehost"},
                "test_setting": "test_value"
            }
            json.dump(test_config, temp_file)
            temp_file_path = temp_file.name
            
        try:
            # Load the config
            config = load_config(temp_file_path)
            assert config.get("database.host") == "filehost"
            assert config.get("test_setting") == "test_value"
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_load_config_file_not_found(self):
        """Test loading config when file doesn't exist"""
        config = load_config("/path/that/does/not/exist.json")
        # Should return default config
        assert config.config == DEFAULT_CONFIG
    
    def test_load_config_invalid_json(self):
        """Test loading config with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            # Create an invalid JSON file
            temp_file.write("{this is not valid JSON")
            temp_file_path = temp_file.name
            
        try:
            # Should handle the error and return default config
            config = load_config(temp_file_path)
            assert config.config == DEFAULT_CONFIG
        finally:
            # Clean up
            os.unlink(temp_file_path)


class TestEnvironmentOverrides:
    """Test suite for environment variable overrides"""
    
    def test_env_var_override(self, monkeypatch):
        """Test environment variables override config values"""
        # Set up environment variable
        monkeypatch.setenv("DPAM_DATABASE_HOST", "envhost")
        
        # Create config with different value
        config = ConfigManager({"database": {"host": "confighost"}})
        
        # Environment variable should take precedence
        assert config.get("database.host") == "envhost"
    
    def test_env_var_type_conversion(self, monkeypatch):
        """Test environment variables are converted to appropriate types"""
        # Set up environment variables
        monkeypatch.setenv("DPAM_DATABASE_PORT", "5433")  # Should be converted to int
        monkeypatch.setenv("DPAM_DEBUG", "true")  # Should be converted to bool
        
        # Create config
        config = ConfigManager({})
        
        # Check type conversion
        assert config.get("database.port") == 5433
        assert isinstance(config.get("database.port"), int)
        assert config.get("debug") is True
        assert isinstance(config.get("debug"), bool)

# Add more tests for other functionality in config.py as needed