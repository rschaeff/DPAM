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
# Add these tests to your existing tests/unit/test_config.py file

class TestAdditionalConfigMethods:
    """Tests for additional ConfigManager methods"""
    
    def test_get_grid_config(self):
        """Test retrieving grid configuration"""
        test_grid_config = {
            "script_path": "/test/path",
            "log_dir": "/test/logs",
            "max_runtime": "12:00:00",
            "memory": "16G"
        }
        
        config = ConfigManager({"grid": test_grid_config})
        grid_config = config.get_grid_config()
        
        assert grid_config == test_grid_config
        assert grid_config["script_path"] == "/test/path"
        assert grid_config["memory"] == "16G"
    
    def test_get_grid_config_missing(self):
        """Test retrieving grid configuration when it's missing"""
        config = ConfigManager({})  # No grid config
        grid_config = config.get_grid_config()
        
        assert grid_config == {}
    
    def test_save_config(self, temp_dir):
        """Test saving configuration to file"""
        test_config = {
            "database": {"host": "savehost"},
            "test_key": "test_value"
        }
        
        config = ConfigManager(test_config)
        save_path = os.path.join(temp_dir, "saved_config.json")
        
        # Save the config
        result_path = config.save(save_path)
        assert result_path == save_path
        assert os.path.exists(save_path)
        
        # Load the saved config and verify
        with open(save_path, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config["database"]["host"] == "savehost"
        assert loaded_config["test_key"] == "test_value"
    
    def test_save_config_default_path(self, monkeypatch):
        """Test saving configuration to default path"""
        # Mock tempfile.gettempdir to return a specific dir
        test_temp_dir = "/mock/temp/dir"
        monkeypatch.setattr(tempfile, "gettempdir", lambda: test_temp_dir)
        
        # Mock os.makedirs to avoid actually creating directories
        monkeypatch.setattr(os, "makedirs", lambda path, exist_ok: None)
        
        # Mock open to avoid actually writing files
        mock_file = Mock()
        mock_open = Mock(return_value=mock_file)
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=None)
        monkeypatch.setattr(builtins, "open", mock_open)
        
        # Test with no path specified
        config = ConfigManager({"test": "value"})
        config.save()
        
        # Verify the expected path
        expected_path = os.path.join(test_temp_dir, "dpam_config.json")
        mock_open.assert_called_with(expected_path, 'w')


class TestGlobalConfigManager:
    """Tests for global configuration manager functions"""
    
    def test_get_config_loads_default(self, monkeypatch):
        """Test get_config loads default config when none found"""
        # Reset global config_manager
        import dpam.config
        monkeypatch.setattr(dpam.config, "config_manager", None)
        
        # Mock load_config to return a specific config
        mock_config = ConfigManager({"test": "global_config"})
        monkeypatch.setattr(dpam.config, "load_config", lambda: mock_config)
        
        # Get the global config
        config = dpam.config.get_config()
        
        # Should be the mock config
        assert config.get("test") == "global_config"
        assert dpam.config.config_manager is config
    
    def test_get_config_reuses_existing(self, monkeypatch):
        """Test get_config reuses existing config manager"""
        # Set a specific global config_manager
        import dpam.config
        mock_config = ConfigManager({"test": "existing_config"})
        monkeypatch.setattr(dpam.config, "config_manager", mock_config)
        
        # Should not call load_config
        load_config_mock = Mock()
        monkeypatch.setattr(dpam.config, "load_config", load_config_mock)
        
        # Get the global config
        config = dpam.config.get_config()
        
        # Should be the existing config
        assert config is mock_config
        assert config.get("test") == "existing_config"
        load_config_mock.assert_not_called()


class TestComplexEnvironmentOverrides:
    """Tests for complex environment variable overrides"""
    
    def test_env_var_nested_override(self, monkeypatch):
        """Test environment variable overrides for nested keys"""
        # Set environment variables for nested keys
        monkeypatch.setenv("DPAM_GRID_MEMORY", "32G")
        monkeypatch.setenv("DPAM_GRID_THREADS", "8")
        monkeypatch.setenv("DPAM_PIPELINE_MIN_DOMAIN_SIZE", "40")
        
        # Create config with different values
        config = ConfigManager({
            "grid": {
                "memory": "8G",
                "threads": 4
            },
            "pipeline": {
                "min_domain_size": 30
            }
        })
        
        # Environment variables should override the config values
        assert config.get("grid.memory") == "32G"
        assert config.get("grid.threads") == 8  # Should be converted to int
        assert config.get("pipeline.min_domain_size") == 40
    
    def test_env_var_array_values(self, monkeypatch):
        """Test environment variables for arrays are correctly handled"""
        # This is a limitation - can't easily set arrays via env vars
        # But the code should handle it gracefully
        monkeypatch.setenv("DPAM_SEARCH_PATHS", "['/path1', '/path2']")
        
        config = ConfigManager({})
        # Should be kept as a string, not parsed as JSON
        assert config.get("search_paths") == "['/path1', '/path2']"
    
    def test_env_var_case_sensitivity(self, monkeypatch):
        """Test environment variable handling is case-insensitive"""
        # Different case variations should work
        monkeypatch.setenv("DPAM_DATABASE_HOST", "lowercase")
        monkeypatch.setenv("DPAM_DATABASE_PORT", "5678")
        monkeypatch.setenv("DPAM_DATA_DIR", "/case/test")
        
        config = ConfigManager({
            "database": {
                "HOST": "original",  # Note the uppercase here
                "Port": 1234  # Mixed case
            },
            "DATA_DIR": "/original"  # Uppercase in config
        })
        
        # Keys in config should be accessed case-sensitively
        assert config.get("database.HOST") == "lowercase"  # Env var should override
        assert config.get("database.Port") == 5678  # Env var should override
        assert config.get("DATA_DIR") == "/case/test"  # Env var should override