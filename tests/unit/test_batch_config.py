# tests/unit/test_batch_config.py
import os
import pytest
from dpam.config import ConfigManager, load_config

class TestBatchConfiguration:
    """Test suite for batch-related configuration"""
    
    def test_batch_directory_config(self):
        """Test batch directory configuration"""
        test_batch_dir = "/test/batch/path"
        config = ConfigManager({
            "batch_dir": test_batch_dir
        })
        
        assert config.get("batch_dir") == test_batch_dir
    
    def test_batch_directory_env_override(self, monkeypatch):
        """Test batch directory override via environment variable"""
        env_batch_dir = "/env/batch/path"
        monkeypatch.setenv("DPAM_BATCH_DIR", env_batch_dir)
        
        config = ConfigManager({
            "batch_dir": "/original/batch/path"
        })
        
        assert config.get("batch_dir") == env_batch_dir
    
    def test_batch_directory_default(self):
        """Test default batch directory is based on data_dir"""
        config = ConfigManager({
            "data_dir": "/data/test",
            # No explicit batch_dir
        })
        
        # Should fall back to the default from DEFAULT_CONFIG
        assert config.get("batch_dir") == "/data/ecod/dpam/batches"  # This matches the DEFAULT_CONFIG value
    
    @pytest.mark.parametrize("batch_size", [10, 50, 100])
    def test_configurable_batch_size(self, batch_size):
        """Test configurable batch size parameter"""
        # Add a batch size parameter - this might need to be added to your config module
        config = ConfigManager({
            "pipeline": {
                "batch_size": batch_size
            }
        })
        
        assert config.get("pipeline.batch_size") == batch_size
    
    def test_batch_execution_config(self):
        """Test batch execution configuration parameters"""
        config = ConfigManager({
            "pipeline": {
                "parallel_batches": 4,
                "retry_failed": True,
                "max_retries": 3,
                "batch_timeout": 3600  # seconds
            }
        })
        
        assert config.get("pipeline.parallel_batches") == 4
        assert config.get("pipeline.retry_failed") is True
        assert config.get("pipeline.max_retries") == 3
        assert config.get("pipeline.batch_timeout") == 3600