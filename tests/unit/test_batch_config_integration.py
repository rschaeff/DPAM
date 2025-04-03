# tests/unit/test_batch_config_integration.py
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import psycopg2
import requests

from dpam.config import ConfigManager, get_config
from dpam.batch.manager import DPAMBatchManager
from dpam.batch.preparation import DPAMBatchPreparation
from dpam.batch.supplement import DPAMBatchSupplementation

class TestBatchConfigIntegration:
    """Test integration between config system and batch modules"""
    
    @pytest.fixture
    def mock_db_connection(self):
        """Fixture to provide a mock DB connection and cursor"""
        mock_conn = Mock(spec=psycopg2.extensions.connection)
        mock_cursor = Mock(spec=psycopg2.extensions.cursor)
        mock_conn.cursor.return_value = mock_cursor
        
        # Configure cursor.fetchone to return appropriate test values
        mock_cursor.fetchone.side_effect = [
            (1,),  # batch_id
            ("test_batch",),  # batch_name
        ]
        
        # Set up psycopg2.connect patch
        with patch('psycopg2.connect', return_value=mock_conn):
            yield mock_conn, mock_cursor
    
    @pytest.fixture
    def mock_requests(self):
        """Fixture to provide mock requests functionality"""
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {"value": "Test Protein"}
                }
            },
            "comments": [{"text": [{"value": "Test description"}]}],
            "sequence": {"value": "MASKGEELFTGVVPILVELDGDVNGHKFSVS"}
        }
        mock_response.content = b'{"predicted_aligned_error": [[0.1, 0.2], [0.2, 0.1]]}'
        
        with patch('requests.get', return_value=mock_response):
            yield
    
    def test_batch_manager_with_config(self, mock_db_connection, mock_requests):
        """Test creating DPAMBatchManager with configuration"""
        # Set up config
        config = ConfigManager({
            "database": {
                "host": "test_host",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
                "schema": "test_schema"
            },
            "api": {
                "host": "api_host",
                "port": 8000
            },
            "batch_dir": "/test/batch/dir"
        })
        
        # Create batch manager with config
        db_config = config.get_db_config()
        api_config = config.get("api")
        
        batch_manager = DPAMBatchManager(db_config, api_config)
        
        # Test creating a batch
        accessions = ["P12345", "Q67890"]
        batch_id = batch_manager.create_batch_from_accessions(accessions, "Test Batch")
        
        # Verify database connection used correct config
        mock_conn, mock_cursor = mock_db_connection
        psycopg2.connect.assert_called_with(
            host="test_host",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password",
            schema="test_schema"
        )
        
        # Verify batch was created
        assert batch_id == 1
        
    def test_batch_preparation_with_config(self, mock_db_connection, monkeypatch, tmp_path):
        """Test DPAMBatchPreparation with configuration"""
        # Mock file system operations
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        
        # Set up config
        config = ConfigManager({
            "database": {
                "host": "test_host",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password"
            },
            "batch_dir": str(tmp_path)
        })
        
        # Mock cursor fetchall for structures
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchall.return_value = [
            (1, "AF-P12345-F1", "P12345"),
            (2, "AF-Q67890-F1", "Q67890")
        ]
        
        # Create batch preparation with config
        db_config = config.get_db_config()
        batch_prep = DPAMBatchPreparation(db_config, config.get("batch_dir"))
        
        # Mock the _process_structure method
        with patch.object(batch_prep, '_process_structure', return_value={'success': True}):
            result = batch_prep.prepare_batch_directory(1)
        
        # Verify database connection used correct config
        psycopg2.connect.assert_called_with(
            host="test_host",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password"
        )
        
        # Verify batch path was created with correct structure
        assert 'batch_path' in result
        
    def test_batch_supplementation_with_config(self, mock_db_connection, mock_requests, monkeypatch, tmp_path):
        """Test DPAMBatchSupplementation with configuration"""
        # Mock file operations
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        mock_open = MagicMock()
        monkeypatch.setattr('builtins.open', mock_open)
        
        # Set up config
        config = ConfigManager({
            "database": {
                "host": "test_host",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password"
            },
            "alphafold": {
                "base_url": "https://custom-alphafold.example.com/files"
            }
        })
        
        # Setup mock cursor results
        mock_conn, mock_cursor = mock_db_connection
        # Mock result for batch_path query
        mock_cursor.fetchone.side_effect = [
            (str(tmp_path),),  # batch path
        ]
        # Mock result for structures query
        mock_cursor.fetchall.return_value = [
            (1, "AF-P12345-F1", "P12345"),
            (2, "AF-Q67890-F1", "Q67890")
        ]
        
        # Create batch supplementation with config
        db_config = config.get_db_config()
        batch_supp = DPAMBatchSupplementation(db_config)
        
        # Override the alphafold URL if specified in config
        if config.get("alphafold.base_url"):
            batch_supp.afdb_base_url = config.get("alphafold.base_url")
        
        # Mock the _fetch_pae_file method
        with patch.object(batch_supp, '_fetch_pae_file', return_value={'success': True}):
            result = batch_supp.fetch_pae_files(1)
        
        # Verify custom URL was used
        assert batch_supp.afdb_base_url == "https://custom-alphafold.example.com/files"
        
        # Verify database connection used correct config
        psycopg2.connect.assert_called_with(
            host="test_host",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password"
        )
        
        # Verify metrics were returned
        assert 'metrics' in result
        
    def test_env_var_overrides_for_batch_config(self, monkeypatch):
        """Test environment variables override batch configuration"""
        # Set environment variables
        monkeypatch.setenv("DPAM_BATCH_DIR", "/env/batch/dir")
        monkeypatch.setenv("DPAM_ALPHAFOLD_BASE_URL", "https://env-alphafold.example.com/files")
        
        # Create config with different values
        config = ConfigManager({
            "batch_dir": "/original/batch/dir",
            "alphafold": {
                "base_url": "https://original-alphafold.example.com/files"
            }
        })
        
        # Environment variables should take precedence
        assert config.get("batch_dir") == "/env/batch/dir"
        assert config.get("alphafold.base_url") == "https://env-alphafold.example.com/files"

    def test_global_config_for_batch_modules(self, monkeypatch):
        """Test using global config for batch modules"""
        # Create a test config
        test_config = ConfigManager({
            "database": {
                "host": "global_host",
                "port": 5432,
                "dbname": "global_db",
                "user": "global_user",
                "password": "global_password"
            },
            "batch_dir": "/global/batch/dir",
            "api": {
                "host": "global_api_host",
                "port": 8000
            }
        })
        
        # Replace global config
        import dpam.config
        monkeypatch.setattr(dpam.config, "config_manager", test_config)
        
        # Get config for batch manager
        global_config = get_config()
        db_config = global_config.get_db_config()
        api_config = global_config.get("api")
        
        # Verify global config values
        assert db_config["host"] == "global_host"
        assert db_config["dbname"] == "global_db"
        assert global_config.get("batch_dir") == "/global/batch/dir"
        assert api_config["host"] == "global_api_host"