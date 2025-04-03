# tests/unit/test_batch_controller.py
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import psycopg2
import requests

from dpam.config import ConfigManager
from dpam.batch.controller import BatchController, get_batch_controller

class TestBatchController:
    """Test suite for the integrated BatchController"""
    
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
            (str("/test/batch/path"),),  # batch_path
        ]
        
        # Configure fetchall for structures
        mock_cursor.fetchall.return_value = [
            (1, "AF-P12345-F1", "P12345"),
            (2, "AF-Q67890-F1", "Q67890")
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
    
    @pytest.fixture
    def test_config(self):
        """Fixture to provide test configuration"""
        return ConfigManager({
            "database": {
                "host": "test_host",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
                "schema": "test_schema"
            },
            "batch_dir": "/test/batch/dir",
            "api": {
                "host": "api_host",
                "port": 8000
            },
            "alphafold": {
                "base_url": "https://test-alphafold.example.com/files"
            }
        })
    
    def test_controller_initialization(self, test_config):
        """Test BatchController initialization with config"""
        controller = BatchController(test_config)
        
        # Check config values were properly extracted
        assert controller.db_config["host"] == "test_host"
        assert controller.db_config["dbname"] == "test_db"
        assert controller.batch_dir == "/test/batch/dir"
        assert controller.api_config["host"] == "api_host"
        
        # Check component instances were created
        assert controller.batch_manager is not None
        assert controller.batch_preparation is not None
        assert controller.batch_supplementation is not None
        
        # Check AlphaFold URL was customized
        assert controller.batch_supplementation.afdb_base_url == "https://test-alphafold.example.com/files"
    
    def test_create_batch(self, test_config, mock_db_connection, mock_requests):
        """Test creating a batch through the controller"""
        controller = BatchController(test_config)
        
        # Test with mock data
        accessions = ["P12345", "Q67890"]
        batch_id = controller.create_batch(accessions, "Test Batch", "Test description")
        
        # Verify batch was created
        assert batch_id == 1
        
        # Verify database was called with correct parameters
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.execute.assert_any_call(
            "INSERT INTO batches (name, description, status, created_at) "
            "VALUES (%s, %s, %s, NOW()) RETURNING batch_id",
            ("Test Batch", "Test description", "INITIALIZED")
        )
    
    def test_prepare_batch(self, test_config, mock_db_connection, monkeypatch):
        """Test preparing a batch through the controller"""
        controller = BatchController(test_config)
        
        # Mock necessary dependencies
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        
        # Mock the _process_structure method in batch_preparation
        with patch.object(controller.batch_preparation, '_process_structure', 
                         return_value={'success': True}):
            result = controller.prepare_batch(1)
        
        # Verify result
        assert result is not None
        
        # Verify database was called
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.execute.assert_any_call(
            "SELECT name FROM batches WHERE batch_id = %s",
            (1,)
        )
    
    def test_supplement_batch(self, test_config, mock_db_connection, mock_requests, monkeypatch):
        """Test supplementing a batch through the controller"""
        controller = BatchController(test_config)
        
        # Mock file operations
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        mock_open = MagicMock()
        monkeypatch.setattr('builtins.open', mock_open)
        
        # Mock the _fetch_pae_file method in batch_supplementation
        with patch.object(controller.batch_supplementation, '_fetch_pae_file', 
                         return_value={'success': True}):
            result = controller.supplement_batch(1)
        
        # Verify result
        assert result is not None
        assert 'metrics' in result
        
        # Verify database was called
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.execute.assert_any_call(
            "SELECT parameters->>'batch_path' FROM batches WHERE batch_id = %s",
            (1,)
        )
    
    def test_run_full_preparation(self, test_config, mock_db_connection, mock_requests, monkeypatch):
        """Test running the full preparation workflow"""
        controller = BatchController(test_config)
        
        # Mock necessary dependencies
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        mock_open = MagicMock()
        monkeypatch.setattr('builtins.open', mock_open)
        
        # Mock methods to control the test
        with patch.object(controller, 'create_batch', return_value=1) as mock_create, \
             patch.object(controller, 'prepare_batch', 
                     return_value={'status': 'READY'}) as mock_prepare, \
             patch.object(controller, 'supplement_batch', 
                     return_value={'status': 'READY_FOR_PROCESSING'}) as mock_supplement:
             
            # Run the full workflow
            result = controller.run_full_preparation(
                ["P12345", "Q67890"], 
                "Test Batch", 
                "Test description"
            )
        
        # Verify all methods were called
        mock_create.assert_called_once()
        mock_prepare.assert_called_once_with(1)
        mock_supplement.assert_called_once_with(1)
        
        # Verify result structure
        assert result['batch_id'] == 1
        assert result['preparation']['status'] == 'READY'
        assert result['supplementation']['status'] == 'READY_FOR_PROCESSING'
        assert result['overall_status'] == 'READY_FOR_PROCESSING'
    
    def test_get_batch_controller(self, test_config, monkeypatch):
        """Test the helper function to get a batch controller"""
        # Use the helper function
        controller = get_batch_controller(test_config)
        
        # Verify controller was created with the config
        assert controller.config == test_config
        assert controller.db_config["host"] == "test_host"
        
        # Test with global config
        import dpam.config
        monkeypatch.setattr(dpam.config, "config_manager", test_config)
        
        # Should use the global config
        global_controller = get_batch_controller()
        assert global_controller.config == test_config