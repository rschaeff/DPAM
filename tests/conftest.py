import pytest
import tempfile
import os
from unittest.mock import Mock

@pytest.fixture
def test_config():
    """Provide test configuration"""
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_pass"
        },
        "data_dir": "/tmp/test_data"
    }

@pytest.fixture
def mock_db_connection():
    """Provide a mock database connection"""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn, mock_cursor

@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's removed after the test"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir