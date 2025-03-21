#!/usr/bin/env python3
"""
Database connection and utility functions for DPAM pipeline.

This module provides database connection management and utility functions
for interacting with the PostgreSQL database used by DPAM.
"""

import os
import logging
import json
import time
import threading
import contextlib
from typing import Dict, Any, Optional, List, Tuple, Union, Generator
from datetime import datetime

import psycopg2
import psycopg2.extras
import psycopg2.pool

from dpam.config import get_config
from dpam.exceptions import (
    DatabaseConnectionError, 
    DatabaseQueryError,
    DatabaseTransactionError,
    DatabasePoolError
)

# Configure logging
logger = logging.getLogger("dpam.database")

# Thread-local storage for connections
_thread_local = threading.local()

# Global connection pool
_connection_pool = None
_pool_lock = threading.Lock()

class DatabaseManager:
    """
    Database connection manager for DPAM pipeline.
    
    Provides connection management and utility functions for
    database operations.
    """
    
    def __init__(self, db_config: Dict[str, Any] = None):
        """
        Initialize database manager.
        
        Args:
            db_config: Database configuration dictionary
        """
        self.db_config = db_config or get_config().get_db_config()
        self.schema = self.db_config.pop('schema', 'dpam_queue')
        self._conn = None
    
    def connect(self) -> psycopg2.extensions.connection:
        """
        Create a new database connection.
        
        Returns:
            Database connection
        
        Raises:
            DatabaseConnectionError: If connection fails
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            # Set autocommit to False by default
            conn.autocommit = False
            
            # Set search path to schema
            with conn.cursor() as cursor:
                cursor.execute(f"SET search_path TO {self.schema}")
            
            logger.debug(f"Connected to database {self.db_config['dbname']}")
            return conn
            
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise DatabaseConnectionError(f"Failed to connect to database: {e}")
    
    def get_connection(self) -> psycopg2.extensions.connection:
        """
        Get database connection, creating if needed.
        
        Returns:
            Database connection
        """
        if self._conn is None or self._conn.closed:
            self._conn = self.connect()
        return self._conn
    
    def close(self) -> None:
        """Close database connection if open."""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")
    
    def execute(self, query: str, params: Optional[Tuple] = None, 
               commit: bool = False) -> int:
        """
        Execute query and return affected row count.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            commit: Whether to commit after execution
            
        Returns:
            Number of affected rows
            
        Raises:
            DatabaseQueryError: If query fails
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                affected = cursor.rowcount
                
            if commit:
                conn.commit()
                
            return affected
                
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Query execution failed: {e}, SQL: {query}, params: {params}")
            raise DatabaseQueryError(f"Query execution failed: {e}", query, params)
    
    def fetchone(self, query: str, params: Optional[Tuple] = None) -> Optional[Dict[str, Any]]:
        """
        Execute query and fetch one result row as dictionary.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Result row as dictionary or None
            
        Raises:
            DatabaseQueryError: If query fails
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                return cursor.fetchone()
                
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Query execution failed: {e}, SQL: {query}, params: {params}")
            raise DatabaseQueryError(f"Query execution failed: {e}", query, params)
    
    def fetchall(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute query and fetch all result rows as dictionaries.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            List of result rows as dictionaries
            
        Raises:
            DatabaseQueryError: If query fails
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
                
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Query execution failed: {e}, SQL: {query}, params: {params}")
            raise DatabaseQueryError(f"Query execution failed: {e}", query, params)
    
    def fetchiter(self, query: str, params: Optional[Tuple] = None, 
                 batch_size: int = 1000) -> Generator[Dict[str, Any], None, None]:
        """
        Execute query and yield result rows as dictionaries.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            batch_size: Number of rows to fetch at once
            
        Yields:
            Result rows as dictionaries
            
        Raises:
            DatabaseQueryError: If query fails
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                while True:
                    results = cursor.fetchmany(batch_size)
                    if not results:
                        break
                    for row in results:
                        yield row
                
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Query execution failed: {e}, SQL: {query}, params: {params}")
            raise DatabaseQueryError(f"Query execution failed: {e}", query, params)
    
    @contextlib.contextmanager
    def transaction(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Context manager for database transactions.
        
        Yields:
            Database connection
            
        Raises:
            DatabaseTransactionError: If transaction fails
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction failed: {e}")
            raise DatabaseTransactionError(f"Transaction failed: {e}")
    
    def insert(self, table: str, data: Dict[str, Any], returning: str = None) -> Optional[Any]:
        """
        Insert row into table.
        
        Args:
            table: Table name
            data: Dictionary of column name -> value
            returning: Column to return from inserted row
            
        Returns:
            Value of returning column if specified
            
        Raises:
            DatabaseQueryError: If query fails
        """
        columns = list(data.keys())
        values = list(data.values())
        placeholders = [f"%s" for _ in range(len(columns))]
        
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        
        if returning:
            query += f" RETURNING {returning}"
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, values)
                conn.commit()
                
                if returning:
                    return cursor.fetchone()[0]
                return None
                
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Insert failed: {e}, table: {table}, data: {data}")
            raise DatabaseQueryError(f"Insert failed: {e}", query, values)
    
    def update(self, table: str, data: Dict[str, Any], condition: str, 
              condition_params: Tuple = None, returning: str = None) -> Optional[Any]:
        """
        Update rows in table.
        
        Args:
            table: Table name
            data: Dictionary of column name -> value
            condition: WHERE condition
            condition_params: Condition parameters
            returning: Column to return from updated rows
            
        Returns:
            Value of returning column if specified
            
        Raises:
            DatabaseQueryError: If query fails
        """
        columns = list(data.keys())
        values = list(data.values())
        set_clause = ", ".join([f"{col} = %s" for col in columns])
        
        query = f"UPDATE {table} SET {set_clause} WHERE {condition}"
        
        if returning:
            query += f" RETURNING {returning}"
        
        params = tuple(values) + (condition_params or ())
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                conn.commit()
                
                if returning:
                    result = cursor.fetchone()
                    return result[0] if result else None
                return None
                
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Update failed: {e}, table: {table}, data: {data}, condition: {condition}")
            raise DatabaseQueryError(f"Update failed: {e}", query, params)
    
    def upsert(self, table: str, data: Dict[str, Any], key_columns: List[str],
              returning: str = None) -> Optional[Any]:
        """
        Upsert row into table (insert or update).
        
        Args:
            table: Table name
            data: Dictionary of column name -> value
            key_columns: Columns identifying unique row
            returning: Column to return from upserted row
            
        Returns:
            Value of returning column if specified
            
        Raises:
            DatabaseQueryError: If query fails
        """
        columns = list(data.keys())
        values = list(data.values())
        placeholders = [f"%s" for _ in range(len(columns))]
        
        # Set clause for ON CONFLICT
        update_columns = [col for col in columns if col not in key_columns]
        if not update_columns:
            # Nothing to update, just do INSERT ... ON CONFLICT DO NOTHING
            set_clause = "DO NOTHING"
        else:
            set_expressions = [f"{col} = EXCLUDED.{col}" for col in update_columns]
            set_clause = f"DO UPDATE SET {', '.join(set_expressions)}"
        
        query = (f"INSERT INTO {table} ({', '.join(columns)}) "
                f"VALUES ({', '.join(placeholders)}) "
                f"ON CONFLICT ({', '.join(key_columns)}) {set_clause}")
        
        if returning:
            query += f" RETURNING {returning}"
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, values)
                conn.commit()
                
                if returning:
                    result = cursor.fetchone()
                    return result[0] if result else None
                return None
                
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Upsert failed: {e}, table: {table}, data: {data}")
            raise DatabaseQueryError(f"Upsert failed: {e}", query, values)
    
    def create_jsonb_patch(self, data: Dict[str, Any]) -> str:
        """
        Create JSONB patch expression for updating JSONB columns.
        
        Args:
            data: Dictionary of key -> value to patch
            
        Returns:
            JSONB patch expression
        """
        return json.dumps(data)
    
    def execute_script(self, script: str) -> None:
        """
        Execute SQL script.
        
        Args:
            script: SQL script to execute
            
        Raises:
            DatabaseQueryError: If script execution fails
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(script)
                conn.commit()
                
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Script execution failed: {e}")
            raise DatabaseQueryError(f"Script execution failed: {e}", script)
    
    def __enter__(self) -> 'DatabaseManager':
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, closing connection."""
        self.close()

def init_connection_pool(min_conn: int = 2, max_conn: int = 10) -> None:
    """
    Initialize database connection pool.
    
    Args:
        min_conn: Minimum number of connections
        max_conn: Maximum number of connections
        
    Raises:
        DatabasePoolError: If pool initialization fails
    """
    global _connection_pool
    
    with _pool_lock:
        if _connection_pool is not None:
            return
            
        try:
            db_config = get_config().get_db_config()
            schema = db_config.pop('schema', 'dpam_queue')
            
            _connection_pool = psycopg2.pool.ThreadedConnectionPool(
                min_conn, max_conn, **db_config
            )
            
            # Test connection
            conn = _connection_pool.getconn()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(f"SET search_path TO {schema}")
                    cursor.execute("SELECT 1")
                    
                logger.info(f"Connection pool initialized with {min_conn}-{max_conn} connections")
                
            finally:
                _connection_pool.putconn(conn)
                
        except psycopg2.Error as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise DatabasePoolError(f"Failed to initialize connection pool: {e}")

def get_db_connection() -> psycopg2.extensions.connection:
    """
    Get database connection from pool.
    
    Returns:
        Database connection
    
    Raises:
        DatabaseConnectionError: If connection fails
    """
    global _connection_pool
    
    # Initialize pool if needed
    if _connection_pool is None:
        init_connection_pool()
    
    # Get thread-local connection
    if not hasattr(_thread_local, 'conn'):
        try:
            _thread_local.conn = _connection_pool.getconn()
            _thread_local.conn.autocommit = False
            
            # Set search path to schema
            db_config = get_config().get_db_config()
            schema = db_config.get('schema', 'dpam_queue')
            
            with _thread_local.conn.cursor() as cursor:
                cursor.execute(f"SET search_path TO {schema}")
                
            logger.debug("Got connection from pool")
            
        except psycopg2.Error as e:
            logger.error(f"Failed to get connection from pool: {e}")
            raise DatabaseConnectionError(f"Failed to get connection from pool: {e}")
    
    return _thread_local.conn

def release_db_connection() -> None:
    """Release database connection back to pool."""
    global _connection_pool
    
    if _connection_pool is not None and hasattr(_thread_local, 'conn'):
        _connection_pool.putconn(_thread_local.conn)
        delattr(_thread_local, 'conn')
        logger.debug("Released connection to pool")

@contextlib.contextmanager
def db_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Context manager for database connections.
    
    Yields:
        Database connection
        
    Raises:
        DatabaseConnectionError: If connection fails
    """
    conn = get_db_connection()
    try:
        yield conn
    finally:
        release_db_connection()

@contextlib.contextmanager
def db_transaction() -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Context manager for database transactions.
    
    Yields:
        Database connection
        
    Raises:
        DatabaseTransactionError: If transaction fails
    """
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Transaction failed: {e}")
        raise DatabaseTransactionError(f"Transaction failed: {e}")
    finally:
        release_db_connection()