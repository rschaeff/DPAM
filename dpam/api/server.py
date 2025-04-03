#!/usr/bin/env python3
"""
REST API server implementation for DPAM pipeline.

This module provides the FastAPI server that exposes the DPAM pipeline
functionality through a RESTful API.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseSettings

# Add DPAM to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dpam.config import ConfigManager, load_config
from dpam.database import DatabaseManager
from dpam.pipeline.controller import DPAMPipelineController
from dpam.batch.manager import DPAMBatchManager
from dpam.batch.preparation import DPAMBatchPreparation
from dpam.api.routes import setup_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dpam.api.server")

class APISettings(BaseSettings):
    """API server settings"""
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    timeout: int = 60
    allow_origins: List[str] = ["*"]
    
    class Config:
        env_prefix = "DPAM_API_"

def create_app(config_path: Optional[str] = None) -> FastAPI:
    """
    Create and configure the FastAPI application
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured FastAPI application
    """
    # Load DPAM configuration
    config = load_config(config_path)
    
    # Check if required configuration exists
    missing = config.validate()
    if missing:
        logger.error(f"Missing required configuration: {missing}")
        raise ValueError(f"Missing required configuration: {missing}")
    
    # Load API settings from configuration
    api_config = config.get('api', {})
    settings = APISettings(
        host=api_config.get('host', 'localhost'),
        port=api_config.get('port', 8000),
        debug=api_config.get('debug', False),
        workers=api_config.get('workers', 4),
        timeout=api_config.get('timeout', 60),
        allow_origins=api_config.get('allow_origins', ["*"])
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="DPAM API",
        description="Domain Parser for AlphaFold Models - REST API",
        version="0.1.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # State for dependency injection
    app.state.config = config
    app.state.settings = settings
    
    # Initialize controllers
    db_config = config.get_db_config()
    data_dir = config.get('data_dir', '/data/dpam')
    batch_dir = config.get('batch_dir', '/data/dpam/batches')
    grid_config = config.get('grid', {})
    
    # Create database manager
    app.state.db_manager = DatabaseManager(db_config)
    
    # Create controllers
    app.state.pipeline_controller = DPAMPipelineController(db_config, grid_config, data_dir)
    app.state.batch_manager = DPAMBatchManager(db_config, {"api_base": api_config.get('base_url', '')})
    app.state.batch_preparation = DPAMBatchPreparation(db_config, batch_dir)
    
    # Setup API routes
    setup_routes(app)
    
    @app.on_event("startup")
    async def startup_event():
        """Run on server startup"""
        logger.info(f"Starting DPAM API server on {settings.host}:{settings.port}")
        
        # Test database connection
        try:
            with app.state.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Run on server shutdown"""
        logger.info("Shutting down DPAM API server")
        
        # Close any resources
        app.state.db_manager.close()
    
    return app

def start_server(config_path: Optional[str] = None):
    """
    Start the API server
    
    Args:
        config_path: Path to configuration file
    """
    app = create_app(config_path)
    settings = app.state.settings
    
    # Run the server
    uvicorn.run(
        "dpam.api.server:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.debug,
        timeout_keep_alive=settings.timeout
    )

# Create app instance for running with uvicorn
app = create_app()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DPAM API Server")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--host", help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Override settings from command line
    app = create_app(args.config)
    settings = app.state.settings
    
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.debug:
        settings.debug = True
    
    # Start server
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        timeout_keep_alive=settings.timeout
    )