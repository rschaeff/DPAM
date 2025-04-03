#!/usr/bin/env python3
"""
DPAM: Domain Parser for AlphaFold Models

A pipeline for automated protein domain identification, 
classification, and analysis using multiple sources of evidence.
"""

import os
import logging
import logging.config

__version__ = '0.1.0'
__author__ = 'DPAM Development Team'
__license__ = 'MIT'

# Configure package-wide logging
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': os.environ.get('DPAM_LOG_FILE', 'dpam.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 10
        }
    },
    'loggers': {
        'dpam': {
            'level': os.environ.get('DPAM_LOG_LEVEL', 'INFO'),
            'handlers': ['console', 'file'],
            'propagate': False
        }
    }
}

try:
    logging.config.dictConfig(logging_config)
except Exception as e:
    # Fallback to basic configuration if the above fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.warning(f"Failed to configure logging using dictConfig: {e}")
    logging.warning("Falling back to basic logging configuration.")

# Create logger for this module
logger = logging.getLogger("dpam")
logger.debug("DPAM package initialization")

# Import primary modules for easy access
try:
    from dpam.config import ConfigManager, load_config
    from dpam.database import get_db_connection, DatabaseManager
    
    # Import pipeline components
    from dpam.pipeline.controller import DPAMPipelineController
    from dpam.pipeline.checkpoints import DPAMBatchCheckpoints
    from dpam.pipeline.quality import DPAMQualityControl
    from dpam.pipeline.errors import DPAMErrorHandler
    
    # Import batch operations
    from dpam.batch.manager import DPAMBatchManager
    from dpam.batch.preparation import DPAMBatchPreparation
    from dpam.batch.supplement import DPAMBatchSupplementation
    
    # Import grid operations
    from dpam.grid.manager import DPAMOpenGridManager
    
    __all__ = [
        'ConfigManager', 'load_config',
        'get_db_connection', 'DatabaseManager',
        'DPAMPipelineController', 'DPAMBatchCheckpoints',
        'DPAMQualityControl', 'DPAMErrorHandler',
        'DPAMBatchManager', 'DPAMBatchPreparation', 'DPAMBatchSupplementation',
        'DPAMOpenGridManager'
    ]
    
except ImportError as e:
    logger.warning(f"Could not import some modules: {e}")
    logger.warning("Some functionality may be unavailable.")