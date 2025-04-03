#!/usr/bin/env python3
"""
Integrated batch management for DPAM pipeline.

This module provides a unified interface to batch operations using the
DPAM configuration system.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from dpam.config import ConfigManager, get_config
from dpam.batch.manager import DPAMBatchManager
from dpam.batch.preparation import DPAMBatchPreparation
from dpam.batch.supplement import DPAMBatchSupplementation

# Configure logging
logger = logging.getLogger("dpam.batch")

class BatchController:
    """
    Unified controller for DPAM batch operations.
    
    This class integrates the various batch management modules with the
    DPAM configuration system.
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the BatchController with configuration.
        
        Args:
            config: Configuration manager instance (uses global config if None)
        """
        self.config = config or get_config()
        
        # Extract needed configurations
        self.db_config = self.config.get_db_config()
        self.api_config = self.config.get("api", {})
        self.batch_dir = self.config.get("batch_dir")
        
        # Ensure batch directory exists
        if self.batch_dir:
            os.makedirs(self.batch_dir, exist_ok=True)
        
        # Initialize the component managers
        self.batch_manager = DPAMBatchManager(self.db_config, self.api_config)
        self.batch_preparation = DPAMBatchPreparation(self.db_config, self.batch_dir)
        self.batch_supplementation = DPAMBatchSupplementation(self.db_config)
        
        # Customize AlphaFold base URL if specified in config
        alphafold_base_url = self.config.get("alphafold.base_url")
        if alphafold_base_url:
            self.batch_supplementation.afdb_base_url = alphafold_base_url
            # If preparation module also has this attribute, update it too
            if hasattr(self.batch_preparation, 'afdb_base_url'):
                self.batch_preparation.afdb_base_url = alphafold_base_url
    
    def create_batch(self, accessions: List[str], batch_name: Optional[str] = None, 
                     description: Optional[str] = None) -> int:
        """
        Create a new batch from UniProt accessions.
        
        Args:
            accessions: List of UniProt accession IDs
            batch_name: Optional name for the batch
            description: Optional description for the batch
            
        Returns:
            Batch ID
        """
        logger.info(f"Creating batch with {len(accessions)} accessions")
        
        # Validate input
        if not accessions:
            raise ValueError("No accessions provided")
        
        # Create batch using batch manager
        batch_id = self.batch_manager.create_batch_from_accessions(
            accessions, 
            batch_name=batch_name, 
            description=description
        )
        
        logger.info(f"Created batch with ID {batch_id}")
        return batch_id
    
    def prepare_batch(self, batch_id: int) -> Dict[str, Any]:
        """
        Prepare batch directory and download structure files.
        
        Args:
            batch_id: ID of the batch to prepare
            
        Returns:
            Dictionary with preparation results
        """
        logger.info(f"Preparing batch {batch_id}")
        
        # Prepare batch using batch preparation
        result = self.batch_preparation.prepare_batch_directory(batch_id)
        
        logger.info(f"Batch {batch_id} preparation completed with status {result.get('status')}")
        return result
    
    def supplement_batch(self, batch_id: int) -> Dict[str, Any]:
        """
        Download supplementary files for a batch (e.g., PAE files).
        
        Args:
            batch_id: ID of the batch to supplement
            
        Returns:
            Dictionary with supplementation results
        """
        logger.info(f"Supplementing batch {batch_id} with PAE files")
        
        # Download PAE files using batch supplementation
        result = self.batch_supplementation.fetch_pae_files(batch_id)
        
        logger.info(f"Batch {batch_id} supplementation completed with status {result.get('status')}")
        return result
    
    def run_full_preparation(self, accessions: List[str], 
                            batch_name: Optional[str] = None,
                            description: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete batch preparation workflow from accessions to ready-for-processing.
        
        Args:
            accessions: List of UniProt accession IDs
            batch_name: Optional name for the batch
            description: Optional description for the batch
            
        Returns:
            Dictionary with complete workflow results
        """
        # Create batch
        batch_id = self.create_batch(accessions, batch_name, description)
        
        # Prepare batch
        prep_result = self.prepare_batch(batch_id)
        
        # Supplement with PAE files if preparation was successful
        if prep_result.get('status') in ['READY', 'PARTIALLY_READY']:
            supp_result = self.supplement_batch(batch_id)
            
            # Combine results
            result = {
                'batch_id': batch_id,
                'preparation': prep_result,
                'supplementation': supp_result,
                'overall_status': supp_result.get('status')
            }
        else:
            result = {
                'batch_id': batch_id,
                'preparation': prep_result,
                'supplementation': None,
                'overall_status': prep_result.get('status')
            }
        
        logger.info(f"Full batch preparation completed with status {result['overall_status']}")
        return result

# Helper function to get a configured batch controller
def get_batch_controller(config: Optional[ConfigManager] = None) -> BatchController:
    """
    Get a configured batch controller instance.
    
    Args:
        config: Optional custom configuration (uses global config if None)
        
    Returns:
        Configured BatchController instance
    """
    return BatchController(config)