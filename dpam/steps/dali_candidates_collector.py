#!/usr/bin/env python3
"""
Upstream module for defining DALI search candidates based on fast search

This module defines the candidate collector class, it is upstream of iterative_dali
"""

import os
import logging
from datetime import datetime

class DaliCandidatesCollector:
    """Collects ECOD domains from HHSearch and FoldSeek for Dali search"""
    
    def __init__(self, config):
        """
        Initialize Dali candidates collector with configuration
        
        Args:
            config (dict): Configuration containing parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.dali_candidates")
    
    def run(self, structure_id, ecod_mapping_path, foldseek_filtered_path, output_dir):
        """
        Collect candidate domains for Dali search
        
        Args:
            structure_id (str): Structure identifier
            ecod_mapping_path (str): Path to ECOD mapping results
            foldseek_filtered_path (str): Path to filtered FoldSeek results
            output_dir (str): Directory for output files
            
        Returns:
            dict: Results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Collecting Dali candidates for structure {structure_id}")
        
        prefix = f"struct_{structure_id}"
        
        try:
            # Create set of unique domains from ECOD mapping
            domains = set()
            
            # Read ECOD mapping results
            if os.path.exists(ecod_mapping_path):
                self.logger.debug(f"Reading ECOD mappings from {ecod_mapping_path}")
                with open(ecod_mapping_path, 'r') as fp:
                    for countl, line in enumerate(fp):
                        if countl > 0:  # Skip header
                            words = line.split()
                            domains.add(words[0])
            
            # Read filtered FoldSeek results
            if os.path.exists(foldseek_filtered_path):
                self.logger.debug(f"Reading filtered FoldSeek results from {foldseek_filtered_path}")
                with open(foldseek_filtered_path, 'r') as fp:
                    for countl, line in enumerate(fp):
                        if countl > 0:  # Skip header
                            words = line.split()
                            domains.add(words[0])
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{prefix}_hits4Dali")
            
            # Write domains to output file
            with open(output_path, 'w') as rp:
                for domain in domains:
                    rp.write(f"{domain}\n")
            
            self.logger.info(f"Wrote {len(domains)} Dali candidates to {output_path}")
            
            # Return success and output paths
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "status": "COMPLETED",
                "structure_id": structure_id,
                "output_files": {
                    "dali_candidates": output_path
                },
                "metrics": {
                    "duration_seconds": duration,
                    "candidate_count": len(domains)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting Dali candidates for {structure_id}: {str(e)}")
            return {
                "status": "FAILED",
                "structure_id": structure_id,
                "error_message": str(e)
            }