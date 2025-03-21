#!/usr/bin/env python3
"""
Wraps HHSearch for domain detection
"""

import os
import subprocess
import logging
import tempfile
import shutil
from datetime import datetime

class HHSearchRunner:
    """Encapsulates execution of HHSearch pipeline for sequence-based homology detection"""
    
    def __init__(self, config):
        """
        Initialize HHSearch runner with configuration
        
        Args:
            config (dict): Configuration containing paths and parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.hhsearch")
        self.data_dir = config.get('data_dir', '/data')
        self.threads = config.get('hhsearch_threads', 4)
    
    def run(self, structure_id, fasta_path, output_dir):
        """
        Run HHSearch pipeline for a given structure
        
        Args:
            structure_id (str): Structure identifier
            fasta_path (str): Path to input FASTA file
            output_dir (str): Directory for output files
            
        Returns:
            dict: Results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Starting HHSearch for structure {structure_id}")
        
        # Create temporary working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = f"struct_{structure_id}"
            
            # Copy fasta to working directory
            shutil.copy(fasta_path, os.path.join(temp_dir, f"{prefix}.fa"))
            
            # Change to working directory
            original_dir = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Run HHBlits to generate MSA
                self.logger.info(f"Running HHBlits for {structure_id}")
                hhblits_cmd = [
                    "hhblits",
                    "-cpu", str(self.threads),
                    "-i", f"{prefix}.fa",
                    "-d", f"{self.data_dir}/UniRef30_2022_02/UniRef30_2022_02",
                    "-oa3m", f"{prefix}.a3m"
                ]
                self._run_command(hhblits_cmd)
                
                # Add secondary structure
                self.logger.info(f"Adding secondary structure for {structure_id}")
                addss_cmd = [
                    "addss.pl",
                    f"{prefix}.a3m",
                    f"{prefix}.a3m.ss",
                    "-a3m"
                ]
                self._run_command(addss_cmd)
                
                # Move the SS augmented MSA back to original name
                os.rename(f"{prefix}.a3m.ss", f"{prefix}.a3m")
                
                # Run HHMake to generate HMM
                self.logger.info(f"Running HHMake for {structure_id}")
                hhmake_cmd = [
                    "hhmake",
                    "-i", f"{prefix}.a3m",
                    "-o", f"{prefix}.hmm"
                ]
                self._run_command(hhmake_cmd)
                
                # Run HHSearch against PDB70
                self.logger.info(f"Running HHSearch for {structure_id}")
                hhsearch_cmd = [
                    "hhsearch",
                    "-cpu", str(self.threads),
                    "-Z", "100000",
                    "-B", "100000",
                    "-i", f"{prefix}.hmm",
                    "-d", f"{self.data_dir}/pdb70/pdb70",
                    "-o", f"{prefix}.hhsearch"
                ]
                self._run_command(hhsearch_cmd)
                
                # Copy results to output directory
                os.makedirs(output_dir, exist_ok=True)
                for ext in [".a3m", ".hmm", ".hhsearch"]:
                    src = os.path.join(temp_dir, f"{prefix}{ext}")
                    dst = os.path.join(output_dir, f"{prefix}{ext}")
                    shutil.copy(src, dst)
                
                # Return success and output paths
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                return {
                    "status": "COMPLETED",
                    "structure_id": structure_id,
                    "output_files": {
                        "a3m": os.path.join(output_dir, f"{prefix}.a3m"),
                        "hmm": os.path.join(output_dir, f"{prefix}.hmm"),
                        "hhsearch": os.path.join(output_dir, f"{prefix}.hhsearch")
                    },
                    "metrics": {
                        "duration_seconds": duration,
                        "threads_used": self.threads
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Error running HHSearch for {structure_id}: {str(e)}")
                return {
                    "status": "FAILED",
                    "structure_id": structure_id,
                    "error_message": str(e)
                }
            
            finally:
                # Return to original directory
                os.chdir(original_dir)
    
    def _run_command(self, cmd):
        """Run a command and handle errors"""
        self.logger.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with code {result.returncode}: {result.stderr}")
        
        return result.stdout