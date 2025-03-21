## dpam/steps/foldseek.py

import os
import subprocess
import logging
import tempfile
import shutil
import time
from datetime import datetime

class FoldSeekRunner:
    """Encapsulates execution of FoldSeek for structure-based homology detection"""
    
    def __init__(self, config):
        """
        Initialize FoldSeek runner with configuration
        
        Args:
            config (dict): Configuration containing paths and parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.foldseek")
        self.data_dir = config.get('data_dir', '/data')
        self.threads = config.get('foldseek_threads', 4)
    
    def run(self, structure_id, pdb_path, output_dir):
        """
        Run FoldSeek for a given structure
        
        Args:
            structure_id (str): Structure identifier
            pdb_path (str): Path to input PDB/CIF file
            output_dir (str): Directory for output files
            
        Returns:
            dict: Results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Starting FoldSeek for structure {structure_id}")
        
        # Create temporary working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = f"struct_{structure_id}"
            
            # Copy PDB to working directory
            pdb_filename = os.path.basename(pdb_path)
            local_pdb_path = os.path.join(temp_dir, f"{prefix}.pdb")
            shutil.copy(pdb_path, local_pdb_path)
            
            # Create temporary directory for FoldSeek indexes
            foldseek_tmp = os.path.join(temp_dir, "foldseek_tmp")
            os.makedirs(foldseek_tmp, exist_ok=True)
            
            # Change to working directory
            original_dir = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Run FoldSeek easy-search
                self.logger.info(f"Running FoldSeek for {structure_id}")
                cmd = [
                    "foldseek", "easy-search",
                    local_pdb_path,
                    f"{self.data_dir}/ECOD_foldseek_DB/ECOD_foldseek_DB",
                    f"{prefix}.foldseek",
                    "foldseek_tmp",
                    "-e", "1000000",
                    "--max-seqs", "1000000",
                    "--threads", str(self.threads)
                ]
                
                # Execute the command and capture output
                self.logger.debug(f"Running: {' '.join(cmd)}")
                log_file = f"{prefix}_foldseek.log"
                with open(log_file, 'w') as f:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
                
                # Check if output file exists
                if not os.path.exists(f"{prefix}.foldseek"):
                    raise RuntimeError(f"FoldSeek did not produce expected output file: {prefix}.foldseek")
                
                # Create output directory and copy results
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{prefix}.foldseek")
                log_output_path = os.path.join(output_dir, f"{prefix}_foldseek.log")
                shutil.copy(f"{prefix}.foldseek", output_path)
                shutil.copy(log_file, log_output_path)
                
                # Return success and output paths
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                return {
                    "status": "COMPLETED",
                    "structure_id": structure_id,
                    "output_files": {
                        "foldseek": output_path,
                        "log": log_output_path
                    },
                    "metrics": {
                        "duration_seconds": duration,
                        "threads_used": self.threads
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Error running FoldSeek for {structure_id}: {str(e)}")
                return {
                    "status": "FAILED",
                    "structure_id": structure_id,
                    "error_message": str(e)
                }
            
            finally:
                # Return to original directory
                os.chdir(original_dir)
                # Clean up temporary folders
                if os.path.exists(foldseek_tmp):
                    shutil.rmtree(foldseek_tmp, ignore_errors=True)