# dpam/steps/hhsearch.py
"""
Implementation of HHSearch step for DPAM pipeline.
"""

import os
import subprocess
import json
from typing import Dict, List, Any, Optional

from dpam.steps.base import StepBase, StepResult
from dpam.gemmi_utils import get_structure_handler

class HHSearchStep(StepBase):
    """Step for running HHblits and HHsearch."""
    
    def run(self) -> StepResult:
        """
        Run the step.
        
        Returns:
            Step result
        """
        # Get parameters
        params = self._load_step_params()
        
        # Get structure handler
        handler = get_structure_handler()
        
        # Get structure path
        structure_path = self._get_structure_path()
        
        # Define output paths
        fasta_path = self._get_output_file(".fa")
        a3m_path = self._get_output_file(".a3m")
        hmm_path = self._get_output_file(".hmm")
        hhsearch_path = self._get_output_file(".hhsearch")
        
        # Extract sequence if not already done
        if not os.path.exists(fasta_path):
            self.logger.info("Extracting sequence from structure")
            
            try:
                # Read structure
                structure = handler.read_structure(structure_path)
                
                # Extract sequence
                sequences = handler.extract_sequence(structure)
                
                # Write FASTA
                with open(fasta_path, 'w') as f:
                    f.write(f">{self.structure_info['pdb_id']}\n")
                    f.write(f"{list(sequences.values())[0]}\n")
                
                self.logger.info(f"Wrote sequence to {fasta_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to extract sequence: {e}")
                return {
                    'status': 'FAILED',
                    'error_message': f"Failed to extract sequence: {e}"
                }
        
        # Run HHblits
        self.logger.info("Running HHblits")
        uniref_db = os.path.join(self.data_dir, "UniRef30_2022_02", "UniRef30_2022_02")
        hhblits_cmd = f"hhblits -cpu {self.threads} -i {fasta_path} -d {uniref_db} -oa3m {a3m_path}"
        
        exit_code, stdout, stderr = self._run_command(hhblits_cmd)
        if exit_code != 0:
            self.logger.error(f"HHblits failed: {stderr}")
            return {
                'status': 'FAILED',
                'error_message': f"HHblits failed: {stderr}"
            }
        
        # Add secondary structure
        self.logger.info("Adding secondary structure")
        addss_cmd = f"addss.pl {a3m_path} {a3m_path}.ss -a3m"
        exit_code, stdout, stderr = self._run_command(addss_cmd)
        
        # Move result
        os.rename(f"{a3m_path}.ss", a3m_path)
        
        # Create HMM
        self.logger.info("Creating HMM")
        hhmake_cmd = f"hhmake -i {a3m_path} -o {hmm_path}"
        exit_code, stdout, stderr = self._run_command(hhmake_cmd)
        if exit_code != 0:
            self.logger.error(f"HHmake failed: {stderr}")
            return {
                'status': 'FAILED',
                'error_message': f"HHmake failed: {stderr}"
            }
        
        # Run HHsearch
        self.logger.info("Running HHsearch")
        pdb70_db = os.path.join(self.data_dir, "pdb70", "pdb70")
        
        # Get parameters
        z_score = params.get('z_score', 100000)
        b_score = params.get('b_score', 100000)
        
        hhsearch_cmd = f"hhsearch -cpu {self.threads} -Z {z_score} -B {b_score} -i {hmm_path} -d {pdb70_db} -o {hhsearch_path}"
        exit_code, stdout, stderr = self._run_command(hhsearch_cmd)
        if exit_code != 0:
            self.logger.error(f"HHsearch failed: {stderr}")
            return {
                'status': 'FAILED',
                'error_message': f"HHsearch failed: {stderr}"
            }
        
        # Check if HHsearch found any hits
        hit_count = 0
        with open(hhsearch_path, 'r') as f:
            for line in f:
                if line.startswith("No "):
                    hit_count += 1
        
        self.logger.info(f"HHsearch found {hit_count} hits")
        
        # Return success result
        return {
            'status': 'COMPLETED',
            'output_files': {
                'fasta': fasta_path,
                'a3m': a3m_path,
                'hmm': hmm_path,
                'hhsearch': hhsearch_path
            },
            'metrics': {
                'cpu_time': self.threads * 3600,  # Estimate in seconds
                'hit_count': hit_count
            }
        }