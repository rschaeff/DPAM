## dpam/steps/iterative_dali.py

import os
import subprocess
import logging
import tempfile
import shutil
import time
import json
from datetime import datetime
from pathlib import Path

class IterativeDaliRunner:
    """Executes iterative Dali searches for remote homology detection"""
    
    def __init__(self, config):
        """
        Initialize Dali runner with configuration
        
        Args:
            config (dict): Configuration containing paths and parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.iterative_dali")
        self.data_dir = config.get('data_dir', '/data')
        self.threads = config.get('dali_threads', 4)
        self.max_hits = config.get('dali_max_hits', 100)
        self.max_iterations = config.get('dali_max_iterations', 3)
        self.z_score_threshold = config.get('dali_z_score_threshold', 2.0)
        self.ecod_domain_db = os.path.join(self.data_dir, 'ECOD_domain_DB')
    
    def run(self, structure_id, structure_path, dali_candidates_path, output_dir):
        """
        Run iterative Dali searches for a structure
        
        Args:
            structure_id (str): Structure identifier
            structure_path (str): Path to query structure file (PDB/mmCIF)
            dali_candidates_path (str): Path to file with Dali search candidates
            output_dir (str): Directory for output files
            
        Returns:
            dict: Results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Starting iterative Dali for structure {structure_id}")
        
        # Create temporary working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = f"struct_{structure_id}"
            
            # Copy input structure to working directory
            local_structure = os.path.join(temp_dir, f"{prefix}.pdb")
            self._convert_to_pdb(structure_path, local_structure)
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                # Load candidate domains
                candidates = self._load_candidates(dali_candidates_path)
                self.logger.info(f"Loaded {len(candidates)} candidate domains for {structure_id}")
                
                if not candidates:
                    self.logger.warning(f"No candidate domains found for {structure_id}")
                    return {
                        "status": "COMPLETED",
                        "structure_id": structure_id,
                        "output_files": {
                            "dali_results": os.path.join(output_dir, f"{prefix}_dali_results.json"),
                            "hits_summary": os.path.join(output_dir, f"{prefix}_dali_hits_summary.tsv")
                        },
                        "metrics": {
                            "duration_seconds": (datetime.now() - start_time).total_seconds(),
                            "iterations": 0,
                            "hits": 0
                        }
                    }
                
                # Run iterative Dali search
                all_hits, iterations = self._run_iterative_dali(
                    local_structure, candidates, temp_dir, prefix
                )
                
                # Write results
                results_json_path = os.path.join(output_dir, f"{prefix}_dali_results.json")
                summary_path = os.path.join(output_dir, f"{prefix}_dali_hits_summary.tsv")
                
                self._write_results(all_hits, iterations, results_json_path, summary_path)
                
                # Return success and output paths
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                return {
                    "status": "COMPLETED",
                    "structure_id": structure_id,
                    "output_files": {
                        "dali_results": results_json_path,
                        "hits_summary": summary_path
                    },
                    "metrics": {
                        "duration_seconds": duration,
                        "iterations": iterations,
                        "hits": len(all_hits)
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Error running Dali for {structure_id}: {str(e)}")
                return {
                    "status": "FAILED",
                    "structure_id": structure_id,
                    "error_message": str(e)
                }
    
    def _convert_to_pdb(self, input_path, output_path):
        """Convert structure to PDB format if needed"""
        # If the input is already PDB, just copy it
        if input_path.endswith('.pdb'):
            shutil.copy(input_path, output_path)
            return
        
        # For mmCIF, use a tool like gemmi to convert
        try:
            # Using gemmi for conversion
            cmd = ["gemmi", "convert", "--to", "pdb", input_path, output_path]
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error converting structure: {e.stderr}")
            raise RuntimeError(f"Failed to convert structure to PDB: {e.stderr}")
    
    def _load_candidates(self, candidates_path):
        """Load candidate domains from file"""
        candidates = []
        with open(candidates_path, 'r') as f:
            for line in f:
                domain_id = line.strip()
                if domain_id:
                    candidates.append(domain_id)
        return candidates
    
    def _run_iterative_dali(self, query_structure, initial_candidates, work_dir, prefix):
        """Run iterative Dali searches until convergence or max iterations"""
        all_hits = {}
        seen_domains = set()
        current_candidates = initial_candidates[:self.max_hits]  # Limit initial search
        iterations = 0
        
        while current_candidates and iterations < self.max_iterations:
            iterations += 1
            self.logger.info(f"Starting Dali iteration {iterations} with {len(current_candidates)} candidates")
            
            # Prepare candidate list file
            candidates_file = os.path.join(work_dir, f"{prefix}_iter{iterations}_candidates.txt")
            with open(candidates_file, 'w') as f:
                for domain in current_candidates:
                    f.write(f"{domain}\n")
            
            # Run Dali for this iteration
            dali_results_file = os.path.join(work_dir, f"{prefix}_iter{iterations}_dali.txt")
            self._run_dali_search(query_structure, candidates_file, dali_results_file)
            
            # Parse results
            new_hits = self._parse_dali_results(dali_results_file, iteration=iterations)
            
            # Add new hits to all hits
            for domain_id, hit_info in new_hits.items():
                if domain_id not in all_hits:
                    all_hits[domain_id] = hit_info
            
            # Extract new candidates for next iteration
            current_candidates = []
            for domain_id, hit_info in new_hits.items():
                seen_domains.add(domain_id)
                
                # Only consider hits above threshold for expansion
                if hit_info['z_score'] >= self.z_score_threshold:
                    # Add related domains that we haven't seen yet
                    # This would use some domain relationship database
                    related_domains = self._get_related_domains(domain_id)
                    for related in related_domains:
                        if related not in seen_domains:
                            current_candidates.append(related)
            
            # Limit candidates for next iteration
            current_candidates = current_candidates[:self.max_hits]
            
            # If no new candidates, we're done
            if not current_candidates:
                self.logger.info(f"No new candidates found after iteration {iterations}, stopping")
                break
        
        return all_hits, iterations
    
    def _run_dali_search(self, query_structure, candidates_file, output_file):
        """Run Dali search against specified candidates"""
        # Command to run Dali
        # In practice, this would use the actual Dali command line options
        cmd = [
            "dali.pl",
            "--query", query_structure,
            "--db", self.ecod_domain_db,
            "--list", candidates_file,
            "--outfile", output_file,
            "--np", str(self.threads)
        ]
        
        self.logger.debug(f"Running Dali command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Dali search failed: {e.stderr}")
            raise RuntimeError(f"Dali search failed: {e.stderr}")
    
    def _parse_dali_results(self, results_file, iteration):
        """Parse Dali results from output file"""
        hits = {}
        
        try:
            with open(results_file, 'r') as f:
                # Skip header lines
                in_results = False
                for line in f:
                    line = line.strip()
                    
                    # Check if we're in the results section
                    if not in_results:
                        if line.startswith("# CHAIN"):
                            in_results = True
                        continue
                    
                    # Parse result line
                    if line and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) >= 10:
                            domain_id = parts[1]
                            z_score = float(parts[2])
                            rmsd = float(parts[3])
                            seq_id = float(parts[4])
                            coverage = float(parts[7])
                            
                            hits[domain_id] = {
                                'domain_id': domain_id,
                                'z_score': z_score,
                                'rmsd': rmsd,
                                'seq_id': seq_id,
                                'coverage': coverage,
                                'iteration_found': iteration
                            }
        except Exception as e:
            self.logger.error(f"Error parsing Dali results: {str(e)}")
            raise RuntimeError(f"Failed to parse Dali results: {str(e)}")
        
        return hits
    
    def _get_related_domains(self, domain_id):
        """Get related domains for expansion in next iteration"""
        # In a real implementation, this would query a domain relationship database
        # For now, return an empty list
        # In practice, you might use ECOD hierarchy or pre-computed domain relationships
        return []
    
    def _write_results(self, all_hits, iterations, results_json_path, summary_path):
        """Write results to output files"""
        # Write full JSON results
        with open(results_json_path, 'w') as f:
            json.dump({
                'hits': all_hits,
                'iterations': iterations,
                'total_hits': len(all_hits)
            }, f, indent=2)
        
        # Write summary TSV file
        with open(summary_path, 'w') as f:
            f.write("domain_id\tz_score\trmsd\tseq_id\tcoverage\titeration_found\n")
            
            # Sort hits by Z-score
            sorted_hits = sorted(
                all_hits.items(), 
                key=lambda x: x[1]['z_score'], 
                reverse=True
            )
            
            for domain_id, hit_info in sorted_hits:
                f.write(f"{domain_id}\t{hit_info['z_score']:.2f}\t{hit_info['rmsd']:.2f}\t"
                       f"{hit_info['seq_id']:.2f}\t{hit_info['coverage']:.2f}\t"
                       f"{hit_info['iteration_found']}\n")