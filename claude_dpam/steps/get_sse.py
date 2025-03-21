## dpam/steps/sse.py

import os
import subprocess
import logging
import tempfile
import shutil
import json
from datetime import datetime
from typing import Dict, List, Any

class SecondaryStructureAssigner:
    """Assigns secondary structure elements using DSSP"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DSSP runner with configuration
        
        Args:
            config: Configuration containing paths and parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.sse")
        self.data_dir = config.get('data_dir', '/data')
        self.dssp_binary = config.get('dssp_binary', 'mkdssp')
        self.min_helix_length = config.get('min_helix_length', 3)
        self.min_strand_length = config.get('min_strand_length', 2)
    
    def run(self, structure_id: str, structure_path: str, domain_support_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Run secondary structure assignment
        
        Args:
            structure_id: Structure identifier
            structure_path: Path to input structure file (PDB/mmCIF)
            domain_support_path: Path to domain support file
            output_dir: Directory for output files
            
        Returns:
            Dict with results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Starting secondary structure assignment for structure {structure_id}")
        
        prefix = f"struct_{structure_id}"
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Create temporary working directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prepare structure for DSSP
                local_structure_path = os.path.join(temp_dir, f"{prefix}.pdb")
                self._prepare_structure(structure_path, local_structure_path)
                
                # Run DSSP
                dssp_output_path = os.path.join(temp_dir, f"{prefix}.dssp")
                self._run_dssp(local_structure_path, dssp_output_path)
                
                # Parse DSSP output
                ss_data = self._parse_dssp_output(dssp_output_path)
                
                # Load domain data
                domains = self._load_domains(domain_support_path)
                
                # Calculate SSE for each domain
                domains_with_sse = self._calculate_domain_sse(domains, ss_data)
                
                # Write results
                results_path = os.path.join(output_dir, f"{prefix}_sse.json")
                summary_path = os.path.join(output_dir, f"{prefix}_sse_summary.tsv")
                
                self._write_results(domains_with_sse, ss_data, results_path, summary_path)
                
                # Copy DSSP output to results directory
                final_dssp_path = os.path.join(output_dir, f"{prefix}.dssp")
                shutil.copy(dssp_output_path, final_dssp_path)
            
            # Return success and output paths
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "status": "COMPLETED",
                "structure_id": structure_id,
                "output_files": {
                    "sse_json": results_path,
                    "sse_summary": summary_path,
                    "dssp_output": final_dssp_path
                },
                "metrics": {
                    "duration_seconds": duration,
                    "domains_processed": len(domains),
                    "residues_processed": len(ss_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error assigning secondary structure for {structure_id}: {str(e)}")
            return {
                "status": "FAILED",
                "structure_id": structure_id,
                "error_message": str(e)
            }
    
    def _prepare_structure(self, input_path: str, output_path: str) -> None:
        """Convert structure to PDB format if needed"""
        # If the input is already PDB, just copy it
        if input_path.endswith('.pdb'):
            shutil.copy(input_path, output_path)
            return
        
        # For mmCIF, use gemmi to convert
        try:
            # Using gemmi for conversion
            cmd = ["gemmi", "convert", "--to", "pdb", input_path, output_path]
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error converting structure: {e.stderr}")
            raise RuntimeError(f"Failed to convert structure to PDB: {e.stderr}")
    
    def _run_dssp(self, input_path: str, output_path: str) -> None:
        """Run DSSP on the input structure"""
        try:
            cmd = [self.dssp_binary, "-i", input_path, "-o", output_path]
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            self.logger.debug(f"DSSP output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"DSSP failed: {e.stderr}")
            raise RuntimeError(f"DSSP failed: {e.stderr}")
    
    def _parse_dssp_output(self, dssp_path: str) -> Dict[int, Dict[str, Any]]:
        """Parse DSSP output file"""
        ss_data = {}
        
        try:
            with open(dssp_path, 'r') as f:
                lines = f.readlines()
                
            # Find the start of the residue data (line with #)
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith("  #  RESIDUE"):
                    data_start = i + 1
                    break
            
            # Parse residue data
            for line in lines[data_start:]:
                if len(line) < 17:  # Skip short lines
                    continue
                    
                try:
                    resnum = int(line[5:10].strip())
                    resname = line[10:14].strip()
                    chain = line[11:12].strip()
                    ss = line[16:17].strip()
                    
                    # Convert DSSP SS codes to simplified categories
                    ss_type = self._simplify_ss_code(ss)
                    
                    # Additional data (if needed)
                    acc = float(line[35:38].strip()) if line[35:38].strip() else 0.0
                    phi = float(line[103:109].strip()) if line[103:109].strip() else 0.0
                    psi = float(line[109:115].strip()) if line[109:115].strip() else 0.0
                    
                    ss_data[resnum] = {
                        'residue_num': resnum,
                        'residue_name': resname,
                        'chain': chain,
                        'ss_code': ss,
                        'ss_type': ss_type,
                        'accessibility': acc,
                        'phi': phi,
                        'psi': psi
                    }
                    
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error parsing DSSP line: {line}: {e}")
                    continue
            
            return ss_data
            
        except Exception as e:
            self.logger.error(f"Error parsing DSSP output: {e}")
            raise
    
    def _simplify_ss_code(self, ss_code: str) -> str:
        """Convert DSSP SS codes to simplified types (H, E, L)"""
        if not ss_code:
            return "L"  # Default to loop
            
        if ss_code in ["H", "G", "I"]:  # Helix types
            return "H"
        elif ss_code in ["E", "B"]:  # Strand types
            return "E"
        else:  # Everything else is loop
            return "L"
    
    def _load_domains(self, domain_support_path: str) -> List[Dict[str, Any]]:
        """Load domain definitions from support file"""
        with open(domain_support_path, 'r') as f:
            data = json.load(f)
        
        return data.get('domains', [])
    
    def _calculate_domain_sse(self, domains: List[Dict[str, Any]], ss_data: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate SSE composition for each domain"""
        result_domains = []
        
        for domain in domains:
            start = domain.get('start', 0)
            end = domain.get('end', 0)
            
            # Count SSE types in domain
            h_count = 0
            e_count = 0
            l_count = 0
            
            # Collect SSE segments
            current_segment = {'type': None, 'start': 0, 'end': 0}
            segments = []
            
            # Process residues in domain
            for resnum in range(start, end + 1):
                if resnum not in ss_data:
                    continue
                
                ss_type = ss_data[resnum]['ss_type']
                
                # Count by type
                if ss_type == 'H':
                    h_count += 1
                elif ss_type == 'E':
                    e_count += 1
                elif ss_type == 'L':
                    l_count += 1
                
                # Track segments
                if current_segment['type'] != ss_type:
                    # End previous segment if it exists
                    if current_segment['type'] is not None:
                        current_segment['end'] = resnum - 1
                        
                        # Add segment if it meets length requirements
                        if (current_segment['type'] == 'H' and 
                            current_segment['end'] - current_segment['start'] + 1 >= self.min_helix_length) or \
                           (current_segment['type'] == 'E' and 
                            current_segment['end'] - current_segment['start'] + 1 >= self.min_strand_length) or \
                           (current_segment['type'] == 'L'):
                            segments.append(current_segment.copy())
                    
                    # Start new segment
                    current_segment = {'type': ss_type, 'start': resnum, 'end': resnum}
                else:
                    # Continue current segment
                    current_segment['end'] = resnum
            
            # Add final segment
            if current_segment['type'] is not None:
                segments.append(current_segment)
            
            # Calculate domain length with SS assignment
            domain_len = h_count + e_count + l_count
            
            # Skip if no SS data for domain
            if domain_len == 0:
                domain_copy = domain.copy()
                domain_copy['ss_composition'] = None
                result_domains.append(domain_copy)
                continue
            
            # Calculate SS composition percentages
            h_percent = h_count / domain_len * 100 if domain_len > 0 else 0
            e_percent = e_count / domain_len * 100 if domain_len > 0 else 0
            l_percent = l_count / domain_len * 100 if domain_len > 0 else 0
            
            # Determine domain class (alpha, beta, alpha/beta, etc.)
            domain_class = self._determine_domain_class(h_percent, e_percent)
            
            # Update domain with SS information
            domain_copy = domain.copy()
            domain_copy.update({
                'ss_composition': {
                    'helix_count': h_count,
                    'strand_count': e_count,
                    'loop_count': l_count,
                    'helix_percent': h_percent,
                    'strand_percent': e_percent,
                    'loop_percent': l_percent,
                    'domain_class': domain_class
                },
                'ss_segments': segments
            })
            
            result_domains.append(domain_copy)
        
        return result_domains
    
    def _determine_domain_class(self, h_percent: float, e_percent: float) -> str:
        """Determine domain class based on secondary structure composition"""
        if h_percent >= 40 and e_percent < 20:
            return "alpha"
        elif e_percent >= 40 and h_percent < 20:
            return "beta"
        elif h_percent >= 20 and e_percent >= 20:
            return "alpha/beta"
        else:
            return "irregular"
    
    def _write_results(self, domains: List[Dict[str, Any]], ss_data: Dict[int, Dict[str, Any]], 
                      results_path: str, summary_path: str) -> None:
        """Write results to output files"""
        # Write full JSON results
        with open(results_path, 'w') as f:
            json.dump({
                'domains': domains,
                'ss_data': ss_data
            }, f, indent=2)
        
        # Write summary TSV
        with open(summary_path, 'w') as f:
            f.write("domain_id\tstart\tend\tsize\thelix_percent\tstrand_percent\tloop_percent\tdomain_class\tsupport_score\n")
            
            for domain in domains:
                domain_id = domain.get('domain_id', 'unknown')
                start = domain.get('start', 0)
                end = domain.get('end', 0)
                size = domain.get('size', 0)
                
                ss_comp = domain.get('ss_composition', {})
                if ss_comp:
                    h_percent = ss_comp.get('helix_percent', 0)
                    e_percent = ss_comp.get('strand_percent', 0)
                    l_percent = ss_comp.get('loop_percent', 0)
                    domain_class = ss_comp.get('domain_class', 'unknown')
                else:
                    h_percent = 0
                    e_percent = 0
                    l_percent = 0
                    domain_class = 'unknown'
                
                support = domain.get('overall_support', 0)
                
                f.write(f"{domain_id}\t{start}\t{end}\t{size}\t"
                       f"{h_percent:.1f}\t{e_percent:.1f}\t{l_percent:.1f}\t"
                       f"{domain_class}\t{support:.3f}\n")