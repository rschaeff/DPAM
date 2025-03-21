#!/usr/bin/env python3
"""
Disorder predictor

Predicts disordered regions using pLDDT scores from AlphaFold models

"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple

class DisorderPredictor:
    """Predicts disordered regions using pLDDT scores from AlphaFold models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize disorder predictor with configuration
        
        Args:
            config: Configuration containing parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.disorder")
        
        # Configuration parameters
        self.disorder_threshold = config.get('disorder_threshold', 70.0)
        self.min_domain_ordered_percent = config.get('min_domain_ordered_percent', 70.0)
        self.min_ordered_segment = config.get('min_ordered_segment', 20)
        self.min_disordered_segment = config.get('min_disordered_segment', 5)
    
    def run(self, structure_id: str, structure_path: str, sse_path: str, 
            pae_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Predict disordered regions and refine domain boundaries
        
        Args:
            structure_id: Structure identifier
            structure_path: Path to input structure file (PDB/mmCIF)
            sse_path: Path to secondary structure results
            pae_path: Path to PAE data
            output_dir: Directory for output files
            
        Returns:
            Dict with results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Starting disorder prediction for structure {structure_id}")
        
        prefix = f"struct_{structure_id}"
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract pLDDT values from structure
            plddt_values = self._extract_plddt_values(structure_path)
            
            # Load domain information with SSE
            domains = self._load_domains_with_sse(sse_path)
            
            # Identify disordered regions
            disorder_regions = self._identify_disorder_regions(plddt_values)
            
            # Refine domain boundaries based on disorder
            refined_domains = self._refine_domains(domains, disorder_regions, plddt_values)
            
            # Write results
            results_path = os.path.join(output_dir, f"{prefix}_disorder.json")
            summary_path = os.path.join(output_dir, f"{prefix}_disorder_summary.tsv")
            
            self._write_results(refined_domains, disorder_regions, plddt_values, results_path, summary_path)
            
            # Return success and output paths
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "status": "COMPLETED",
                "structure_id": structure_id,
                "output_files": {
                    "disorder_json": results_path,
                    "disorder_summary": summary_path
                },
                "metrics": {
                    "duration_seconds": duration,
                    "domains_processed": len(domains),
                    "domains_refined": len(refined_domains),
                    "disorder_regions": len(disorder_regions)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting disorder for {structure_id}: {str(e)}")
            return {
                "status": "FAILED",
                "structure_id": structure_id,
                "error_message": str(e)
            }
    
    def _extract_plddt_values(self, structure_path: str) -> Dict[int, float]:
        """Extract pLDDT values from structure B-factors"""
        from dpam.gemmi_utils import get_structure_handler
        
        handler = get_structure_handler()
        structure = handler.read_structure(structure_path)
        
        # Calculate pLDDT from B-factors
        plddt_values = handler.calculate_plddt(structure)
        
        self.logger.debug(f"Extracted pLDDT values for {len(plddt_values)} residues")
        return plddt_values
    
    def _load_domains_with_sse(self, sse_path: str) -> List[Dict[str, Any]]:
        """Load domain information with secondary structure assignments"""
        with open(sse_path, 'r') as f:
            data = json.load(f)
        
        return data.get('domains', [])
    
    def _identify_disorder_regions(self, plddt_values: Dict[int, float]) -> List[Dict[str, Any]]:
        """Identify disordered regions based on pLDDT values"""
        disorder_regions = []
        
        if not plddt_values:
            return disorder_regions
            
        # Sort residues by position
        sorted_residues = sorted(plddt_values.keys())
        
        # Identify contiguous regions of disorder
        current_region = None
        
        for resnum in sorted_residues:
            plddt = plddt_values[resnum]
            
            # Check if residue is disordered
            is_disordered = plddt < self.disorder_threshold
            
            if is_disordered:
                # Start new region or extend current
                if current_region is None:
                    current_region = {
                        'start': resnum, 
                        'end': resnum, 
                        'avg_plddt': plddt,
                        'plddt_values': [plddt]
                    }
                else:
                    # Check if this residue is contiguous with current region
                    if resnum <= current_region['end'] + 1:
                        current_region['end'] = resnum
                        current_region['plddt_values'].append(plddt)
                        current_region['avg_plddt'] = sum(current_region['plddt_values']) / len(current_region['plddt_values'])
                    else:
                        # Save current region if it meets minimum size
                        if current_region['end'] - current_region['start'] + 1 >= self.min_disordered_segment:
                            disorder_regions.append(current_region)
                        
                        # Start new region
                        current_region = {
                            'start': resnum, 
                            'end': resnum, 
                            'avg_plddt': plddt,
                            'plddt_values': [plddt]
                        }
            elif current_region is not None:
                # Save current region if it meets minimum size
                if current_region['end'] - current_region['start'] + 1 >= self.min_disordered_segment:
                    disorder_regions.append(current_region)
                
                # Reset current region
                current_region = None
        
        # Add final region if it exists and meets minimum size
        if current_region is not None and current_region['end'] - current_region['start'] + 1 >= self.min_disordered_segment:
            disorder_regions.append(current_region)
        
        # Add length to each region
        for region in disorder_regions:
            region['length'] = region['end'] - region['start'] + 1
            
            # Remove plddt_values list to save space
            if 'plddt_values' in region:
                del region['plddt_values']
        
        return disorder_regions
    
    def _refine_domains(self, domains: List[Dict[str, Any]], 
                      disorder_regions: List[Dict[str, Any]], 
                      plddt_values: Dict[int, float]) -> List[Dict[str, Any]]:
        """Refine domain boundaries based on disorder and evaluate domain quality"""
        refined_domains = []
        
        for domain in domains:
            domain_start = domain.get('start', 0)
            domain_end = domain.get('end', 0)
            domain_id = domain.get('domain_id', 'unknown')
            
            self.logger.debug(f"Refining domain {domain_id} ({domain_start}-{domain_end})")
            
            # Check for overlapping disorder regions
            overlapping_regions = []
            for region in disorder_regions:
                # Check for any overlap
                if (region['start'] <= domain_end and region['end'] >= domain_start):
                    overlapping_regions.append(region)
            
            # Calculate domain-specific metrics
            domain_metrics = self._calculate_domain_disorder_metrics(
                domain_start, domain_end, plddt_values, overlapping_regions
            )
            
            # Decide if and how to refine domain
            new_domain = domain.copy()
            new_domain.update({
                'disorder_metrics': domain_metrics,
                'overlapping_disorder': len(overlapping_regions) > 0
            })
            
            # Add refined boundaries if needed
            if domain_metrics['percent_ordered'] < self.min_domain_ordered_percent:
                # Domain has too much disorder, try to refine boundaries
                refined_boundaries = self._find_refined_boundaries(
                    domain_start, domain_end, plddt_values, overlapping_regions
                )
                
                if refined_boundaries:
                    new_domain.update({
                        'original_boundaries': {'start': domain_start, 'end': domain_end},
                        'refined_boundaries': refined_boundaries,
                        'boundary_refinement': 'disorder_based'
                    })
            
            refined_domains.append(new_domain)
        
        return refined_domains
    
    def _calculate_domain_disorder_metrics(self, domain_start: int, domain_end: int, 
                                         plddt_values: Dict[int, float], 
                                         overlapping_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate disorder-related metrics for a domain"""
        # Count residues with pLDDT values
        domain_residues = [r for r in range(domain_start, domain_end + 1) if r in plddt_values]
        
        if not domain_residues:
            return {
                'avg_plddt': None,
                'min_plddt': None,
                'max_plddt': None,
                'percent_ordered': 0,
                'disordered_residues': 0,
                'total_residues': 0
            }
        
        # Calculate pLDDT statistics
        domain_plddt = [plddt_values[r] for r in domain_residues]
        avg_plddt = sum(domain_plddt) / len(domain_plddt)
        min_plddt = min(domain_plddt)
        max_plddt = max(domain_plddt)
        
        # Count disordered residues
        disordered_residues = sum(1 for plddt in domain_plddt if plddt < self.disorder_threshold)
        percent_ordered = 100 - (disordered_residues / len(domain_residues) * 100)
        
        # Calculate metrics specific to overlapping regions
        overlapping_residues = set()
        for region in overlapping_regions:
            for r in range(max(domain_start, region['start']), min(domain_end, region['end']) + 1):
                overlapping_residues.add(r)
        
        return {
            'avg_plddt': avg_plddt,
            'min_plddt': min_plddt,
            'max_plddt': max_plddt,
            'percent_ordered': percent_ordered,
            'disordered_residues': disordered_residues,
            'total_residues': len(domain_residues),
            'disorder_overlap_residues': len(overlapping_residues)
        }
    
    def _find_refined_boundaries(self, domain_start: int, domain_end: int, 
                               plddt_values: Dict[int, float],
                               overlapping_regions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Find refined domain boundaries to exclude disorder"""
        if not overlapping_regions:
            return None
        
        # Get ordered segments within domain
        ordered_segments = self._find_ordered_segments(
            domain_start, domain_end, plddt_values, overlapping_regions
        )
        
        if not ordered_segments:
            return None
        
        # Find largest ordered segment
        largest_segment = max(ordered_segments, key=lambda s: s['length'])
        
        # Check if it meets minimum size requirement
        if largest_segment['length'] < self.min_ordered_segment:
            return None
            
        # Domain should be at least 80% of the original size to be considered valid
        original_size = domain_end - domain_start + 1
        if largest_segment['length'] < 0.8 * original_size:
            return None
        
        return {
            'start': largest_segment['start'],
            'end': largest_segment['end']
        }
    
    def _find_ordered_segments(self, domain_start: int, domain_end: int, 
                             plddt_values: Dict[int, float],
                             overlapping_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find ordered segments within a domain"""
        # Create a set of disordered residues
        disordered_residues = set()
        for region in overlapping_regions:
            for r in range(region['start'], region['end'] + 1):
                disordered_residues.add(r)
        
        # Identify ordered segments
        ordered_segments = []
        current_segment = None
        
        for resnum in range(domain_start, domain_end + 1):
            # Skip residues without pLDDT
            if resnum not in plddt_values:
                continue
                
            # Check if residue is ordered
            is_ordered = resnum not in disordered_residues and plddt_values[resnum] >= self.disorder_threshold
            
            if is_ordered:
                # Start new segment or extend current
                if current_segment is None:
                    current_segment = {'start': resnum, 'end': resnum}
                else:
                    # Check if this residue is contiguous with current segment
                    if resnum <= current_segment['end'] + 1:
                        current_segment['end'] = resnum
                    else:
                        # Save current segment and start new one
                        current_segment['length'] = current_segment['end'] - current_segment['start'] + 1
                        ordered_segments.append(current_segment)
                        current_segment = {'start': resnum, 'end': resnum}
            elif current_segment is not None:
                # Save current segment
                current_segment['length'] = current_segment['end'] - current_segment['start'] + 1
                ordered_segments.append(current_segment)
                current_segment = None
        
        # Add final segment if it exists
        if current_segment is not None:
            current_segment['length'] = current_segment['end'] - current_segment['start'] + 1
            ordered_segments.append(current_segment)
        
        return ordered_segments
    
    def _write_results(self, domains: List[Dict[str, Any]], 
                      disorder_regions: List[Dict[str, Any]],
                      plddt_values: Dict[int, float],
                      results_path: str, summary_path: str) -> None:
        """Write results to output files"""
        # Write full JSON results
        with open(results_path, 'w') as f:
            json.dump({
                'domains': domains,
                'disorder_regions': disorder_regions,
                'disorder_threshold': self.disorder_threshold
            }, f, indent=2)
        
        # Write summary TSV
        with open(summary_path, 'w') as f:
            f.write("domain_id\tstart\tend\trefined_start\trefined_end\tsize\tavg_plddt\tpercent_ordered\toverlapping_disorder\tdomain_class\n")
            
            for domain in domains:
                domain_id = domain.get('domain_id', 'unknown')
                start = domain.get('start', 0)
                end = domain.get('end', 0)
                size = end - start + 1
                
                # Get refined boundaries if available
                refined = domain.get('refined_boundaries', {})
                refined_start = refined.get('start', start) if refined else start
                refined_end = refined.get('end', end) if refined else end
                
                # Get disorder metrics
                metrics = domain.get('disorder_metrics', {})
                avg_plddt = metrics.get('avg_plddt', 0)
                percent_ordered = metrics.get('percent_ordered', 0)
                
                # Get domain class from SS composition if available
                ss_comp = domain.get('ss_composition', {})
                domain_class = ss_comp.get('domain_class', 'unknown') if ss_comp else 'unknown'
                
                # Write row
                f.write(f"{domain_id}\t{start}\t{end}\t{refined_start}\t{refined_end}\t"
                       f"{size}\t{avg_plddt:.1f}\t{percent_ordered:.1f}\t"
                       f"{domain.get('overlapping_disorder', False)}\t{domain_class}\n")
                       
        # Generate visualizable data
        self._generate_plddt_profile(plddt_values, disorder_regions, domains, results_path.replace(".json", "_profile.json"))
        
    def _generate_plddt_profile(self, plddt_values: Dict[int, float], 
                              disorder_regions: List[Dict[str, Any]],
                              domains: List[Dict[str, Any]],
                              output_path: str) -> None:
        """Generate pLDDT profile for visualization"""
        # Create list of residues with pLDDT values
        residues = []
        
        for resnum in sorted(plddt_values.keys()):
            plddt = plddt_values[resnum]
            
            # Determine if in a disorder region
            in_disorder = False
            for region in disorder_regions:
                if region['start'] <= resnum <= region['end']:
                    in_disorder = True
                    break
            
            # Determine domain memberships
            domain_ids = []
            for domain in domains:
                start = domain.get('start', 0)
                end = domain.get('end', 0)
                if start <= resnum <= end:
                    domain_ids.append(domain.get('domain_id', 'unknown'))
            
            residues.append({
                'resnum': resnum,
                'plddt': plddt,
                'disordered': plddt < self.disorder_threshold,
                'in_disorder_region': in_disorder,
                'domain_ids': domain_ids
            })
        
        # Create domain boundaries
        domain_boundaries = []
        for domain in domains:
            domain_id = domain.get('domain_id', 'unknown')
            start = domain.get('start', 0)
            end = domain.get('end', 0)
            
            # Add original boundaries
            domain_boundaries.append({
                'domain_id': domain_id,
                'type': 'original',
                'start': start,
                'end': end
            })
            
            # Add refined boundaries if available
            refined = domain.get('refined_boundaries', {})
            if refined:
                domain_boundaries.append({
                    'domain_id': domain_id,
                    'type': 'refined',
                    'start': refined.get('start', start),
                    'end': refined.get('end', end)
                })
        
        # Write visualization data
        with open(output_path, 'w') as f:
            json.dump({
                'residues': residues,
                'domain_boundaries': domain_boundaries,
                'disorder_threshold': self.disorder_threshold,
                'disorder_regions': disorder_regions
            }, f, indent=2)
            
    def analyze_structure_disorder(self, structure_id: str, plddt_values: Dict[int, float]) -> Dict[str, Any]:
        """Analyze overall disorder content of structure"""
        if not plddt_values:
            return {
                'total_residues': 0,
                'ordered_residues': 0,
                'disordered_residues': 0,
                'percent_ordered': 0,
                'percent_disordered': 0,
                'avg_plddt': 0,
                'disorder_regions': 0
            }
        
        # Calculate basic statistics
        total_residues = len(plddt_values)
        disordered_residues = sum(1 for plddt in plddt_values.values() if plddt < self.disorder_threshold)
        ordered_residues = total_residues - disordered_residues
        
        percent_ordered = (ordered_residues / total_residues) * 100 if total_residues > 0 else 0
        percent_disordered = (disordered_residues / total_residues) * 100 if total_residues > 0 else 0
        
        avg_plddt = sum(plddt_values.values()) / total_residues if total_residues > 0 else 0
        
        # Identify disorder regions
        disorder_regions = self._identify_disorder_regions(plddt_values)
        
        return {
            'structure_id': structure_id,
            'total_residues': total_residues,
            'ordered_residues': ordered_residues,
            'disordered_residues': disordered_residues,
            'percent_ordered': percent_ordered,
            'percent_disordered': percent_disordered,
            'avg_plddt': avg_plddt,
            'disorder_regions': len(disorder_regions),
            'disorder_region_details': disorder_regions
        }## dpam/steps/disorder.py

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple

class DisorderPredictor:
    """Predicts disordered regions using pLDDT scores from AlphaFold models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize disorder predictor with configuration
        
        Args:
            config: Configuration containing parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.disorder")
        
        # Configuration parameters
        self.disorder_threshold = config.get('disorder_threshold', 70.0)
        self.min_domain_ordered_percent = config.get('min_domain_ordered_percent', 70.0)
        self.min_ordered_segment = config.get('min_ordered_segment', 20)
        self.min_disordered_segment = config.get('min_disordered_segment', 5)
    
    def run(self, structure_id: str, structure_path: str, sse_path: str, 
            pae_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Predict disordered regions and refine domain boundaries
        
        Args:
            structure_id: Structure identifier
            structure_path: Path to input structure file (PDB/mmCIF)
            sse_path: Path to secondary structure results
            pae_path: Path to PAE data
            output_dir: Directory for output files
            
        Returns:
            Dict with results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Starting disorder prediction for structure {structure_id}")
        
        prefix = f"struct_{structure_id}"
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract pLDDT values from structure
            plddt_values = self._extract_plddt_values(structure_path)
            
            # Load domain information with SSE
            domains = self._load_domains_with_sse(sse_path)
            
            # Identify disordered regions
            disorder_regions = self._identify_disorder_regions(plddt_values)
            
            # Refine domain boundaries based on disorder
            refined_domains = self._refine_domains(domains, disorder_regions, plddt_values)
            
            # Write results
            results_path = os.path.join(output_dir, f"{prefix}_disorder.json")
            summary_path = os.path.join(output_dir, f"{prefix}_disorder_summary.tsv")
            
            self._write_results(refined_domains, disorder_regions, plddt_values, results_path, summary_path)
            
            # Return success and output paths
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "status": "COMPLETED",
                "structure_id": structure_id,
                "output_files": {
                    "disorder_json": results_path,
                    "disorder_summary": summary_path
                },
                "metrics": {
                    "duration_seconds": duration,
                    "domains_processed": len(domains),
                    "domains_refined": len(refined_domains),
                    "disorder_regions": len(disorder_regions)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting disorder for {structure_id}: {str(e)}")
            return {
                "status": "FAILED",
                "structure_id": structure_id,
                "error_message": str(e)
            }
    
    def _extract_plddt_values(self, structure_path: str) -> Dict[int, float]:
        """Extract pLDDT values from structure B-factors"""
        from dpam.gemmi_utils import get_structure_handler
        
        handler = get_structure_handler()
        structure = handler.read_structure(structure_path)
        
        # Calculate pLDDT from B-factors
        plddt_values = handler.calculate_plddt(structure)
        
        self.logger.debug(f"Extracted pLDDT values for {len(plddt_values)} residues")
        return plddt_values
    
    def _load_domains_with_sse(self, sse_path: str) -> List[Dict[str, Any]]:
        """Load domain information with secondary structure assignments"""
        with open(sse_path, 'r') as f:
            data = json.load(f)
        
        return data.get('domains', [])
    
    def _identify_disorder_regions(self, plddt_values: Dict[int, float]) -> List[Dict[str, Any]]:
        """Identify disordered regions based on pLDDT values"""
        disorder_regions = []
        
        if not plddt_values:
            return disorder_regions
            
        # Sort residues by position
        sorted_residues = sorted(plddt_values.keys())
        
        # Identify contiguous regions of disorder
        current_region = None
        
        for resnum in sorted_residues:
            plddt = plddt_values[resnum]
            
            # Check if residue is disordered
            is_disordered = plddt < self.disorder_threshold
            
            if is_disordered:
                # Start new region or extend current
                if current_region is None:
                    current_region = {
                        'start': resnum, 
                        'end': resnum, 
                        'avg_plddt': plddt,
                        'plddt_values': [plddt]
                    }
                else:
                    # Check if this residue is contiguous with current region
                    if resnum <= current_region['end'] + 1:
                        current_region['end'] = resnum
                        current_region['plddt_values'].append(plddt)
                        current_region['avg_plddt'] = sum(current_region['plddt_values']) / len(current_region['plddt_values'])
                    else:
                        # Save current region if it meets minimum size
                        if current_region['end'] - current_region['start'] + 1 >= self.min_disordered_segment:
                            disorder_regions.append(current_region)
                        
                        # Start new region
                        current_region = {
                            'start': resnum, 
                            'end': resnum, 
                            'avg_plddt': plddt,
                            'plddt_values': [plddt]
                        }
            elif current_region is not None:
                # Save current region if it meets minimum size
                if current_region['end'] - current_region['start'] + 1 >= self.min_disordered_segment:
                    disorder_regions.append(current_region)
                
                # Reset current region
                current_region = None
        
        # Add final region if it exists and meets minimum size
        if current_region is not None and current_region['end'] - current_region['start'] + 1 >= self.min_disordered_segment:
            disorder_regions.append(current_region)
        
        # Add length to each region
        for region in disorder_regions:
            region['length'] = region['end'] - region['start'] + 1
            
            # Remove plddt_values list to save space
            if 'plddt_values' in region:
                del region['plddt_values']
        
        return disorder_regions
    
    def _refine_domains(self, domains: List[Dict[str, Any]], 
                      disorder_regions: List[Dict[str, Any]], 
                      plddt_values: Dict[int, float]) -> List[Dict[str, Any]]:
        """Refine domain boundaries based on disorder and evaluate domain quality"""
        refined_domains = []
        
        for domain in domains:
            domain_start = domain.get('start', 0)
            domain_end = domain.get('end', 0)
            domain_id = domain.get('domain_id', 'unknown')
            
            self.logger.debug(f"Refining domain {domain_id} ({domain_start}-{domain_end})")
            
            # Check for overlapping disorder regions
            overlapping_regions = []
            for region in disorder_regions:
                # Check for any overlap
                if (region['start'] <= domain_end and region['end'] >= domain_start):
                    overlapping_regions.append(region)
            
            # Calculate domain-specific metrics
            domain_metrics = self._calculate_domain_disorder_metrics(
                domain_start, domain_end, plddt_values, overlapping_regions
            )
            
            # Decide if and how to refine domain
            new_domain = domain.copy()
            new_domain.update({
                'disorder_metrics': domain_metrics,
                'overlapping_disorder': len(overlapping_regions) > 0
            })
            
            # Add refined boundaries if needed
            if domain_metrics['percent_ordered'] < self.min_domain_ordered_percent:
                # Domain has too much disorder, try to refine boundaries
                refined_boundaries = self._find_refined_boundaries(
                    domain_start, domain_end, plddt_values, overlapping_regions
                )
                
                if refined_boundaries:
                    new_domain.update({
                        'original_boundaries': {'start': domain_start, 'end': domain_end},
                        'refined_boundaries': refined_boundaries,
                        'boundary_refinement': 'disorder_based'
                    })
            
            refined_domains.append(new_domain)
        
        return refined_domains
    
    def _calculate_domain_disorder_metrics(self, domain_start: int, domain_end: int, 
                                         plddt_values: Dict[int, float], 
                                         overlapping_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate disorder-related metrics for a domain"""
        # Count residues with pLDDT values
        domain_residues = [r for r in range(domain_start, domain_end + 1) if r in plddt_values]
        
        if not domain_residues:
            return {
                'avg_plddt': None,
                'min_plddt': None,
                'max_plddt': None,
                'percent_ordered': 0,
                'disordered_residues': 0,
                'total_residues': 0
            }
        
        # Calculate pLDDT statistics
        domain_plddt = [plddt_values[r] for r in domain_residues]
        avg_plddt = sum(domain_plddt) / len(domain_plddt)
        min_plddt = min(domain_plddt)
        max_plddt = max(domain_plddt)
        
        # Count disordered residues
        disordered_residues = sum(1 for plddt in domain_plddt if plddt < self.disorder_threshold)
        percent_ordered = 100 - (disordered_residues / len(domain_residues) * 100)
        
        # Calculate metrics specific to overlapping regions
        overlapping_residues = set()
        for region in overlapping_regions:
            for r in range(max(domain_start, region['start']), min(domain_end, region['end']) + 1):
                overlapping_residues.add(r)
        
        return {
            'avg_plddt': avg_plddt,
            'min_plddt': min_plddt,
            'max_plddt': max_plddt,
            'percent_ordered': percent_ordered,
            'disordered_residues': disordered_residues,
            'total_residues': len(domain_residues),
            'disorder_overlap_residues': len(overlapping_residues)
        }
    
    def _find_refined_boundaries(self, domain_start: int, domain_end: int, 
                               plddt_values: Dict[int, float],
                               overlapping_regions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Find refined domain boundaries to exclude disorder"""
        if not overlapping_regions:
            return None
        
        # Get ordered segments within domain
        ordered_segments = self._find_ordered_segments(
            domain_start, domain_end, plddt_values, overlapping_regions
        )
        
        if not ordered_segments:
            return None
        
        # Find largest ordered segment
        largest_segment = max(ordered_segments, key=lambda s: s['length'])
        
        # Check if it meets minimum size requirement
        if largest_segment['length'] < self.min_ordered_segment:
            return None
            
        # Domain should be at least 80% of the original size to be considered valid
        original_size = domain_end - domain_start + 1
        if largest_segment['length'] < 0.8 * original_size:
            return None
        
        return {
            'start': largest_segment['start'],
            'end': largest_segment['end']
        }
    
    def _find_ordered_segments(self, domain_start: int, domain_end: int, 
                             plddt_values: Dict[int, float],
                             overlapping_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find ordered segments within a domain"""
        # Create a set of disordered residues
        disordered_residues = set()
        for region in overlapping_regions:
            for r in range(region['start'], region['end'] + 1):
                disordered_residues.add(r)
        
        # Identify ordered segments
        ordered_segments = []
        current_segment = None
        
        for resnum in range(domain_start, domain_end + 1):
            # Skip residues without pLDDT
            if resnum not in plddt_values:
                continue
                
            # Check if residue is ordered
            is_ordered = resnum not in disordered_residues and plddt_values[resnum] >= self.disorder_threshold
            
            if is_ordered:
                # Start new segment or extend current
                if current_segment is None:
                    current_segment = {'start': resnum, 'end': resnum}
                else:
                    # Check if this residue is contiguous with current segment
                    if resnum <= current_segment['end'] + 1:
                        current_segment['end'] = resnum
                    else:
                        # Save current segment and start new one
                        current_segment['length'] = current_segment['end'] - current_segment['start'] + 1
                        ordered_segments.append(current_segment)
                        current_segment = {'start': resnum, 'end': resnum}
            elif current_segment is not None:
                # Save current segment
                current_segment['length'] = current_segment['end'] - current_segment['start'] + 1
                ordered_segments.append(current_segment)
                current_segment = None
        
        # Add final segment if it exists
        if current_segment is not None:
            current_segment['length'] = current_segment['end'] - current_segment['start'] + 1
            ordered_segments.append(current_segment)
        
        return ordered_segments
    
    def _write_results(self, domains: List[Dict[str, Any]], 
                      disorder_regions: List[Dict[str, Any]],
                      plddt_values: Dict[int, float],
                      results_path: str, summary_path: str) -> None:
        """Write results to output files"""
        # Write full JSON results
        with open(results_path, 'w') as f:
            json.dump({
                'domains': domains,
                'disorder_regions': disorder_regions,
                'disorder_threshold': self.disorder_threshold
            }, f, indent=2)
        
        # Write summary TSV
        with open(summary_path, 'w') as f:
            f.write("domain_id\tstart\tend\trefined_start\trefined_end\tsize\tavg_plddt\tpercent_ordered\toverlapping_disorder\tdomain_class\n")
            
            for domain in domains:
                domain_id = domain.get('domain_id', 'unknown')
                start = domain.get('start', 0)
                end = domain.get('end', 0)
                size = end - start + 1
                
                # Get refined boundaries if available
                refined = domain.get('refined_boundaries', {})
                refined_start = refined.get('start', start) if refined else start
                refined_end = refined.get('end', end) if refined else end
                
                # Get disorder metrics
                metrics = domain.get('disorder_metrics', {})
                avg_plddt = metrics.get('avg_plddt', 0)
                percent_ordered = metrics.get('percent_ordered', 0)
                
                # Get domain class from SS composition if available
                ss_comp = domain.get('ss_composition', {})
                domain_class = ss_comp.get('domain_class', 'unknown') if ss_comp else 'unknown'
                
                # Write row
                f.write(f"{domain_id}\t{start}\t{end}\t{refined_start}\t{refined_end}\t"
                       f"{size}\t{avg_plddt:.1f}\t{percent_ordered:.1f}\t"
                       f"{domain.get('overlapping_disorder', False)}\t{domain_class}\n")
                       
        # Generate visualizable data
        self._generate_plddt_profile(plddt_values, disorder_regions, domains, results_path.replace(".json", "_profile.json"))
        
    def _generate_plddt_profile(self, plddt_values: Dict[int, float], 
                              disorder_regions: List[Dict[str, Any]],
                              domains: List[Dict[str, Any]],
                              output_path: str) -> None:
        """Generate pLDDT profile for visualization"""
        # Create list of residues with pLDDT values
        residues = []
        
        for resnum in sorted(plddt_values.keys()):
            plddt = plddt_values[resnum]
            
            # Determine if in a disorder region
            in_disorder = False
            for region in disorder_regions:
                if region['start'] <= resnum <= region['end']:
                    in_disorder = True
                    break
            
            # Determine domain memberships
            domain_ids = []
            for domain in domains:
                start = domain.get('start', 0)
                end = domain.get('end', 0)
                if start <= resnum <= end:
                    domain_ids.append(domain.get('domain_id', 'unknown'))
            
            residues.append({
                'resnum': resnum,
                'plddt': plddt,
                'disordered': plddt < self.disorder_threshold,
                'in_disorder_region': in_disorder,
                'domain_ids': domain_ids
            })
        
        # Create domain boundaries
        domain_boundaries = []
        for domain in domains:
            domain_id = domain.get('domain_id', 'unknown')
            start = domain.get('start', 0)
            end = domain.get('end', 0)
            
            # Add original boundaries
            domain_boundaries.append({
                'domain_id': domain_id,
                'type': 'original',
                'start': start,
                'end': end
            })
            
            # Add refined boundaries if available
            refined = domain.get('refined_boundaries', {})
            if refined:
                domain_boundaries.append({
                    'domain_id': domain_id,
                    'type': 'refined',
                    'start': refined.get('start', start),
                    'end': refined.get('end', end)
                })
        
        # Write visualization data
        with open(output_path, 'w') as f:
            json.dump({
                'residues': residues,
                'domain_boundaries': domain_boundaries,
                'disorder_threshold': self.disorder_threshold,
                'disorder_regions': disorder_regions
            }, f, indent=2)
            
    def analyze_structure_disorder(self, structure_id: str, plddt_values: Dict[int, float]) -> Dict[str, Any]:
        """Analyze overall disorder content of structure"""
        if not plddt_values:
            return {
                'total_residues': 0,
                'ordered_residues': 0,
                'disordered_residues': 0,
                'percent_ordered': 0,
                'percent_disordered': 0,
                'avg_plddt': 0,
                'disorder_regions': 0
            }
        
        # Calculate basic statistics
        total_residues = len(plddt_values)
        disordered_residues = sum(1 for plddt in plddt_values.values() if plddt < self.disorder_threshold)
        ordered_residues = total_residues - disordered_residues
        
        percent_ordered = (ordered_residues / total_residues) * 100 if total_residues > 0 else 0
        percent_disordered = (disordered_residues / total_residues) * 100 if total_residues > 0 else 0
        
        avg_plddt = sum(plddt_values.values()) / total_residues if total_residues > 0 else 0
        
        # Identify disorder regions
        disorder_regions = self._identify_disorder_regions(plddt_values)
        
        return {
            'structure_id': structure_id,
            'total_residues': total_residues,
            'ordered_residues': ordered_residues,
            'disordered_residues': disordered_residues,
            'percent_ordered': percent_ordered,
            'percent_disordered': percent_disordered,
            'avg_plddt': avg_plddt,
            'disorder_regions': len(disorder_regions),
            'disorder_region_details': disorder_regions
        }