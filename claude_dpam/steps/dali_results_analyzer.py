#!/usr/bin/env python3
"""
Step module for processing dali search results

This module defines the parameters and contains parsing routines for reading DALI result files
"""
import os
import json
import logging
import tempfile
import numpy as np
from datetime import datetime
from collections import defaultdict

class DaliResultsAnalyzer:
    """Analyzes Dali search results to identify domain boundaries"""
    
    def __init__(self, config):
        """
        Initialize Dali analyzer with configuration
        
        Args:
            config (dict): Configuration containing parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.analyze_dali")
        
        # Analysis parameters
        self.min_z_score = config.get('dali_min_z_score', 8.0)
        self.significant_z_score = config.get('dali_significant_z_score', 4.0)
        self.min_coverage = config.get('dali_min_coverage', 0.5)
        self.max_rmsd = config.get('dali_max_rmsd', 5.0)
        self.min_domain_size = config.get('min_domain_size', 30)
    
    def run(self, structure_id, structure_path, dali_results_path, ecod_mapping_path, output_dir):
        """
        Analyze Dali results to identify domain boundaries
        
        Args:
            structure_id (str): Structure identifier
            structure_path (str): Path to query structure file (PDB/mmCIF)
            dali_results_path (str): Path to Dali results JSON
            ecod_mapping_path (str): Path to ECOD mapping results
            output_dir (str): Directory for output files
            
        Returns:
            dict: Results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Starting Dali analysis for structure {structure_id}")
        
        prefix = f"struct_{structure_id}"
        
        try:
            # Load Dali results
            with open(dali_results_path, 'r') as f:
                dali_data = json.load(f)
            
            hits = dali_data.get('hits', {})
            self.logger.info(f"Loaded {len(hits)} Dali hits for {structure_id}")
            
            # Load ECOD mapping data if available
            ecod_mappings = self._load_ecod_mappings(ecod_mapping_path)
            
            # Filter significant hits
            significant_hits = self._filter_significant_hits(hits)
            self.logger.info(f"Found {len(significant_hits)} significant hits")
            
            # Extract residue-based coverage
            residue_coverage = self._analyze_residue_coverage(significant_hits, structure_path)
            
            # Group regions for domain identification
            domain_regions = self._identify_domain_regions(residue_coverage)
            self.logger.info(f"Identified {len(domain_regions)} potential domain regions")
            
            # Refine domain boundaries
            domains = self._refine_domain_boundaries(domain_regions, ecod_mappings)
            self.logger.info(f"Defined {len(domains)} domains after refinement")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Write analysis results
            analysis_path = os.path.join(output_dir, f"{prefix}_dali_analysis.json")
            coverage_path = os.path.join(output_dir, f"{prefix}_residue_coverage.tsv")
            domains_path = os.path.join(output_dir, f"{prefix}_domains.tsv")
            
            # Write coverage and analysis files
            self._write_residue_coverage(residue_coverage, coverage_path)
            self._write_domains(domains, domains_path)
            
            # Write full analysis
            with open(analysis_path, 'w') as f:
                json.dump({
                    'structure_id': structure_id,
                    'significant_hits': len(significant_hits),
                    'total_hits': len(hits),
                    'domains': domains,
                    'residue_coverage': {str(k): v for k, v in residue_coverage.items()}
                }, f, indent=2)
            
            # Return success and output paths
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "status": "COMPLETED",
                "structure_id": structure_id,
                "output_files": {
                    "analysis": analysis_path,
                    "coverage": coverage_path,
                    "domains": domains_path
                },
                "metrics": {
                    "duration_seconds": duration,
                    "significant_hits": len(significant_hits),
                    "total_hits": len(hits),
                    "domains_found": len(domains)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing Dali results for {structure_id}: {str(e)}")
            return {
                "status": "FAILED",
                "structure_id": structure_id,
                "error_message": str(e)
            }
    
    def _load_ecod_mappings(self, ecod_mapping_path):
        """Load ECOD mapping data if available"""
        if not os.path.exists(ecod_mapping_path):
            return {}
        
        mappings = {}
        try:
            with open(ecod_mapping_path, 'r') as f:
                # Skip header
                next(f)
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 12:
                        domain_id = parts[0]
                        query_range = parts[11]
                        
                        # Parse residue ranges
                        residues = set()
                        for range_str in query_range.split(','):
                            if '-' in range_str:
                                start, end = map(int, range_str.split('-'))
                                for i in range(start, end + 1):
                                    residues.add(i)
                            else:
                                residues.add(int(range_str))
                        
                        mappings[domain_id] = residues
            return mappings
        except Exception as e:
            self.logger.warning(f"Could not load ECOD mappings: {str(e)}")
            return {}
    
    def _filter_significant_hits(self, hits):
        """Filter for significant Dali hits based on Z-score and other metrics"""
        significant = {}
        
        for domain_id, hit_info in hits.items():
            z_score = hit_info.get('z_score', 0)
            rmsd = hit_info.get('rmsd', float('inf'))
            coverage = hit_info.get('coverage', 0)
            
            # Check if hit meets significance criteria
            if (z_score >= self.significant_z_score and 
                rmsd <= self.max_rmsd and 
                coverage >= self.min_coverage):
                significant[domain_id] = hit_info
        
        return significant
    
    def _analyze_residue_coverage(self, hits, structure_path):
        """
        Analyze which residues are covered by Dali hits
        
        This is a simplified version as we don't have actual alignment data.
        In a real implementation, this would parse detailed alignment files.
        """
        # Create a coverage counter for each residue
        residue_coverage = defaultdict(int)
        
        # In a real implementation, this would use actual alignments
        # For now, we'll simulate coverage with a simplified model
        structure_length = self._get_structure_length(structure_path)
        
        if structure_length <= 0:
            self.logger.warning(f"Could not determine structure length")
            return residue_coverage
        
        for domain_id, hit_info in hits.items():
            z_score = hit_info.get('z_score', 0)
            coverage = hit_info.get('coverage', 0)
            
            # Weight factor based on Z-score
            weight = min(1.0, (z_score - self.significant_z_score) / 10.0)
            if weight <= 0:
                continue
            
            # For simplicity, spread coverage evenly
            # In a real implementation, this would use actual alignment regions
            aligned_length = int(structure_length * coverage)
            start = int((structure_length - aligned_length) / 2)
            end = start + aligned_length
            
            for i in range(start, end):
                residue_coverage[i+1] += weight
        
        return residue_coverage
    
    def _get_structure_length(self, structure_path):
        """Get the length of the structure in residues"""
        # In a real implementation, this would parse the PDB/mmCIF file
        # For now, we'll return a default value
        return 300  # Placeholder
    
    def _identify_domain_regions(self, residue_coverage):
        """Identify potential domain regions based on residue coverage"""
        regions = []
        current_region = None
        
        # Sort residues by position
        sorted_residues = sorted(residue_coverage.keys())
        
        for i, res in enumerate(sorted_residues):
            coverage = residue_coverage[res]
            
            if coverage > 0:
                # Start new region or extend current
                if current_region is None:
                    current_region = {'start': res, 'end': res, 'coverage': [coverage]}
                else:
                    # Check if this residue is contiguous with current region
                    if res <= current_region['end'] + 3:  # Allow small gaps
                        current_region['end'] = res
                        current_region['coverage'].append(coverage)
                    else:
                        # Save current region and start new one
                        regions.append(current_region)
                        current_region = {'start': res, 'end': res, 'coverage': [coverage]}
            elif current_region is not None:
                # End of a region
                regions.append(current_region)
                current_region = None
        
        # Add final region if exists
        if current_region is not None:
            regions.append(current_region)
        
        # Filter regions by size and calculate metrics
        filtered_regions = []
        for region in regions:
            size = region['end'] - region['start'] + 1
            
            if size >= self.min_domain_size:
                # Calculate coverage statistics
                avg_coverage = sum(region['coverage']) / len(region['coverage'])
                region['size'] = size
                region['avg_coverage'] = avg_coverage
                filtered_regions.append(region)
        
        return filtered_regions
    
    def _refine_domain_boundaries(self, regions, ecod_mappings):
        """Refine domain boundaries using ECOD mappings if available"""
        domains = []
        
        # If we have ECOD mappings, use them to refine boundaries
        if ecod_mappings:
            # Try to match regions to ECOD domains
            for region in regions:
                best_match = None
                best_overlap = 0
                
                region_residues = set(range(region['start'], region['end'] + 1))
                
                for domain_id, ecod_residues in ecod_mappings.items():
                    overlap = len(region_residues.intersection(ecod_residues))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = {
                            'domain_id': domain_id,
                            'start': min(ecod_residues),
                            'end': max(ecod_residues),
                            'size': len(ecod_residues),
                            'source': 'ecod',
                            'overlap': overlap / len(region_residues)
                        }
                
                if best_match and best_match['overlap'] >= 0.7:
                    # Use ECOD boundaries if good overlap
                    domains.append(best_match)
                else:
                    # Use our detected boundaries
                    domains.append({
                        'domain_id': f"dom_{len(domains) + 1}",
                        'start': region['start'],
                        'end': region['end'],
                        'size': region['size'],
                        'source': 'dali',
                        'avg_coverage': region['avg_coverage']
                    })
        else:
            # Without ECOD mappings, just use our detected boundaries
            for i, region in enumerate(regions):
                domains.append({
                    'domain_id': f"dom_{i + 1}",
                    'start': region['start'],
                    'end': region['end'],
                    'size': region['size'],
                    'source': 'dali',
                    'avg_coverage': region['avg_coverage']
                })
        
        # Check for overlapping domains and resolve
        domains = self._resolve_domain_overlaps(domains)
        
        return domains
    
    def _resolve_domain_overlaps(self, domains):
        """Resolve overlapping domain boundaries"""
        if not domains or len(domains) <= 1:
            return domains
        
        # Sort domains by start position
        sorted_domains = sorted(domains, key=lambda d: d['start'])
        
        # Check for overlaps
        resolved = [sorted_domains[0]]
        
        for domain in sorted_domains[1:]:
            prev = resolved[-1]
            
            # Check for overlap
            if domain['start'] <= prev['end']:
                # Calculate overlap size
                overlap = min(domain['end'], prev['end']) - domain['start'] + 1
                overlap_percent = overlap / min(domain['size'], prev['size'])
                
                if overlap_percent > 0.5:
                    # Significant overlap, merge domains
                    merged = {
                        'domain_id': f"{prev['domain_id']}_{domain['domain_id']}",
                        'start': min(prev['start'], domain['start']),
                        'end': max(prev['end'], domain['end']),
                        'source': 'merged',
                    }
                    merged['size'] = merged['end'] - merged['start'] + 1
                    
                    # Replace previous domain with merged one
                    resolved[-1] = merged
                else:
                    # Adjust boundaries to eliminate overlap
                    midpoint = (prev['end'] + domain['start']) // 2
                    prev['end'] = midpoint
                    domain['start'] = midpoint + 1
                    
                    # Update sizes
                    prev['size'] = prev['end'] - prev['start'] + 1
                    domain['size'] = domain['end'] - domain['start'] + 1
                    
                    # Add adjusted domain
                    if domain['size'] >= self.min_domain_size:
                        resolved.append(domain)
            else:
                # No overlap, add domain
                resolved.append(domain)
        
        return resolved
    
    def _write_residue_coverage(self, residue_coverage, output_path):
        """Write residue coverage to TSV file"""
        with open(output_path, 'w') as f:
            f.write("residue\tcoverage\n")
            
            for residue in sorted(residue_coverage.keys()):
                f.write(f"{residue}\t{residue_coverage[residue]:.4f}\n")
    
    def _write_domains(self, domains, output_path):
        """Write domain definitions to TSV file"""
        with open(output_path, 'w') as f:
            f.write("domain_id\tstart\tend\tsize\tsource\n")
            
            for domain in domains:
                f.write(f"{domain['domain_id']}\t{domain['start']}\t{domain['end']}\t"
                       f"{domain['size']}\t{domain['source']}\n")