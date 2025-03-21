#!/usr/bin/env python3
"""
Calculates support for domains using multiple evidence sources

"""
import os
import json
import logging
import math
import numpy as np
from datetime import datetime
from collections import defaultdict

class DomainSupportCalculator:
    """Calculates support scores for predicted domains using multiple evidence sources"""
    
    def __init__(self, config):
        """
        Initialize domain support calculator with configuration
        
        Args:
            config (dict): Configuration containing parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.support")
        
        # Support calculation parameters
        self.min_domain_size = config.get('min_domain_size', 30)
        self.ecod_weight = config.get('ecod_weight', 2.0)
        self.dali_weight = config.get('dali_weight', 1.5)
        self.foldseek_weight = config.get('foldseek_weight', 1.0)
        self.pae_weight = config.get('pae_weight', 2.0)
        self.min_support_score = config.get('min_support_score', 0.5)
    
    def run(self, structure_id, structure_path, dali_analysis_path, ecod_mapping_path, 
            foldseek_filtered_path, pae_path, output_dir):
        """
        Calculate support for domain boundaries
        
        Args:
            structure_id (str): Structure identifier
            structure_path (str): Path to query structure file (PDB/mmCIF)
            dali_analysis_path (str): Path to Dali analysis results
            ecod_mapping_path (str): Path to ECOD mapping results
            foldseek_filtered_path (str): Path to filtered FoldSeek results
            pae_path (str): Path to PAE data (optional)
            output_dir (str): Directory for output files
            
        Returns:
            dict: Results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Calculating domain support for structure {structure_id}")
        
        prefix = f"struct_{structure_id}"
        
        try:
            # Load domain predictions from Dali analysis
            with open(dali_analysis_path, 'r') as f:
                dali_data = json.load(f)
            
            domains = dali_data.get('domains', [])
            self.logger.info(f"Loaded {len(domains)} domains from Dali analysis")
            
            # Load ECOD mappings
            ecod_mappings = self._load_ecod_mappings(ecod_mapping_path)
            
            # Load FoldSeek hits
            foldseek_hits = self._load_foldseek_hits(foldseek_filtered_path)
            
            # Load PAE data if available
            pae_matrix = self._load_pae_data(pae_path)
            
            # Calculate support for each domain
            domains_with_support = self._calculate_domain_support(
                domains, ecod_mappings, foldseek_hits, pae_matrix, structure_path
            )
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Write results
            support_path = os.path.join(output_dir, f"{prefix}_domain_support.json")
            support_tsv_path = os.path.join(output_dir, f"{prefix}_domain_support.tsv")
            
            self._write_support_results(domains_with_support, support_path, support_tsv_path)
            
            # Return success and output paths
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "status": "COMPLETED",
                "structure_id": structure_id,
                "output_files": {
                    "support_json": support_path,
                    "support_tsv": support_tsv_path
                },
                "metrics": {
                    "duration_seconds": duration,
                    "domains_evaluated": len(domains),
                    "ecod_mappings": len(ecod_mappings),
                    "foldseek_hits": len(foldseek_hits),
                    "pae_data_available": pae_matrix is not None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating domain support for {structure_id}: {str(e)}")
            return {
                "status": "FAILED",
                "structure_id": structure_id,
                "error_message": str(e)
            }
    
    def _load_ecod_mappings(self, ecod_mapping_path):
        """Load ECOD mapping results"""
        mappings = []
        
        if not os.path.exists(ecod_mapping_path):
            return mappings
        
        try:
            with open(ecod_mapping_path, 'r') as f:
                # Skip header
                next(f)
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 12:
                        domain_id = parts[0]
                        ecod_key = parts[1]
                        query_range = parts[11]
                        
                        # Parse residue ranges
                        ranges = []
                        for range_str in query_range.split(','):
                            if '-' in range_str:
                                start, end = map(int, range_str.split('-'))
                                ranges.append((start, end))
                            else:
                                pos = int(range_str)
                                ranges.append((pos, pos))
                        
                        mappings.append({
                            'domain_id': domain_id,
                            'ecod_key': ecod_key,
                            'ranges': ranges
                        })
            return mappings
        except Exception as e:
            self.logger.warning(f"Could not load ECOD mappings: {str(e)}")
            return []
    
    def _load_foldseek_hits(self, foldseek_path):
        """Load filtered FoldSeek hits"""
        hits = []
        
        if not os.path.exists(foldseek_path):
            return hits
        
        try:
            with open(foldseek_path, 'r') as f:
                # Skip header
                next(f)
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        domain_id = parts[0]
                        evalue = float(parts[1])
                        range_str = parts[2]
                        
                        # Parse range
                        if '-' in range_str:
                            start, end = map(int, range_str.split('-'))
                            hits.append({
                                'domain_id': domain_id,
                                'evalue': evalue,
                                'start': start,
                                'end': end
                            })
            return hits
        except Exception as e:
            self.logger.warning(f"Could not load FoldSeek hits: {str(e)}")
            return []
    
    def _load_pae_data(self, pae_path):
        """Load PAE (Predicted Aligned Error) data if available"""
        if not pae_path or not os.path.exists(pae_path):
            return None
        
        try:
            with open(pae_path, 'r') as f:
                pae_data = json.load(f)
                
            # Extract PAE matrix
            if 'predicted_aligned_error' in pae_data:
                return np.array(pae_data['predicted_aligned_error'])
            else:
                self.logger.warning(f"PAE data does not contain predicted_aligned_error field")
                return None
        except Exception as e:
            self.logger.warning(f"Could not load PAE data: {str(e)}")
            return None
    
    def _calculate_domain_support(self, domains, ecod_mappings, foldseek_hits, pae_matrix, structure_path):
        """Calculate support scores for each domain"""
        domains_with_support = []
        
        for domain in domains:
            support_scores = {}
            domain_start = domain.get('start', 0)
            domain_end = domain.get('end', 0)
            
            # Calculate ECOD support
            ecod_support = self._calculate_ecod_support(domain_start, domain_end, ecod_mappings)
            support_scores['ecod_support'] = ecod_support
            
            # Calculate FoldSeek support
            foldseek_support = self._calculate_foldseek_support(domain_start, domain_end, foldseek_hits)
            support_scores['foldseek_support'] = foldseek_support
            
            # Calculate PAE support
            pae_support = self._calculate_pae_support(domain_start, domain_end, pae_matrix)
            support_scores['pae_support'] = pae_support
            
            # Calculate structural support based on domain compactness
            structure_support = self._calculate_structure_support(domain_start, domain_end, structure_path)
            support_scores['structure_support'] = structure_support
            
            # Calculate overall support score (weighted average)
            weights = {
                'ecod_support': self.ecod_weight,
                'foldseek_support': self.foldseek_weight,
                'pae_support': self.pae_weight,
                'structure_support': 1.0
            }
            
            denominator = 0
            numerator = 0
            
            for key, score in support_scores.items():
                if score is not None:
                    weight = weights.get(key, 1.0)
                    numerator += score * weight
                    denominator += weight
            
            overall_support = numerator / denominator if denominator > 0 else 0
            
            # Add support scores to domain
            domain_copy = domain.copy()
            domain_copy.update({
                'ecod_support': ecod_support,
                'foldseek_support': foldseek_support,
                'pae_support': pae_support,
                'structure_support': structure_support,
                'overall_support': overall_support
            })
            
            domains_with_support.append(domain_copy)
        
        # Sort domains by overall support
        domains_with_support.sort(key=lambda d: d.get('overall_support', 0), reverse=True)
        
        return domains_with_support
    
    def _calculate_ecod_support(self, domain_start, domain_end, ecod_mappings):
        """Calculate support from ECOD mappings"""
        if not ecod_mappings:
            return None
        
        domain_size = domain_end - domain_start + 1
        domain_residues = set(range(domain_start, domain_end + 1))
        
        max_overlap = 0
        best_match = None
        
        for mapping in ecod_mappings:
            mapping_residues = set()
            for start, end in mapping['ranges']:
                mapping_residues.update(range(start, end + 1))
            
            overlap = len(domain_residues.intersection(mapping_residues))
            overlap_ratio = overlap / domain_size if domain_size > 0 else 0
            
            if overlap_ratio > max_overlap:
                max_overlap = overlap_ratio
                best_match = mapping
        
        # Scale support score
        if max_overlap >= 0.8:
            return 1.0
        elif max_overlap >= 0.5:
            return 0.5 + (max_overlap - 0.5) * 1.0
        elif max_overlap > 0:
            return max_overlap
        else:
            return 0.0
    
    def _calculate_foldseek_support(self, domain_start, domain_end, foldseek_hits):
        """Calculate support from FoldSeek hits"""
        if not foldseek_hits:
            return None
        
        domain_size = domain_end - domain_start + 1
        domain_residues = set(range(domain_start, domain_end + 1))
        
        covered_residues = set()
        for hit in foldseek_hits:
            hit_start = hit.get('start', 0)
            hit_end = hit.get('end', 0)
            hit_residues = set(range(hit_start, hit_end + 1))
            
            # Check overlap with domain
            overlap = domain_residues.intersection(hit_residues)
            covered_residues.update(overlap)
        
        coverage_ratio = len(covered_residues) / domain_size if domain_size > 0 else 0
        
        # Scale support score
        if coverage_ratio >= 0.8:
            return 1.0
        elif coverage_ratio >= 0.5:
            return 0.5 + (coverage_ratio - 0.5) * 1.0
        elif coverage_ratio > 0:
            return coverage_ratio
        else:
            return 0.0
    
    def _calculate_pae_support(self, domain_start, domain_end, pae_matrix):
        """Calculate support from PAE (Predicted Aligned Error) data"""
        if pae_matrix is None:
            return None
        
        # Extract domain region from PAE matrix
        domain_indices = range(domain_start - 1, domain_end)  # Convert to 0-based
        
        # Check if domain indices are valid for the PAE matrix
        if domain_start < 1 or domain_end > len(pae_matrix):
            return None
        
        # Extract domain sub-matrix
        domain_pae = pae_matrix[np.ix_(domain_indices, domain_indices)]
        
        # Calculate within-domain average PAE
        within_domain_avg_pae = np.mean(domain_pae)
        
        # Calculate between-domain average PAE (if structure is longer)
        if len(pae_matrix) > len(domain_indices):
            # Create mask for non-domain regions
            mask = np.ones(len(pae_matrix), dtype=bool)
            mask[domain_indices] = False
            
            # Calculate average PAE between domain and non-domain regions
            between_domain_avg_pae = np.mean([
                np.mean(pae_matrix[domain_indices, :][:, mask]),
                np.mean(pae_matrix[mask, :][:, domain_indices])
            ])
            
            # Higher contrast between within and between PAE indicates better domain
            pae_contrast = between_domain_avg_pae / within_domain_avg_pae if within_domain_avg_pae > 0 else 1.0
            
            # Scale support score
            if pae_contrast >= 2.0:
                return 1.0
            elif pae_contrast >= 1.5:
                return 0.5 + (pae_contrast - 1.5) * 1.0
            elif pae_contrast > 1.0:
                return (pae_contrast - 1.0) * 1.0
            else:
                return 0.0
        else:
            # If domain is the entire structure, use absolute PAE value
            # Lower PAE is better
            if within_domain_avg_pae <= 5.0:
                return 1.0
            elif within_domain_avg_pae <= 10.0:
                return 1.0 - (within_domain_avg_pae - 5.0) / 5.0
            else:
                return 0.0
    
    def _calculate_structure_support(self, domain_start, domain_end, structure_path):
        """
        Calculate support based on structural properties
        
        This is a simplified placeholder - in a real implementation, 
        this would analyze the actual structure for domain-like properties
        """
        # In a real implementation, this would analyze:
        # - Domain compactness (radius of gyration)
        # - Secondary structure elements distribution
        # - Contact density within domain vs. between domains
        # - Hydrophobic core presence
        
        # For now, just return a placeholder value
        domain_size = domain_end - domain_start + 1
        
        if domain_size < self.min_domain_size:
            return 0.0
        elif domain_size > 300:
            return 0.6  # Very large domains are less likely
        else:
            return 0.8  # Default reasonable support
    
    def _write_support_results(self, domains, json_path, tsv_path):
        """Write support results to output files"""
        # Write JSON results
        with open(json_path, 'w') as f:
            json.dump({
                'domains': domains,
                'total_domains': len(domains)
            }, f, indent=2)
        
        # Write TSV results
        with open(tsv_path, 'w') as f:
            f.write("domain_id\tstart\tend\tsize\tecod_support\tfoldseek_support\t"
                   "pae_support\tstructure_support\toverall_support\n")
            
            for domain in domains:
                f.write(f"{domain.get('domain_id', 'unknown')}\t"
                       f"{domain.get('start', 0)}\t{domain.get('end', 0)}\t"
                       f"{domain.get('size', 0)}\t"
                       f"{domain.get('ecod_support', 'NA')}\t"
                       f"{domain.get('foldseek_support', 'NA')}\t"
                       f"{domain.get('pae_support', 'NA')}\t"
                       f"{domain.get('structure_support', 'NA')}\t"
                       f"{domain.get('overall_support', 0.0)}\n")