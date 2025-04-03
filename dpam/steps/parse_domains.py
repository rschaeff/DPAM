## dpam/steps/parse_domains.py

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

class DomainParser:
    """Parses and finalizes domain definitions from various sources"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize domain parser with configuration
        
        Args:
            config: Configuration containing parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.parse_domains")
        
        # Configuration parameters
        self.min_domain_size = config.get('min_domain_size', 30)
        self.min_support_score = config.get('min_support_score', 0.5)
        self.min_ordered_percent = config.get('min_ordered_percent', 70.0)
        self.maximum_domain_overlap = config.get('maximum_domain_overlap', 20)
        self.maximum_overlap_percent = config.get('maximum_overlap_percent', 0.2)
    
    def run(self, structure_id: str, structure_path: str, domain_support_path: str, 
            sse_path: str, disorder_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Parse and finalize domain definitions
        
        Args:
            structure_id: Structure identifier
            structure_path: Path to input structure file (PDB/mmCIF)
            domain_support_path: Path to domain support file
            sse_path: Path to secondary structure results
            disorder_path: Path to disorder prediction results
            output_dir: Directory for output files
            
        Returns:
            Dict with results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Parsing final domains for structure {structure_id}")
        
        prefix = f"struct_{structure_id}"
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Load domain support data
            domains_with_support = self._load_domains_with_support(domain_support_path)
            
            # Load SSE data
            domains_with_sse = self._load_domains_with_sse(sse_path)
            
            # Load disorder data
            domains_with_disorder = self._load_domains_with_disorder(disorder_path)
            
            # Merge domain information
            merged_domains = self._merge_domain_data(
                domains_with_support, domains_with_sse, domains_with_disorder
            )
            
            # Filter and finalize domains
            final_domains = self._filter_and_finalize_domains(merged_domains)
            
            # Resolve domain overlaps
            non_overlapping_domains = self._resolve_domain_overlaps(final_domains)
            
            # Get final domain ranges for visualization
            domain_ranges = self._get_domain_ranges(non_overlapping_domains)
            
            # Write results
            results_path = os.path.join(output_dir, f"{prefix}_domains.json")
            summary_path = os.path.join(output_dir, f"{prefix}_domains_summary.tsv")
            ranges_path = os.path.join(output_dir, f"{prefix}_domain_ranges.txt")
            
            self._write_results(non_overlapping_domains, results_path, summary_path, ranges_path)
            
            # Return success and output paths
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "status": "COMPLETED",
                "structure_id": structure_id,
                "output_files": {
                    "domains_json": results_path,
                    "domains_summary": summary_path,
                    "domain_ranges": ranges_path
                },
                "metrics": {
                    "duration_seconds": duration,
                    "total_domains": len(non_overlapping_domains),
                    "initial_domains": len(merged_domains),
                    "filtered_domains": len(final_domains)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing domains for {structure_id}: {str(e)}")
            return {
                "status": "FAILED",
                "structure_id": structure_id,
                "error_message": str(e)
            }
    
    def _load_domains_with_support(self, path: str) -> List[Dict[str, Any]]:
        """Load domains with support scores"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return data.get('domains', [])
    
    def _load_domains_with_sse(self, path: str) -> List[Dict[str, Any]]:
        """Load domains with secondary structure information"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return data.get('domains', [])
    
    def _load_domains_with_disorder(self, path: str) -> List[Dict[str, Any]]:
        """Load domains with disorder information"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return data.get('domains', [])
    
    def _merge_domain_data(self, support_domains: List[Dict[str, Any]], 
                         sse_domains: List[Dict[str, Any]],
                         disorder_domains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge domain information from multiple sources"""
        # Create lookup table by domain ID
        sse_lookup = {d.get('domain_id', ''): d for d in sse_domains if 'domain_id' in d}
        disorder_lookup = {d.get('domain_id', ''): d for d in disorder_domains if 'domain_id' in d}
        
        # Start with support domains as base
        merged_domains = []
        
        for domain in support_domains:
            domain_id = domain.get('domain_id', '')
            if not domain_id:
                continue
                
            # Create merged domain
            merged_domain = domain.copy()
            
            # Add SSE data if available
            if domain_id in sse_lookup:
                sse_domain = sse_lookup[domain_id]
                merged_domain['ss_composition'] = sse_domain.get('ss_composition')
                merged_domain['ss_segments'] = sse_domain.get('ss_segments')
            
            # Add disorder data if available
            if domain_id in disorder_lookup:
                disorder_domain = disorder_lookup[domain_id]
                merged_domain['disorder_metrics'] = disorder_domain.get('disorder_metrics')
                merged_domain['overlapping_disorder'] = disorder_domain.get('overlapping_disorder')
                
                # Use refined boundaries if available
                if 'refined_boundaries' in disorder_domain:
                    merged_domain['original_boundaries'] = {
                        'start': merged_domain.get('start', 0),
                        'end': merged_domain.get('end', 0)
                    }
                    refined = disorder_domain['refined_boundaries']
                    merged_domain['start'] = refined.get('start')
                    merged_domain['end'] = refined.get('end')
                    merged_domain['size'] = merged_domain['end'] - merged_domain['start'] + 1
                    merged_domain['boundary_refinement'] = 'disorder_based'
            
            merged_domains.append(merged_domain)
        
        return merged_domains
    
    def _filter_and_finalize_domains(self, domains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter domains based on various quality criteria"""
        filtered_domains = []
        
        for domain in domains:
            # Check domain size
            domain_size = domain.get('size', 0)
            if domain_size < self.min_domain_size:
                self.logger.debug(f"Domain {domain.get('domain_id')} filtered out: too small ({domain_size})")
                continue
            
            # Check support score
            support_score = domain.get('overall_support', 0)
            if support_score < self.min_support_score:
                self.logger.debug(f"Domain {domain.get('domain_id')} filtered out: low support ({support_score})")
                continue
            
            # Check disorder percentage
            disorder_metrics = domain.get('disorder_metrics', {})
            if disorder_metrics:
                percent_ordered = disorder_metrics.get('percent_ordered', 0)
                if percent_ordered < self.min_ordered_percent:
                    self.logger.debug(f"Domain {domain.get('domain_id')} filtered out: too disordered ({percent_ordered:.1f}%)")
                    continue
            
            # Add domain quality score
            quality_score = self._calculate_domain_quality(domain)
            domain['quality_score'] = quality_score
            
            filtered_domains.append(domain)
        
        # Sort domains by quality score (descending)
        filtered_domains.sort(key=lambda d: d.get('quality_score', 0), reverse=True)
        
        return filtered_domains
    
    def _calculate_domain_quality(self, domain: Dict[str, Any]) -> float:
        """Calculate an overall domain quality score"""
        # Start with support score
        quality = domain.get('overall_support', 0) * 0.5
        
        # Add contribution from disorder metrics
        disorder_metrics = domain.get('disorder_metrics', {})
        if disorder_metrics:
            # Higher pLDDT means better quality
            avg_plddt = disorder_metrics.get('avg_plddt', 0)
            if avg_plddt:
                quality += min(1.0, avg_plddt / 100) * 0.3
            
            # Higher ordered percentage means better quality
            percent_ordered = disorder_metrics.get('percent_ordered', 0)
            if percent_ordered:
                quality += min(1.0, percent_ordered / 100) * 0.2
        
        # Bonus for alpha/beta domains (more likely to be real domains)
        ss_comp = domain.get('ss_composition', {})
        if ss_comp:
            domain_class = ss_comp.get('domain_class', '')
            if domain_class == 'alpha/beta':
                quality += 0.1
        
        return min(1.0, quality)
    
    def _resolve_domain_overlaps(self, domains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve overlaps between domains"""
        if not domains:
            return []
            
        # Start with highest quality domain
        sorted_domains = sorted(domains, key=lambda d: d.get('quality_score', 0), reverse=True)
        accepted_domains = [sorted_domains[0]]
        
        # Process remaining domains
        for domain in sorted_domains[1:]:
            # Check for significant overlap with accepted domains
            has_significant_overlap = False
            
            for accepted in accepted_domains:
                overlap = self._calculate_domain_overlap(domain, accepted)
                
                # Skip if significant overlap
                if overlap['overlap_residues'] > self.maximum_domain_overlap and \
                   overlap['overlap_percent'] > self.maximum_overlap_percent:
                    has_significant_overlap = True
                    break
            
            if not has_significant_overlap:
                accepted_domains.append(domain)
        
        return accepted_domains
    
    def _calculate_domain_overlap(self, domain1: Dict[str, Any], 
                                domain2: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overlap between two domains"""
        start1 = domain1.get('start', 0)
        end1 = domain1.get('end', 0)
        start2 = domain2.get('start', 0)
        end2 = domain2.get('end', 0)
        
        # Check for overlap
        if end1 < start2 or end2 < start1:
            return {
                'overlap_residues': 0,
                'overlap_percent': 0,
                'domain1_size': end1 - start1 + 1,
                'domain2_size': end2 - start2 + 1
            }
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_residues = overlap_end - overlap_start + 1
        
        # Calculate overlap percentage (relative to smaller domain)
        domain1_size = end1 - start1 + 1
        domain2_size = end2 - start2 + 1
        smaller_size = min(domain1_size, domain2_size)
        overlap_percent = overlap_residues / smaller_size
        
        return {
            'overlap_residues': overlap_residues,
            'overlap_percent': overlap_percent,
            'domain1_size': domain1_size,
            'domain2_size': domain2_size
        }
    
    def _get_domain_ranges(self, domains: List[Dict[str, Any]]) -> List[str]:
        """Get domain ranges in format suitable for visualization"""
        ranges = []
        
        for domain in domains:
            domain_id = domain.get('domain_id', 'unknown')
            start = domain.get('start', 0)
            end = domain.get('end', 0)
            
            ranges.append(f"{domain_id}\t{start}-{end}")
        
        return ranges
    
    def _write_results(self, domains: List[Dict[str, Any]], 
                      results_path: str, summary_path: str, ranges_path: str) -> None:
        """Write results to output files"""
        # Write full JSON results
        with open(results_path, 'w') as f:
            json.dump({
                'domains': domains,
                'total_domains': len(domains)
            }, f, indent=2)
        
        # Write domain summary
        with open(summary_path, 'w') as f:
            f.write("domain_id\tstart\tend\tsize\tquality_score\toverall_support\t"
                   "domain_class\tpercent_ordered\tavg_plddt\n")
            
            for domain in domains:
                domain_id = domain.get('domain_id', 'unknown')
                start = domain.get('start', 0)
                end = domain.get('end', 0)
                size = domain.get('size', 0)
                quality = domain.get('quality_score', 0)
                support = domain.get('overall_support', 0)
                
                # Get secondary structure class
                ss_comp = domain.get('ss_composition', {})
                domain_class = ss_comp.get('domain_class', 'unknown') if ss_comp else 'unknown'
                
                # Get disorder metrics
                metrics = domain.get('disorder_metrics', {})
                percent_ordered = metrics.get('percent_ordered', 0) if metrics else 0
                avg_plddt = metrics.get('avg_plddt', 0) if metrics else 0
                
                f.write(f"{domain_id}\t{start}\t{end}\t{size}\t{quality:.3f}\t"
                       f"{support:.3f}\t{domain_class}\t{percent_ordered:.1f}\t"
                       f"{avg_plddt:.1f}\n")
        
        # Write domain ranges for visualization
        with open(ranges_path, 'w') as f:
            for domain in domains:
                domain_id = domain.get('domain_id', 'unknown')
                start = domain.get('start', 0)
                end = domain.get('end', 0)
                quality = domain.get('quality_score', 0)
                
                f.write(f"{domain_id}\t{start}-{end}\t{quality:.3f}\n")