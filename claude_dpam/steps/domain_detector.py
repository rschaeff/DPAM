#!/usr/bin/env python3
"""
DPAM domain detector

This module reads alignment data and provides the domain definitions

"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union

class DomainDetector:
    """Detects and finalizes domain definitions based on multiple sources of evidence"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize domain detector with configuration
        
        Args:
            config: Configuration containing parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.domains")
        
        # Configuration parameters
        self.min_domain_size = config.get('min_domain_size', 30)
        self.max_domain_overlap = config.get('max_domain_overlap', 10)
        self.min_domain_coverage = config.get('min_domain_coverage', 0.8)
        self.min_support_score = config.get('min_support_score', 0.5)
        self.min_domain_ordered_percent = config.get('min_domain_ordered_percent', 70.0)
        
        # Confidence thresholds
        self.high_confidence_threshold = config.get('high_confidence_threshold', 0.8)
        self.medium_confidence_threshold = config.get('medium_confidence_threshold', 0.6)
    
    def run(self, structure_id: str, structure_path: str, support_path: str, 
            sse_path: str, disorder_path: str, dali_path: str, 
            output_dir: str) -> Dict[str, Any]:
        """
        Detect and finalize domain definitions
        
        Args:
            structure_id: Structure identifier
            structure_path: Path to input structure file (PDB/mmCIF)
            support_path: Path to domain support file
            sse_path: Path to secondary structure results
            disorder_path: Path to disorder prediction results
            dali_path: Path to Dali analysis results
            output_dir: Directory for output files
            
        Returns:
            Dict with results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Starting domain detection for structure {structure_id}")
        
        prefix = f"struct_{structure_id}"
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Load domain data from different sources
            support_domains = self._load_support_domains(support_path)
            sse_domains = self._load_sse_domains(sse_path)
            disorder_domains = self._load_disorder_domains(disorder_path)
            dali_domains = self._load_dali_domains(dali_path)
            
            # Merge and integrate domains from all sources
            integrated_domains = self._integrate_domain_data(
                support_domains, sse_domains, disorder_domains, dali_domains
            )
            
            # Resolve overlapping domains
            final_domains = self._resolve_domain_overlaps(integrated_domains)
            
            # Assign confidence levels
            domains_with_confidence = self._assign_confidence_levels(final_domains)
            
            # Filter out low-confidence domains
            filtered_domains = self._filter_low_confidence_domains(domains_with_confidence)
            
            # Write domain results
            domains_json_path = os.path.join(output_dir, f"{prefix}_domains.json")
            domains_tsv_path = os.path.join(output_dir, f"{prefix}_domains.tsv")
            visual_path = os.path.join(output_dir, f"{prefix}_domains_visual.json")
            
            self._write_domain_results(filtered_domains, domains_json_path, domains_tsv_path, visual_path)
            
            # Return success and output paths
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "status": "COMPLETED",
                "structure_id": structure_id,
                "output_files": {
                    "domains_json": domains_json_path,
                    "domains_tsv": domains_tsv_path,
                    "domains_visual": visual_path
                },
                "metrics": {
                    "duration_seconds": duration,
                    "support_domains": len(support_domains),
                    "sse_domains": len(sse_domains),
                    "disorder_domains": len(disorder_domains),
                    "dali_domains": len(dali_domains),
                    "integrated_domains": len(integrated_domains),
                    "final_domains": len(filtered_domains)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting domains for {structure_id}: {str(e)}")
            return {
                "status": "FAILED",
                "structure_id": structure_id,
                "error_message": str(e)
            }
    
    def _load_support_domains(self, support_path: str) -> List[Dict[str, Any]]:
        """Load domains from support file"""
        if not os.path.exists(support_path):
            return []
            
        with open(support_path, 'r') as f:
            data = json.load(f)
            
        return data.get('domains', [])
    
    def _load_sse_domains(self, sse_path: str) -> List[Dict[str, Any]]:
        """Load domains with secondary structure information"""
        if not os.path.exists(sse_path):
            return []
            
        with open(sse_path, 'r') as f:
            data = json.load(f)
            
        return data.get('domains', [])
    
    def _load_disorder_domains(self, disorder_path: str) -> List[Dict[str, Any]]:
        """Load domains with disorder information"""
        if not os.path.exists(disorder_path):
            return []
            
        with open(disorder_path, 'r') as f:
            data = json.load(f)
            
        return data.get('domains', [])
    
    def _load_dali_domains(self, dali_path: str) -> List[Dict[str, Any]]:
        """Load domains from Dali analysis"""
        if not os.path.exists(dali_path):
            return []
            
        with open(dali_path, 'r') as f:
            data = json.load(f)
            
        return data.get('domains', [])
    
    def _integrate_domain_data(self, support_domains: List[Dict[str, Any]],
                             sse_domains: List[Dict[str, Any]],
                             disorder_domains: List[Dict[str, Any]],
                             dali_domains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Integrate domain information from multiple sources"""
        # Create lookup dictionaries
        sse_lookup = {d.get('domain_id', ''): d for d in sse_domains if 'domain_id' in d}
        disorder_lookup = {d.get('domain_id', ''): d for d in disorder_domains if 'domain_id' in d}
        dali_lookup = {d.get('domain_id', ''): d for d in dali_domains if 'domain_id' in d}
        
        # Start with support domains as base
        integrated_domains = []
        
        # Process each domain from support file
        for base_domain in support_domains:
            domain_id = base_domain.get('domain_id', '')
            if not domain_id:
                continue
                
            # Create merged domain starting with support data
            merged_domain = base_domain.copy()
            
            # Add SSE data if available
            if domain_id in sse_lookup:
                sse_data = sse_lookup[domain_id]
                if 'ss_composition' in sse_data:
                    merged_domain['ss_composition'] = sse_data['ss_composition']
                if 'ss_segments' in sse_data:
                    merged_domain['ss_segments'] = sse_data['ss_segments']
            
            # Add disorder data if available
            if domain_id in disorder_lookup:
                disorder_data = disorder_lookup[domain_id]
                if 'disorder_metrics' in disorder_data:
                    merged_domain['disorder_metrics'] = disorder_data['disorder_metrics']
                if 'overlapping_disorder' in disorder_data:
                    merged_domain['overlapping_disorder'] = disorder_data['overlapping_disorder']
                
                # Use refined boundaries if available
                if 'refined_boundaries' in disorder_data:
                    # Store original boundaries
                    if 'original_boundaries' not in merged_domain:
                        merged_domain['original_boundaries'] = {
                            'start': merged_domain.get('start', 0),
                            'end': merged_domain.get('end', 0),
                            'source': merged_domain.get('source', 'support')
                        }
                    
                    # Apply refined boundaries
                    refined = disorder_data['refined_boundaries']
                    merged_domain['start'] = refined.get('start')
                    merged_domain['end'] = refined.get('end')
                    merged_domain['size'] = merged_domain['end'] - merged_domain['start'] + 1
                    if 'boundary_sources' not in merged_domain:
                        merged_domain['boundary_sources'] = []
                    merged_domain['boundary_sources'].append('disorder')
            
            # Add Dali data if available
            if domain_id in dali_lookup:
                dali_data = dali_lookup[domain_id]
                if 'dali_hits' in dali_data:
                    merged_domain['dali_hits'] = dali_data['dali_hits']
                if 'dali_homology' in dali_data:
                    merged_domain['dali_homology'] = dali_data['dali_homology']
            
            # Add domain to integrated list
            integrated_domains.append(merged_domain)
            
        # Add domains that are only in Dali but not in support
        for domain_id, dali_domain in dali_lookup.items():
            # Skip if already processed
            if any(d.get('domain_id') == domain_id for d in integrated_domains):
                continue
            
            # Need minimum evidence to add a domain
            if dali_domain.get('z_score', 0) < 10:
                continue
                
            # Create new domain from Dali data
            new_domain = dali_domain.copy()
            new_domain['source'] = 'dali'
            
            # Add to integrated list
            integrated_domains.append(new_domain)
        
        return integrated_domains
    
    def _resolve_domain_overlaps(self, domains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve overlapping domains by ranking and selecting non-overlapping subsets"""
        if not domains:
            return []
            
        # Sort domains by quality/support score in descending order
        sorted_domains = sorted(
            domains, 
            key=lambda d: d.get('overall_support', 0) + d.get('quality_score', 0), 
            reverse=True
        )
        
        # Initialize set of accepted domains
        accepted_domains = []
        
        # Process domains in order of quality
        for domain in sorted_domains:
            domain_start = domain.get('start', 0)
            domain_end = domain.get('end', 0)
            domain_size = domain_end - domain_start + 1
            
            # Skip if domain is too small
            if domain_size < self.min_domain_size:
                continue
                
            # Check for significant overlap with accepted domains
            has_significant_overlap = False
            
            for accepted in accepted_domains:
                accepted_start = accepted.get('start', 0)
                accepted_end = accepted.get('end', 0)
                
                # Check for overlap
                if domain_start <= accepted_end and domain_end >= accepted_start:
                    # Calculate overlap size
                    overlap_start = max(domain_start, accepted_start)
                    overlap_end = min(domain_end, accepted_end)
                    overlap_size = overlap_end - overlap_start + 1
                    
                    # Skip if overlap is too large
                    if overlap_size > self.max_domain_overlap:
                        has_significant_overlap = True
                        break
            
            # Add to accepted domains if no significant overlap
            if not has_significant_overlap:
                accepted_domains.append(domain)
        
        return accepted_domains
    
    def _assign_confidence_levels(self, domains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign confidence levels based on evidence quality"""
        domains_with_confidence = []
        
        for domain in domains:
            # Extract evidence metrics
            support_score = domain.get('overall_support', 0)
            
            # Get disorder metrics if available
            disorder_metrics = domain.get('disorder_metrics', {})
            percent_ordered = disorder_metrics.get('percent_ordered', 0) if disorder_metrics else 0
            
            # Get secondary structure metrics if available
            ss_comp = domain.get('ss_composition', {})
            has_secondary_structure = ss_comp is not None and (
                ss_comp.get('helix_percent', 0) > 10 or ss_comp.get('strand_percent', 0) > 10
            )
            
            # Calculate confidence level
            confidence = "low"
            confidence_score = 0.0
            
            # Support score has highest weight
            if support_score >= self.high_confidence_threshold:
                confidence_score += 0.6
            elif support_score >= self.medium_confidence_threshold:
                confidence_score += 0.4
            else:
                confidence_score += 0.2
                
            # Disorder metrics contribute
            if percent_ordered >= self.min_domain_ordered_percent:
                confidence_score += 0.2
            else:
                confidence_score += 0.1 * (percent_ordered / self.min_domain_ordered_percent)
                
            # Secondary structure contributes
            if has_secondary_structure:
                confidence_score += 0.2
            
            # Determine final confidence level
            if confidence_score >= self.high_confidence_threshold:
                confidence = "high"
            elif confidence_score >= self.medium_confidence_threshold:
                confidence = "medium"
            
            # Add confidence to domain
            domain_with_confidence = domain.copy()
            domain_with_confidence.update({
                'confidence_level': confidence,
                'confidence_score': round(confidence_score, 2)
            })
            
            domains_with_confidence.append(domain_with_confidence)
        
        return domains_with_confidence
    
    def _filter_low_confidence_domains(self, domains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out low confidence domains"""
        return [d for d in domains if d.get('confidence_level', 'low') != 'low']
    
    def _write_domain_results(self, domains: List[Dict[str, Any]], 
                            json_path: str, tsv_path: str, 
                            visual_path: str) -> None:
        """Write domain results to output files"""
        # Write full JSON results
        with open(json_path, 'w') as f:
            json.dump({
                'domains': domains,
                'total_domains': len(domains)
            }, f, indent=2)
        
        # Write TSV summary
        with open(tsv_path, 'w') as f:
            f.write("domain_id\tstart\tend\tsize\tsource\tconfidence_level\t"
                   "support_score\tpercent_ordered\tdomain_class\n")
            
            for domain in domains:
                domain_id = domain.get('domain_id', 'unknown')
                start = domain.get('start', 0)
                end = domain.get('end', 0)
                size = domain.get('size', end - start + 1)
                source = domain.get('source', 'unknown')
                confidence = domain.get('confidence_level', 'low')
                support = domain.get('overall_support', 0)
                
                # Get disorder metrics if available
                disorder_metrics = domain.get('disorder_metrics', {})
                percent_ordered = disorder_metrics.get('percent_ordered', 0) if disorder_metrics else 0
                
                # Get domain class if available
                ss_comp = domain.get('ss_composition', {})
                domain_class = ss_comp.get('domain_class', 'unknown') if ss_comp else 'unknown'
                
                f.write(f"{domain_id}\t{start}\t{end}\t{size}\t{source}\t{confidence}\t"
                       f"{support:.2f}\t{percent_ordered:.1f}\t{domain_class}\n")
        
        # Write visualization data
        self._write_visual_data(domains, visual_path)
    
    def _write_visual_data(self, domains: List[Dict[str, Any]], visual_path: str) -> None:
        """Write domain data in format suitable for visualization"""
        visual_data = {
            'domains': [],
            'boundaries': [],
            'residue_classes': []
        }
        
        # Process each domain
        for domain in domains:
            domain_id = domain.get('domain_id', 'unknown')
            start = domain.get('start', 0)
            end = domain.get('end', 0)
            confidence = domain.get('confidence_level', 'low')
            
            # Add domain information
            visual_data['domains'].append({
                'id': domain_id,
                'start': start,
                'end': end,
                'confidence': confidence,
                'source': domain.get('source', 'unknown'),
                'class': domain.get('ss_composition', {}).get('domain_class', 'unknown')
            })
            
            # Add boundary information
            visual_data['boundaries'].append({
                'domain_id': domain_id,
                'type': 'domain',
                'position': start,
                'confidence': confidence
            })
            
            visual_data['boundaries'].append({
                'domain_id': domain_id,
                'type': 'domain',
                'position': end,
                'confidence': confidence
            })
            
            # Add original boundaries if available
            original = domain.get('original_boundaries', {})
            if original:
                orig_start = original.get('start', 0)
                orig_end = original.get('end', 0)
                
                if orig_start != start:
                    visual_data['boundaries'].append({
                        'domain_id': domain_id,
                        'type': 'original',
                        'position': orig_start,
                        'confidence': 'low'
                    })
                
                if orig_end != end:
                    visual_data['boundaries'].append({
                        'domain_id': domain_id,
                        'type': 'original',
                        'position': orig_end,
                        'confidence': 'low'
                    })
            
            # Add residue classifications
            for res in range(start, end + 1):
                visual_data['residue_classes'].append({
                    'position': res,
                    'domain_id': domain_id,
                    'confidence': confidence
                })
        
        # Write to file
        with open(visual_path, 'w') as f:
            json.dump(visual_data, f, indent=2)
    
    def combine_small_domains(self, domains: List[Dict[str, Any]], 
                             min_domain_size: int = None) -> List[Dict[str, Any]]:
        """
        Combine small domains that are adjacent or close
        
        This method is useful for post-processing domains that are too small
        individually but might represent parts of the same domain.
        
        Args:
            domains: List of domain dictionaries
            min_domain_size: Minimum domain size (defaults to class attribute)
            
        Returns:
            List of combined domains
        """
        if min_domain_size is None:
            min_domain_size = self.min_domain_size
        
        # Sort domains by start position
        sorted_domains = sorted(domains, key=lambda d: d.get('start', 0))
        
        # Initialize result list
        combined_domains = []
        current_domain = None
        
        for domain in sorted_domains:
            domain_size = domain.get('end', 0) - domain.get('start', 0) + 1
            
            # Skip domains that are already large enough
            if domain_size >= min_domain_size:
                # Finalize current combined domain if exists
                if current_domain is not None:
                    combined_domains.append(current_domain)
                    current_domain = None
                
                # Add domain to result
                combined_domains.append(domain)
                continue
            
            # Handle small domain
            if current_domain is None:
                # Start new combined domain
                current_domain = domain.copy()
            else:
                # Check if this domain is close to current combined domain
                curr_end = current_domain.get('end', 0)
                domain_start = domain.get('start', 0)
                
                if domain_start <= curr_end + 20:  # Allow small gaps (20 residues)
                    # Extend current domain
                    current_domain['end'] = max(curr_end, domain.get('end', 0))
                    current_domain['size'] = current_domain['end'] - current_domain['start'] + 1
                    
                    # Combine IDs
                    if current_domain.get('domain_id') != domain.get('domain_id'):
                        current_domain['domain_id'] = f"{current_domain.get('domain_id')}+{domain.get('domain_id')}"
                    
                    # Mark as combined
                    current_domain['combined'] = True
                    
                    # Take maximum support score
                    current_support = current_domain.get('overall_support', 0)
                    domain_support = domain.get('overall_support', 0)
                    current_domain['overall_support'] = max(current_support, domain_support)
                else:
                    # Finalize current combined domain if large enough
                    combined_size = current_domain.get('end', 0) - current_domain.get('start', 0) + 1
                    if combined_size >= min_domain_size:
                        combined_domains.append(current_domain)
                    
                    # Start new combined domain
                    current_domain = domain.copy()
        
        # Add final combined domain if exists and large enough
        if current_domain is not None:
            combined_size = current_domain.get('end', 0) - current_domain.get('start', 0) + 1
            if combined_size >= min_domain_size:
                combined_domains.append(current_domain)
        
        return combined_domains