#!/usr/bin/env python3
"""
Domain quality filtering step for DPAM pipeline.

This module provides functionality for filtering candidate domains
based on sequence and structure evidence, retaining only those with
sufficient support to be considered real domains.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union

class DomainQualityFilter:
    """Filters domains based on quality criteria from multiple evidence sources"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize domain quality filter with configuration
        
        Args:
            config: Configuration containing parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.get_good_domains")
        
        # Configuration parameters
        self.data_dir = config.get('data_dir', '/data')
        self.min_domain_size = config.get('min_domain_size', 25)
        self.min_segment_size = config.get('min_segment_size', 5)
        self.max_segment_gap = config.get('max_segment_gap', 10)
        self.min_znorm_threshold = config.get('min_znorm_threshold', 0.225)
        self.min_qscore_threshold = config.get('min_qscore_threshold', 0.5)
        
        # Load ECOD normalization values
        self.ecod_norms = self._load_ecod_norms()
    
    def run(self, structure_id: str, domain_support_path: str, 
            sequence_results_path: str, structure_results_path: str,
            output_dir: str) -> Dict[str, Any]:
        """
        Filter domains based on quality criteria
        
        Args:
            structure_id: Structure identifier
            domain_support_path: Path to domain support file
            sequence_results_path: Path to sequence analysis results
            structure_results_path: Path to structure analysis results
            output_dir: Directory for output files
            
        Returns:
            Dict with results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Starting domain quality filtering for structure {structure_id}")
        
        prefix = f"struct_{structure_id}"
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            good_domains = []
            
            # Process sequence-based domains
            if os.path.exists(sequence_results_path):
                sequence_domains = self._process_sequence_domains(sequence_results_path)
                good_domains.extend(sequence_domains)
                self.logger.info(f"Found {len(sequence_domains)} good sequence-based domains")
            
            # Process structure-based domains
            if os.path.exists(structure_results_path):
                structure_domains = self._process_structure_domains(structure_results_path)
                good_domains.extend(structure_domains)
                self.logger.info(f"Found {len(structure_domains)} good structure-based domains")
            
            # Write results
            results_path = os.path.join(output_dir, f"{prefix}_good_domains.json")
            summary_path = os.path.join(output_dir, f"{prefix}_good_domains.tsv")
            
            self._write_results(good_domains, results_path, summary_path)
            
            # Return success and output paths
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "status": "COMPLETED",
                "structure_id": structure_id,
                "output_files": {
                    "good_domains_json": results_path,
                    "good_domains_tsv": summary_path
                },
                "metrics": {
                    "duration_seconds": duration,
                    "good_domains_found": len(good_domains)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error filtering domains for {structure_id}: {str(e)}")
            return {
                "status": "FAILED",
                "structure_id": structure_id,
                "error_message": str(e)
            }
    
    def _load_ecod_norms(self) -> Dict[str, float]:
        """Load ECOD normalization values"""
        ecod_norms = {}
        norms_path = os.path.join(self.data_dir, 'ECOD_norms')
        
        if not os.path.exists(norms_path):
            self.logger.warning(f"ECOD norms file not found: {norms_path}")
            return ecod_norms
        
        try:
            with open(norms_path, 'r') as f:
                for line in f:
                    words = line.split()
                    if len(words) >= 2:
                        ecod_norms[words[0]] = float(words[1])
            
            self.logger.debug(f"Loaded {len(ecod_norms)} ECOD normalization values")
            return ecod_norms
            
        except Exception as e:
            self.logger.warning(f"Error loading ECOD norms: {str(e)}")
            return {}
    
    def _filter_segments(self, segment_str: str) -> Tuple[List[str], int]:
        """
        Filter segments based on size and gap criteria
        
        Args:
            segment_str: Comma-separated list of residue ranges (e.g., "10-30,45-60")
            
        Returns:
            Tuple of (filtered segment strings, total residue count)
        """
        filtered_segments = []
        
        # Parse segments into residue lists
        segments = []
        for seg in segment_str.split(','):
            if '-' in seg:
                start, end = map(int, seg.split('-'))
                current_residues = list(range(start, end + 1))
                
                # Add to segments with gap checking
                if not segments:
                    segments.append(current_residues)
                else:
                    # Check if this segment should be merged with the last one
                    if current_residues[0] <= segments[-1][-1] + self.max_segment_gap:
                        # Merge by extending the last segment
                        segments[-1].extend([r for r in current_residues if r > segments[-1][-1]])
                    else:
                        # Start a new segment
                        segments.append(current_residues)
        
        # Convert segments back to range strings and filter by size
        total_residues = 0
        for segment in segments:
            if len(segment) >= self.min_segment_size:
                start = segment[0]
                end = segment[-1]
                total_residues += (end - start + 1)
                filtered_segments.append(f"{start}-{end}")
        
        return filtered_segments, total_residues
    
    def _process_sequence_domains(self, sequence_results_path: str) -> List[Dict[str, Any]]:
        """Process sequence-based domain candidates"""
        good_domains = []
        
        try:
            with open(sequence_results_path, 'r') as f:
                for line in f:
                    words = line.strip().split('\t')
                    if len(words) < 7:
                        continue
                    
                    # Extract information
                    domain_id = words[0]
                    coverage = words[6]
                    
                    # Filter segments
                    filtered_segments, total_residues = self._filter_segments(coverage)
                    
                    # Check if domain meets size criteria
                    if total_residues >= self.min_domain_size and filtered_segments:
                        domain = {
                            'source': 'sequence',
                            'domain_id': domain_id,
                            'quality': 'medium',
                            'original_segments': coverage,
                            'filtered_segments': ','.join(filtered_segments),
                            'total_residues': total_residues,
                            'evidence': {
                                'type': 'sequence',
                                'data': {word_idx: word for word_idx, word in enumerate(words)}
                            }
                        }
                        good_domains.append(domain)
            
            return good_domains
            
        except Exception as e:
            self.logger.warning(f"Error processing sequence domains: {str(e)}")
            return []
    
    def _process_structure_domains(self, structure_results_path: str) -> List[Dict[str, Any]]:
        """Process structure-based domain candidates"""
        good_domains = []
        
        try:
            with open(structure_results_path, 'r') as f:
                for line in f:
                    words = line.strip().split('\t')
                    if len(words) < 10:
                        continue
                    
                    # Extract information
                    ecodnum = words[0].split('_')[0]
                    edomain = words[1]
                    zscore = float(words[3])
                    qscore = float(words[4])
                    ztile = float(words[5])
                    qtile = float(words[6])
                    rank = float(words[7])
                    bestprob = float(words[8])
                    bestcov = float(words[9])
                    segments = words[10] if len(words) > 10 else ""
                    
                    # Calculate normalized Z-score
                    try:
                        znorm = round(zscore / self.ecod_norms.get(ecodnum, 1.0), 2)
                    except (KeyError, ZeroDivisionError):
                        znorm = 0.0
                    
                    # Judge domain quality
                    quality_score = 0
                    
                    # Structure-based criteria
                    if rank < 1.5:
                        quality_score += 1
                    if qscore > self.min_qscore_threshold:
                        quality_score += 1
                    if 0 <= ztile < 0.75:
                        quality_score += 1
                    if 0 <= qtile < 0.75:
                        quality_score += 1
                    if znorm > self.min_znorm_threshold:
                        quality_score += 1
                    
                    # Sequence-based criteria
                    seq_quality = 'no'
                    if bestprob >= 20 and bestcov >= 0.2:
                        quality_score += 1
                        seq_quality = 'low'
                    if bestprob >= 50 and bestcov >= 0.3:
                        quality_score += 1
                        seq_quality = 'medium'
                    if bestprob >= 80 and bestcov >= 0.4:
                        quality_score += 1
                        seq_quality = 'high'
                    if bestprob >= 95 and bestcov >= 0.6:
                        quality_score += 1
                        seq_quality = 'superb'
                    
                    # Filter segments
                    filtered_segments, total_residues = self._filter_segments(segments)
                    
                    # Check if domain meets quality and size criteria
                    if quality_score > 0 and total_residues >= self.min_domain_size and filtered_segments:
                        # Map quality score to confidence level
                        if quality_score >= 6:
                            quality = 'high'
                        elif quality_score >= 3:
                            quality = 'medium'
                        else:
                            quality = 'low'
                        
                        domain = {
                            'source': 'structure',
                            'domain_id': edomain,
                            'quality': quality,
                            'seq_quality': seq_quality,
                            'znorm': znorm,
                            'qscore': qscore,
                            'quality_score': quality_score,
                            'original_segments': segments,
                            'filtered_segments': ','.join(filtered_segments),
                            'total_residues': total_residues,
                            'evidence': {
                                'type': 'structure',
                                'data': {
                                    'zscore': zscore,
                                    'qscore': qscore,
                                    'ztile': ztile,
                                    'qtile': qtile,
                                    'rank': rank,
                                    'bestprob': bestprob,
                                    'bestcov': bestcov
                                }
                            }
                        }
                        good_domains.append(domain)
            
            return good_domains
            
        except Exception as e:
            self.logger.warning(f"Error processing structure domains: {str(e)}")
            return []
    
    def _write_results(self, domains: List[Dict[str, Any]], 
                      json_path: str, tsv_path: str) -> None:
        """Write results to output files"""
        # Write full JSON results
        with open(json_path, 'w') as f:
            json.dump({
                'domains': domains,
                'total_domains': len(domains)
            }, f, indent=2)
        
        # Write summary TSV
        with open(tsv_path, 'w') as f:
            f.write("source\tquality\tdomain_id\tznorm\tqscore\ttotal_residues\tfiltered_segments\n")
            
            for domain in domains:
                source = domain.get('source', 'unknown')
                quality = domain.get('quality', 'unknown')
                domain_id = domain.get('domain_id', 'unknown')
                znorm = domain.get('znorm', 0.0)
                qscore = domain.get('qscore', 0.0)
                total_residues = domain.get('total_residues', 0)
                filtered_segments = domain.get('filtered_segments', '')
                
                f.write(f"{source}\t{quality}\t{domain_id}\t{znorm}\t{qscore}\t"
                       f"{total_residues}\t{filtered_segments}\n")
    
    def get_domain_boundaries(self, domains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert good domains to domain boundary format for downstream processing
        
        Args:
            domains: List of good domains
            
        Returns:
            List of domain boundaries
        """
        boundaries = []
        
        for i, domain in enumerate(domains):
            filtered_segments = domain.get('filtered_segments', '')
            
            # Split into individual segments and build residue list
            domain_residues = set()
            for segment in filtered_segments.split(','):
                if '-' in segment:
                    start, end = map(int, segment.split('-'))
                    domain_residues.update(range(start, end + 1))
            
            if domain_residues:
                # Get overall domain boundaries
                start = min(domain_residues)
                end = max(domain_residues)
                
                boundary = {
                    'domain_id': f"dom_{i+1}",
                    'source': domain.get('source', 'unknown'),
                    'quality': domain.get('quality', 'unknown'),
                    'start': start,
                    'end': end,
                    'size': len(domain_residues),
                    'segments': filtered_segments,
                    'evidence': domain.get('evidence', {})
                }
                
                boundaries.append(boundary)
        
        return boundaries