## dpam/steps/mapping.py

import os
import json
import logging
import subprocess
import tempfile
import requests
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union

class ECODMapper:
    """ECOD mapping utilities for protein domains"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ECOD mapper with configuration
        
        Args:
            config: Configuration containing parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.mapping")
        
        # Configuration parameters
        self.data_dir = config.get('data_dir', '/data')
        self.ecod_db_path = config.get('ecod_db_path', os.path.join(self.data_dir, 'ECOD'))
        self.min_domain_residues = config.get('min_domain_residues', 10)
        
        # API endpoints
        self.ecod_api_base = config.get('ecod_api_base', 'http://prodata.swmed.edu/ecod/rest/api')
        
        # Taxonomy levels in ECOD
        self.ecod_levels = ['X', 'H', 'T', 'F', 'D']
    
    def run(self, structure_id: str, domains_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Map domains to ECOD classifications
        
        Args:
            structure_id: Structure identifier
            domains_path: Path to final domain definitions
            output_dir: Directory for output files
            
        Returns:
            Dict with results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Starting ECOD mapping for structure {structure_id}")
        
        prefix = f"struct_{structure_id}"
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Load domain definitions
            domains = self._load_domains(domains_path)
            
            if not domains:
                self.logger.warning(f"No domains found for structure {structure_id}")
                return {
                    "status": "COMPLETED",
                    "structure_id": structure_id,
                    "output_files": {},
                    "metrics": {
                        "domains_processed": 0,
                        "domains_mapped": 0,
                        "duration_seconds": (datetime.now() - start_time).total_seconds()
                    }
                }
            
            # Map domains to ECOD
            mapped_domains = self._map_domains_to_ecod(domains)
            
            # Add hierarchical classification
            classified_domains = self._add_hierarchical_classification(mapped_domains)
            
            # Write mapping results
            mapping_json_path = os.path.join(output_dir, f"{prefix}_ecod_mapping.json")
            mapping_tsv_path = os.path.join(output_dir, f"{prefix}_ecod_mapping.tsv")
            
            self._write_mapping_results(classified_domains, mapping_json_path, mapping_tsv_path)
            
            # Return success and output paths
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Count mapped domains
            mapped_count = sum(1 for d in classified_domains if d.get('ecod_mapping', {}).get('ecod_domain_id'))
            
            return {
                "status": "COMPLETED",
                "structure_id": structure_id,
                "output_files": {
                    "mapping_json": mapping_json_path,
                    "mapping_tsv": mapping_tsv_path
                },
                "metrics": {
                    "duration_seconds": duration,
                    "domains_processed": len(domains),
                    "domains_mapped": mapped_count
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error mapping domains to ECOD for {structure_id}: {str(e)}")
            return {
                "status": "FAILED",
                "structure_id": structure_id,
                "error_message": str(e)
            }
    
    def _load_domains(self, domains_path: str) -> List[Dict[str, Any]]:
        """Load domain definitions"""
        if not os.path.exists(domains_path):
            return []
            
        with open(domains_path, 'r') as f:
            data = json.load(f)
            
        return data.get('domains', [])
    
    def _map_domains_to_ecod(self, domains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map domains to ECOD classifications using multiple methods"""
        mapped_domains = []
        
        for domain in domains:
            # Try different mapping methods in order of preference
            ecod_mapping = (
                self._map_domain_by_dali(domain) or 
                self._map_domain_by_sequence(domain) or 
                self._map_domain_by_structure(domain) or
                {}
            )
            
            # Add mapping information to domain
            domain_with_mapping = domain.copy()
            domain_with_mapping['ecod_mapping'] = ecod_mapping
            
            mapped_domains.append(domain_with_mapping)
        
        return mapped_domains
    
    def _map_domain_by_dali(self, domain: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map domain to ECOD based on Dali hits"""
        # Check if domain has Dali hits
        dali_hits = domain.get('dali_hits', [])
        if not dali_hits:
            return None
        
        # Find best hit with ECOD mapping
        best_ecod_hit = None
        best_z_score = 0
        
        for hit in dali_hits:
            hit_id = hit.get('domain_id', '')
            z_score = hit.get('z_score', 0)
            
            # Skip if not ECOD domain ID
            if not hit_id.startswith('e'):
                continue
                
            # Update best hit if z-score is higher
            if z_score > best_z_score:
                best_z_score = z_score
                best_ecod_hit = hit
        
        # Return mapping if found
        if best_ecod_hit:
            domain_id = best_ecod_hit.get('domain_id', '')
            mapping = {
                'ecod_domain_id': domain_id,
                'mapping_method': 'dali',
                'z_score': best_z_score,
                'confidence': 'high' if best_z_score > 10 else 'medium'
            }
            return mapping
            
        return None
    
    def _map_domain_by_sequence(self, domain: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map domain to ECOD based on sequence similarity"""
        # Check if domain has sequence
        sequence = domain.get('sequence', '')
        if not sequence:
            return None
            
        # Use ECOD API for sequence search
        try:
            url = f"{self.ecod_api_base}/search_seq/"
            response = requests.post(url, data={'sequence': sequence})
            
            if response.status_code != 200:
                self.logger.warning(f"ECOD API sequence search failed: {response.status_code}")
                return None
                
            # Parse response
            results = response.json()
            hits = results.get('hits', [])
            
            if not hits:
                return None
                
            # Get best hit
            best_hit = hits[0]
            domain_id = best_hit.get('ecod_domain_id', '')
            e_value = best_hit.get('e_value', 1.0)
            
            # Return mapping if good hit
            if domain_id and e_value < 0.001:
                mapping = {
                    'ecod_domain_id': domain_id,
                    'mapping_method': 'sequence',
                    'e_value': e_value,
                    'confidence': 'high' if e_value < 1e-10 else 'medium'
                }
                return mapping
                
        except Exception as e:
            self.logger.warning(f"Error in sequence-based mapping: {str(e)}")
            
        return None
    
    def _map_domain_by_structure(self, domain: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map domain to ECOD based on structural similarity"""
        # This would normally use a structural alignment tool
        # For now, return None to indicate no mapping
        return None
    
    def _add_hierarchical_classification(self, domains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add hierarchical classification information to mapped domains"""
        # Load ECOD classification data
        ecod_hierarchy = self._load_ecod_hierarchy()
        
        # Process each domain
        for domain in domains:
            ecod_mapping = domain.get('ecod_mapping', {})
            ecod_domain_id = ecod_mapping.get('ecod_domain_id', '')
            
            if not ecod_domain_id:
                continue
                
            # Get hierarchical classification
            hierarchy = ecod_hierarchy.get(ecod_domain_id, {})
            if hierarchy:
                ecod_mapping['hierarchy'] = hierarchy
        
        return domains
    
    def _load_ecod_hierarchy(self) -> Dict[str, Dict[str, Any]]:
        """Load ECOD hierarchical classification data"""
        ecod_hierarchy = {}
        
        # Try to load from cached file
        cache_path = os.path.join(self.ecod_db_path, 'ecod_hierarchy.json')
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load ECOD hierarchy from cache: {str(e)}")
        
        # Load from domain list file
        domain_list_path = os.path.join(self.ecod_db_path, 'ecod.latest.domains.txt')
        
        if not os.path.exists(domain_list_path):
            self.logger.warning(f"ECOD domain list not found: {domain_list_path}")
            return ecod_hierarchy
        
        try:
            with open(domain_list_path, 'r') as f:
                # Skip header
                next(f)
                
                # Parse domains
                for line in f:
                    fields = line.strip().split('\t')
                    if len(fields) < 8:
                        continue
                        
                    domain_id = fields[0]
                    x_group = fields[4]
                    h_group = fields[5]
                    t_group = fields[6]
                    f_group = fields[7]
                    
                    ecod_hierarchy[domain_id] = {
                        'domain_id': domain_id,
                        'x_group': x_group,
                        'h_group': h_group,
                        't_group': t_group,
                        'f_group': f_group
                    }
                    
            # Cache the hierarchy
            with open(cache_path, 'w') as f:
                json.dump(ecod_hierarchy, f)
                
        except Exception as e:
            self.logger.error(f"Error loading ECOD hierarchy: {str(e)}")
        
        return ecod_hierarchy
    
    def _write_mapping_results(self, domains: List[Dict[str, Any]], 
                             json_path: str, tsv_path: str) -> None:
        """Write mapping results to output files"""
        # Write full JSON results
        with open(json_path, 'w') as f:
            json.dump({
                'domains': domains,
                'total_domains': len(domains)
            }, f, indent=2)
        
        # Write TSV summary
        with open(tsv_path, 'w') as f:
            f.write("domain_id\tstart\tend\tecod_domain_id\tmapping_method\t"
                   "confidence\tx_group\th_group\tt_group\tf_group\n")
            
            for domain in domains:
                domain_id = domain.get('domain_id', 'unknown')
                start = domain.get('start', 0)
                end = domain.get('end', 0)
                
                # Get ECOD mapping information
                mapping = domain.get('ecod_mapping', {})
                ecod_domain_id = mapping.get('ecod_domain_id', '-')
                mapping_method = mapping.get('mapping_method', '-')
                confidence = mapping.get('confidence', '-')
                
                # Get hierarchy information
                hierarchy = mapping.get('hierarchy', {})
                x_group = hierarchy.get('x_group', '-')
                h_group = hierarchy.get('h_group', '-')
                t_group = hierarchy.get('t_group', '-')
                f_group = hierarchy.get('f_group', '-')
                
                f.write(f"{domain_id}\t{start}\t{end}\t{ecod_domain_id}\t{mapping_method}\t"
                       f"{confidence}\t{x_group}\t{h_group}\t{t_group}\t{f_group}\n")
    
    def fetch_ecod_domain_info(self, ecod_domain_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information about an ECOD domain
        
        Args:
            ecod_domain_id: ECOD domain identifier
            
        Returns:
            Dictionary with domain information or None if not found
        """
        try:
            # Use ECOD API
            url = f"{self.ecod_api_base}/domain/{ecod_domain_id}/"
            response = requests.get(url)
            
            if response.status_code != 200:
                self.logger.warning(f"ECOD API domain lookup failed: {response.status_code}")
                return None
                
            # Parse response
            domain_info = response.json()
            return domain_info
            
        except Exception as e:
            self.logger.warning(f"Error fetching ECOD domain info: {str(e)}")
            return None
    
    def get_representative_structures(self, x_group: str, h_group: str = None,
                                    t_group: str = None, f_group: str = None) -> List[str]:
        """
        Get representative structures for an ECOD classification level
        
        Args:
            x_group: X-group identifier
            h_group: H-group identifier (optional)
            t_group: T-group identifier (optional)
            f_group: F-group identifier (optional)
            
        Returns:
            List of representative domain IDs
        """
        try:
            # Build query parameters
            params = {'x_group': x_group}
            if h_group:
                params['h_group'] = h_group
            if t_group:
                params['t_group'] = t_group
            if f_group:
                params['f_group'] = f_group
                
            # Use ECOD API
            url = f"{self.ecod_api_base}/representatives/"
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                self.logger.warning(f"ECOD API representatives lookup failed: {response.status_code}")
                return []
                
            # Parse response
            results = response.json()
            return results.get('representatives', [])
            
        except Exception as e:
            self.logger.warning(f"Error fetching ECOD representatives: {str(e)}")
            return []
    
    def get_domain_statistics(self, domains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics for mapped domains
        
        Args:
            domains: List of domains with ECOD mapping
            
        Returns:
            Dictionary with mapping statistics
        """
        stats = {
            'total_domains': len(domains),
            'mapped_domains': 0,
            'mapping_methods': {},
            'confidence_levels': {},
            'x_groups': {},
            'domain_types': {}
        }
        
        for domain in domains:
            mapping = domain.get('ecod_mapping', {})
            ecod_domain_id = mapping.get('ecod_domain_id', '')
            
            if not ecod_domain_id:
                continue
                
            # Count mapped domains
            stats['mapped_domains'] += 1
            
            # Count mapping methods
            method = mapping.get('mapping_method', 'unknown')
            stats['mapping_methods'][method] = stats['mapping_methods'].get(method, 0) + 1
            
            # Count confidence levels
            confidence = mapping.get('confidence', 'unknown')
            stats['confidence_levels'][confidence] = stats['confidence_levels'].get(confidence, 0) + 1
            
            # Count X-groups
            hierarchy = mapping.get('hierarchy', {})
            x_group = hierarchy.get('x_group', 'unknown')
            stats['x_groups'][x_group] = stats['x_groups'].get(x_group, 0) + 1
            
            # Count domain classes
            domain_class = domain.get('ss_composition', {}).get('domain_class', 'unknown')
            stats['domain_types'][domain_class] = stats['domain_types'].get(domain_class, 0) + 1
        
        # Calculate percentages
        if stats['total_domains'] > 0:
            stats['percent_mapped'] = (stats['mapped_domains'] / stats['total_domains']) * 100
        else:
            stats['percent_mapped'] = 0
            
        return stats