## dpam/steps/filter_foldseek.py

import os
import logging
from datetime import datetime

class FoldSeekFilter:
    """Filters FoldSeek results to retain significant matches"""
    
    def __init__(self, config):
        """
        Initialize FoldSeek filter with configuration
        
        Args:
            config (dict): Configuration containing filtering parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.filter_foldseek")
        self.max_hit_count = config.get('foldseek_max_hit_count', 100)
        self.min_good_residues = config.get('foldseek_min_good_residues', 10)
    
    def run(self, structure_id, fasta_path, foldseek_path, output_dir):
        """
        Filter FoldSeek results to identify significant hits
        
        Args:
            structure_id (str): Structure identifier
            fasta_path (str): Path to query sequence FASTA file
            foldseek_path (str): Path to FoldSeek results file
            output_dir (str): Directory for output files
            
        Returns:
            dict: Results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Filtering FoldSeek results for structure {structure_id}")
        
        prefix = f"struct_{structure_id}"
        
        try:
            # Read query sequence
            query_seq = self._read_fasta_sequence(fasta_path)
            qlen = len(query_seq)
            self.logger.debug(f"Query sequence length: {qlen}")
            
            # Read FoldSeek hits
            hits = self._parse_foldseek_results(foldseek_path)
            self.logger.debug(f"Found {len(hits)} FoldSeek hits")
            
            # Initialize residue count dictionary
            qres2count = {res: 0 for res in range(1, qlen + 1)}
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{prefix}.foldseek.flt.result")
            
            # Filter and write results
            filtered_hit_count = self._filter_and_write_hits(hits, qres2count, output_path)
            self.logger.info(f"Retained {filtered_hit_count} filtered hits")
            
            # Return success and output paths
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "status": "COMPLETED",
                "structure_id": structure_id,
                "output_files": {
                    "filtered_foldseek": output_path
                },
                "metrics": {
                    "duration_seconds": duration,
                    "total_hits": len(hits),
                    "filtered_hits": filtered_hit_count
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error filtering FoldSeek results for {structure_id}: {str(e)}")
            return {
                "status": "FAILED",
                "structure_id": structure_id,
                "error_message": str(e)
            }
    
    def _read_fasta_sequence(self, fasta_path):
        """Read sequence from FASTA file"""
        query_seq = ''
        with open(fasta_path, 'r') as fp:
            for line in fp:
                if line[0] != '>':
                    query_seq += line.strip()
        return query_seq
    
    def _parse_foldseek_results(self, foldseek_path):
        """Parse FoldSeek results file"""
        hits = []
        with open(foldseek_path, 'r') as fp:
            for line in fp:
                words = line.split()
                dnum = words[1].split('.')[0]
                qstart = int(words[6])
                qend = int(words[7])
                qresids = set(range(qstart, qend + 1))
                evalue = float(words[10])
                hits.append([dnum, evalue, qstart, qend, qresids])
        
        # Sort hits by E-value
        hits.sort(key=lambda x: x[1])
        return hits
    
    def _filter_and_write_hits(self, hits, qres2count, output_path):
        """Filter hits and write results"""
        filtered_hit_count = 0
        
        with open(output_path, 'w') as rp:
            rp.write('ecodnum\tevalue\trange\n')
            
            for hit in hits:
                dnum, evalue, qstart, qend, qresids = hit
                
                # Update residue counts
                for res in qresids:
                    qres2count[res] += 1
                
                # Count residues that haven't been seen too often
                good_res = sum(1 for res in qresids if qres2count[res] <= self.max_hit_count)
                
                # Output hits with sufficient good residues
                if good_res >= self.min_good_residues:
                    rp.write(f"{dnum}\t{evalue}\t{qstart}-{qend}\n")
                    filtered_hit_count += 1
        
        return filtered_hit_count