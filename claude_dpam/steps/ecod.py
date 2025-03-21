## dpam/steps/ecod.py

import os
import logging
from datetime import datetime

class ECODMapper:
    """Maps HHSearch results to ECOD domains"""
    
    def __init__(self, config):
        """
        Initialize ECOD mapper with configuration
        
        Args:
            config (dict): Configuration containing paths and parameters
        """
        self.config = config
        self.logger = logging.getLogger("dpam.steps.ecod")
        self.data_dir = config.get('data_dir', '/data')
        self.min_domain_residues = config.get('ecod_min_domain_residues', 10)
    
    def run(self, structure_id, hhsearch_path, output_dir):
        """
        Map HHSearch results to ECOD domains
        
        Args:
            structure_id (str): Structure identifier
            hhsearch_path (str): Path to HHSearch results file
            output_dir (str): Directory for output files
            
        Returns:
            dict: Results including paths to output files and status
        """
        start_time = datetime.now()
        self.logger.info(f"Mapping HHSearch results to ECOD for structure {structure_id}")
        
        prefix = f"struct_{structure_id}"
        
        try:
            # Parse HHSearch results
            hits, need_pdbchains, need_pdbs = self._parse_hhsearch_results(hhsearch_path)
            self.logger.debug(f"Found {len(hits)} HHSearch hits")
            
            # Load ECOD mappings
            pdb2ecod, good_hids = self._load_ecod_pdb_mappings(need_pdbchains, need_pdbs)
            self.logger.debug(f"Loaded ECOD mappings for {len(good_hids)} PDB chains")
            
            # Load ECOD domain information
            ecod2key, ecod2len = self._load_ecod_domain_info()
            self.logger.debug(f"Loaded information for {len(ecod2key)} ECOD domains")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{prefix}.map2ecod.result")
            
            # Map and write results
            mapping_count = self._map_and_write_results(
                hits, good_hids, pdb2ecod, ecod2key, ecod2len, output_path
            )
            self.logger.info(f"Mapped {mapping_count} HHSearch hits to ECOD domains")
            
            # Return success and output paths
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "status": "COMPLETED",
                "structure_id": structure_id,
                "output_files": {
                    "ecod_mapping": output_path
                },
                "metrics": {
                    "duration_seconds": duration,
                    "total_hits": len(hits),
                    "mapped_hits": mapping_count
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error mapping to ECOD for {structure_id}: {str(e)}")
            return {
                "status": "FAILED",
                "structure_id": structure_id,
                "error_message": str(e)
            }
    
    def _parse_hhsearch_results(self, hhsearch_path):
        """Parse HHSearch results file"""
        with open(hhsearch_path, 'r') as fp:
            info = fp.read().split('\n>')
        
        allhits = []
        need_pdbchains = set()
        need_pdbs = set()
        
        for hit in info[1:]:
            lines = hit.split('\n')
            qstart = 0
            qend = 0
            qseq = ''
            hstart = 0
            hend = 0
            hseq = ''
            hid = None
            hh_prob = None
            hh_eval = None
            hh_score = None
            aligned_cols = None
            idents = None
            similarities = None
            sum_probs = None
            
            for line in lines:
                if len(line) >= 6:
                    if line[:6] == 'Probab':
                        words = line.split()
                        for word in words:
                            subwords = word.split('=')
                            if subwords[0] == 'Probab':
                                hh_prob = subwords[1]
                            elif subwords[0] == 'E-value':
                                hh_eval = subwords[1]
                            elif subwords[0] == 'Score':
                                hh_score = subwords[1]
                            elif subwords[0] == 'Aligned_cols':
                                aligned_cols = subwords[1]
                            elif subwords[0] == 'Identities':
                                idents = subwords[1]
                            elif subwords[0] == 'Similarity':
                                similarities = subwords[1]
                            elif subwords[0] == 'Sum_probs':
                                sum_probs = subwords[1]

                    elif line[:2] == 'Q ':
                        words = line.split()
                        if words[1] != 'ss_pred' and words[1] != 'Consensus':
                            qseq += words[3]
                            if not qstart:
                                qstart = int(words[2])
                            qend = int(words[4])

                    elif line[:2] == 'T ':
                        words = line.split()
                        if words[1] != 'Consensus' and words[1] != 'ss_dssp' and words[1] != 'ss_pred':
                            hid = words[1]
                            hseq += words[3]
                            if not hstart:
                                hstart = int(words[2])
                            hend = int(words[4])
            
            if hid:
                allhits.append([hid, hh_prob, hh_eval, hh_score, aligned_cols, idents, 
                               similarities, sum_probs, qstart, qend, qseq, hstart, hend, hseq])
                need_pdbchains.add(hid)
                need_pdbs.add(hid.split('_')[0].lower())
        
        return allhits, need_pdbchains, need_pdbs
    
    def _load_ecod_pdb_mappings(self, need_pdbchains, need_pdbs):
        """Load ECOD to PDB mappings"""
        pdb2ecod = {}
        good_hids = set()
        
        ecod_pdbmap_path = os.path.join(self.data_dir, 'ECOD_pdbmap')
        with open(ecod_pdbmap_path, 'r') as fp:
            for line in fp:
                words = line.split()
                pdbid = words[1]
                
                # Skip if we don't need this PDB
                if pdbid.lower() not in need_pdbs:
                    continue
                    
                segments = words[2].split(',')
                chainids = set()
                resids = []
                
                for segment in segments:
                    chainids.add(segment.split(':')[0])
                    if '-' in segment:
                        start = int(segment.split(':')[1].split('-')[0])
                        end = int(segment.split(':')[1].split('-')[1])
                        for res in range(start, end + 1):
                            resids.append(res)
                    else:
                        resid = int(segment.split(':')[1])
                        resids.append(resid)
                
                if len(chainids) == 1:
                    chainid = list(chainids)[0]
                    pdbchain = pdbid.upper() + '_' + chainid
                    
                    if pdbchain in need_pdbchains:
                        good_hids.add(pdbchain)
                        pdb2ecod[pdbchain] = {}
                        
                        for i, resid in enumerate(resids):
                            pdb2ecod[pdbchain][resid] = words[0] + ':' + str(i + 1)
        
        return pdb2ecod, good_hids
    
    def _load_ecod_domain_info(self):
        """Load ECOD domain information"""
        ecod2key = {}
        ecod2len = {}
        
        ecod_length_path = os.path.join(self.data_dir, 'ECOD_length')
        with open(ecod_length_path, 'r') as fp:
            for line in fp:
                words = line.split()
                ecod2key[words[0]] = words[1]
                ecod2len[words[0]] = int(words[2])
        
        return ecod2key, ecod2len
    
    def _get_range(self, resids, chainid):
        """Convert residue list to range string"""
        resids = sorted(resids)
        segs = []
        
        for resid in resids:
            if not segs:
                segs.append([resid])
            else:
                if resid > segs[-1][-1] + 1:
                    segs.append([resid])
                else:
                    segs[-1].append(resid)
        
        ranges = []
        for seg in segs:
            if chainid:
                ranges.append(f"{chainid}:{seg[0]}-{seg[-1]}")
            else:
                ranges.append(f"{seg[0]}-{seg[-1]}")
        
        return ','.join(ranges)
    
    def _map_and_write_results(self, hits, good_hids, pdb2ecod, ecod2key, ecod2len, output_path):
        """Map hits to ECOD domains and write results"""
        mapping_count = 0
        
        with open(output_path, 'w') as rp:
            rp.write('uid\tecod_domain_id\thh_prob\thh_eval\thh_score\taligned_cols\tidents\t'
                    'similarities\tsum_probs\tcoverage\tungapped_coverage\tquery_range\t'
                    'template_range\ttemplate_seqid_range\n')
            
            for hit in hits:
                hid = hit[0]
                
                if hid not in good_hids:
                    continue
                    
                pdbid = hid.split('_')[0]
                chainid = hid.split('_')[1]
                
                # Get ECOD domains for this PDB chain
                ecods = []
                ecod2hres = {}
                ecod2hresmap = {}
                
                for pdbres in pdb2ecod[hid].keys():
                    for item in pdb2ecod[hid][pdbres].split(','):
                        ecod = item.split(':')[0]
                        ecodres = int(item.split(':')[1])
                        
                        if ecod not in ecod2hres:
                            ecods.append(ecod)
                            ecod2hres[ecod] = set()
                            ecod2hresmap[ecod] = {}
                            
                        ecod2hres[ecod].add(pdbres)
                        ecod2hresmap[ecod][pdbres] = ecodres
                
                # Extract hit information
                hh_prob = hit[1]
                hh_eval = hit[2]
                hh_score = hit[3]
                aligned_cols = hit[4]
                idents = hit[5]
                similarities = hit[6]
                sum_probs = hit[7]
                qstart = hit[8]
                qseq = hit[10]
                hstart = hit[11]
                hseq = hit[13]
                
                # Process each ECOD domain
                for ecod in ecods:
                    if ecod not in ecod2key or ecod not in ecod2len:
                        continue
                        
                    ecodkey = ecod2key[ecod]
                    ecodlen = ecod2len[ecod]
                    
                    qposi = qstart - 1
                    hposi = hstart - 1
                    qresids = []
                    hresids = []
                    eresids = []
                    
                    # Align sequences and map residues
                    if len(qseq) == len(hseq):
                        for i in range(len(hseq)):
                            if qseq[i] != '-':
                                qposi += 1
                            if hseq[i] != '-':
                                hposi += 1
                            if qseq[i] != '-' and hseq[i] != '-':
                                if hposi in ecod2hres[ecod]:
                                    eposi = ecod2hresmap[ecod][hposi]
                                    qresids.append(qposi)
                                    hresids.append(hposi)
                                    eresids.append(eposi)
                        
                        # Write mapping if enough residues map
                        if (len(qresids) >= self.min_domain_residues and 
                            len(eresids) >= self.min_domain_residues):
                            
                            qrange = self._get_range(qresids, '')
                            hrange = self._get_range(hresids, chainid)
                            erange = self._get_range(eresids, '')
                            
                            coverage = round(len(eresids) / ecodlen, 3)
                            ungapped_coverage = round((max(eresids) - min(eresids) + 1) / ecodlen, 3)
                            
                            rp.write(f"{ecod}\t{ecodkey}\t{hh_prob}\t{hh_eval}\t{hh_score}\t"
                                    f"{aligned_cols}\t{idents}\t{similarities}\t{sum_probs}\t"
                                    f"{coverage}\t{ungapped_coverage}\t{qrange}\t{erange}\t{hrange}\n")
                            
                            mapping_count += 1