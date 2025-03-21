#!/usr/bin/env python3
"""
Gemmi-based utilities for handling protein structures in the DPAM pipeline.
"""

import os
import gzip
import logging
import tempfile
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from pathlib import Path

import gemmi
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class StructureHandler:
    """
    Handles protein structure operations using Gemmi library.
    Provides utilities for structure reading, validation, and manipulation.
    """
    
    def __init__(self, validation_level: str = 'standard'):
        """
        Initialize the structure handler.
        
        Args:
            validation_level: Level of structure validation ('minimal', 'standard', 'strict')
        """
        self.validation_level = validation_level
        self.three_to_one = self._init_residue_mapping()
    
    def _init_residue_mapping(self) -> Dict[str, str]:
        """
        Initialize standard amino acid mapping (3-letter to 1-letter codes).
        
        Returns:
            Dictionary mapping 3-letter codes to 1-letter codes
        """
        mapping = {}
        for code in gemmi.expand_protein_one_letter_codes():
            residue = gemmi.find_tabulated_residue(code)
            if residue.is_amino_acid():
                mapping[residue.name] = code
        
        # Add common non-standard residues
        mapping["MSE"] = "M"  # Selenomethionine
        
        return mapping
    
    def read_structure(self, 
                      file_path: str, 
                      structure_format: Optional[str] = None
                      ) -> gemmi.Structure:
        """
        Read a structure file (PDB or mmCIF).
        
        Args:
            file_path: Path to structure file
            structure_format: Format of the file ('mmcif', 'pdb', or None for auto-detection)
            
        Returns:
            Gemmi Structure object
        """
        # Determine file format if not specified
        if structure_format is None:
            if file_path.endswith(('.cif', '.mmcif', '.cif.gz', '.mmcif.gz')):
                structure_format = 'mmcif'
            elif file_path.endswith(('.pdb', '.ent', '.pdb.gz', '.ent.gz')):
                structure_format = 'pdb'
            else:
                raise ValueError(f"Cannot determine format for file: {file_path}")
        
        # Read the file
        try:
            # Handle gzipped files
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt') as f:
                    if structure_format == 'mmcif':
                        structure = gemmi.read_structure_from_string(f.read(), structure_format)
                    else:
                        structure = gemmi.read_pdb_string(f.read())
            else:
                if structure_format == 'mmcif':
                    structure = gemmi.cif.read(file_path).sole_block().find_mmcif_structure()
                else:
                    structure = gemmi.read_pdb(file_path)
            
            return structure
        
        except Exception as e:
            logger.error(f"Failed to read structure file {file_path}: {e}")
            raise
    
    def extract_sequence(self, 
                        structure: gemmi.Structure, 
                        chain_id: Optional[str] = None,
                        model_num: int = 0
                        ) -> Dict[str, str]:
        """
        Extract protein sequence from structure.
        
        Args:
            structure: Gemmi Structure object
            chain_id: Chain ID to extract (None for all chains)
            model_num: Model number to use
            
        Returns:
            Dictionary mapping chain IDs to sequences
        """
        sequences = {}
        
        # Check if model exists
        if model_num >= len(structure):
            model_num = 0
            logger.warning(f"Model {model_num} not found, using model 0 instead.")
        
        model = structure[model_num]
        
        # Process specified chains or all chains
        chains_to_process = [model[chain_id]] if chain_id else model
        
        for chain in chains_to_process:
            chain_id = chain.name
            sequence = ""
            
            for residue in chain:
                if not residue.is_water() and residue.is_amino_acid():
                    # Get one-letter code from standard mapping
                    code = self.three_to_one.get(residue.name, 'X')
                    
                    # Try to handle modified residues
                    if code == 'X' and hasattr(residue, 'parent_name'):
                        parent_code = self.three_to_one.get(residue.parent_name, 'X')
                        if parent_code != 'X':
                            code = parent_code
                    
                    sequence += code
            
            if sequence:
                sequences[chain_id] = sequence
        
        return sequences
    
    def validate_structure(self, 
                          structure: gemmi.Structure
                          ) -> Dict[str, Any]:
        """
        Validate a protein structure for quality issues.
        
        Args:
            structure: Gemmi Structure object
            
        Returns:
            Dictionary with validation metrics and issues
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        # Basic structure checks
        if len(structure) == 0:
            validation_results['valid'] = False
            validation_results['errors'].append("No models in structure")
            return validation_results
        
        model = structure[0]
        if len(model) == 0:
            validation_results['valid'] = False
            validation_results['errors'].append("No chains in model")
            return validation_results
        
        # Count amino acid residues
        aa_count = 0
        for chain in model:
            for residue in chain:
                if not residue.is_water() and residue.is_amino_acid():
                    aa_count += 1
        
        validation_results['metrics']['residue_count'] = aa_count
        
        if aa_count == 0:
            validation_results['valid'] = False
            validation_results['errors'].append("No amino acid residues found")
            return validation_results
        
        # Check for CA atoms in amino acids
        missing_ca = 0
        for chain in model:
            for residue in chain:
                if not residue.is_water() and residue.is_amino_acid():
                    if not residue.find_atom("CA", "*"):
                        missing_ca += 1
        
        if missing_ca > 0:
            validation_results['warnings'].append(f"Missing CA atoms in {missing_ca} residues")
            validation_results['metrics']['missing_ca_count'] = missing_ca
            
            if missing_ca / aa_count > 0.1 and self.validation_level == 'strict':
                validation_results['valid'] = False
                validation_results['errors'].append("More than 10% of residues missing CA atoms")
        
        # Check for alternate conformations
        alt_conf_residues = 0
        for chain in model:
            for residue in chain:
                if residue.has_altloc():
                    alt_conf_residues += 1
        
        if alt_conf_residues > 0:
            validation_results['warnings'].append(f"Found {alt_conf_residues} residues with alternate conformations")
            validation_results['metrics']['alt_conf_count'] = alt_conf_residues
        
        # Standard geometry checks if validation level is strict
        if self.validation_level == 'strict':
            # Bond length check (simplified)
            unusual_bonds = 0
            for chain in model:
                for residue in chain:
                    if not residue.is_water() and residue.is_amino_acid():
                        ca = residue.find_atom("CA", "*")
                        n = residue.find_atom("N", "*")
                        c = residue.find_atom("C", "*")
                        
                        if ca and n:
                            dist = ca.pos.dist(n.pos)
                            if dist < 1.3 or dist > 1.6:
                                unusual_bonds += 1
                        
                        if ca and c:
                            dist = ca.pos.dist(c.pos)
                            if dist < 1.3 or dist > 1.6:
                                unusual_bonds += 1
            
            if unusual_bonds > 0:
                validation_results['warnings'].append(f"Found {unusual_bonds} unusual bond lengths")
                validation_results['metrics']['unusual_bonds'] = unusual_bonds
                
                if unusual_bonds / aa_count > 0.1:
                    validation_results['valid'] = False
                    validation_results['errors'].append("More than 10% of residues have unusual bond geometries")
        
        return validation_results
    
    def standardize_structure(self, 
                             structure: gemmi.Structure,
                             select_best_altloc: bool = True,
                             remove_hydrogens: bool = True,
                             remove_waters: bool = True,
                             remove_ligands: bool = False
                             ) -> gemmi.Structure:
        """
        Standardize a structure for consistent processing.
        
        Args:
            structure: Gemmi Structure object
            select_best_altloc: Select best alternate conformation
            remove_hydrogens: Remove hydrogen atoms
            remove_waters: Remove water molecules
            remove_ligands: Remove ligands and non-standard residues
            
        Returns:
            Standardized structure
        """
        # Work on a copy
        std_structure = gemmi.Structure(structure)
        
        # Process only first model
        if len(std_structure) > 1:
            # Keep only first model
            model = std_structure[0]
            std_structure.models.clear()
            std_structure.models.push_back(model)
        
        model = std_structure[0]
        
        # Process chains
        for chain in model:
            # Process residues
            residues_to_remove = []
            
            for residue in chain:
                # Handle water
                if residue.is_water() and remove_waters:
                    residues_to_remove.append(residue)
                    continue
                
                # Handle non-amino acid residues
                if not residue.is_amino_acid() and not residue.is_water() and remove_ligands:
                    residues_to_remove.append(residue)
                    continue
                
                # Handle alternate conformations
                if select_best_altloc and residue.has_altloc():
                    self._select_best_altloc(residue)
                
                # Remove hydrogens
                if remove_hydrogens:
                    atoms_to_remove = []
                    for atom in residue:
                        if atom.element.is_hydrogen():
                            atoms_to_remove.append(atom)
                    
                    for atom in atoms_to_remove:
                        residue.erase(atom)
            
            # Remove marked residues
            for residue in residues_to_remove:
                chain.erase(residue)
        
        return std_structure
    
    def _select_best_altloc(self, residue: gemmi.Residue) -> None:
        """
        Select the best alternate conformation of a residue based on occupancy.
        Modifies the residue in place.
        
        Args:
            residue: Gemmi Residue object
        """
        # Group atoms by altloc
        altloc_groups = {}
        
        for atom in residue:
            if atom.has_altloc():
                alt = atom.altloc
                if alt not in altloc_groups:
                    altloc_groups[alt] = []
                altloc_groups[alt].append(atom)
        
        if not altloc_groups:
            return
        
        # Calculate average occupancy for each altloc
        avg_occupancies = {}
        for alt, atoms in altloc_groups.items():
            total_occ = sum(atom.occ for atom in atoms)
            avg_occupancies[alt] = total_occ / len(atoms)
        
        # Find best altloc
        best_alt = max(avg_occupancies.items(), key=lambda x: x[1])[0]
        
        # Remove non-best altlocs
        atoms_to_remove = []
        for atom in residue:
            if atom.has_altloc() and atom.altloc != best_alt:
                atoms_to_remove.append(atom)
            elif atom.has_altloc() and atom.altloc == best_alt:
                # Clear altloc flag from best conformation
                atom.altloc = ''
        
        for atom in atoms_to_remove:
            residue.erase(atom)
    
    def calculate_plddt(self, 
                       structure: gemmi.Structure,
                       model_num: int = 0
                       ) -> Dict[int, float]:
        """
        Extract pLDDT values from B-factors (for AlphaFold models).
        
        Args:
            structure: Gemmi Structure object
            model_num: Model number to use
            
        Returns:
            Dictionary mapping residue numbers to pLDDT values
        """
        plddt_values = {}
        
        if model_num >= len(structure):
            model_num = 0
        
        model = structure[model_num]
        
        for chain in model:
            for residue in chain:
                if not residue.is_water() and residue.is_amino_acid():
                    ca = residue.find_atom("CA", "*")
                    if ca:
                        # In AlphaFold models, B-factor contains pLDDT
                        plddt = ca.b_iso
                        res_num = residue.seqid.num
                        plddt_values[res_num] = plddt
        
        return plddt_values
    
    def extract_pae_matrix(self, 
                          json_file: str
                          ) -> Tuple[np.ndarray, int]:
        """
        Extract PAE (Predicted Aligned Error) matrix from AlphaFold JSON file.
        
        Args:
            json_file: Path to JSON file with PAE data
            
        Returns:
            Tuple of (PAE matrix as numpy array, sequence length)
        """
        import json
        
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            if 'predicted_aligned_error' in data:
                pae = np.array(data['predicted_aligned_error'])
                return pae, pae.shape[0]
            elif 'distance' in data:
                # Handle alternative format
                residue1 = data['residue1']
                residue2 = data['residue2']
                distances = data['distance']
                
                # Determine sequence length
                seq_len = max(max(residue1), max(residue2))
                
                # Create empty matrix
                pae = np.zeros((seq_len, seq_len))
                
                # Fill matrix with values
                for i, (res1, res2, dist) in enumerate(zip(residue1, residue2, distances)):
                    pae[res1-1, res2-1] = dist
                
                return pae, seq_len
            else:
                raise ValueError("No PAE data found in JSON file")
                
        except Exception as e:
            logger.error(f"Failed to read PAE data from {json_file}: {e}")
            raise
    
    def save_structure(self, 
                      structure: gemmi.Structure, 
                      output_path: str, 
                      format: str = 'pdb',
                      gzip_output: bool = True
                      ) -> str:
        """
        Save a structure to a file.
        
        Args:
            structure: Gemmi Structure object
            output_path: Path to output file
            format: Output format ('pdb' or 'mmcif')
            gzip_output: Whether to gzip the output
            
        Returns:
            Path to saved file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Prepare file path with extension
        if format == 'pdb' and not output_path.endswith(('.pdb', '.pdb.gz')):
            output_path = output_path + '.pdb'
        elif format == 'mmcif' and not output_path.endswith(('.cif', '.mmcif', '.cif.gz', '.mmcif.gz')):
            output_path = output_path + '.cif'
        
        if gzip_output and not output_path.endswith('.gz'):
            output_path = output_path + '.gz'
        
        try:
            if gzip_output:
                # Write to temporary file first
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                    temp_path = temp_file.name
                    
                    if format == 'pdb':
                        structure.write_pdb(temp_path)
                    else:  # mmcif
                        structure.write_mmcif(temp_path)
                
                # Gzip the file
                with open(temp_path, 'rb') as f_in:
                    with gzip.open(output_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                
                # Clean up temporary file
                os.unlink(temp_path)
            else:
                # Write directly
                if format == 'pdb':
                    structure.write_pdb(output_path)
                else:  # mmcif
                    structure.write_mmcif(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save structure to {output_path}: {e}")
            raise
    
    def superpose_structures(self, 
                            mobile: gemmi.Structure, 
                            reference: gemmi.Structure,
                            mobile_chain: Optional[str] = None,
                            reference_chain: Optional[str] = None
                            ) -> Tuple[gemmi.Structure, float]:
        """
        Superpose a mobile structure onto a reference structure.
        
        Args:
            mobile: Mobile Gemmi Structure
            reference: Reference Gemmi Structure
            mobile_chain: Chain ID in mobile structure (None for first chain)
            reference_chain: Chain ID in reference structure (None for first chain)
            
        Returns:
            Tuple of (superposed structure, RMSD)
        """
        # Extract chains
        if mobile_chain is None:
            mobile_chain = mobile[0][0].name
        
        if reference_chain is None:
            reference_chain = reference[0][0].name
        
        # Create superposition
        sup = gemmi.calculate_superposition(
            mobile[0][mobile_chain], 
            reference[0][reference_chain],
            gemmi.SupSelect.CA,  # Use CA atoms
            gemmi.SupSelect.CA
        )
        
        # Apply transformation
        transformed = gemmi.Structure(mobile)
        sup.apply(transformed[0])
        
        return transformed, sup.rmsd
    
    def extract_domains(self, 
                       structure: gemmi.Structure, 
                       domains: List[Dict[str, Any]]
                       ) -> Dict[str, gemmi.Structure]:
        """
        Extract domain structures from a full structure.
        
        Args:
            structure: Gemmi Structure object
            domains: List of domain specifications with 'id' and 'ranges' keys
                     Ranges should be in format "start-end,start-end,..."
            
        Returns:
            Dictionary mapping domain IDs to structures
        """
        domain_structures = {}
        
        for domain_spec in domains:
            domain_id = domain_spec['id']
            ranges_str = domain_spec.get('ranges', domain_spec.get('domain_range', ''))
            
            if not ranges_str:
                logger.warning(f"No range specified for domain {domain_id}")
                continue
            
            # Parse ranges
            residue_ids = set()
            for range_str in ranges_str.split(','):
                if '-' in range_str:
                    start, end = map(int, range_str.split('-'))
                    residue_ids.update(range(start, end + 1))
                else:
                    residue_ids.add(int(range_str))
            
            # Create domain structure
            domain_structure = gemmi.Structure()
            domain_structure.name = domain_id
            model = gemmi.Model(structure[0].name)
            
            # Copy chains with selected residues
            for chain in structure[0]:
                new_chain = gemmi.Chain(chain.name)
                
                for residue in chain:
                    if residue.is_amino_acid() and residue.seqid.num in residue_ids:
                        new_chain.add_residue(gemmi.Residue(residue))
                
                if len(new_chain) > 0:
                    model.add_chain(new_chain)
            
            if len(model) > 0:
                domain_structure.add_model(model)
                domain_structures[domain_id] = domain_structure
            else:
                logger.warning(f"No residues found for domain {domain_id}")
        
        return domain_structures

# Function to get residue contact map from structure
def get_contact_map(structure: gemmi.Structure, 
                   cutoff: float = 8.0,
                   ca_only: bool = True
                   ) -> np.ndarray:
    """
    Generate a contact map from a protein structure.
    
    Args:
        structure: Gemmi Structure object
        cutoff: Distance cutoff for contacts (Angstroms)
        ca_only: Use only CA atoms for distance calculation
        
    Returns:
        Binary contact map as numpy array
    """
    # Get first model and chain
    model = structure[0]
    chain = model[0]
    
    # Count residues
    residue_count = 0
    for residue in chain:
        if residue.is_amino_acid():
            residue_count += 1
    
    # Create empty contact map
    contact_map = np.zeros((residue_count, residue_count), dtype=np.int8)
    
    # Populate residue list
    residues = []
    for residue in chain:
        if residue.is_amino_acid():
            residues.append(residue)
    
    # Calculate contacts
    for i, res1 in enumerate(residues):
        for j, res2 in enumerate(residues[i:], i):
            if i == j:
                contact_map[i, j] = 1  # Diagonal
                continue
            
            # Calculate minimum distance between residues
            min_dist = float('inf')
            
            if ca_only:
                # CA-CA distance
                ca1 = res1.find_atom("CA", "*")
                ca2 = res2.find_atom("CA", "*")
                
                if ca1 and ca2:
                    min_dist = ca1.pos.dist(ca2.pos)
            else:
                # All-atom minimum distance
                for atom1 in res1:
                    for atom2 in res2:
                        dist = atom1.pos.dist(atom2.pos)
                        min_dist = min(min_dist, dist)
            
            # Set contact if distance is below cutoff
            if min_dist <= cutoff:
                contact_map[i, j] = 1
                contact_map[j, i] = 1  # Symmetric
    
    return contact_map

# Function to download structure from AlphaFold DB
def download_alphafold_structure(uniprot_id: str, 
                                output_dir: str,
                                version: str = 'v4'
                                ) -> Tuple[Optional[str], Optional[str]]:
    """
    Download structure and PAE data from AlphaFold DB.
    
    Args:
        uniprot_id: UniProt accession
        output_dir: Directory to save downloaded files
        version: AlphaFold DB version
        
    Returns:
        Tuple of (path to structure file, path to PAE file)
    """
    import requests
    from urllib.error import HTTPError
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    structure_path = os.path.join(output_dir, f"AF-{uniprot_id}-F1-model_{version}.cif.gz")
    pae_path = os.path.join(output_dir, f"AF-{uniprot_id}-F1-predicted_aligned_error_{version}.json")
    
    # Base URL
    base_url = "https://alphafold.ebi.ac.uk/files"
    
    # Download structure
    structure_url = f"{base_url}/AF-{uniprot_id}-F1-model_{version}.cif"
    
    try:
        response = requests.get(structure_url)
        response.raise_for_status()
        
        # Save to file (gzipped)
        with gzip.open(structure_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded structure for {uniprot_id} to {structure_path}")
    except (requests.RequestException, HTTPError) as e:
        logger.error(f"Failed to download structure for {uniprot_id}: {e}")
        structure_path = None
    
    # Download PAE
    pae_url = f"{base_url}/AF-{uniprot_id}-F1-predicted_aligned_error_{version}.json"
    
    try:
        response = requests.get(pae_url)
        response.raise_for_status()
        
        # Save to file
        with open(pae_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded PAE for {uniprot_id} to {pae_path}")
    except (requests.RequestException, HTTPError) as e:
        logger.error(f"Failed to download PAE for {uniprot_id}: {e}")
        pae_path = None
    
    return structure_path, pae_path

# Singleton instance
_structure_handler = None

def get_structure_handler() -> StructureHandler:
    """
    Get the structure handler singleton instance.
    
    Returns:
        StructureHandler instance
    """
    global _structure_handler
    if _structure_handler is None:
        _structure_handler = StructureHandler()
    return _structure_handler