## dpam/batch/preparation.py

import os
import gzip
import gemmi
import shutil
import hashlib
from concurrent.futures import ThreadPoolExecutor

class DPAMBatchPreparation:
    def __init__(self, db_config, batch_dir):
        self.db_config = db_config
        self.batch_dir = batch_dir
        self.afdb_base_url = "https://alphafold.ebi.ac.uk/files"
    
    def prepare_batch_directory(self, batch_id):
        """Prepare directories for a batch and assess file integrity"""
        # Connect to database
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            # Get batch information
            cursor.execute(
                "SELECT name FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            batch_name = cursor.fetchone()[0]
            
            # Create batch directory structure
            batch_path = os.path.join(self.batch_dir, f"batch_{batch_id}_{batch_name}")
            os.makedirs(batch_path, exist_ok=True)
            
            # Create subdirectories
            structures_dir = os.path.join(batch_path, "structures")
            pae_dir = os.path.join(batch_path, "pae")
            results_dir = os.path.join(batch_path, "results")
            
            os.makedirs(structures_dir, exist_ok=True)
            os.makedirs(pae_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            
            # Get all structures in the batch
            cursor.execute(
                """
                SELECT s.structure_id, s.pdb_id, p.uniprot_id
                FROM batch_items bi
                JOIN structures s ON bi.structure_id = s.structure_id
                JOIN proteins p ON s.protein_id = p.protein_id
                WHERE bi.batch_id = %s
                """,
                (batch_id,)
            )
            structures = cursor.fetchall()
            
            # Update batch path in database
            cursor.execute(
                "UPDATE batches SET parameters = jsonb_set(COALESCE(parameters, '{}'::jsonb), '{batch_path}', %s) "
                "WHERE batch_id = %s",
                (json.dumps(batch_path), batch_id)
            )
            
            # Update batch status
            cursor.execute(
                "UPDATE batches SET status = %s WHERE batch_id = %s",
                ("PREPARING", batch_id)
            )
            
            conn.commit()
            
            # Process structures in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for structure_id, pdb_id, uniprot_id in structures:
                    futures.append(
                        executor.submit(
                            self._process_structure,
                            structure_id,
                            pdb_id,
                            uniprot_id,
                            structures_dir
                        )
                    )
                
                # Wait for all downloads to complete
                results = [future.result() for future in futures]
            
            # Update batch status based on results
            success_count = sum(1 for result in results if result['success'])
            if success_count == 0:
                new_status = "FAILED"
            elif success_count < len(structures):
                new_status = "PARTIALLY_READY"
            else:
                new_status = "READY"
                
            cursor.execute(
                "UPDATE batches SET status = %s WHERE batch_id = %s",
                (new_status, batch_id)
            )
            
            # Update metrics
            metrics = {
                'total': len(structures),
                'success': success_count,
                'failed': len(structures) - success_count
            }
            
            cursor.execute(
                "UPDATE batches SET parameters = jsonb_set(COALESCE(parameters, '{}'::jsonb), '{metrics}', %s) "
                "WHERE batch_id = %s",
                (json.dumps(metrics), batch_id)
            )
            
            conn.commit()
            
            return {
                'batch_path': batch_path,
                'metrics': metrics,
                'status': new_status
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def _process_structure(self, structure_id, pdb_id, uniprot_id, structures_dir):
        """Download and validate AlphaFold structure files"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            # Define file paths
            cif_url = f"{self.afdb_base_url}/AF-{uniprot_id}-F1-model_v4.cif"
            local_cif_path = os.path.join(structures_dir, f"{pdb_id}.cif.gz")
            
            # Download file
            success = self._download_and_validate_cif(cif_url, local_cif_path)
            
            if success:
                # Extract sequence from structure
                sequence = self._extract_sequence_from_cif(local_cif_path)
                
                # Update structure record
                cursor.execute(
                    """
                    UPDATE structures 
                    SET structure_path = %s, processing_status = %s, 
                        resolution = %s, processing_date = NOW() 
                    WHERE structure_id = %s
                    """,
                    (local_cif_path, "READY", 0.0, structure_id)
                )
                
                # Update batch item status
                cursor.execute(
                    "UPDATE batch_items SET status = %s WHERE structure_id = %s",
                    ("READY", structure_id)
                )
            else:
                # Update failure in database
                cursor.execute(
                    "UPDATE structures SET processing_status = %s WHERE structure_id = %s",
                    ("FAILED", structure_id)
                )
                
                cursor.execute(
                    "UPDATE batch_items SET status = %s, error_message = %s WHERE structure_id = %s",
                    ("FAILED", "Failed to download or validate structure file", structure_id)
                )
            
            conn.commit()
            
            return {
                'structure_id': structure_id,
                'pdb_id': pdb_id,
                'success': success
            }
            
        except Exception as e:
            conn.rollback()
            # Log error
            try:
                cursor.execute(
                    "UPDATE batch_items SET status = %s, error_message = %s WHERE structure_id = %s",
                    ("FAILED", str(e), structure_id)
                )
                conn.commit()
            except:
                pass
                
            return {
                'structure_id': structure_id,
                'pdb_id': pdb_id,
                'success': False,
                'error': str(e)
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def _download_and_validate_cif(self, url, local_path):
        """Download and validate CIF file"""
        try:
            # Download file
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                return False
                
            # Save to temporary file
            temp_path = local_path + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validate CIF file
            try:
                # Check if we can read it with Gemmi
                with gzip.open(temp_path, 'rt') as f:
                    structure = gemmi.read_structure(f)
                    
                # Basic validation
                if len(structure) == 0 or len(structure[0]) == 0:
                    raise ValueError("Empty structure")
                    
                # If validation passes, move to final location
                shutil.move(temp_path, local_path)
                return True
                
            except Exception as e:
                # Remove temp file if validation fails
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                print(f"CIF validation error: {str(e)}")
                return False
                
        except Exception as e:
            print(f"Download error: {str(e)}")
            return False
    
    def _extract_sequence_from_cif(self, cif_path):
        """Extract sequence from CIF file using Gemmi"""
        try:
            with gzip.open(cif_path, 'rt') as f:
                structure = gemmi.read_structure(f)
                
            # Get the first model and chain
            model = structure[0]
            chain = model[0]
            
            sequence = ""
            for residue in chain:
                if residue.is_amino_acid():
                    code = gemmi.find_tabulated_residue(residue.name).one_letter_code
                    sequence += code if code != '?' else 'X'
                    
            return sequence
            
        except Exception as e:
            print(f"Error extracting sequence: {str(e)}")
            return ""