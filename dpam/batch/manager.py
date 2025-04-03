## dpam/batch/manager.py

import psycopg2
import requests
import json
from datetime import datetime

class DPAMBatchManager:
    def __init__(self, db_config, api_config):
        self.db_config = db_config
        self.api_config = api_config
    
    def create_batch_from_accessions(self, accessions, batch_name=None, description=None):
        """Create a new batch from a list of UniProt accessions"""
        if batch_name is None:
            batch_name = f"UniProt_Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Connect to database
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            # Create batch record
            cursor.execute(
                "INSERT INTO batches (name, description, status, created_at) "
                "VALUES (%s, %s, %s, NOW()) RETURNING batch_id",
                (batch_name, description, "INITIALIZED")
            )
            batch_id = cursor.fetchone()[0]
            
            # Process each accession
            for accession in accessions:
                # Check if protein already exists
                cursor.execute(
                    "SELECT protein_id FROM proteins WHERE uniprot_id = %s",
                    (accession,)
                )
                result = cursor.fetchone()
                
                if result:
                    protein_id = result[0]
                else:
                    # Fetch basic protein info from UniProt
                    protein_info = self._fetch_uniprot_info(accession)
                    
                    # Create protein record
                    cursor.execute(
                        "INSERT INTO proteins (uniprot_id, name, description, sequence, length) "
                        "VALUES (%s, %s, %s, %s, %s) RETURNING protein_id",
                        (
                            accession, 
                            protein_info.get('name', accession),
                            protein_info.get('description', ''),
                            protein_info.get('sequence', ''),
                            len(protein_info.get('sequence', ''))
                        )
                    )
                    protein_id = cursor.fetchone()[0]
                
                # Map AlphaFold DB IDs
                afdb_id = f"AF-{accession}-F1"
                
                # Create structure record (will be populated later)
                cursor.execute(
                    "INSERT INTO structures (protein_id, pdb_id, format, processing_status) "
                    "VALUES (%s, %s, %s, %s) RETURNING structure_id",
                    (protein_id, afdb_id, "mmCIF", "PENDING")
                )
                structure_id = cursor.fetchone()[0]
                
                # Add to batch items
                cursor.execute(
                    "INSERT INTO batch_items (batch_id, structure_id, status) "
                    "VALUES (%s, %s, %s)",
                    (batch_id, structure_id, "PENDING")
                )
            
            # Update batch status
            cursor.execute(
                "UPDATE batches SET status = %s WHERE batch_id = %s",
                ("READY_FOR_DOWNLOAD", batch_id)
            )
            
            conn.commit()
            return batch_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
    
    def _fetch_uniprot_info(self, accession):
        """Fetch basic protein information from UniProt API"""
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'name': data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', accession),
                    'description': data.get('comments', [{}])[0].get('text', [{}])[0].get('value', ''),
                    'sequence': data.get('sequence', {}).get('value', '')
                }
            else:
                return {}
                
        except Exception as e:
            print(f"Error fetching UniProt info for {accession}: {str(e)}")
            return {}