## dpam/batch/supplement.py

class DPAMBatchSupplementation:
    def __init__(self, db_config):
        self.db_config = db_config
        self.afdb_base_url = "https://alphafold.ebi.ac.uk/files"
    
    def fetch_pae_files(self, batch_id):
        """Fetch PAE (Predicted Aligned Error) files for a batch"""
        # Connect to database
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            # Get batch path
            cursor.execute(
                "SELECT parameters->>'batch_path' FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            batch_path = cursor.fetchone()[0]
            
            # Get all ready structures in the batch
            cursor.execute(
                """
                SELECT s.structure_id, s.pdb_id, p.uniprot_id
                FROM batch_items bi
                JOIN structures s ON bi.structure_id = s.structure_id
                JOIN proteins p ON s.protein_id = p.protein_id
                WHERE bi.batch_id = %s AND bi.status = 'READY'
                """,
                (batch_id,)
            )
            structures = cursor.fetchall()
            
            # Set batch to PAE downloading status
            cursor.execute(
                "UPDATE batches SET status = %s WHERE batch_id = %s",
                ("DOWNLOADING_PAE", batch_id)
            )
            conn.commit()
            
            # Define PAE directory
            pae_dir = os.path.join(batch_path, "pae")
            os.makedirs(pae_dir, exist_ok=True)
            
            # Process PAE files in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for structure_id, pdb_id, uniprot_id in structures:
                    futures.append(
                        executor.submit(
                            self._fetch_pae_file,
                            structure_id,
                            pdb_id,
                            uniprot_id,
                            pae_dir
                        )
                    )
                
                # Wait for all downloads to complete
                results = [future.result() for future in futures]
            
            # Update batch status
            success_count = sum(1 for result in results if result['success'])
            if success_count == len(structures):
                new_status = "READY_FOR_PROCESSING"
            else:
                new_status = "READY_WITH_PARTIAL_PAE"
                
            cursor.execute(
                "UPDATE batches SET status = %s WHERE batch_id = %s",
                (new_status, batch_id)
            )
            
            # Update metrics
            metrics = {
                'total_pae': len(structures),
                'success_pae': success_count,
                'failed_pae': len(structures) - success_count
            }
            
            cursor.execute(
                "UPDATE batches SET parameters = jsonb_set(COALESCE(parameters, '{}'::jsonb), '{pae_metrics}', %s) "
                "WHERE batch_id = %s",
                (json.dumps(metrics), batch_id)
            )
            
            conn.commit()
            
            return {
                'metrics': metrics,
                'status': new_status
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def _fetch_pae_file(self, structure_id, pdb_id, uniprot_id, pae_dir):
        """Download PAE file for a structure"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            # Define file paths
            pae_url = f"{self.afdb_base_url}/AF-{uniprot_id}-F1-predicted_aligned_error_v4.json"
            local_pae_path = os.path.join(pae_dir, f"{pdb_id}_pae.json")
            
            # Download file
            success = False
            try:
                response = requests.get(pae_url)
                if response.status_code == 200:
                    # Save PAE file
                    with open(local_pae_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Validate JSON
                    with open(local_pae_path, 'r') as f:
                        pae_data = json.load(f)
                    
                    # Basic validation that it contains PAE data
                    if 'predicted_aligned_error' in pae_data:
                        success = True
                    else:
                        raise ValueError("Missing PAE data in JSON")
            except Exception as e:
                print(f"Error downloading PAE for {pdb_id}: {str(e)}")
                if os.path.exists(local_pae_path):
                    os.remove(local_pae_path)
                success = False
            
            # Update structure record
            if success:
                cursor.execute(
                    "UPDATE structures SET parameters = jsonb_set(COALESCE(parameters, '{}'::jsonb), '{pae_path}', %s) "
                    "WHERE structure_id = %s",
                    (json.dumps(local_pae_path), structure_id)
                )
                
                # Add log entry
                cursor.execute(
                    "INSERT INTO step_logs (batch_item_id, step_id, started_at, completed_at, status, output) "
                    "SELECT bi.batch_item_id, ps.step_id, NOW(), NOW(), 'COMPLETED', 'PAE file downloaded successfully' "
                    "FROM batch_items bi, pipeline_steps ps "
                    "WHERE bi.structure_id = %s AND ps.name = 'download_pae'",
                    (structure_id,)
                )
            else:
                # Log failure
                cursor.execute(
                    "INSERT INTO step_logs (batch_item_id, step_id, started_at, completed_at, status, output) "
                    "SELECT bi.batch_item_id, ps.step_id, NOW(), NOW(), 'FAILED', 'Failed to download PAE file' "
                    "FROM batch_items bi, pipeline_steps ps "
                    "WHERE bi.structure_id = %s AND ps.name = 'download_pae'",
                    (structure_id,)
                )
            
            conn.commit()
            
            return {
                'structure_id': structure_id,
                'pdb_id': pdb_id,
                'success': success,
                'pae_path': local_pae_path if success else None
            }
            
        except Exception as e:
            conn.rollback()
            print(f"Error processing PAE for {pdb_id}: {str(e)}")
            return {
                'structure_id': structure_id,
                'pdb_id': pdb_id,
                'success': False,
                'error': str(e)
            }
            
        finally:
            cursor.close()
            conn.close()