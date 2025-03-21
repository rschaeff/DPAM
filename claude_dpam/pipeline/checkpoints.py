## dpam/pipeline/checkpoints.py

class DPAMBatchCheckpoints:
    """Manages checkpoints and recovery for DPAM pipeline"""
    
    def __init__(self, db_config):
        self.db_config = db_config
    
    def get_db_connection(self):
        """Get a connection to the PostgreSQL database"""
        return psycopg2.connect(**self.db_config)
    
    def set_checkpoint(self, batch_id, step_name, status="COMPLETED"):
        """Set a checkpoint for a batch at a specific step"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Record checkpoint
            cursor.execute(
                """
                INSERT INTO batch_checkpoints (batch_id, step_name, status, created_at)
                VALUES (%s, %s, %s, NOW())
                """,
                (batch_id, step_name, status)
            )
            
            # Update batch status
            cursor.execute(
                "UPDATE batches SET status = %s WHERE batch_id = %s",
                (f"{status}_{step_name.upper()}", batch_id)
            )
            
            conn.commit()
            
        finally:
            cursor.close()
            conn.close()
    
    def get_last_checkpoint(self, batch_id):
        """Get the last successful checkpoint for a batch"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                SELECT step_name, status, created_at
                FROM batch_checkpoints
                WHERE batch_id = %s AND status = 'COMPLETED'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (batch_id,)
            )
            
            result = cursor.fetchone()
            if result:
                return {
                    'step_name': result[0],
                    'status': result[1],
                    'created_at': result[2]
                }
            return None
            
        finally:
            cursor.close()
            conn.close()
    
    def can_resume_from_step(self, batch_id, step_name):
        """Check if a batch can resume from a specific step"""
        # Define step dependencies
        step_dependencies = {
            'run_hhsearch': [],
            'run_foldseek': [],
            'filter_foldseek': ['run_foldseek'],
            'map_to_ecod': ['run_hhsearch'],
            'get_dali_candidates': ['map_to_ecod', 'filter_foldseek'],
            'run_iterative_dali': ['get_dali_candidates'],
            'analyze_dali': ['run_iterative_dali'],
            'get_support': ['analyze_dali', 'map_to_ecod'],
            'get_good_domains': ['get_support'],
            'get_sse': ['get_good_domains'],
            'get_diso': ['get_sse'],
            'parse_domains': ['get_diso']
        }
        
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Check dependencies
            dependencies = step_dependencies.get(step_name, [])
            for dependency in dependencies:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM batch_checkpoints
                    WHERE batch_id = %s AND step_name = %s AND status = 'COMPLETED'
                    """,
                    (batch_id, dependency)
                )
                
                count = cursor.fetchone()[0]
                if count == 0:
                    return False, f"Missing dependency: {dependency}"
            
            return True, "All dependencies satisfied"
            
        finally:
            cursor.close()
            conn.close()
    
    def recover_batch(self, batch_id):
        """Recover a batch from the last checkpoint"""
        last_checkpoint = self.get_last_checkpoint(batch_id)
        if not last_checkpoint:
            return False, "No checkpoint found"
            
        # Define step order
        step_order = [
            'run_hhsearch',
            'run_foldseek',
            'filter_foldseek',
            'map_to_ecod',
            'get_dali_candidates',
            'run_iterative_dali',
            'analyze_dali',
            'get_support',
            'get_good_domains',
            'get_sse',
            'get_diso',
            'parse_domains'
        ]
        
        # Find next step
        last_step = last_checkpoint['step_name']
        try:
            next_step_index = step_order.index(last_step) + 1
            if next_step_index < len(step_order):
                next_step = step_order[next_step_index]
                can_resume, message = self.can_resume_from_step(batch_id, next_step)
                
                if can_resume:
                    return True, next_step
                else:
                    return False, message
            else:
                return False, "Batch already completed all steps"
                
        except ValueError:
            return False, f"Unknown step: {last_step}"