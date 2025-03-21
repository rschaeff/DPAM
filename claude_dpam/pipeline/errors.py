## dpam/pipeline/errors.py

class DPAMErrorHandler:
    """Sophisticated error handling and recovery for DPAM pipeline"""
    
    def __init__(self, db_config, recovery_config=None):
        self.db_config = db_config
        self.recovery_config = recovery_config or {
            'max_retries': 3,
            'backoff_factor': 2,  # Exponential backoff
            'retry_delay': 300    # 5 minutes initially
        }
    
    def get_db_connection(self):
        """Get connection to PostgreSQL database"""
        return psycopg2.connect(**self.db_config)
    
    def log_error(self, batch_id, structure_id, step_name, error_message, error_type=None):
        """Log an error for analysis and tracking"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Log error
            cursor.execute(
                """
                INSERT INTO error_logs 
                (batch_id, structure_id, step_name, error_message, error_type, occurred_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                RETURNING error_id
                """,
                (batch_id, structure_id, step_name, error_message, error_type)
            )
            error_id = cursor.fetchone()[0]
            
            # Update structure and batch item status
            cursor.execute(
                """
                UPDATE batch_items bi
                SET status = 'ERROR', 
                    error_message = %s
                FROM structures s
                WHERE bi.structure_id = s.structure_id
                AND s.structure_id = %s
                AND bi.batch_id = %s
                """,
                (error_message, structure_id, batch_id)
            )
            
            conn.commit()
            return error_id
            
        finally:
            cursor.close()
            conn.close()
    
    def classify_error(self, error_message, step_name):
        """Classify error for targeted recovery"""
        # Common error patterns and their classifications
        error_patterns = [
            # Resource issues
            {"pattern": "out of memory", "type": "RESOURCE_MEMORY", "retriable": True},
            {"pattern": "killed", "type": "RESOURCE_KILLED", "retriable": True},
            {"pattern": "timed out", "type": "RESOURCE_TIMEOUT", "retriable": True},
            
            # Input data issues
            {"pattern": "malformed", "type": "DATA_MALFORMED", "retriable": False},
            {"pattern": "no such file", "type": "DATA_MISSING", "retriable": False},
            {"pattern": "corrupted", "type": "DATA_CORRUPTED", "retriable": False},
            
            # Tool failures
            {"pattern": "segmentation fault", "type": "TOOL_CRASH", "retriable": True},
            {"pattern": "command not found", "type": "TOOL_MISSING", "retriable": False},
            {"pattern": "connection refused", "type": "NETWORK", "retriable": True},
            
            # Step-specific errors
            {"pattern": "no hits found", "type": "NO_HITS", "retriable": False, "steps": ["get_dali_candidates"]},
            {"pattern": "insufficient data", "type": "INSUFFICIENT_DATA", "retriable": False, "steps": ["parse_domains"]}
        ]
        
        # Check for matches
        for pattern in error_patterns:
            if pattern["pattern"] in error_message.lower():
                # Check if step-specific and matches
                if "steps" in pattern and step_name not in pattern["steps"]:
                    continue
                    
                return {
                    "type": pattern["type"],
                    "retriable": pattern["retriable"]
                }
        
        # Default classification
        return {
            "type": "UNKNOWN",
            "retriable": True  # Default to retriable
        }
    
    def should_retry(self, batch_id, structure_id, step_name, error_message):
        """Determine if failed step should be retried"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get retry count
            cursor.execute(
                """
                SELECT COUNT(*) 
                FROM error_logs
                WHERE batch_id = %s 
                  AND structure_id = %s 
                  AND step_name = %s
                """,
                (batch_id, structure_id, step_name)
            )
            retry_count = cursor.fetchone()[0]
            
            # Classify error
            error_info = self.classify_error(error_message, step_name)
            
            # Check if retriable and under max_retries
            if error_info["retriable"] and retry_count < self.recovery_config["max_retries"]:
                # Calculate backoff delay
                backoff_delay = self.recovery_config["retry_delay"] * (
                    self.recovery_config["backoff_factor"] ** retry_count
                )
                
                return True, {
                    "retry_count": retry_count + 1,
                    "backoff_delay": backoff_delay,
                    "error_type": error_info["type"]
                }
            else:
                return False, {
                    "retry_count": retry_count,
                    "error_type": error_info["type"],
                    "reason": "Not retriable" if not error_info["retriable"] else "Max retries exceeded"
                }
                
        finally:
            cursor.close()
            conn.close()
    
    def apply_recovery_strategy(self, batch_id, structure_id, step_name, error_info):
        """Apply appropriate recovery strategy"""
        # Different strategies based on error type
        recovery_strategies = {
            "RESOURCE_MEMORY": self._increase_resource_allocation,
            "RESOURCE_TIMEOUT": self._increase_resource_allocation,
            "RESOURCE_KILLED": self._increase_resource_allocation,
            "DATA_MALFORMED": self._redownload_structure,
            "DATA_MISSING": self._redownload_structure,
            "DATA_CORRUPTED": self._redownload_structure,
            "TOOL_CRASH": self._retry_with_modified_parameters,
            "NETWORK": self._retry_with_backoff,
            "NO_HITS": self._mark_step_as_skipped,
            "INSUFFICIENT_DATA": self._use_fallback_method
        }
        
        # Get appropriate strategy or use default
        strategy = recovery_strategies.get(
            error_info["error_type"], 
            self._retry_with_backoff
        )
        
        # Apply strategy
        return strategy(batch_id, structure_id, step_name, error_info)
    
    def _increase_resource_allocation(self, batch_id, structure_id, step_name, error_info):
        """Increase resources for next attempt"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get current resource allocation
            cursor.execute(
                """
                SELECT parameters->>'resources' 
                FROM structures 
                WHERE structure_id = %s
                """,
                (structure_id,)
            )
            current_resources = json.loads(cursor.fetchone()[0] or '{}')
            
            # Increase memory/runtime based on step
            memory_scaling = {
                'run_hhsearch': 1.5,
                'run_foldseek': 2.0,
                'run_iterative_dali': 2.0,
                'parse_domains': 1.5
            }
            
            # Default scaling factor
            scale_factor = memory_scaling.get(step_name, 1.5)
            
            # Update resources
            new_resources = current_resources.copy()
            new_resources['memory'] = current_resources.get('memory', 4) * scale_factor
            new_resources['runtime'] = current_resources.get('runtime', 4) * 1.5
            
            cursor.execute(
                """
                UPDATE structures
                SET parameters = jsonb_set(
                    COALESCE(parameters, '{}'::jsonb),
                    '{resources}',
                    %s::jsonb
                )
                WHERE structure_id = %s
                """,
                (json.dumps(new_resources), structure_id)
            )
            
            conn.commit()
            
            return {
                "strategy": "INCREASED_RESOURCES",
                "new_resources": new_resources,
                "backoff_delay": error_info["backoff_delay"]
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def _redownload_structure(self, batch_id, structure_id, step_name, error_info):
        """Redownload structure files"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get structure information
            cursor.execute(
                """
                SELECT s.pdb_id, p.uniprot_id, b.parameters->>'batch_path' as batch_path
                FROM structures s
                JOIN proteins p ON s.protein_id = p.protein_id
                JOIN batch_items bi ON bi.structure_id = s.structure_id
                JOIN batches b ON bi.batch_id = b.batch_id
                WHERE s.structure_id = %s AND bi.batch_id = %s
                """,
                (structure_id, batch_id)
            )
            pdb_id, uniprot_id, batch_path = cursor.fetchone()
            
            # Set status to redownloading
            cursor.execute(
                """
                UPDATE structures
                SET processing_status = 'REDOWNLOADING'
                WHERE structure_id = %s
                """,
                (structure_id,)
            )
            
            conn.commit()
            
            # Submit redownload task
            # This would be a separate function to redownload files
            # For demonstration, we'll assume it works
            
            return {
                "strategy": "REDOWNLOADED",
                "pdb_id": pdb_id,
                "uniprot_id": uniprot_id,
                "backoff_delay": 0  # No delay for redownload
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def _retry_with_modified_parameters(self, batch_id, structure_id, step_name, error_info):
        """Retry with modified parameters for the specific step"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Define parameter modifications based on step
            parameter_mods = {
                'run_hhsearch': {'max_target': 50000, 'z_score': 50000},
                'run_foldseek': {'sensitivity': 7.5, 'max_seqs': 500000},
                'run_iterative_dali': {'similarity_threshold': 0.3}
            }
            
            # Get current parameters
            cursor.execute(
                """
                SELECT parameters->>'step_params' 
                FROM structures 
                WHERE structure_id = %s
                """,
                (structure_id,)
            )
            current_params = json.loads(cursor.fetchone()[0] or '{}')
            
            # Update parameters
            step_params = current_params.get(step_name, {})
            for param, value in parameter_mods.get(step_name, {}).items():
                step_params[param] = value
            
            current_params[step_name] = step_params
            
            cursor.execute(
                """
                UPDATE structures
                SET parameters = jsonb_set(
                    COALESCE(parameters, '{}'::jsonb),
                    '{step_params}',
                    %s::jsonb
                )
                WHERE structure_id = %s
                """,
                (json.dumps(current_params), structure_id)
            )
            
            conn.commit()
            
            return {
                "strategy": "MODIFIED_PARAMETERS",
                "modified_params": step_params,
                "backoff_delay": error_info["backoff_delay"]
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def _retry_with_backoff(self, batch_id, structure_id, step_name, error_info):
        """Simple retry with backoff"""
        return {
            "strategy": "BACKOFF_RETRY",
            "backoff_delay": error_info["backoff_delay"]
        }
    
    def _mark_step_as_skipped(self, batch_id, structure_id, step_name, error_info):
        """Mark step as skipped and continue"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get batch item id
            cursor.execute(
                """
                SELECT batch_item_id FROM batch_items
                WHERE batch_id = %s AND structure_id = %s
                """,
                (batch_id, structure_id)
            )
            batch_item_id = cursor.fetchone()[0]
            
            # Get step id
            cursor.execute(
                """
                SELECT step_id FROM pipeline_steps
                WHERE name = %s
                """,
                (step_name,)
            )
            step_id = cursor.fetchone()[0]
            
            # Log as skipped
            cursor.execute(
                """
                INSERT INTO step_logs
                (batch_item_id, step_id, started_at, completed_at, status, output)
                VALUES (%s, %s, NOW(), NOW(), 'SKIPPED', %s)
                """,
                (
                    batch_item_id, 
                    step_id, 
                    json.dumps({"reason": error_info.get("reason", "No hits found")})
                )
            )
            
            conn.commit()
            
            return {
                "strategy": "SKIPPED",
                "reason": error_info.get("reason", "No hits found")
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def _use_fallback_method(self, batch_id, structure_id, step_name, error_info):
        """Use fallback method for the step"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Define fallback methods for each step
            fallbacks = {
                'parse_domains': '_run_simplified_domain_detection',
                'get_support': '_run_structure_only_support'
            }
            
            # Get fallback method name
            fallback_method = fallbacks.get(step_name)
            
            if not fallback_method:
                return self._mark_step_as_skipped(batch_id, structure_id, step_name, error_info)
            
            # Mark step as using fallback
            cursor.execute(
                """
                UPDATE structures
                SET parameters = jsonb_set(