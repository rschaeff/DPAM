## dpam/pipeline/controller.py

class DPAMPipelineController:
    """Main controller for DPAM pipeline execution"""
    
    def __init__(self, db_config, grid_config, data_dir):
        self.db_config = db_config
        self.grid_config = grid_config
        self.data_dir = data_dir
        
        # Initialize components
        self.grid_manager = DPAMOpenGridManager(db_config, grid_config, data_dir)
        self.error_handler = DPAMErrorHandler(db_config)
        self.checkpoints = DPAMBatchCheckpoints(db_config)
    
    def get_db_connection(self):
        """Get connection to PostgreSQL database"""
        return psycopg2.connect(**self.db_config)
    
    def prepare_batch_pipeline(self, batch_id):
        """Prepare batch for pipeline execution"""
        # Initialize batch path and directories
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Check if batch exists and is ready
            cursor.execute(
                "SELECT status FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            status = cursor.fetchone()[0]
            
            if not status.startswith('READY'):
                return {
                    'success': False,
                    'message': f"Batch not ready for pipeline (status: {status})"
                }
            
            # Prepare grid resources
            grid_config = self.grid_manager.prepare_batch_for_grid(batch_id)
            
            # Define pipeline steps
            steps = [
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
            
            # Store pipeline configuration
            cursor.execute(
                """
                UPDATE batches 
                SET parameters = jsonb_set(
                    COALESCE(parameters, '{}'::jsonb),
                    '{pipeline_config}',
                    %s::jsonb
                ),
                status = 'PIPELINE_READY'
                WHERE batch_id = %s
                """,
                (
                    json.dumps({
                        'steps': steps,
                        'current_step_index': -1,
                        'total_steps': len(steps)
                    }),
                    batch_id
                )
            )
            
            conn.commit()
            
            return {
                'success': True,
                'message': "Batch prepared for pipeline execution",
                'grid_config': grid_config,
                'steps': steps
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def start_pipeline(self, batch_id, start_from=None):
        """Start pipeline execution from a specific step"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get pipeline configuration
            cursor.execute(
                "SELECT parameters->>'pipeline_config' FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            pipeline_config = json.loads(cursor.fetchone()[0])
            steps = pipeline_config['steps']
            
            # Determine starting step
            start_idx = 0
            if start_from:
                try:
                    start_idx = steps.index(start_from)
                except ValueError:
                    return {
                        'success': False,
                        'message': f"Unknown step: {start_from}"
                    }
            
            # Update pipeline configuration
            pipeline_config['current_step_index'] = start_idx
            cursor.execute(
                """
                UPDATE batches 
                SET parameters = jsonb_set(
                    COALESCE(parameters, '{}'::jsonb),
                    '{pipeline_config}',
                    %s::jsonb
                ),
                status = 'PIPELINE_RUNNING'
                WHERE batch_id = %s
                """,
                (
                    json.dumps(pipeline_config),
                    batch_id
                )
            )
            
            conn.commit()
            
            # Submit first step
            step = steps[start_idx]
            result = self.grid_manager.submit_pipeline_job(batch_id, step)
            
            return {
                'success': True,
                'message': f"Pipeline started at step: {step}",
                'job_id': result['job_id'],
                'grid_job_id': result['grid_job_id']
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def monitor_and_advance_pipeline(self, batch_id):
        """Monitor current step and advance to next when ready"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get batch status and pipeline configuration
            cursor.execute(
                """
                SELECT status, parameters->>'pipeline_config', parameters->>'current_grid_job'
                FROM batches WHERE batch_id = %s
                """,
                (batch_id,)
            )
            status, pipeline_config_json, current_job_json = cursor.fetchone()
            
            # Check if pipeline is running
            if not status.startswith(('PIPELINE_RUNNING', 'GRID_RUNNING', 'GRID_COMPLETED')):
                return {
                    'success': False,
                    'message': f"Pipeline not running (status: {status})"
                }
            
            pipeline_config = json.loads(pipeline_config_json)
            current_job = json.loads(current_job_json) if current_job_json else None
            
            # If no current job, this is unexpected
            if not current_job:
                return {
                    'success': False,
                    'message': "No current job found"
                }
            
            # Check job status
            grid_job_id = current_job['grid_job_id']
            job_status = self.grid_manager.check_job_status(grid_job_id)
            
            # If job is still running, return status
            if job_status['status'] == 'RUNNING':
                return {
                    'success': True,
                    'message': "Current step still running",
                    'status': job_status,
                    'current_step': current_job['step_name']
                }
            
            # Handle job completion or failure
            if job_status['status'] == 'COMPLETED':
                # Set checkpoint
                self.checkpoints.set_checkpoint(batch_id, current_job['step_name'])
                
                # Get next step
                current_step_index = pipeline_config['current_step_index']
                steps = pipeline_config['steps']
                
                if current_step_index + 1 < len(steps):
                    # Advance to next step
                    next_step_index = current_step_index + 1
                    next_step = steps[next_step_index]
                    
                    # Update pipeline configuration
                    pipeline_config['current_step_index'] = next_step_index
                    cursor.execute(
                        """
                        UPDATE batches 
                        SET parameters = jsonb_set(
                            COALESCE(parameters, '{}'::jsonb),
                            '{pipeline_config}',
                            %s::jsonb
                        )
                        WHERE batch_id = %s
                        """,
                        (
                            json.dumps(pipeline_config),
                            batch_id
                        )
                    )
                    
                    conn.commit()
                    
                    # Submit next step
                    result = self.grid_manager.submit_pipeline_job(batch_id, next_step)
                    
                    return {
                        'success': True,
                        'message': f"Advanced to next step: {next_step}",
                        'job_id': result['job_id'],
                        'grid_job_id': result['grid_job_id']
                    }
                else:
                    # Pipeline completed
                    cursor.execute(
                        "UPDATE batches SET status = 'PIPELINE_COMPLETED', completed_at = NOW() WHERE batch_id = %s",
                        (batch_id,)
                    )
                    
                    conn.commit()
                    
                    return {
                        'success': True,
                        'message': "Pipeline completed successfully",
                        'status': 'COMPLETED'
                    }
            else:
                # Job failed
                return {
                    'success': False,
                    'message': f"Step {current_job['step_name']} failed",
                    'status': job_status
                }
                
        finally:
            cursor.close()
            conn.close()
    
    def recover_failed_pipeline(self, batch_id):
        """Attempt to recover a failed pipeline"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get batch status and pipeline configuration
            cursor.execute(
                """
                SELECT status, parameters->>'pipeline_config', parameters->>'current_grid_job'
                FROM batches WHERE batch_id = %s
                """,
                (batch_id,)
            )
            status, pipeline_config_json, current_job_json = cursor.fetchone()
            
            # Check if pipeline is failed
            if not status.startswith(('GRID_FAILED', 'PIPELINE_FAILED')):
                return {
                    'success': False,
                    'message': f"Pipeline not failed (status: {status})"
                }
            
            pipeline_config = json.loads(pipeline_config_json)
            current_job = json.loads(current_job_json) if current_job_json else None
            
            # If no current job, this is unexpected
            if not current_job:
                return {
                    'success': False,
                    'message': "No current job found"
                }
            
            # Get failed step
            failed_step = current_job['step_name']
            
            # Get failed structures
            cursor.execute(
                """
                SELECT s.structure_id, sl.error_message
                FROM step_logs sl
                JOIN batch_items bi ON sl.batch_item_id = bi.batch_item_id
                JOIN structures s ON bi.structure_id = s.structure_id
                JOIN pipeline_steps ps ON sl.step_id = ps.step_id
                WHERE bi.batch_id = %s AND ps.name = %s AND sl.status = 'FAILED'
                """,
                (batch_id, failed_step)
            )
            failed_structures = cursor.fetchall()
            
            # Process each failed structure
            recovery_plans = []
            for structure_id, error_message in failed_structures:
                # Check if should retry
                should_retry, retry_info = self.error_handler.should_retry(
                    batch_id, structure_id, failed_step, error_message
                )
                
                if should_retry:
                    # Apply recovery strategy
                    recovery_result = self.error_handler.apply_recovery_strategy(
                        batch_id, structure_id, failed_step, retry_info
                    )
                    
                    recovery_plans.append({
                        'structure_id': structure_id,
                        'strategy': recovery_result['strategy'],
                        'retry_info': retry_info
                    })
            
            # If no recovery plans, skip step for failed structures
            if not recovery_plans:
                # Mark failed structures as skipped for this step
                for structure_id, _ in failed_structures:
                    self.error_handler._mark_step_as_skipped(
                        batch_id, structure_id, failed_step, {'reason': 'No recovery plan available'}
                    )
                
                # Set checkpoint with partial completion
                self.checkpoints.set_checkpoint(batch_id, failed_step, status="PARTIALLY_COMPLETED")
                
                # Get next step
                current_step_index = pipeline_config['current_step_index']
                steps = pipeline_config['steps']
                
                if current_step_index + 1 < len(steps):
                    # Advance to next step
                    next_step_index = current_step_index + 1
                    next_step = steps[next_step_index]
                    
                    # Update pipeline configuration
                    pipeline_config['current_step_index'] = next_step_index
                    cursor.execute(
                        """
                        UPDATE batches 
                        SET parameters = jsonb_set(
                            COALESCE(parameters, '{}'::jsonb),
                            '{pipeline_config}',
                            %s::jsonb
                        )
                        WHERE batch_id = %s
                        """,
                        (
                            json.dumps(pipeline_config),
                            batch_id
                        )
                    )
                    
                    conn.commit()
                    
                    # Submit next step
                    result = self.grid_manager.submit_pipeline_job(batch_id, next_step)
                    
                    return {
                        'success': True,
                        'message': f"Skipped failed structures and advanced to: {next_step}",
                        'job_id': result['job_id'],
                        'grid_job_id': result['grid_job_id']
                    }
                else:
                    # Pipeline completed with failures
                    cursor.execute(
                        "UPDATE batches SET status = 'PIPELINE_COMPLETED_WITH_FAILURES', completed_at = NOW() WHERE batch_id = %s",
                        (batch_id,)
                    )
                    
                    conn.commit()
                    
                    return {
                        'success': True,
                        'message': "Pipeline completed with some failures",
                        'status': 'COMPLETED_WITH_FAILURES'
                    }
            
            # Otherwise, retry the step with recovery plans
            cursor.execute(
                "UPDATE batches SET status = 'PIPELINE_RECOVERING' WHERE batch_id = %s",
                (batch_id,)
            )
            
            conn.commit()
            
            # Resubmit job for just the failed structures
            structure_ids = [plan['structure_id'] for plan in recovery_plans]
            result = self._submit_recovery_job(batch_id, failed_step, structure_ids)
            
            return {
                'success': True,
                'message': f"Recovering {len(recovery_plans)} structures for step: {failed_step}",
                'job_id': result['job_id'],
                'grid_job_id': result['grid_job_id'],
                'recovery_plans': recovery_plans
            }
                
        finally:
            cursor.close()
            conn.close()
    
    def _submit_recovery_job(self, batch_id, step_name, structure_ids):
        """Submit a job for recovery of specific structures"""
        # Implementation similar to grid_manager.submit_pipeline_job but for specific structures
        # ...
        
        # For brevity, we'll assume it returns similar result
        return {'job_id': 'recovery_job', 'grid_job_id': 123}