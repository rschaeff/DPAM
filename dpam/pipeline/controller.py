## dpam/pipeline/controller.py

import os
import logging
import json
import time
import psycopg2
from datetime import datetime

from dpam.batch.manager import DPAMBatchManager
from dpam.batch.preparation import DPAMBatchPreparation
from dpam.batch.supplement import DPAMBatchSupplementation
from dpam.grid.manager import DPAMOpenGridManager
from dpam.pipeline.checkpoints import DPAMBatchCheckpoints
from dpam.pipeline.errors import DPAMErrorHandler
from dpam.pipeline.quality import DPAMQualityControl

# Import step implementations
from dpam.steps.hhsearch import HHSearchRunner
from dpam.steps.foldseek import FoldSeekRunner
from dpam.steps.filter_foldseek import FoldSeekFilter
from dpam.steps.ecod import ECODMapper
from dpam.steps.dali_candidates import DaliCandidatesCollector
from dpam.steps.get_good_domains import DomainQualityFilter


class DPAMPipelineController:
    """Main controller for DPAM pipeline execution"""
    
    def __init__(self, db_config, grid_config, data_dir):
        """
        Initialize pipeline controller with configuration
        
        Args:
            db_config (dict): Database connection configuration
            grid_config (dict): Grid/cluster configuration
            data_dir (str): Path to data directory
        """
        self.db_config = db_config
        self.grid_config = grid_config
        self.data_dir = data_dir
        self.logger = logging.getLogger("dpam.pipeline.controller")
        
        # Initialize components
        self.grid_manager = DPAMOpenGridManager(db_config, grid_config, data_dir)
        self.error_handler = DPAMErrorHandler(db_config)
        self.checkpoints = DPAMBatchCheckpoints(db_config)
        self.quality_control = DPAMQualityControl(db_config)
        
        # Initialize step runners
        self.step_runners = {
            'run_hhsearch': HHSearchRunner({
                'data_dir': data_dir,
                'hhsearch_threads': grid_config.get('hhsearch_threads', 4)
            }),
            'run_foldseek': FoldSeekRunner({
                'data_dir': data_dir,
                'foldseek_threads': grid_config.get('foldseek_threads', 4)
            }),
            'filter_foldseek': FoldSeekFilter({
                'foldseek_max_hit_count': 100,
                'foldseek_min_good_residues': 10
            }),
            'map_to_ecod': ECODMapper({
                'data_dir': data_dir,
                'ecod_min_domain_residues': 10
            }),
            'get_dali_candidates': DaliCandidatesCollector({})
            # Other step runners will be added as they are implemented
        }
        
        # Define pipeline steps and dependencies
        self.pipeline_steps = [
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
        
        # Step dependencies for validation and recovery
        self.step_dependencies = {
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
            'get_good_domain': DomainQualityFilter({
                'data_dir': data_dir,
                'min_domain_size': pipeline_config.get('min_domain_size', 25),
                'min_segment_size': pipeline_config.get('min_segment_size', 5),
                'max_segment_gap': pipeline_config.get('max_segment_gap', 10)
            })

        }
    
    def get_db_connection(self):
        """Get connection to PostgreSQL database"""
        return psycopg2.connect(**self.db_config)
    
    def prepare_batch_pipeline(self, batch_id):
        """
        Prepare batch for pipeline execution
        
        Args:
            batch_id (int): Batch ID
            
        Returns:
            dict: Preparation result
        """
        self.logger.info(f"Preparing batch {batch_id} for pipeline execution")
        
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Check if batch exists and is ready
            cursor.execute(
                "SELECT status FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return {
                    'success': False,
                    'message': f"Batch {batch_id} not found"
                }
                
            status = result[0]
            if not status.startswith('READY'):
                return {
                    'success': False,
                    'message': f"Batch not ready for pipeline (status: {status})"
                }
            
            # Prepare grid resources
            grid_config = self.grid_manager.prepare_batch_for_grid(batch_id)
            
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
                        'steps': self.pipeline_steps,
                        'current_step_index': -1,
                        'total_steps': len(self.pipeline_steps),
                        'prepared_at': datetime.now().isoformat()
                    }),
                    batch_id
                )
            )
            
            conn.commit()
            
            return {
                'success': True,
                'message': "Batch prepared for pipeline execution",
                'grid_config': grid_config,
                'steps': self.pipeline_steps
            }
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error preparing batch {batch_id}: {str(e)}")
            return {
                'success': False,
                'message': f"Error preparing batch: {str(e)}"
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def start_pipeline(self, batch_id, start_from=None, restart=False):
        """
        Start pipeline execution from a specific step
        
        Args:
            batch_id (int): Batch ID
            start_from (str, optional): Step name to start from
            restart (bool, optional): Whether to restart failed steps
            
        Returns:
            dict: Start result
        """
        self.logger.info(f"Starting pipeline for batch {batch_id}")
        
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Check if batch exists
            cursor.execute(
                "SELECT status, parameters->>'pipeline_config' FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return {
                    'success': False,
                    'message': f"Batch {batch_id} not found"
                }
                
            status, pipeline_config_json = result
            
            # Check if batch is ready
            if status != 'PIPELINE_READY' and not restart:
                return {
                    'success': False,
                    'message': f"Batch not ready for pipeline (status: {status})"
                }
            
            # Load pipeline configuration
            pipeline_config = json.loads(pipeline_config_json) if pipeline_config_json else {
                'steps': self.pipeline_steps,
                'current_step_index': -1,
                'total_steps': len(self.pipeline_steps)
            }
            
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
            
            # Validate step dependencies
            if start_idx > 0 and not restart:
                for dep_step in self.step_dependencies[steps[start_idx]]:
                    can_proceed, message = self.checkpoints.can_resume_from_step(batch_id, dep_step)
                    if not can_proceed:
                        return {
                            'success': False,
                            'message': f"Cannot start from {steps[start_idx]}: {message}"
                        }
            
            # Update pipeline configuration
            pipeline_config['current_step_index'] = start_idx
            pipeline_config['started_at'] = datetime.now().isoformat()
            
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
            
            self.logger.info(f"Pipeline started for batch {batch_id} at step {step}")
            
            return {
                'success': True,
                'message': f"Pipeline started at step: {step}",
                'job_id': result['job_id'],
                'grid_job_id': result['grid_job_id'],
                'step': step,
                'step_index': start_idx
            }
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error starting pipeline for batch {batch_id}: {str(e)}")
            return {
                'success': False,
                'message': f"Error starting pipeline: {str(e)}"
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def monitor_and_advance_pipeline(self, batch_id):
        """
        Monitor current step and advance to next when ready
        
        Args:
            batch_id (int): Batch ID
            
        Returns:
            dict: Monitoring result
        """
        self.logger.info(f"Monitoring pipeline for batch {batch_id}")
        
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
            result = cursor.fetchone()
            
            if not result:
                return {
                    'success': False,
                    'message': f"Batch {batch_id} not found"
                }
                
            status, pipeline_config_json, current_job_json = result
            
            # Check if pipeline is running
            if not status.startswith(('PIPELINE_RUNNING', 'GRID_RUNNING', 'GRID_COMPLETED')):
                return {
                    'success': False,
                    'message': f"Pipeline not running (status: {status})"
                }
            
            # Load pipeline configuration
            pipeline_config = json.loads(pipeline_config_json) if pipeline_config_json else None
            current_job = json.loads(current_job_json) if current_job_json else None
            
            # If no configuration or current job, this is unexpected
            if not pipeline_config or not current_job:
                return {
                    'success': False,
                    'message': "Pipeline configuration or current job not found"
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
                
                # Check if this was the last step
                if current_step_index + 1 >= len(steps):
                    # Pipeline completed
                    cursor.execute(
                        "UPDATE batches SET status = 'PIPELINE_COMPLETED', completed_at = NOW() WHERE batch_id = %s",
                        (batch_id,)
                    )
                    
                    conn.commit()
                    
                    # Generate quality report
                    self.quality_control.generate_batch_report(batch_id)
                    
                    return {
                        'success': True,
                        'message': "Pipeline completed successfully",
                        'status': 'COMPLETED'
                    }
                
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
                
                self.logger.info(f"Advanced pipeline for batch {batch_id} to step {next_step}")
                
                return {
                    'success': True,
                    'message': f"Advanced to next step: {next_step}",
                    'job_id': result['job_id'],
                    'grid_job_id': result['grid_job_id'],
                    'current_step_index': next_step_index,
                    'total_steps': len(steps)
                }
            
            # Job failed
            cursor.execute(
                "UPDATE batches SET status = 'PIPELINE_FAILED' WHERE batch_id = %s",
                (batch_id,)
            )
            
            conn.commit()
            
            self.logger.error(f"Pipeline step {current_job['step_name']} failed for batch {batch_id}")
            
            return {
                'success': False,
                'message': f"Step {current_job['step_name']} failed",
                'status': job_status,
                'recovery_options': {
                    'restart_step': current_job['step_name'],
                    'recover': True
                }
            }
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error monitoring pipeline for batch {batch_id}: {str(e)}")
            return {
                'success': False,
                'message': f"Error monitoring pipeline: {str(e)}"
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def recover_failed_pipeline(self, batch_id):
        """
        Attempt to recover a failed pipeline
        
        Args:
            batch_id (int): Batch ID
            
        Returns:
            dict: Recovery result
        """
        self.logger.info(f"Attempting to recover failed pipeline for batch {batch_id}")
        
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
            result = cursor.fetchone()
            
            if not result:
                return {
                    'success': False,
                    'message': f"Batch {batch_id} not found"
                }
                
            status, pipeline_config_json, current_job_json = result
            
            # Check if pipeline is failed
            if not status.startswith(('GRID_FAILED', 'PIPELINE_FAILED')):
                return {
                    'success': False,
                    'message': f"Pipeline not failed (status: {status})"
                }
            
            # Load pipeline configuration
            pipeline_config = json.loads(pipeline_config_json) if pipeline_config_json else None
            current_job = json.loads(current_job_json) if current_job_json else None
            
            # If no configuration or current job, this is unexpected
            if not pipeline_config or not current_job:
                return {
                    'success': False,
                    'message': "Pipeline configuration or current job not found"
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
                
                # Check if this was the last step
                if current_step_index + 1 >= len(steps):
                    # Pipeline completed with failures
                    cursor.execute(
                        "UPDATE batches SET status = 'PIPELINE_COMPLETED_WITH_FAILURES', completed_at = NOW() WHERE batch_id = %s",
                        (batch_id,)
                    )
                    
                    conn.commit()
                    
                    # Generate quality report
                    self.quality_control.generate_batch_report(batch_id)
                    
                    return {
                        'success': True,
                        'message': "Pipeline completed with some failures",
                        'status': 'COMPLETED_WITH_FAILURES'
                    }
                
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
                
                self.logger.info(f"Skipped failed structures and advanced to step {next_step} for batch {batch_id}")
                
                return {
                    'success': True,
                    'message': f"Skipped failed structures and advanced to: {next_step}",
                    'job_id': result['job_id'],
                    'grid_job_id': result['grid_job_id'],
                    'structures_skipped': len(failed_structures)
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
            
            self.logger.info(f"Recovering {len(recovery_plans)} structures for step {failed_step} for batch {batch_id}")
            
            return {
                'success': True,
                'message': f"Recovering {len(recovery_plans)} structures for step: {failed_step}",
                'job_id': result['job_id'],
                'grid_job_id': result['grid_job_id'],
                'recovery_plans': recovery_plans
            }
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error recovering pipeline for batch {batch_id}: {str(e)}")
            return {
                'success': False,
                'message': f"Error recovering pipeline: {str(e)}"
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def _submit_recovery_job(self, batch_id, step_name, structure_ids):
        """
        Submit a job for recovery of specific structures
        
        Args:
            batch_id (int): Batch ID
            step_name (str): Step name
            structure_ids (list): List of structure IDs to recover
            
        Returns:
            dict: Job submission result
        """
        self.logger.info(f"Submitting recovery job for batch {batch_id}, step {step_name}")
        
        # Create structure manifest
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get batch path
            cursor.execute(
                "SELECT parameters->>'batch_path' FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            batch_path = cursor.fetchone()[0]
            
            # Create recovery manifest
            recovery_manifest_path = os.path.join(batch_path, f"recovery_manifest_{step_name}.json")
            
            # Get structure information
            structure_data = {}
            placeholders = ','.join(['%s'] * len(structure_ids))
            cursor.execute(
                f"""
                SELECT s.structure_id, s.pdb_id, s.structure_path 
                FROM structures s
                WHERE s.structure_id IN ({placeholders})
                """,
                structure_ids
            )
            
            for structure_id, pdb_id, structure_path in cursor.fetchall():
                structure_data[str(structure_id)] = {
                    'pdb_id': pdb_id,
                    'structure_path': structure_path,
                    'recovery': True
                }
            
            # Write manifest
            with open(recovery_manifest_path, 'w') as f:
                json.dump({
                    'batch_id': batch_id,
                    'step_name': step_name,
                    'structures': structure_data,
                    'total_structures': len(structure_ids),
                    'recovery': True
                }, f, indent=2)
            
            # Submit specific grid job for recovery
            # This would be a modified version of grid_manager.submit_pipeline_job
            # that uses the recovery manifest instead of the full batch
            
            # For this implementation, we'll just simulate the job submission
            # In a real implementation, this would call a specific method on the grid manager
            
            return {
                'job_id': f"recovery_{batch_id}_{step_name}_{int(time.time())}",
                'grid_job_id': 9999  # Placeholder for actual grid job ID
            }
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error submitting recovery job: {str(e)}")
            raise
            
        finally:
            cursor.close()
            conn.close()
    
    def run_step_locally(self, batch_id, step_name, structure_id):
        """
        Run a specific pipeline step locally for a single structure
        
        This is useful for testing and debugging without using the grid
        
        Args:
            batch_id (int): Batch ID
            step_name (str): Step name
            structure_id (int): Structure ID
            
        Returns:
            dict: Step result
        """
        self.logger.info(f"Running step {step_name} locally for structure {structure_id}")
        
        # Get structure information
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get batch path
            cursor.execute(
                "SELECT parameters->>'batch_path' FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            batch_path = cursor.fetchone()[0]
            
            # Get structure information
            cursor.execute(
                """
                SELECT s.pdb_id, s.structure_path, p.sequence
                FROM structures s
                JOIN proteins p ON s.protein_id = p.protein_id
                WHERE s.structure_id = %s
                """,
                (structure_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return {
                    'success': False,
                    'message': f"Structure {structure_id} not found"
                }
                
            pdb_id, structure_path, sequence = result
            
            # Create output directory
            output_dir = os.path.join(batch_path, "results", step_name, f"struct_{structure_id}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Write FASTA file if needed
            fasta_path = os.path.join(output_dir, f"struct_{structure_id}.fa")
            with open(fasta_path, 'w') as f:
                f.write(f">{pdb_id}\n{sequence}\n")
            
            # Get step runner
            try:
                step_runner = self.step_runners[step_name]
            except KeyError:
                return {
                    'success': False,
                    'message': f"Step {step_name} not implemented"
                }
            
            # Run step based on step name
            if step_name == 'run_hhsearch':
                result = step_runner.run(structure_id, fasta_path, output_dir)
                
            elif step_name == 'run_foldseek':
                result = step_runner.run(structure_id, structure_path, output_dir)
                
            elif step_name == 'filter_foldseek':
                # Get FoldSeek results path
                foldseek_dir = os.path.join(batch_path, "results", "run_foldseek", f"struct_{structure_id}")
                foldseek_path = os.path.join(foldseek_dir, f"struct_{structure_id}.foldseek")
                
                result = step_runner.run(structure_id, fasta_path, foldseek_path, output_dir)
                
            elif step_name == 'map_to_ecod':
                # Get HHSearch results path
                hhsearch_dir = os.path.join(batch_path, "results", "run_hhsearch", f"struct_{structure_id}")
                hhsearch_path = os.path.join(hhsearch_dir, f"struct_{structure_id}.hhsearch")
                
                result = step_runner.run(structure_id, hhsearch_path, output_dir)
                
            elif step_name == 'get_dali_candidates':
                # Get required input paths
                ecod_dir = os.path.join(batch_path, "results", "map_to_ecod", f"struct_{structure_id}")
                ecod_path = os.path.join(ecod_dir, f"struct_{structure_id}.map2ecod.result")
                
                foldseek_dir = os.path.join(batch_path, "results", "filter_foldseek", f"struct_{structure_id}")
                foldseek_filtered_path = os.path.join(foldseek_dir, f"struct_{structure_id}.foldseek.flt.result")
                
                result = step_runner.run(structure_id, ecod_path, foldseek_filtered_path, output_dir)
                
            else:
                return {
                    'success': False,
                    'message': f"Step {step_name} execution not implemented"
                }
            
            # Update step log in database
            cursor.execute(
                """
                INSERT INTO step_logs 
                (batch_item_id, step_id, started_at, completed_at, status, output, metrics, error_message)
                SELECT 
                    bi.batch_item_id, 
                    ps.step_id, 
                    NOW() - INTERVAL '1 minute', 
                    NOW(), 
                    %s, 
                    %s, 
                    %s, 
                    %s
                FROM batch_items bi, pipeline_steps ps
                WHERE bi.structure_id = %s 
                  AND bi.batch_id = %s
                  AND ps.name = %s
                """,
                (
                    result.get('status'),
                    json.dumps(result.get('output_files', {})),
                    json.dumps(result.get('metrics', {})),
                    result.get('error_message'),
                    structure_id,
                    batch_id,
                    step_name
                )
            )
            
            conn.commit()
            
            return {
                'success': True,
                'step': step_name,
                'structure_id': structure_id,
                'result': result
            }
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error running step {step_name} locally: {str(e)}")
            return {
                'success': False,
                'message': f"Error running step locally: {str(e)}"
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def get_pipeline_status(self, batch_id):
        """
        Get detailed status of pipeline for a batch
        
        Args:
            batch_id (int): Batch ID
            
        Returns:
            dict: Pipeline status information
        """
        self.logger.debug(f"Getting pipeline status for batch {batch_id}")
        
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get batch information
            cursor.execute(
                """
                SELECT b.status, b.created_at, b.completed_at, 
                       b.parameters->>'pipeline_config', b.parameters->>'current_grid_job',
                       COUNT(bi.batch_item_id) as total_structures
                FROM batches b
                JOIN batch_items bi ON b.batch_id = bi.batch_id
                WHERE b.batch_id = %s
                GROUP BY b.batch_id
                """,
                (batch_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return {
                    'success': False,
                    'message': f"Batch {batch_id} not found"
                }
                
            status, created_at, completed_at, pipeline_config_json, current_job_json, total_structures = result
            
            # Load configurations
            pipeline_config = json.loads(pipeline_config_json) if pipeline_config_json else None
            current_job = json.loads(current_job_json) if current_job_json else None
            
            # Get step statistics
            cursor.execute(
                """
                SELECT ps.name, sl.status, COUNT(sl.log_id)
                FROM step_logs sl
                JOIN pipeline_steps ps ON sl.step_id = ps.step_id
                JOIN batch_items bi ON sl.batch_item_id = bi.batch_item_id
                WHERE bi.batch_id = %s
                GROUP BY ps.name, sl.status
                ORDER BY ps.name, sl.status
                """,
                (batch_id,)
            )
            steps_results = cursor.fetchall()
            
            # Format step statistics
            step_status = {}
            for step, status, count in steps_results:
                if step not in step_status:
                    step_status[step] = {}
                step_status[step][status] = count
            
            # Get checkpoint information
            cursor.execute(
                """
                SELECT step_name, status, created_at
                FROM batch_checkpoints
                WHERE batch_id = %s
                ORDER BY created_at
                """,
                (batch_id,)
            )
            checkpoints = [
                {
                    'step': row[0],
                    'status': row[1],
                    'timestamp': row[2].isoformat() if row[2] else None
                }
                for row in cursor.fetchall()
            ]
            
            # Build response
            response = {
                'success': True,
                'batch_id': batch_id,
                'status': status,
                'created_at': created_at.isoformat() if created_at else None,
                'completed_at': completed_at.isoformat() if completed_at else None,
                'total_structures': total_structures,
                'checkpoints': checkpoints,
                'step_status': step_status
            }
            
            # Add pipeline progress if available
            if pipeline_config:
                steps = pipeline_config['steps']
                current_step_index = pipeline_config.get('current_step_index', -1)
                
                response.update({
                    'pipeline_steps': steps,
                    'total_steps': len(steps),
                    'current_step_index': current_step_index,
                    'current_step': steps[current_step_index] if 0 <= current_step_index < len(steps) else None,
                    'progress_percent': round((current_step_index + 1) / len(steps) * 100, 1) if current_step_index >= 0 else 0
                })
            
            # Add current job information if available
            if current_job:
                response['current_job'] = current_job
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline status for batch {batch_id}: {str(e)}")
            return {
                'success': False,
                'message': f"Error getting pipeline status: {str(e)}"
            }
            
        finally:
            cursor.close()
            conn.close()