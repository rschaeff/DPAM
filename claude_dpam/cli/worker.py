#!/usr/bin/env python3
"""
Worker node execution script for DPAM pipeline.

This module provides the worker node functionality for executing
pipeline steps in a distributed grid environment.
"""

import os
import sys
import json
import time
import logging
import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dpam.config import load_config
from dpam.database import DatabaseManager
from dpam.exceptions import StepExecutionError, ToolExecutionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dpam.cli.worker")

class DPAMWorker:
    """Worker node implementation for DPAM pipeline steps"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize worker with configuration
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Get basic configuration
        self.db_config = self.config.get_db_config()
        self.data_dir = self.config.get('data_dir', '/data/dpam')
        self.grid_config = self.config.get('grid', {})
        
        # Create database manager
        self.db_manager = DatabaseManager(self.db_config)
        
        # Initialize step runners
        self._init_step_runners()
    
    def _init_step_runners(self):
        """Initialize step runners for all pipeline steps"""
        self.step_runners = {}
        
        # Import step implementations
        try:
            from dpam.steps.hhsearch import HHSearchStep
            from dpam.steps.foldseek import FoldSeekRunner
            from dpam.steps.filter_foldseek import FoldSeekFilter
            from dpam.steps.ecod import ECODMapper
            from dpam.steps.dali_candidates import DaliCandidatesCollector
            from dpam.steps.iterative_dali import IterativeDaliRunner
            from dpam.steps.analyze_dali import DaliResultsAnalyzer
            from dpam.steps.support import DomainSupportCalculator
            from dpam.steps.sse import SecondaryStructureAssigner
            from dpam.steps.disorder import DisorderPredictor
            from dpam.steps.domains import DomainDetector
            from dpam.steps.mapping import ECODMapper as ECODDomainMapper
            
            # Create step runners with appropriate configuration
            binaries_config = self.config.get('binaries', {})
            pipeline_config = self.config.get('pipeline', {})
            
            # Create configuration for each step
            hhsearch_config = {
                'data_dir': self.data_dir,
                'hhsearch_threads': self.grid_config.get('hhsearch_threads', 4),
                'hhsearch_binary': binaries_config.get('hhsearch', 'hhsearch'),
                'hhblits_binary': binaries_config.get('hhblits', 'hhblits')
            }
            
            foldseek_config = {
                'data_dir': self.data_dir,
                'foldseek_threads': self.grid_config.get('foldseek_threads', 4),
                'foldseek_binary': binaries_config.get('foldseek', 'foldseek')
            }
            
            dali_config = {
                'data_dir': self.data_dir,
                'dali_threads': self.grid_config.get('threads', 4),
                'dali_binary': binaries_config.get('dali', 'dali.pl'),
                'dali_min_z_score': pipeline_config.get('dali_min_z_score', 8.0)
            }
            
            domain_config = {
                'min_domain_size': pipeline_config.get('min_domain_size', 30),
                'max_domains': pipeline_config.get('max_domains', 20),
                'min_support_score': pipeline_config.get('min_support_score', 0.5),
                'disorder_threshold': pipeline_config.get('disorder_threshold', 70.0),
                'ecod_weight': pipeline_config.get('ecod_weight', 2.0),
                'dali_weight': pipeline_config.get('dali_weight', 1.5),
                'foldseek_weight': pipeline_config.get('foldseek_weight', 1.0),
                'pae_weight': pipeline_config.get('pae_weight', 2.0),
                'dssp_binary': binaries_config.get('dssp', 'mkdssp')
            }
            
            # Register step runners
            self.step_runners = {
                'run_hhsearch': HHSearchStep,
                'run_foldseek': FoldSeekRunner(foldseek_config),
                'filter_foldseek': FoldSeekFilter({
                    'foldseek_max_hit_count': 100,
                    'foldseek_min_good_residues': 10
                }),
                'map_to_ecod': ECODMapper({
                    'data_dir': self.data_dir,
                    'ecod_min_domain_residues': 10
                }),
                'get_dali_candidates': DaliCandidatesCollector({}),
                'run_iterative_dali': IterativeDaliRunner(dali_config),
                'analyze_dali': DaliResultsAnalyzer(dali_config),
                'get_support': DomainSupportCalculator(domain_config),
                'get_sse': SecondaryStructureAssigner(domain_config),
                'get_diso': DisorderPredictor(domain_config),
                'parse_domains': DomainDetector(domain_config),
                'map_domains': ECODDomainMapper(domain_config)
            }
            
        except ImportError as e:
            logger.error(f"Failed to import step implementations: {e}")
            self.step_runners = {}
    
    def load_manifest(self, manifest_path: str) -> Dict[str, Any]:
        """
        Load structure manifest file
        
        Args:
            manifest_path: Path to manifest file
            
        Returns:
            Manifest data
        """
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to load manifest from {manifest_path}: {e}")
            raise
    
    def get_structure_info(self, structure_id: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get structure information from manifest
        
        Args:
            structure_id: Structure ID to retrieve
            manifest: Manifest data
            
        Returns:
            Structure information
        """
        # Check if structure exists in manifest
        if structure_id not in manifest.get('structures', {}):
            raise ValueError(f"Structure {structure_id} not found in manifest")
            
        return manifest['structures'][structure_id]
    
    def execute_step(self, step_name: str, structure_id: str, 
                    task_id: int, manifest_path: str, 
                    output_dir: str, threads: int = 1) -> Dict[str, Any]:
        """
        Execute a specific pipeline step for a structure
        
        Args:
            step_name: Pipeline step to execute
            structure_id: Structure ID to process
            task_id: Task ID in the array job
            manifest_path: Path to structure manifest
            output_dir: Directory for output files
            threads: Number of threads to use
            
        Returns:
            Step execution result
        """
        start_time = time.time()
        wall_start_time = datetime.now()
        
        try:
            # Load manifest
            manifest = self.load_manifest(manifest_path)
            batch_id = manifest.get('batch_id')
            
            # Get structure information
            structure_info = self.get_structure_info(structure_id, manifest)
            
            # Get runner for step
            if step_name not in self.step_runners:
                raise ValueError(f"Step {step_name} not implemented or not found")
                
            runner = self.step_runners[step_name]
            
            # Create working directory
            work_dir = tempfile.mkdtemp(prefix=f"dpam_worker_{structure_id}_{step_name}_")
            
            # Create step-specific output directory
            step_output_dir = os.path.join(output_dir, step_name, f"struct_{structure_id}")
            os.makedirs(step_output_dir, exist_ok=True)
            
            # Execute step
            logger.info(f"Executing {step_name} for structure {structure_id}")
            
            # For class-based step runners
            if hasattr(runner, '__class__') and not callable(runner):
                # Direct method call for class instances
                result = runner.run(structure_id, **self._get_step_inputs(
                    step_name, structure_info, work_dir, step_output_dir
                ))
            else:
                # For executable class (like HHSearchStep)
                step_instance = runner(
                    structure_info=structure_info,
                    data_dir=self.data_dir,
                    batch_id=batch_id,
                    work_dir=work_dir,
                    threads=threads
                )
                result = step_instance.run()
            
            # Update result with timing information
            end_time = time.time()
            wall_end_time = datetime.now()
            
            result['structure_id'] = structure_id
            result['batch_id'] = batch_id
            result['step_name'] = step_name
            result['task_id'] = task_id
            result['started_at'] = wall_start_time.isoformat()
            result['completed_at'] = wall_end_time.isoformat()
            result['cpu_time'] = (end_time - start_time) * threads
            result['wall_time'] = (wall_end_time - wall_start_time).total_seconds()
            
            # Log step completion
            logger.info(f"Completed {step_name} for structure {structure_id} in {result['wall_time']:.2f}s")
            
            # Save result to output directory
            result_path = os.path.join(step_output_dir, f"struct_{structure_id}_result.json")
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Update database if possible
            self._update_database(result)
            
            # Return success
            return result
            
        except Exception as e:
            logger.error(f"Error executing {step_name} for structure {structure_id}: {e}")
            
            # Create error result
            error_result = {
                'status': 'FAILED',
                'structure_id': structure_id,
                'batch_id': manifest.get('batch_id') if 'manifest' in locals() else None,
                'step_name': step_name,
                'task_id': task_id,
                'started_at': wall_start_time.isoformat(),
                'completed_at': datetime.now().isoformat(),
                'error_message': str(e),
                'cpu_time': (time.time() - start_time) * threads,
                'wall_time': (datetime.now() - wall_start_time).total_seconds()
            }
            
            # Try to save error result
            try:
                step_output_dir = os.path.join(output_dir, step_name, f"struct_{structure_id}")
                os.makedirs(step_output_dir, exist_ok=True)
                
                error_path = os.path.join(step_output_dir, f"struct_{structure_id}_error.json")
                with open(error_path, 'w') as f:
                    json.dump(error_result, f, indent=2)
            except Exception as save_error:
                logger.error(f"Failed to save error result: {save_error}")
            
            # Try to update database
            try:
                self._update_database(error_result)
            except Exception as db_error:
                logger.error(f"Failed to update database: {db_error}")
            
            # Re-raise the exception
            if isinstance(e, (StepExecutionError, ToolExecutionError)):
                raise
            else:
                raise StepExecutionError(str(e), step_name=step_name, structure_id=structure_id)
    
    def _get_step_inputs(self, step_name: str, structure_info: Dict[str, Any], 
                       work_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Get input parameters for a specific step
        
        Args:
            step_name: Pipeline step name
            structure_info: Structure information
            work_dir: Working directory
            output_dir: Output directory
            
        Returns:
            Dictionary of input parameters for the step
        """
        # Get structure path
        structure_path = structure_info.get('structure_path', '')
        
        # Define common step parameters
        common_params = {
            'output_dir': output_dir
        }
        
        # Define step-specific parameters
        step_params = {
            'run_hhsearch': {
                'structure_path': structure_path
            },
            'run_foldseek': {
                'pdb_path': structure_path
            },
            'filter_foldseek': {
                'fasta_path': os.path.join(output_dir, '..', 'run_hhsearch', f"struct_{structure_info.get('structure_id')}", 
                                          f"struct_{structure_info.get('structure_id')}.fa"),
                'foldseek_path': os.path.join(output_dir, '..', 'run_foldseek', f"struct_{structure_info.get('structure_id')}", 
                                             f"struct_{structure_info.get('structure_id')}.foldseek")
            },
            'map_to_ecod': {
                'hhsearch_path': os.path.join(output_dir, '..', 'run_hhsearch', f"struct_{structure_info.get('structure_id')}", 
                                            f"struct_{structure_info.get('structure_id')}.hhsearch")
            },
            'get_dali_candidates': {
                'ecod_mapping_path': os.path.join(output_dir, '..', 'map_to_ecod', f"struct_{structure_info.get('structure_id')}", 
                                                f"struct_{structure_info.get('structure_id')}.map2ecod.result"),
                'foldseek_filtered_path': os.path.join(output_dir, '..', 'filter_foldseek', f"struct_{structure_info.get('structure_id')}", 
                                                     f"struct_{structure_info.get('structure_id')}.foldseek.flt.result")
            },
            'run_iterative_dali': {
                'structure_path': structure_path,
                'dali_candidates_path': os.path.join(output_dir, '..', 'get_dali_candidates', f"struct_{structure_info.get('structure_id')}", 
                                                   f"struct_{structure_info.get('structure_id')}_hits4Dali")
            },
            'analyze_dali': {
                'structure_path': structure_path,
                'dali_results_path': os.path.join(output_dir, '..', 'run_iterative_dali', f"struct_{structure_info.get('structure_id')}", 
                                                f"struct_{structure_info.get('structure_id')}_dali_results.json"),
                'ecod_mapping_path': os.path.join(output_dir, '..', 'map_to_ecod', f"struct_{structure_info.get('structure_id')}", 
                                                f"struct_{structure_info.get('structure_id')}.map2ecod.result")
            },
            'get_support': {
                'structure_path': structure_path,
                'dali_analysis_path': os.path.join(output_dir, '..', 'analyze_dali', f"struct_{structure_info.get('structure_id')}", 
                                                 f"struct_{structure_info.get('structure_id')}_dali_analysis.json"),
                'ecod_mapping_path': os.path.join(output_dir, '..', 'map_to_ecod', f"struct_{structure_info.get('structure_id')}", 
                                                f"struct_{structure_info.get('structure_id')}.map2ecod.result"),
                'foldseek_filtered_path': os.path.join(output_dir, '..', 'filter_foldseek', f"struct_{structure_info.get('structure_id')}", 
                                                     f"struct_{structure_info.get('structure_id')}.foldseek.flt.result"),
                'pae_path': structure_info.get('parameters', {}).get('pae_path')
            },
            'get_sse': {
                'structure_path': structure_path,
                'domain_support_path': os.path.join(output_dir, '..', 'get_support', f"struct_{structure_info.get('structure_id')}", 
                                                  f"struct_{structure_info.get('structure_id')}_domain_support.json")
            },
            'get_diso': {
                'structure_path': structure_path,
                'sse_path': os.path.join(output_dir, '..', 'get_sse', f"struct_{structure_info.get('structure_id')}", 
                                       f"struct_{structure_info.get('structure_id')}_sse.json"),
                'pae_path': structure_info.get('parameters', {}).get('pae_path')
            },
            'parse_domains': {
                'structure_path': structure_path,
                'domain_support_path': os.path.join(output_dir, '..', 'get_support', f"struct_{structure_info.get('structure_id')}", 
                                                  f"struct_{structure_info.get('structure_id')}_domain_support.json"),
                'sse_path': os.path.join(output_dir, '..', 'get_sse', f"struct_{structure_info.get('structure_id')}", 
                                       f"struct_{structure_info.get('structure_id')}_sse.json"),
                'disorder_path': os.path.join(output_dir, '..', 'get_diso', f"struct_{structure_info.get('structure_id')}", 
                                           f"struct_{structure_info.get('structure_id')}_disorder.json")
            },
            'map_domains': {
                'structure_id': structure_info.get('structure_id'),
                'domains_path': os.path.join(output_dir, '..', 'parse_domains', f"struct_{structure_info.get('structure_id')}", 
                                           f"struct_{structure_info.get('structure_id')}_domains.json")
            }
        }
        
        # Combine common and step-specific parameters
        return {**common_params, **(step_params.get(step_name, {}))}
    
    def _update_database(self, result: Dict[str, Any]) -> None:
        """
        Update database with step result
        
        Args:
            result: Step execution result
        """
        if not result.get('batch_id') or not result.get('structure_id'):
            logger.warning("Missing batch_id or structure_id in result, skipping database update")
            return
            
        try:
            batch_id = result['batch_id']
            structure_id = result['structure_id']
            step_name = result['step_name']
            status = result['status']
            
            # Get step ID
            step_id = self.db_manager.fetchone(
                "SELECT step_id FROM pipeline_steps WHERE name = %s",
                (step_name,)
            )
            
            if not step_id:
                logger.warning(f"Step {step_name} not found in database, skipping database update")
                return
                
            step_id = step_id['step_id']
            
            # Get batch item ID
            batch_item_id = self.db_manager.fetchone(
                "SELECT batch_item_id FROM batch_items WHERE batch_id = %s AND structure_id = %s",
                (batch_id, structure_id)
            )
            
            if not batch_item_id:
                logger.warning(f"Batch item not found for batch {batch_id}, structure {structure_id}")
                return
                
            batch_item_id = batch_item_id['batch_item_id']
            
            # Insert step log
            self.db_manager.execute(
                """
                INSERT INTO step_logs 
                (batch_item_id, step_id, started_at, completed_at, status, output, metrics, error_message, cpu_time, wall_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    batch_item_id,
                    step_id,
                    result.get('started_at'),
                    result.get('completed_at'),
                    status,
                    json.dumps(result.get('output_files', {})),
                    json.dumps(result.get('metrics', {})),
                    result.get('error_message'),
                    result.get('cpu_time'),
                    result.get('wall_time')
                ),
                commit=True
            )
            
            # Update batch item status if step failed
            if status == 'FAILED':
                self.db_manager.execute(
                    "UPDATE batch_items SET status = %s, error_message = %s WHERE batch_item_id = %s",
                    ('FAILED', result.get('error_message'), batch_item_id),
                    commit=True
                )
                
            # Update structure with step output paths if completed
            if status == 'COMPLETED' and 'output_files' in result:
                self.db_manager.execute(
                    """
                    UPDATE structures
                    SET parameters = jsonb_set(
                        COALESCE(parameters, '{}'::jsonb),
                        '{step_outputs, %s}',
                        %s::jsonb
                    )
                    WHERE structure_id = %s
                    """,
                    (
                        step_name,
                        json.dumps(result['output_files']),
                        structure_id
                    ),
                    commit=True
                )
            
            logger.info(f"Updated database with result for {step_name} on structure {structure_id}")
            
        except Exception as e:
            logger.error(f"Failed to update database: {e}")
            raise

def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(description='DPAM Worker')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--batch-id', type=int, required=True, help='Batch ID')
    parser.add_argument('--step', required=True, help='Pipeline step to execute')
    parser.add_argument('--task-id', type=int, required=True, help='Task ID in array job')
    parser.add_argument('--manifest', required=True, help='Path to structure manifest')
    parser.add_argument('--data-dir', help='Path to data directory')
    parser.add_argument('--output-dir', help='Path to output directory')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use')
    
    args = parser.parse_args()
    
    # Create worker
    worker = DPAMWorker(args.config)
    
    # Override data directory if specified
    if args.data_dir:
        worker.data_dir = args.data_dir
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(worker.data_dir, 'batches', f"batch_{args.batch_id}", "results")
    
    # Load manifest
    manifest = worker.load_manifest(args.manifest)
    
    # Get structure ID for task
    task_index = args.task_id - 1  # Convert from 1-indexed to 0-indexed
    structure_ids = list(manifest.get('structures', {}).keys())
    
    if task_index < 0 or task_index >= len(structure_ids):
        logger.error(f"Task ID {args.task_id} out of range (1-{len(structure_ids)})")
        sys.exit(1)
        
    structure_id = structure_ids[task_index]
    
    try:
        # Execute step
        result = worker.execute_step(
            args.step,
            structure_id,
            args.task_id,
            args.manifest,
            output_dir,
            args.threads
        )
        
        # Exit with appropriate code
        if result['status'] == 'COMPLETED':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error executing step: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()