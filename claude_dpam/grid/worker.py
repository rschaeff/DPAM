#!/usr/bin/env python3
"""
Worker script for executing DPAM pipeline steps on OpenGrid nodes.
"""

import os
import sys
import json
import time
import gzip
import logging
import argparse
import traceback
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dpam.steps.base import StepBase, StepResult
import dpam.steps  # Import all step implementations
from dpam.gemmi_utils import get_structure_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dpam_worker.log')
    ]
)
logger = logging.getLogger('dpam.grid.worker')

class DPAMWorker:
    """Worker for executing DPAM pipeline steps."""
    
    def __init__(self, 
                task_id: int, 
                batch_id: int, 
                step_name: str, 
                manifest_path: str,
                data_dir: str,
                output_dir: str,
                threads: int = 1,
                verbose: bool = False):
        """
        Initialize the worker.
        
        Args:
            task_id: Task ID (SGE_TASK_ID)
            batch_id: Batch ID
            step_name: Pipeline step to execute
            manifest_path: Path to batch manifest file
            data_dir: Path to reference data directory
            output_dir: Path to output directory
            threads: Number of threads to use
            verbose: Whether to enable verbose logging
        """
        self.task_id = task_id
        self.batch_id = batch_id
        self.step_name = step_name
        self.manifest_path = manifest_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.threads = threads
        self.verbose = verbose
        
        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize handler
        self.structure_handler = get_structure_handler()
        
        # Load manifest
        self.manifest = self._load_manifest()
        
        # Map task_id to structure_id
        self.structure_info = self._get_structure_info()
        
        # Set up working directory
        self.work_dir = self._setup_working_directory()
        
        # Initialize step instance
        self.step_instance = self._initialize_step()
    
    def _load_manifest(self) -> Dict[str, Any]:
        """
        Load the batch manifest file.
        
        Returns:
            Manifest data
        """
        try:
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
            
            logger.debug(f"Loaded manifest with {len(manifest['structures'])} structures")
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            raise
    
    def _get_structure_info(self) -> Dict[str, Any]:
        """
        Get structure information for the current task.
        
        Returns:
            Structure information
        """
        structures = self.manifest['structures']
        structure_ids = list(structures.keys())
        
        if self.task_id <= 0 or self.task_id > len(structure_ids):
            raise ValueError(f"Task ID {self.task_id} out of range (1-{len(structure_ids)})")
        
        structure_id = structure_ids[self.task_id - 1]  # Convert to 0-based
        structure_info = structures[structure_id]
        structure_info['structure_id'] = structure_id
        
        logger.info(f"Processing structure {structure_info['pdb_id']} (ID: {structure_id})")
        return structure_info
    
    def _setup_working_directory(self) -> str:
        """
        Set up the working directory for the task.
        
        Returns:
            Path to working directory
        """
        # Create unique working directory
        work_dir = os.path.join(
            self.output_dir,
            f"batch_{self.batch_id}",
            "work",
            f"{self.structure_info['pdb_id']}_{self.step_name}"
        )
        
        os.makedirs(work_dir, exist_ok=True)
        logger.debug(f"Created working directory: {work_dir}")
        
        return work_dir
    
    def _initialize_step(self) -> StepBase:
        """
        Initialize the step instance.
        
        Returns:
            Step instance
        """
        # Find step class by name (dynamically)
        step_class = None
        step_module_name = self.step_name.lower()
        
        # Import the specific module
        try:
            # Try to import with specific capitalization patterns
            module_candidates = [
                f"dpam.steps.{step_module_name}",
                f"dpam.steps.{step_module_name.replace('_', '')}",
                f"dpam.steps.{step_module_name.replace('_', '.')}"
            ]
            
            for module_name in module_candidates:
                try:
                    module = __import__(module_name, fromlist=[''])
                    # Look for class with pattern StepName, StepNameStep, etc.
                    for attr_name in dir(module):
                        if not attr_name.startswith('_') and attr_name.lower().replace('step', '') == step_module_name.replace('_', '').lower():
                            step_class = getattr(module, attr_name)
                            break
                    
                    if step_class:
                        break
                except ImportError:
                    continue
                
            if not step_class:
                raise ImportError(f"Could not find step class for {self.step_name}")
                
        except ImportError as e:
            logger.error(f"Failed to import step module: {e}")
            raise
        
        # Initialize the step
        return step_class(
            structure_info=self.structure_info,
            data_dir=self.data_dir,
            batch_id=self.batch_id,
            work_dir=self.work_dir,
            threads=self.threads
        )
    
    def execute(self) -> StepResult:
        """
        Execute the step.
        
        Returns:
            Result of step execution
        """
        logger.info(f"Starting execution of step {self.step_name} for structure {self.structure_info['pdb_id']}")
        
        start_time = time.time()
        
        try:
            # Execute the step
            result = self.step_instance.run()
            
            # Record execution time
            execution_time = time.time() - start_time
            if 'metrics' not in result:
                result['metrics'] = {}
            result['metrics']['execution_time'] = execution_time
            
            # Add structure information
            result['structure_id'] = self.structure_info['structure_id']
            result['pdb_id'] = self.structure_info['pdb_id']
            
            # Add step information
            result['step_name'] = self.step_name
            result['started_at'] = datetime.fromtimestamp(start_time).isoformat()
            result['completed_at'] = datetime.fromtimestamp(time.time()).isoformat()
            
            logger.info(f"Step {self.step_name} completed in {execution_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            # Log the error
            logger.error(f"Error executing step {self.step_name}: {e}")
            logger.error(traceback.format_exc())
            
            # Return error result
            return {
                'status': 'FAILED',
                'structure_id': self.structure_info['structure_id'],
                'pdb_id': self.structure_info['pdb_id'],
                'step_name': self.step_name,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'started_at': datetime.fromtimestamp(start_time).isoformat(),
                'completed_at': datetime.fromtimestamp(time.time()).isoformat(),
                'execution_time': time.time() - start_time
            }
    
    def save_result(self, result: StepResult) -> str:
        """
        Save the step result to a file.
        
        Args:
            result: Step execution result
            
        Returns:
            Path to result file
        """
        # Create results directory
        results_dir = os.path.join(
            self.output_dir,
            f"batch_{self.batch_id}",
            "grid_results",
            self.step_name
        )
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Save result
        result_path = os.path.join(
            results_dir,
            f"{self.structure_info['pdb_id']}_{self.step_name}_result.json"
        )
        
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved result to {result_path}")
        return result_path
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        if hasattr(self.step_instance, 'cleanup'):
            self.step_instance.cleanup()
            
        # Clean up working directory if configured
        if os.environ.get('DPAM_CLEAN_WORK_DIR', '').lower() in ('true', '1', 'yes'):
            import shutil
            logger.info(f"Cleaning up working directory: {self.work_dir}")
            shutil.rmtree(self.work_dir, ignore_errors=True)

def main():
    """Main entry point for the worker script."""
    parser = argparse.ArgumentParser(description='DPAM Pipeline Step Executor')
    parser.add_argument('--batch-id', type=int, required=True, help='Batch ID')
    parser.add_argument('--step', type=str, required=True, help='Pipeline step to execute')
    parser.add_argument('--task-id', type=int, required=True, help='Task ID (SGE_TASK_ID)')
    parser.add_argument('--manifest', type=str, required=True, help='Path to batch manifest file')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        # Initialize worker
        worker = DPAMWorker(
            task_id=args.task_id,
            batch_id=args.batch_id,
            step_name=args.step,
            manifest_path=args.manifest,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            threads=args.threads,
            verbose=args.verbose
        )
        
        # Execute step
        result = worker.execute()
        
        # Save result
        worker.save_result(result)
        
        # Clean up
        worker.cleanup()
        
        # Exit with appropriate code
        if result.get('status') == 'FAILED':
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Worker execution failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()