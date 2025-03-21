#!/usr/bin/env python3
"""
Base class for DPAM pipeline steps.

This module defines the abstract base class for all pipeline steps,
providing common functionality and interfaces.
"""

import os
import json
import time
import logging
import tempfile
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union

# Define step result type
StepResult = Dict[str, Any]

class StepBase(ABC):
    """Abstract base class for DPAM pipeline steps"""
    
    def __init__(self, structure_info: Dict[str, Any], data_dir: str, 
                batch_id: int, work_dir: str, threads: int = 1):
        """
        Initialize step with common parameters.
        
        Args:
            structure_info: Dictionary with structure information
            data_dir: Path to data directory
            batch_id: Batch ID
            work_dir: Working directory for temporary files
            threads: Number of threads to use
        """
        self.structure_info = structure_info
        self.structure_id = str(structure_info.get('structure_id', ''))
        self.pdb_id = structure_info.get('pdb_id', '')
        self.data_dir = data_dir
        self.batch_id = batch_id
        self.work_dir = work_dir
        self.threads = threads
        
        # Set up logging
        self.logger = logging.getLogger(f"dpam.steps.{self.__class__.__name__}")
        
        # Ensure working directory exists
        os.makedirs(self.work_dir, exist_ok=True)
    
    @abstractmethod
    def run(self) -> StepResult:
        """
        Run the step.
        
        This method must be implemented by subclasses.
        
        Returns:
            Dictionary with results including paths to output files and status
        """
        pass
    
    def cleanup(self) -> None:
        """
        Clean up temporary files.
        
        This method can be overridden by subclasses if needed.
        """
        pass
    
    def _get_structure_path(self) -> str:
        """
        Get path to input structure file.
        
        Returns:
            Path to structure file
        """
        return self.structure_info.get('structure_path', '')
    
    def _get_output_file(self, extension: str) -> str:
        """
        Generate path for output file in working directory.
        
        Args:
            extension: File extension (including dot)
            
        Returns:
            Path to output file
        """
        return os.path.join(self.work_dir, f"struct_{self.structure_id}{extension}")
    
    def _load_step_params(self) -> Dict[str, Any]:
        """
        Load step-specific parameters from structure record.
        
        Returns:
            Dictionary with parameters or empty dict if none defined
        """
        params = self.structure_info.get('parameters', {})
        step_params = params.get('step_params', {})
        step_name = self.__class__.__name__.lower()
        
        return step_params.get(step_name, {})
    
    def _run_command(self, command: str, timeout: Optional[int] = None, 
                    check: bool = False) -> Tuple[int, str, str]:
        """
        Run shell command and return result.
        
        Args:
            command: Command to run
            timeout: Timeout in seconds (None for no timeout)
            check: Whether to raise exception on non-zero exit code
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        self.logger.debug(f"Running command: {command}")
        
        try:
            process = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=check
            )
            
            return process.returncode, process.stdout, process.stderr
            
        except subprocess.SubprocessError as e:
            self.logger.error(f"Command failed: {e}")
            return 1, "", str(e)
    
    def _copy_to_output(self, source_path: str, destination_path: str) -> None:
        """
        Copy file to output location.
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
        """
        # Create destination directory if needed
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # Copy file
        shutil.copy2(source_path, destination_path)
        self.logger.debug(f"Copied {source_path} to {destination_path}")
    
    def _load_json(self, path: str) -> Dict[str, Any]:
        """
        Load JSON file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            Loaded JSON data as dictionary
        """
        with open(path, 'r') as f:
            return json.load(f)
    
    def _save_json(self, data: Dict[str, Any], path: str) -> None:
        """
        Save data as JSON file.
        
        Args:
            data: Data to save
            path: Output file path
        """
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _measure_time(self, func: callable, *args, **kwargs) -> Tuple[Any, float]:
        """
        Measure execution time of a function.
        
        Args:
            func: Function to call
            args: Positional arguments to function
            kwargs: Keyword arguments to function
            
        Returns:
            Tuple of (function result, execution time in seconds)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        return result, execution_time
    
    def _create_temp_directory(self) -> str:
        """
        Create temporary directory for step processing.
        
        Returns:
            Path to temporary directory
        """
        temp_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_{self.structure_id}_", 
                                   dir=self.work_dir)
        return temp_dir
    
    def _handle_error(self, error: Exception, message: str) -> StepResult:
        """
        Handle step execution error.
        
        Args:
            error: Exception that occurred
            message: Error message prefix
            
        Returns:
            Step result with error status
        """
        error_message = f"{message}: {str(error)}"
        self.logger.error(error_message)
        
        return {
            'status': 'FAILED',
            'error_message': error_message,
            'structure_id': self.structure_id,
            'batch_id': self.batch_id
        }