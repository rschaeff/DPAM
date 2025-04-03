#!/usr/bin/env python3
"""
Custom exception classes for DPAM pipeline.

This module defines custom exception classes used throughout the DPAM pipeline
to provide more specific error handling and context.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union

# Configure logging
logger = logging.getLogger("dpam.exceptions")

class DPAMError(Exception):
    """Base class for all DPAM exceptions."""
    
    def __init__(self, message: str, *args, **kwargs):
        """
        Initialize exception.
        
        Args:
            message: Error message
            args: Additional positional arguments
            kwargs: Additional keyword arguments
        """
        self.message = message
        super().__init__(message, *args)
        
        # Log the exception at creation time
        logger.error(f"{self.__class__.__name__}: {message}")

# Database Exceptions
class DatabaseError(DPAMError):
    """Base class for database-related exceptions."""
    pass

class DatabaseConnectionError(DatabaseError):
    """Exception raised when database connection fails."""
    pass

class DatabaseQueryError(DatabaseError):
    """Exception raised when database query fails."""
    
    def __init__(self, message: str, query: Optional[str] = None, 
                params: Optional[Union[Tuple, Dict]] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            query: SQL query that failed
            params: Query parameters
        """
        self.query = query
        self.params = params
        super().__init__(message)

class DatabaseTransactionError(DatabaseError):
    """Exception raised when database transaction fails."""
    pass

class DatabasePoolError(DatabaseError):
    """Exception raised when connection pool operations fail."""
    pass

# Pipeline Exceptions
class PipelineError(DPAMError):
    """Base class for pipeline-related exceptions."""
    pass

class PipelineConfigError(PipelineError):
    """Exception raised when pipeline configuration is invalid."""
    pass

class PipelineExecutionError(PipelineError):
    """Exception raised when pipeline execution fails."""
    
    def __init__(self, message: str, batch_id: Optional[int] = None,
                step: Optional[str] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            batch_id: Batch ID
            step: Pipeline step that failed
        """
        self.batch_id = batch_id
        self.step = step
        super().__init__(message)

class CheckpointError(PipelineError):
    """Exception raised when checkpoint operations fail."""
    pass

class RecoveryError(PipelineError):
    """Exception raised when recovery operations fail."""
    pass

# Batch Exceptions
class BatchError(DPAMError):
    """Base class for batch-related exceptions."""
    pass

class BatchCreationError(BatchError):
    """Exception raised when batch creation fails."""
    pass

class BatchPreparationError(BatchError):
    """Exception raised when batch preparation fails."""
    pass

class BatchProcessingError(BatchError):
    """Exception raised when batch processing fails."""
    
    def __init__(self, message: str, batch_id: Optional[int] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            batch_id: Batch ID
        """
        self.batch_id = batch_id
        super().__init__(message)

# Grid Exceptions
class GridError(DPAMError):
    """Base class for grid-related exceptions."""
    pass

class GridSubmissionError(GridError):
    """Exception raised when job submission to grid fails."""
    
    def __init__(self, message: str, command: Optional[str] = None,
                stderr: Optional[str] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            command: Command that failed
            stderr: Standard error output
        """
        self.command = command
        self.stderr = stderr
        super().__init__(message)

class GridMonitoringError(GridError):
    """Exception raised when job monitoring fails."""
    pass

class GridJobError(GridError):
    """Exception raised when grid job fails."""
    
    def __init__(self, message: str, job_id: Optional[str] = None,
                task_id: Optional[int] = None, exit_code: Optional[int] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            job_id: Grid job ID
            task_id: Grid task ID
            exit_code: Job exit code
        """
        self.job_id = job_id
        self.task_id = task_id
        self.exit_code = exit_code
        super().__init__(message)

# Step Exceptions
class StepError(DPAMError):
    """Base class for step-related exceptions."""
    pass

class StepInitializationError(StepError):
    """Exception raised when step initialization fails."""
    
    def __init__(self, message: str, step_name: Optional[str] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            step_name: Step name
        """
        self.step_name = step_name
        super().__init__(message)

class StepExecutionError(StepError):
    """Exception raised when step execution fails."""
    
    def __init__(self, message: str, step_name: Optional[str] = None,
                structure_id: Optional[str] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            step_name: Step name
            structure_id: Structure ID
        """
        self.step_name = step_name
        self.structure_id = structure_id
        super().__init__(message)

class ToolExecutionError(StepError):
    """Exception raised when external tool execution fails."""
    
    def __init__(self, message: str, tool: Optional[str] = None,
                command: Optional[str] = None, exit_code: Optional[int] = None,
                stderr: Optional[str] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            tool: Tool name
            command: Command that failed
            exit_code: Tool exit code
            stderr: Standard error output
        """
        self.tool = tool
        self.command = command
        self.exit_code = exit_code
        self.stderr = stderr
        super().__init__(message)

# Structure Exceptions
class StructureError(DPAMError):
    """Base class for structure-related exceptions."""
    pass

class StructureReadError(StructureError):
    """Exception raised when structure reading fails."""
    
    def __init__(self, message: str, file_path: Optional[str] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            file_path: Structure file path
        """
        self.file_path = file_path
        super().__init__(message)

class StructureValidationError(StructureError):
    """Exception raised when structure validation fails."""
    
    def __init__(self, message: str, file_path: Optional[str] = None,
                validation_issues: Optional[List[str]] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            file_path: Structure file path
            validation_issues: List of validation issues
        """
        self.file_path = file_path
        self.validation_issues = validation_issues or []
        super().__init__(message)

class StructureProcessingError(StructureError):
    """Exception raised when structure processing fails."""
    pass

# Domain Exceptions
class DomainError(DPAMError):
    """Base class for domain-related exceptions."""
    pass

class DomainIdentificationError(DomainError):
    """Exception raised when domain identification fails."""
    pass

class DomainRefinementError(DomainError):
    """Exception raised when domain refinement fails."""
    pass

# API Exceptions
class APIError(DPAMError):
    """Base class for API-related exceptions."""
    pass

class APIRequestError(APIError):
    """Exception raised when API request fails."""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            status_code: HTTP status code
        """
        self.status_code = status_code
        super().__init__(message)

class APIAuthenticationError(APIError):
    """Exception raised when API authentication fails."""
    pass

class APIRateLimitError(APIError):
    """Exception raised when API rate limit is exceeded."""
    pass

# Utility Exceptions
class ConfigError(DPAMError):
    """Exception raised when configuration is invalid."""
    
    def __init__(self, message: str, missing_keys: Optional[List[str]] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            missing_keys: List of missing configuration keys
        """
        self.missing_keys = missing_keys or []
        super().__init__(message)

class FileError(DPAMError):
    """Exception raised when file operations fail."""
    
    def __init__(self, message: str, file_path: Optional[str] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            file_path: File path
        """
        self.file_path = file_path
        super().__init__(message)

class NetworkError(DPAMError):
    """Exception raised when network operations fail."""
    
    def __init__(self, message: str, url: Optional[str] = None,
                status_code: Optional[int] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            url: URL that failed
            status_code: HTTP status code
        """
        self.url = url
        self.status_code = status_code
        super().__init__(message)

class TimeoutError(DPAMError):
    """Exception raised when operation times out."""
    
    def __init__(self, message: str, operation: Optional[str] = None,
                timeout: Optional[float] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            operation: Operation that timed out
            timeout: Timeout value in seconds
        """
        self.operation = operation
        self.timeout = timeout
        super().__init__(message)