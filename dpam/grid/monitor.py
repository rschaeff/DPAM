#!/usr/bin/env python3
"""
Grid job monitoring utilities for DPAM pipeline.

This module provides functionality for monitoring and reporting on
the status of jobs submitted to the grid computing environment.
"""

import os
import sys
import json
import time
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

import psycopg2
import psycopg2.extras

class DPAMGridMonitor:
    """Monitors and reports on grid job status for DPAM pipeline"""
    
    def __init__(self, db_config: Dict[str, Any], grid_config: Dict[str, Any], data_dir: str):
        """
        Initialize grid monitor with configuration
        
        Args:
            db_config: Database connection configuration
            grid_config: Grid system configuration
            data_dir: Path to data directory
        """
        self.db_config = db_config
        self.grid_config = grid_config
        self.data_dir = data_dir
        self.logger = logging.getLogger("dpam.grid.monitor")
        
        # Set default polling interval
        self.polling_interval = grid_config.get('polling_interval', 300)  # 5 minutes
        
        # Status mapping from grid to internal representation
        self.status_mapping = {
            'r': 'RUNNING',
            'qw': 'QUEUED',
            'Eqw': 'ERROR',
            't': 'TRANSFERRING',
            'd': 'DELETING',
            'h': 'HELD'
        }
        
        # Terminal states that don't require additional checking
        self.terminal_states = ['COMPLETED', 'FAILED', 'CANCELED']
    
    def get_db_connection(self):
        """Get connection to PostgreSQL database"""
        return psycopg2.connect(**self.db_config)
    
    def get_pending_jobs(self) -> List[Dict[str, Any]]:
        """
        Get all pending grid jobs from database
        
        Returns:
            List of pending job records
        """
        conn = self.get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        try:
            cursor.execute(
                """
                SELECT grid_job_id, batch_id, job_id, step_name, array_size, status, 
                       submitted_at, updated_at
                FROM grid_jobs 
                WHERE status NOT IN %s
                ORDER BY submitted_at
                """,
                (tuple(self.terminal_states),)
            )
            
            jobs = cursor.fetchall()
            return jobs
            
        finally:
            cursor.close()
            conn.close()
    
    def get_batch_jobs(self, batch_id: int) -> List[Dict[str, Any]]:
        """
        Get all grid jobs for a specific batch
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            List of batch job records
        """
        conn = self.get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        try:
            cursor.execute(
                """
                SELECT grid_job_id, batch_id, job_id, step_name, array_size, status, 
                       submitted_at, updated_at, completed_at, task_stats
                FROM grid_jobs 
                WHERE batch_id = %s
                ORDER BY submitted_at
                """,
                (batch_id,)
            )
            
            jobs = cursor.fetchall()
            return jobs
            
        finally:
            cursor.close()
            conn.close()
    
    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check status of a specific grid job
        
        Args:
            job_id: Grid job identifier (e.g., SGE job ID)
            
        Returns:
            Dictionary with job status information
        """
        # First try qstat to see if job is still in the queue
        cmd = f"qstat -j {job_id}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Job is still in the queue, parse status
            return self._parse_qstat_output(result.stdout, job_id)
        else:
            # Job not in queue, check accounting data
            return self._check_job_accounting(job_id)
    
    def _parse_qstat_output(self, output: str, job_id: str) -> Dict[str, Any]:
        """
        Parse qstat output to determine job status
        
        Args:
            output: qstat command output
            job_id: Grid job identifier
            
        Returns:
            Dictionary with job status
        """
        # Initialize counters for array job tasks
        task_stats = {
            'running': 0,
            'queued': 0,
            'held': 0,
            'error': 0,
            'other': 0
        }
        
        # Check if this is an array job
        is_array_job = False
        if '-t' in output:
            is_array_job = True
            
            # For array jobs, we need to run qstat -t to get task status
            cmd = f"qstat -t {job_id}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Process each task status
                for line in result.stdout.strip().split('\n'):
                    if job_id in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            status = parts[4]
                            if status in self.status_mapping:
                                mapped_status = self.status_mapping[status]
                                if mapped_status == 'RUNNING':
                                    task_stats['running'] += 1
                                elif mapped_status == 'QUEUED':
                                    task_stats['queued'] += 1
                                elif mapped_status == 'HELD':
                                    task_stats['held'] += 1
                                elif mapped_status == 'ERROR':
                                    task_stats['error'] += 1
                                else:
                                    task_stats['other'] += 1
        
        # Determine overall status
        overall_status = 'RUNNING'
        if task_stats['running'] == 0 and task_stats['queued'] > 0:
            overall_status = 'QUEUED'
        elif task_stats['error'] > 0:
            overall_status = 'RUNNING_WITH_ERRORS'
        
        return {
            'job_id': job_id,
            'status': overall_status,
            'is_array_job': is_array_job,
            'task_stats': task_stats
        }
    
    def _check_job_accounting(self, job_id: str) -> Dict[str, Any]:
        """
        Check job accounting data to determine final status
        
        Args:
            job_id: Grid job identifier
            
        Returns:
            Dictionary with job status
        """
        cmd = f"qacct -j {job_id}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            # No accounting data available yet
            return {
                'job_id': job_id,
                'status': 'UNKNOWN',
                'task_stats': {},
                'message': 'Job not found in queue or accounting data'
            }
        
        # Parse accounting data
        task_stats = {
            'completed': 0,
            'failed': 0,
            'unknown': 0
        }
        
        task_sections = result.stdout.split("\n=============================================================\n")
        for section in task_sections:
            if not section.strip():
                continue
                
            lines = section.strip().split('\n')
            task_id = None
            exit_status = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('taskid'):
                    task_id = line.split()[1]
                elif line.startswith('exit_status'):
                    exit_status = int(line.split()[1])
            
            if exit_status is not None:
                if exit_status == 0:
                    task_stats['completed'] += 1
                else:
                    task_stats['failed'] += 1
            else:
                task_stats['unknown'] += 1
        
        # Determine overall status
        if task_stats['completed'] > 0 and task_stats['failed'] == 0:
            overall_status = 'COMPLETED'
        elif task_stats['failed'] > 0 and task_stats['completed'] == 0:
            overall_status = 'FAILED'
        elif task_stats['failed'] > 0 and task_stats['completed'] > 0:
            overall_status = 'PARTIALLY_COMPLETED'
        else:
            overall_status = 'UNKNOWN'
        
        return {
            'job_id': job_id,
            'status': overall_status,
            'task_stats': task_stats
        }
    
    def update_job_status_in_db(self, grid_job_id: int, status_info: Dict[str, Any]) -> None:
        """
        Update job status in database
        
        Args:
            grid_job_id: Grid job ID in database
            status_info: Status information from check_job_status
        """
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Update grid job record
            update_fields = [
                "status = %s",
                "updated_at = NOW()",
                "task_stats = %s"
            ]
            
            # Set completion time if job is in terminal state
            if status_info['status'] in self.terminal_states:
                update_fields.append("completed_at = NOW()")
            
            update_sql = f"""
                UPDATE grid_jobs 
                SET {', '.join(update_fields)}
                WHERE grid_job_id = %s
            """
            
            cursor.execute(
                update_sql,
                (
                    status_info['status'],
                    json.dumps(status_info['task_stats']),
                    grid_job_id
                )
            )
            
            # Get batch and step information for this job
            cursor.execute(
                "SELECT batch_id, step_name FROM grid_jobs WHERE grid_job_id = %s",
                (grid_job_id,)
            )
            
            batch_id, step_name = cursor.fetchone()
            
            # If job completed or failed, update batch status
            if status_info['status'] in self.terminal_states:
                cursor.execute(
                    "UPDATE batches SET status = %s WHERE batch_id = %s",
                    (f"GRID_{status_info['status']}_{step_name}", batch_id)
                )
            
            conn.commit()
            self.logger.info(f"Updated status for grid job {grid_job_id} to {status_info['status']}")
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error updating job status: {str(e)}")
            raise
            
        finally:
            cursor.close()
            conn.close()
    
    def monitor_jobs(self, continuous: bool = False, interval: Optional[int] = None) -> None:
        """
        Monitor all pending jobs and update their status
        
        Args:
            continuous: Whether to run in continuous mode
            interval: Polling interval in seconds (overrides default)
        """
        if interval is not None:
            self.polling_interval = interval
            
        self.logger.info(f"Starting job monitoring" + 
                        (f" (continuous mode, interval: {self.polling_interval}s)" if continuous else ""))
        
        try:
            while True:
                # Get pending jobs
                pending_jobs = self.get_pending_jobs()
                self.logger.info(f"Found {len(pending_jobs)} pending jobs")
                
                # Check each job
                for job in pending_jobs:
                    try:
                        job_id = job['job_id']
                        grid_job_id = job['grid_job_id']
                        
                        # Skip jobs updated recently (within polling interval)
                        if job['updated_at'] and continuous:
                            last_update = job['updated_at']
                            if datetime.now() - last_update < timedelta(seconds=self.polling_interval):
                                continue
                        
                        # Check status
                        status_info = self.check_job_status(job_id)
                        
                        # Update in database
                        self.update_job_status_in_db(grid_job_id, status_info)
                        
                    except Exception as e:
                        self.logger.error(f"Error checking job {job['job_id']}: {str(e)}")
                
                # Exit if not in continuous mode
                if not continuous:
                    break
                    
                # Sleep before next poll
                time.sleep(self.polling_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Job monitoring interrupted")
    
    def generate_batch_report(self, batch_id: int) -> Dict[str, Any]:
        """
        Generate a report for a batch
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Dictionary with batch report
        """
        # Get batch information
        conn = self.get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        try:
            # Get batch details
            cursor.execute(
                """
                SELECT b.batch_id, b.name, b.status, b.created_at, b.completed_at, 
                       COUNT(bi.batch_item_id) as total_structures
                FROM batches b
                LEFT JOIN batch_items bi ON b.batch_id = bi.batch_id
                WHERE b.batch_id = %s
                GROUP BY b.batch_id
                """,
                (batch_id,)
            )
            
            batch_info = cursor.fetchone()
            if not batch_info:
                return {"error": f"Batch {batch_id} not found"}
            
            # Get job statistics
            cursor.execute(
                """
                SELECT step_name, status, COUNT(*) as count
                FROM grid_jobs
                WHERE batch_id = %s
                GROUP BY step_name, status
                """,
                (batch_id,)
            )
            
            job_stats = cursor.fetchall()
            
            # Format job statistics
            step_stats = {}
            for stat in job_stats:
                step = stat['step_name']
                status = stat['status']
                count = stat['count']
                
                if step not in step_stats:
                    step_stats[step] = {}
                    
                step_stats[step][status] = count
            
            # Get structure status counts
            cursor.execute(
                """
                SELECT status, COUNT(*) as count
                FROM batch_items
                WHERE batch_id = %s
                GROUP BY status
                """,
                (batch_id,)
            )
            
            structure_stats = cursor.fetchall()
            
            # Format structure statistics
            structure_status = {}
            for stat in structure_stats:
                structure_status[stat['status']] = stat['count']
            
            # Calculate execution time
            execution_time = None
            if batch_info['created_at'] and batch_info['completed_at']:
                execution_time = (batch_info['completed_at'] - batch_info['created_at']).total_seconds() / 3600.0
                
            # Build report
            report = {
                "batch_id": batch_id,
                "name": batch_info['name'],
                "status": batch_info['status'],
                "total_structures": batch_info['total_structures'],
                "created_at": batch_info['created_at'].strftime('%Y-%m-%d %H:%M:%S') if batch_info['created_at'] else None,
                "completed_at": batch_info['completed_at'].strftime('%Y-%m-%d %H:%M:%S') if batch_info['completed_at'] else None,
                "execution_time_hours": round(execution_time, 2) if execution_time else None,
                "step_statistics": step_stats,
                "structure_statistics": structure_status
            }
            
            return report
            
        finally:
            cursor.close()
            conn.close()
    
    def get_job_details(self, grid_job_id: int) -> Dict[str, Any]:
        """
        Get detailed information for a specific job
        
        Args:
            grid_job_id: Grid job ID in database
            
        Returns:
            Dictionary with job details
        """
        conn = self.get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        try:
            # Get job details
            cursor.execute(
                """
                SELECT grid_job_id, batch_id, job_id, step_name, array_size, status, 
                       submitted_at, updated_at, completed_at, task_stats
                FROM grid_jobs 
                WHERE grid_job_id = %s
                """,
                (grid_job_id,)
            )
            
            job_info = cursor.fetchone()
            if not job_info:
                return {"error": f"Job {grid_job_id} not found"}
            
            # Check current status in grid
            current_status = self.check_job_status(job_info['job_id'])
            
            # Format timestamps
            for key in ['submitted_at', 'updated_at', 'completed_at']:
                if job_info[key]:
                    job_info[key] = job_info[key].strftime('%Y-%m-%d %H:%M:%S')
            
            # Combine information
            job_details = dict(job_info)
            job_details['current_status'] = current_status
            
            # Get log file path if available
            log_dir = self.grid_config.get('log_dir', '/var/log/dpam')
            log_pattern = f"{log_dir}/dpam_{job_info['batch_id']}_{job_info['step_name']}.*"
            
            cmd = f"ls -1 {log_pattern} 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                log_files = result.stdout.strip().split('\n')
                job_details['log_files'] = log_files
            
            return job_details
            
        finally:
            cursor.close()
            conn.close()

def main():
    """Command-line entry point"""
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='DPAM Grid Monitor')
    parser.add_argument('--config', help='Path to configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor grid jobs')
    monitor_parser.add_argument('--continuous', action='store_true', help='Run in continuous mode')
    monitor_parser.add_argument('--interval', type=int, help='Polling interval in seconds')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate batch report')
    report_parser.add_argument('--batch-id', type=int, required=True, help='Batch ID')
    report_parser.add_argument('--output', help='Output file for report')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check job status')
    status_parser.add_argument('--job-id', help='Grid job ID')
    status_parser.add_argument('--batch-id', type=int, help='Batch ID')
    
    args = parser.parse_args()
    
    # Load configuration
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from dpam.config import load_config
    
    config = load_config(args.config)
    
    # Create monitor
    monitor = DPAMGridMonitor(
        config.get_db_config(),
        config.get('grid', {}),
        config.get('data_dir', '/data/dpam')
    )
    
    # Execute command
    if args.command == 'monitor':
        monitor.monitor_jobs(
            continuous=args.continuous,
            interval=args.interval
        )
    elif args.command == 'report':
        report = monitor.generate_batch_report(args.batch_id)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {args.output}")
        else:
            print(json.dumps(report, indent=2))
    elif args.command == 'status':
        if args.job_id:
            # Get job details from database
            job_details = monitor.get_job_details(int(args.job_id))
            print(json.dumps(job_details, indent=2))
        elif args.batch_id:
            # Get all jobs for batch
            jobs = monitor.get_batch_jobs(args.batch_id)
            print(json.dumps(jobs, indent=2))
        else:
            parser.error("Either --job-id or --batch-id is required")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()