#!/usr/bin/env python3
"""
DPAM Manager - Command-line tool for managing the DPAM pipeline.

This script provides a command-line interface for creating, managing,
and monitoring batches in the DPAM domain prediction pipeline.
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dpam.config import load_config, ConfigManager
from dpam.database import DatabaseManager
from dpam.batch.manager import DPAMBatchManager
from dpam.batch.preparation import DPAMBatchPreparation
from dpam.batch.supplement import DPAMBatchSupplementation
from dpam.pipeline.controller import DPAMPipelineController
from dpam.pipeline.checkpoints import DPAMBatchCheckpoints
from dpam.pipeline.quality import DPAMQualityControl
from dpam.pipeline.errors import DPAMErrorHandler
from dpam.grid.manager import DPAMOpenGridManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dpam.manager')

def setup_argparse() -> argparse.ArgumentParser:
    """Set up argument parser with subcommands"""
    parser = argparse.ArgumentParser(
        description='DPAM Pipeline Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new batch from a file of UniProt accessions
  dpam-manager batch create --name "Test Batch" --accessions accessions.txt
  
  # Prepare batch for processing
  dpam-manager batch prepare --batch-id 123
  
  # Start pipeline for a batch
  dpam-manager pipeline start --batch-id 123
  
  # Monitor pipeline progress
  dpam-manager pipeline status --batch-id 123
  
  # List all batches
  dpam-manager batch list
  
  # View batch details
  dpam-manager batch view --batch-id 123
  
  # Generate batch report
  dpam-manager batch report --batch-id 123
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Batch commands
    batch_parser = subparsers.add_parser('batch', help='Batch management commands')
    batch_subparsers = batch_parser.add_subparsers(dest='subcommand', help='Batch subcommand')
    
    # batch create
    create_parser = batch_subparsers.add_parser('create', help='Create a new batch')
    create_parser.add_argument('--name', required=True, help='Batch name')
    create_parser.add_argument('--description', help='Batch description')
    group = create_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--accessions', help='File with UniProt accessions (one per line)')
    group.add_argument('--accession-list', help='Comma-separated list of UniProt accessions')
    create_parser.add_argument('--config', help='Path to configuration file')
    
    # batch prepare
    prepare_parser = batch_subparsers.add_parser('prepare', help='Prepare batch for processing')
    prepare_parser.add_argument('--batch-id', required=True, type=int, help='Batch ID')
    
    # batch supplement
    supplement_parser = batch_subparsers.add_parser('supplement', help='Add supplementary data to batch')
    supplement_parser.add_argument('--batch-id', required=True, type=int, help='Batch ID')
    supplement_parser.add_argument('--pae', action='store_true', help='Fetch PAE files')
    
    # batch list
    list_parser = batch_subparsers.add_parser('list', help='List all batches')
    list_parser.add_argument('--status', help='Filter by status')
    list_parser.add_argument('--limit', type=int, default=20, help='Maximum number of batches to show')
    
    # batch view
    view_parser = batch_subparsers.add_parser('view', help='View batch details')
    view_parser.add_argument('--batch-id', required=True, type=int, help='Batch ID')
    
    # batch report
    report_parser = batch_subparsers.add_parser('report', help='Generate batch report')
    report_parser.add_argument('--batch-id', required=True, type=int, help='Batch ID')
    report_parser.add_argument('--output', help='Output directory for report files')
    
    # batch delete
    delete_parser = batch_subparsers.add_parser('delete', help='Delete batch')
    delete_parser.add_argument('--batch-id', required=True, type=int, help='Batch ID')
    delete_parser.add_argument('--force', action='store_true', help='Force deletion without confirmation')
    
    # Pipeline commands
    pipeline_parser = subparsers.add_parser('pipeline', help='Pipeline execution commands')
    pipeline_subparsers = pipeline_parser.add_subparsers(dest='subcommand', help='Pipeline subcommand')
    
    # pipeline prepare
    pipeline_prepare_parser = pipeline_subparsers.add_parser('prepare', help='Prepare pipeline for batch')
    pipeline_prepare_parser.add_argument('--batch-id', required=True, type=int, help='Batch ID')
    
    # pipeline start
    start_parser = pipeline_subparsers.add_parser('start', help='Start pipeline execution')
    start_parser.add_argument('--batch-id', required=True, type=int, help='Batch ID')
    start_parser.add_argument('--start-from', help='Step to start from')
    start_parser.add_argument('--restart', action='store_true', help='Restart pipeline')
    
    # pipeline status
    status_parser = pipeline_subparsers.add_parser('status', help='Check pipeline status')
    status_parser.add_argument('--batch-id', required=True, type=int, help='Batch ID')
    
    # pipeline step
    step_parser = pipeline_subparsers.add_parser('step', help='Run specific pipeline step')
    step_parser.add_argument('--batch-id', required=True, type=int, help='Batch ID')
    step_parser.add_argument('--step', required=True, help='Step name')
    step_parser.add_argument('--structure-id', required=True, help='Structure ID')
    step_parser.add_argument('--local', action='store_true', help='Run locally')
    
    # pipeline recover
    recover_parser = pipeline_subparsers.add_parser('recover', help='Recover failed pipeline')
    recover_parser.add_argument('--batch-id', required=True, type=int, help='Batch ID')
    
    # Grid commands
    grid_parser = subparsers.add_parser('grid', help='Grid management commands')
    grid_subparsers = grid_parser.add_subparsers(dest='subcommand', help='Grid subcommand')
    
    # grid status
    grid_status_parser = grid_subparsers.add_parser('status', help='Check grid job status')
    grid_status_parser.add_argument('--job-id', help='Grid job ID')
    grid_status_parser.add_argument('--batch-id', type=int, help='Batch ID')
    
    # grid list
    grid_list_parser = grid_subparsers.add_parser('list', help='List grid jobs')
    grid_list_parser.add_argument('--batch-id', type=int, help='Filter by batch ID')
    grid_list_parser.add_argument('--status', help='Filter by status')
    grid_list_parser.add_argument('--limit', type=int, default=20, help='Maximum number of jobs to show')
    
    # Config commands
    config_parser = subparsers.add_parser('config', help='Configuration commands')
    config_subparsers = config_parser.add_subparsers(dest='subcommand', help='Config subcommand')
    
    # config show
    config_show_parser = config_subparsers.add_parser('show', help='Show configuration')
    config_show_parser.add_argument('--section', help='Configuration section to show')
    
    # config validate
    config_validate_parser = config_subparsers.add_parser('validate', help='Validate configuration')
    config_validate_parser.add_argument('--config', help='Path to configuration file')
    
    # Version command
    parser.add_argument('--version', action='store_true', help='Show version and exit')
    
    return parser

def load_configuration(args: argparse.Namespace) -> ConfigManager:
    """Load configuration from file"""
    config_path = None
    if hasattr(args, 'config') and args.config:
        config_path = args.config
    
    return load_config(config_path)

def create_batch(args: argparse.Namespace, config: ConfigManager) -> None:
    """Create a new batch from accessions"""
    # Get accessions from file or argument
    accessions = []
    if args.accessions:
        with open(args.accessions, 'r') as f:
            accessions = [line.strip() for line in f if line.strip()]
    elif args.accession_list:
        accessions = [acc.strip() for acc in args.accession_list.split(',') if acc.strip()]
    
    if not accessions:
        logger.error("No accessions provided")
        sys.exit(1)
    
    # Create batch manager
    batch_manager = DPAMBatchManager(
        config.get_db_config(),
        {'api_base': config.get('api.base_url')}
    )
    
    # Create batch
    try:
        batch_id = batch_manager.create_batch_from_accessions(
            accessions,
            batch_name=args.name,
            description=args.description
        )
        
        print(f"Created batch {batch_id} with {len(accessions)} structures")
        
    except Exception as e:
        logger.error(f"Failed to create batch: {e}")
        sys.exit(1)

def prepare_batch(args: argparse.Namespace, config: ConfigManager) -> None:
    """Prepare batch for processing"""
    batch_preparer = DPAMBatchPreparation(
        config.get_db_config(),
        config.get('batch_dir')
    )
    
    try:
        result = batch_preparer.prepare_batch_directory(args.batch_id)
        
        print(f"Prepared batch {args.batch_id}")
        print(f"Batch path: {result['batch_path']}")
        print(f"Status: {result['status']}")
        print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
        
    except Exception as e:
        logger.error(f"Failed to prepare batch: {e}")
        sys.exit(1)

def supplement_batch(args: argparse.Namespace, config: ConfigManager) -> None:
    """Add supplementary data to batch"""
    batch_supplementer = DPAMBatchSupplementation(config.get_db_config())
    
    try:
        if args.pae:
            result = batch_supplementer.fetch_pae_files(args.batch_id)
            
            print(f"Supplemented batch {args.batch_id} with PAE files")
            print(f"Status: {result['status']}")
            print(f"Metrics: {json.dumps(result.get('metrics', {}), indent=2)}")
        else:
            print("No supplementary data specified")
            
    except Exception as e:
        logger.error(f"Failed to supplement batch: {e}")
        sys.exit(1)

def list_batches(args: argparse.Namespace, config: ConfigManager) -> None:
    """List all batches"""
    db = DatabaseManager(config.get_db_config())
    
    try:
        # Build query
        query = """
            SELECT b.batch_id, b.name, b.status, b.created_at, b.completed_at,
                   COUNT(bi.batch_item_id) as structure_count
            FROM batches b
            LEFT JOIN batch_items bi ON b.batch_id = bi.batch_id
        """
        params = []
        
        # Add status filter if specified
        if args.status:
            query += " WHERE b.status = %s"
            params.append(args.status)
        
        # Group by and order
        query += """
            GROUP BY b.batch_id, b.name, b.status, b.created_at, b.completed_at
            ORDER BY b.created_at DESC
            LIMIT %s
        """
        params.append(args.limit)
        
        # Execute query
        batches = db.fetchall(query, tuple(params))
        
        # Print results
        if not batches:
            print("No batches found")
            return
        
        print("\nBatches:")
        print(f"{'ID':<8} {'Name':<20} {'Status':<20} {'Created':<20} {'Structures':<10}")
        print("-" * 80)
        
        for batch in batches:
            created_at = batch['created_at'].strftime('%Y-%m-%d %H:%M') if batch['created_at'] else "-"
            print(f"{batch['batch_id']:<8} {batch['name'][:18]:<20} {batch['status'][:18]:<20} "
                  f"{created_at:<20} {batch['structure_count']:<10}")
        
        print()
        
    except Exception as e:
        logger.error(f"Failed to list batches: {e}")
        sys.exit(1)
    finally:
        db.close()

def view_batch(args: argparse.Namespace, config: ConfigManager) -> None:
    """View batch details"""
    db = DatabaseManager(config.get_db_config())
    
    try:
        # Get batch information
        batch = db.fetchone(
            """
            SELECT b.*, COUNT(bi.batch_item_id) as structure_count
            FROM batches b
            LEFT JOIN batch_items bi ON b.batch_id = bi.batch_id
            WHERE b.batch_id = %s
            GROUP BY b.batch_id
            """,
            (args.batch_id,)
        )
        
        if not batch:
            print(f"Batch {args.batch_id} not found")
            return
        
        # Get batch item status counts
        status_counts = db.fetchall(
            """
            SELECT status, COUNT(*) as count
            FROM batch_items
            WHERE batch_id = %s
            GROUP BY status
            """,
            (args.batch_id,)
        )
        
        # Print batch information
        print("\nBatch Information:")
        print(f"ID:          {batch['batch_id']}")
        print(f"Name:        {batch['name']}")
        print(f"Description: {batch.get('description', '')}")
        print(f"Status:      {batch['status']}")
        print(f"Created:     {batch['created_at'].strftime('%Y-%m-%d %H:%M') if batch['created_at'] else '-'}")
        print(f"Completed:   {batch['completed_at'].strftime('%Y-%m-%d %H:%M') if batch['completed_at'] else '-'}")
        print(f"Structures:  {batch['structure_count']}")
        
        # Print status counts
        print("\nStructure Status:")
        for status in status_counts:
            print(f"  {status['status']}: {status['count']}")
        
        # Get batch parameters
        params = batch.get('parameters', {})
        if params:
            if isinstance(params, str):
                params = json.loads(params)
                
            # Print batch path if available
            if 'batch_path' in params:
                print(f"\nBatch Path: {params['batch_path']}")
            
            # Print metrics if available
            if 'metrics' in params:
                print("\nMetrics:")
                for key, value in params['metrics'].items():
                    print(f"  {key}: {value}")
            
            # Print analysis summary if available
            if 'analysis_summary' in params:
                summary = params['analysis_summary']
                print("\nAnalysis Summary:")
                print(f"  Total Structures: {summary.get('total_structures', 0)}")
                print(f"  With Domains:    {summary.get('structures_with_domains', 0)}")
                print(f"  Avg Domains/Structure: {summary.get('avg_domains_per_structure', 0):.2f}")
        
        print()
        
    except Exception as e:
        logger.error(f"Failed to view batch: {e}")
        sys.exit(1)
    finally:
        db.close()

def generate_batch_report(args: argparse.Namespace, config: ConfigManager) -> None:
    """Generate batch report"""
    quality_control = DPAMQualityControl(config.get_db_config())
    
    try:
        result = quality_control.generate_batch_report(args.batch_id)
        
        if result['success']:
            print(f"Generated report for batch {args.batch_id}")
            print(f"Report path: {result['report_path']}")
            print(f"Domains report path: {result['domains_report_path']}")
            
            # Copy to output directory if specified
            if args.output:
                import shutil
                os.makedirs(args.output, exist_ok=True)
                
                report_name = os.path.basename(result['report_path'])
                domains_name = os.path.basename(result['domains_report_path'])
                
                shutil.copy2(result['report_path'], os.path.join(args.output, report_name))
                shutil.copy2(result['domains_report_path'], os.path.join(args.output, domains_name))
                
                print(f"Copied reports to {args.output}")
        else:
            print(f"Failed to generate report: {result['message']}")
            
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        sys.exit(1)

def delete_batch(args: argparse.Namespace, config: ConfigManager) -> None:
    """Delete batch"""
    db = DatabaseManager(config.get_db_config())
    
    try:
        # Get batch information
        batch = db.fetchone(
            "SELECT name FROM batches WHERE batch_id = %s",
            (args.batch_id,)
        )
        
        if not batch:
            print(f"Batch {args.batch_id} not found")
            return
        
        # Confirm deletion
        if not args.force:
            confirm = input(f"Are you sure you want to delete batch {args.batch_id} ({batch['name']})? [y/N] ")
            if confirm.lower() != 'y':
                print("Deletion cancelled")
                return
        
        # Delete batch
        with db.transaction() as conn:
            cursor = conn.cursor()
            
            # Delete batch items first
            cursor.execute(
                "DELETE FROM batch_items WHERE batch_id = %s",
                (args.batch_id,)
            )
            
            # Delete batch
            cursor.execute(
                "DELETE FROM batches WHERE batch_id = %s",
                (args.batch_id,)
            )
        
        print(f"Deleted batch {args.batch_id}")
        
    except Exception as e:
        logger.error(f"Failed to delete batch: {e}")
        sys.exit(1)
    finally:
        db.close()

def prepare_pipeline(args: argparse.Namespace, config: ConfigManager) -> None:
    """Prepare pipeline for batch"""
    controller = DPAMPipelineController(
        config.get_db_config(),
        config.get('grid', {}),
        config.get('data_dir')
    )
    
    try:
        result = controller.prepare_batch_pipeline(args.batch_id)
        
        if result['success']:
            print(f"Prepared pipeline for batch {args.batch_id}")
            print(f"Grid configuration: {json.dumps(result['grid_config'], indent=2)}")
            print(f"Pipeline steps: {', '.join(result['steps'])}")
        else:
            print(f"Failed to prepare pipeline: {result['message']}")
            
    except Exception as e:
        logger.error(f"Failed to prepare pipeline: {e}")
        sys.exit(1)

def start_pipeline(args: argparse.Namespace, config: ConfigManager) -> None:
    """Start pipeline execution"""
    controller = DPAMPipelineController(
        config.get_db_config(),
        config.get('grid', {}),
        config.get('data_dir')
    )
    
    try:
        result = controller.start_pipeline(
            args.batch_id,
            start_from=args.start_from,
            restart=args.restart
        )
        
        if result['success']:
            print(f"Started pipeline for batch {args.batch_id}")
            print(f"Step: {result['step']}")
            print(f"Job ID: {result['job_id']}")
            print(f"Grid Job ID: {result['grid_job_id']}")
        else:
            print(f"Failed to start pipeline: {result['message']}")
            
    except Exception as e:
        logger.error(f"Failed to start pipeline: {e}")
        sys.exit(1)

def check_pipeline_status(args: argparse.Namespace, config: ConfigManager) -> None:
    """Check pipeline status"""
    controller = DPAMPipelineController(
        config.get_db_config(),
        config.get('grid', {}),
        config.get('data_dir')
    )
    
    try:
        result = controller.get_pipeline_status(args.batch_id)
        
        if result['success']:
            print(f"\nPipeline Status for Batch {args.batch_id}:")
            print(f"Status: {result['status']}")
            print(f"Progress: {result.get('progress_percent', 0):.1f}%")
            
            if 'current_step' in result:
                print(f"Current Step: {result['current_step']} "
                      f"({result['current_step_index'] + 1}/{result['total_steps']})")
            
            if 'current_job' in result:
                job = result['current_job']
                print(f"Current Job: {job.get('job_id', '-')}")
                print(f"Submitted At: {job.get('submitted_at', '-')}")
            
            # Print checkpoints
            if 'checkpoints' in result and result['checkpoints']:
                print("\nCheckpoints:")
                for cp in result['checkpoints']:
                    print(f"  {cp['step']}: {cp['status']} ({cp['timestamp']})")
            
            # Print step status
            if 'step_status' in result:
                print("\nStep Status:")
                for step, status in result['step_status'].items():
                    status_str = ", ".join([f"{s}: {c}" for s, c in status.items()])
                    print(f"  {step}: {status_str}")
            
        else:
            print(f"Failed to get pipeline status: {result['message']}")
            
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {e}")
        sys.exit(1)

def run_pipeline_step(args: argparse.Namespace, config: ConfigManager) -> None:
    """Run specific pipeline step"""
    controller = DPAMPipelineController(
        config.get_db_config(),
        config.get('grid', {}),
        config.get('data_dir')
    )
    
    try:
        if args.local:
            result = controller.run_step_locally(
                args.batch_id,
                args.step,
                args.structure_id
            )
            
            if result['success']:
                print(f"Step {args.step} completed for structure {args.structure_id}")
                print(f"Result: {json.dumps(result['result'], indent=2)}")
            else:
                print(f"Failed to run step: {result['message']}")
        else:
            # Submit to grid
            grid_manager = DPAMOpenGridManager(
                config.get_db_config(),
                config.get('grid', {}),
                config.get('data_dir')
            )
            
            # Create structure-specific manifest
            manifest_path = f"/tmp/dpam_manifest_{args.batch_id}_{args.structure_id}.json"
            with open(manifest_path, 'w') as f:
                json.dump({
                    'batch_id': args.batch_id,
                    'structures': {
                        args.structure_id: {
                            'structure_id': args.structure_id
                        }
                    }
                }, f)
            
            # Submit job
            job_script = grid_manager._create_job_script(
                args.batch_id,
                args.step,
                manifest_path
            )
            
            job_path = f"/tmp/dpam_job_{args.batch_id}_{args.structure_id}_{args.step}.sh"
            with open(job_path, 'w') as f:
                f.write(job_script)
            
            print(f"Job script written to {job_path}")
            print(f"To run: qsub -t 1 {job_path}")
            
    except Exception as e:
        logger.error(f"Failed to run step: {e}")
        sys.exit(1)

def recover_pipeline(args: argparse.Namespace, config: ConfigManager) -> None:
    """Recover failed pipeline"""
    controller = DPAMPipelineController(
        config.get_db_config(),
        config.get('grid', {}),
        config.get('data_dir')
    )
    
    try:
        result = controller.recover_failed_pipeline(args.batch_id)
        
        if result['success']:
            print(f"Recovery initiated for batch {args.batch_id}")
            if 'recovery_plans' in result:
                print(f"Recovery plans: {len(result['recovery_plans'])}")
                for i, plan in enumerate(result['recovery_plans'][:5], 1):
                    print(f"  {i}. Structure {plan['structure_id']}: {plan['strategy']}")
                if len(result['recovery_plans']) > 5:
                    print(f"  ... and {len(result['recovery_plans']) - 5} more")
            else:
                print(f"Skipped failed structures and advanced to: {result.get('message', '')}")
        else:
            print(f"Failed to recover pipeline: {result['message']}")
            
    except Exception as e:
        logger.error(f"Failed to recover pipeline: {e}")
        sys.exit(1)

def check_grid_status(args: argparse.Namespace, config: ConfigManager) -> None:
    """Check grid job status"""
    grid_manager = DPAMOpenGridManager(
        config.get_db_config(),
        config.get('grid', {}),
        config.get('data_dir')
    )
    
    try:
        if args.job_id:
            # Get job by ID
            db = DatabaseManager(config.get_db_config())
            grid_job = db.fetchone(
                "SELECT grid_job_id FROM grid_jobs WHERE job_id = %s",
                (args.job_id,)
            )
            
            if not grid_job:
                print(f"Job {args.job_id} not found")
                return
                
            grid_job_id = grid_job['grid_job_id']
            result = grid_manager.check_job_status(grid_job_id)
            
            print(f"\nStatus for Job {args.job_id}:")
            print(f"Status: {result['status']}")
            print(f"Task Stats: {json.dumps(result['task_stats'], indent=2)}")
            
        elif args.batch_id:
            # Get all jobs for batch
            results = grid_manager.check_all_pending_jobs()
            
            # Filter by batch ID
            db = DatabaseManager(config.get_db_config())
            batch_jobs = db.fetchall(
                "SELECT grid_job_id, job_id, step_name, status FROM grid_jobs WHERE batch_id = %s",
                (args.batch_id,)
            )
            
            print(f"\nJobs for Batch {args.batch_id}:")
            if not batch_jobs:
                print("No jobs found")
                return
                
            for job in batch_jobs:
                grid_job_id = job['grid_job_id']
                job_status = job['status']
                
                # Check status if job is in results
                if grid_job_id in results:
                    job_status = results[grid_job_id]['status']
                    stats = results[grid_job_id]['task_stats']
                    stats_str = ", ".join([f"{k}: {v}" for k, v in stats.items()])
                else:
                    stats_str = "N/A"
                
                print(f"  {job['job_id']} ({job['step_name']}): {job_status} - {stats_str}")
                
        else:
            print("Either --job-id or --batch-id must be specified")
            
    except Exception as e:
        logger.error(f"Failed to check grid status: {e}")
        sys.exit(1)

def list_grid_jobs(args: argparse.Namespace, config: ConfigManager) -> None:
    """List grid jobs"""
    db = DatabaseManager(config.get_db_config())
    
    try:
        # Build query
        query = """
            SELECT gj.grid_job_id, gj.batch_id, gj.job_id, gj.step_name, 
                   gj.status, gj.submitted_at, gj.array_size
            FROM grid_jobs gj
        """
        params = []
        
        # Add filters
        filters = []
        
        if args.batch_id:
            filters.append("gj.batch_id = %s")
            params.append(args.batch_id)
        
        if args.status:
            filters.append("gj.status = %s")
            params.append(args.status)
        
        if filters:
            query += " WHERE " + " AND ".join(filters)
        
        # Add order and limit
        query += """
            ORDER BY gj.submitted_at DESC
            LIMIT %s
        """
        params.append(args.limit)
        
        # Execute query
        jobs = db.fetchall(query, tuple(params))
        
        # Print results
        if not jobs:
            print("No grid jobs found")
            return
        
        print("\nGrid Jobs:")
        print(f"{'ID':<8} {'Batch':<8} {'Job ID':<12} {'Step':<20} {'Status':<12} {'Submitted':<20} {'Tasks':<6}")
        print("-" * 90)
        
        for job in jobs:
            submitted_at = job['submitted_at'].strftime('%Y-%m-%d %H:%M') if job['submitted_at'] else "-"
            job_id_short = job['job_id'][:10] if job['job_id'] else "-"
            print(f"{job['grid_job_id']:<8} {job['batch_id']:<8} {job_id_short:<12} "
                  f"{job['step_name'][:18]:<20} {job['status'][:10]:<12} {submitted_at:<20} {job['array_size']:<6}")
        
        print()
        
    except Exception as e:
        logger.error(f"Failed to list grid jobs: {e}")
        sys.exit(1)
    finally:
        db.close()

def show_config(args: argparse.Namespace, config: ConfigManager) -> None:
    """Show configuration"""
    if args.section:
        section_parts = args.section.split('.')
        value = config.config
        
        for part in section_parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                print(f"Section '{args.section}' not found in configuration")
                return
        
        print(f"\nConfiguration Section: {args.section}")
        print(json.dumps(value, indent=2))
        
    else:
        print("\nCurrent Configuration:")
        print(json.dumps(config.config, indent=2))

def validate_config(args: argparse.Namespace, config: ConfigManager) -> None:
    """Validate configuration"""
    if args.config:
        config = load_config(args.config)
    
    missing = config.validate()
    
    if missing:
        print("\nConfiguration validation failed!")
        print("Missing required keys:")
        for key in missing:
            print(f"  - {key}")
        sys.exit(1)
    else:
        print("\nConfiguration is valid.")
        
        # Test database connection
        try:
            db = DatabaseManager(config.get_db_config())
            db.fetchone("SELECT 1")
            db.close()
            print("Database connection successful.")
        except Exception as e:
            print(f"Database connection failed: {e}")
            sys.exit(1)
        
        # Check directories
        dirs_to_check = [
            ('data_dir', "Data directory"),
            ('batch_dir', "Batch directory")
        ]
        
        for key, desc in dirs_to_check:
            path = config.get(key)
            if path and os.path.exists(path):
                print(f"{desc} exists: {path}")
            elif path:
                print(f"{desc} does not exist: {path}")
                print(f"  Creating {desc}...")
                try:
                    os.makedirs(path, exist_ok=True)
                    print(f"  Created {desc}")
                except Exception as e:
                    print(f"  Failed to create {desc}: {e}")

def show_version() -> None:
    """Show version information"""
    try:
        from dpam import __version__
        version = __version__
    except ImportError:
        version = "development"
    
    print(f"\nDPAM Pipeline Manager version {version}")
    print("Domain Prediction and Analysis for Macromolecules")
    print("\nCopyright (c) 2023")

def main() -> None:
    """Main entry point"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Show version if requested
    if args.version:
        show_version()
        return
    
    # Show help if no command specified
    if not args.command:
        parser.print_help()
        return
    
    # Load configuration
    config = load_configuration(args)
    
    # Handle commands
    if args.command == 'batch':
        if args.subcommand == 'create':
            create_batch(args, config)
        elif args.subcommand == 'prepare':
            prepare_batch(args, config)
        elif args.subcommand == 'supplement':
            supplement_batch(args, config)
        elif args.subcommand == 'list':
            list_batches(args, config)
        elif args.subcommand == 'view':
            view_batch(args, config)
        elif args.subcommand == 'report':
            generate_batch_report(args, config)
        elif args.subcommand == 'delete':
            delete_batch(args, config)
        else:
            print("Unknown batch subcommand")
            
    elif args.command == 'pipeline':
        if args.subcommand == 'prepare':
            prepare_pipeline(args, config)
        elif args.subcommand == 'start':
            start_pipeline(args, config)
        elif args.subcommand == 'status':
            check_pipeline_status(args, config)
        elif args.subcommand == 'step':
            run_pipeline_step(args, config)
        elif args.subcommand == 'recover':
            recover_pipeline(args, config)
        else:
            print("Unknown pipeline subcommand")
            
    elif args.command == 'grid':
        if args.subcommand == 'status':
            check_grid_status(args, config)
        elif args.subcommand == 'list':
            list_grid_jobs(args, config)
        else:
            print("Unknown grid subcommand")
            
    elif args.command == 'config':
        if args.subcommand == 'show':
            show_config(args, config)
        elif args.subcommand == 'validate':
            validate_config(args, config)
        else:
            print("Unknown config subcommand")
            
    else:
        print(f"Unknown command: {args.command}")

if __name__ == '__main__':
    main()