## dpam/grid/manager.py

class DPAMOpenGridManager:
    """Advanced OpenGrid management for DPAM pipeline"""
    
    def __init__(self, db_config, grid_config, data_dir):
        self.db_config = db_config
        self.grid_config = grid_config
        self.data_dir = data_dir
    
    def get_db_connection(self):
        """Get connection to PostgreSQL database"""
        return psycopg2.connect(**self.db_config)
    
    def prepare_batch_for_grid(self, batch_id):
        """Prepare batch data for distributed processing on OpenGrid"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get batch information
            cursor.execute(
                "SELECT parameters->>'batch_path' FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            batch_path = cursor.fetchone()[0]
            
            # Get all structures in batch
            cursor.execute(
                """
                SELECT s.structure_id, s.pdb_id, s.structure_path 
                FROM batch_items bi
                JOIN structures s ON bi.structure_id = s.structure_id
                WHERE bi.batch_id = %s AND bi.status = 'READY'
                """,
                (batch_id,)
            )
            structures = cursor.fetchall()
            
            # Create structure manifest
            manifest_path = f"{batch_path}/grid_manifest.json"
            structure_data = {
                str(structure_id): {
                    'pdb_id': pdb_id,
                    'structure_path': structure_path
                }
                for structure_id, pdb_id, structure_path in structures
            }
            
            with open(manifest_path, 'w') as f:
                json.dump({
                    'batch_id': batch_id,
                    'structures': structure_data,
                    'data_dir': self.data_dir,
                    'total_structures': len(structures)
                }, f, indent=2)
            
            # Create grid scripts directory
            grid_dir = f"{batch_path}/grid_scripts"
            os.makedirs(grid_dir, exist_ok=True)
            
            # Update batch information
            cursor.execute(
                """
                UPDATE batches 
                SET parameters = jsonb_set(
                    COALESCE(parameters, '{}'::jsonb),
                    '{grid_config}',
                    %s::jsonb
                )
                WHERE batch_id = %s
                """,
                (
                    json.dumps({
                        'manifest_path': manifest_path,
                        'grid_dir': grid_dir,
                        'total_structures': len(structures)
                    }),
                    batch_id
                )
            )
            
            conn.commit()
            
            return {
                'manifest_path': manifest_path,
                'grid_dir': grid_dir,
                'total_structures': len(structures)
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def submit_pipeline_job(self, batch_id, step_name):
        """Submit a specific pipeline step to OpenGrid"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get batch grid configuration
            cursor.execute(
                "SELECT parameters->>'grid_config' FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            grid_config = json.loads(cursor.fetchone()[0])
            total_structures = grid_config['total_structures']
            
            # Create job script
            job_script = self._create_job_script(batch_id, step_name, grid_config['manifest_path'])
            job_path = f"{grid_config['grid_dir']}/dpam_{batch_id}_{step_name}.sh"
            
            with open(job_path, 'w') as f:
                f.write(job_script)
            
            # Submit job
            cmd = f"qsub -t 1-{total_structures} -N dpam_{batch_id}_{step_name} {job_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to submit job: {result.stderr}")
                
            # Extract job ID
            job_id = result.stdout.strip()
            
            # Log job submission
            cursor.execute(
                """
                INSERT INTO grid_jobs (batch_id, step_name, job_id, array_size, status, submitted_at)
                VALUES (%s, %s, %s, %s, 'SUBMITTED', NOW())
                RETURNING grid_job_id
                """,
                (batch_id, step_name, job_id, total_structures)
            )
            grid_job_id = cursor.fetchone()[0]
            
            # Update batch status
            cursor.execute(
                """
                UPDATE batches 
                SET status = %s,
                    parameters = jsonb_set(
                        COALESCE(parameters, '{}'::jsonb),
                        '{current_grid_job}',
                        %s::jsonb
                    )
                WHERE batch_id = %s
                """,
                (
                    f"GRID_RUNNING_{step_name.upper()}",
                    json.dumps({
                        'grid_job_id': grid_job_id,
                        'job_id': job_id,
                        'step_name': step_name,
                        'submitted_at': datetime.now().isoformat()
                    }),
                    batch_id
                )
            )
            
            conn.commit()
            
            return {
                'grid_job_id': grid_job_id,
                'job_id': job_id
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def check_job_status(self, grid_job_id):
        """Check status of a grid job and update database"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get job information
            cursor.execute(
                "SELECT batch_id, step_name, job_id, array_size FROM grid_jobs WHERE grid_job_id = %s",
                (grid_job_id,)
            )
            batch_id, step_name, job_id, array_size = cursor.fetchone()
            
            # Check job status in grid
            cmd = f"qstat -j {job_id}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                # Job not found, check if completed
                status, task_stats = self._analyze_job_completion(job_id, array_size)
            else:
                # Job still running
                status = "RUNNING"
                task_stats = self._analyze_running_job(result.stdout, array_size)
            
            # Update job status
            cursor.execute(
                """
                UPDATE grid_jobs 
                SET status = %s, 
                    updated_at = NOW(),
                    task_stats = %s
                WHERE grid_job_id = %s
                """,
                (
                    status,
                    json.dumps(task_stats),
                    grid_job_id
                )
            )
            
            # If job completed or failed, update batch status
            if status in ["COMPLETED", "FAILED"]:
                cursor.execute(
                    "UPDATE batches SET status = %s WHERE batch_id = %s",
                    (f"GRID_{status}_{step_name.upper()}", batch_id)
                )
                
                # Process results if completed
                if status == "COMPLETED":
                    self._process_completed_job(batch_id, step_name)
            
            conn.commit()
            
            return {
                'status': status,
                'task_stats': task_stats
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def check_all_pending_jobs(self):
        """Check status of all pending grid jobs"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT grid_job_id FROM grid_jobs WHERE status = 'SUBMITTED' OR status = 'RUNNING'"
            )
            job_ids = [row[0] for row in cursor.fetchall()]
            
            results = {}
            for job_id in job_ids:
                results[job_id] = self.check_job_status(job_id)
            
            return results
            
        finally:
            cursor.close()
            conn.close()
    
    def _analyze_job_completion(self, job_id, array_size):
        """Analyze completion status of a job"""
        # Check accounting logs
        cmd = f"qacct -j {job_id}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            return "UNKNOWN", {"unknown": array_size}
        
        # Parse output
        success_count = 0
        failed_count = 0
        task_stats = {}
        
        for line in result.stdout.split('\n'):
            if line.startswith("taskid"):
                task_id = line.split()[1]
            elif line.startswith("exit_status"):
                exit_code = int(line.split()[1])
                if exit_code == 0:
                    success_count += 1
                    task_stats[task_id] = "COMPLETED"
                else:
                    failed_count += 1
                    task_stats[task_id] = f"FAILED_WITH_CODE_{exit_code}"
        
        # Determine overall status
        if success_count + failed_count == 0:
            return "UNKNOWN", {"unknown": array_size}
        elif failed_count == 0:
            return "COMPLETED", {"completed": success_count, "failed": 0}
        elif success_count == 0:
            return "FAILED", {"completed": 0, "failed": failed_count}
        else:
            return "PARTIALLY_COMPLETED", {"completed": success_count, "failed": failed_count}
    
    def _analyze_running_job(self, qstat_output, array_size):
        """Analyze status of a running job"""
        # Parse qstat output to count running/waiting tasks
        running_count = qstat_output.count("r")
        waiting_count = qstat_output.count("qw")
        
        return {
            "running": running_count,
            "waiting": waiting_count,
            "pending": array_size - running_count - waiting_count
        }
    
    def _process_completed_job(self, batch_id, step_name):
        """Process results of a completed job"""
        # Create a task to update database with results
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self._update_database_with_results, batch_id, step_name)
    
    def _update_database_with_results(self, batch_id, step_name):
        """Update database with job results"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get batch path
            cursor.execute(
                "SELECT parameters->>'batch_path' FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            batch_path = cursor.fetchone()[0]
            
            # Get results directory
            results_dir = f"{batch_path}/grid_results/{step_name}"
            
            # Process result files
            if os.path.exists(results_dir):
                for result_file in os.listdir(results_dir):
                    if result_file.endswith(".json"):
                        with open(f"{results_dir}/{result_file}") as f:
                            result_data = json.load(f)
                            
                            structure_id = result_data['structure_id']
                            status = result_data['status']
                            output_files = result_data.get('output_files', {})
                            metrics = result_data.get('metrics', {})
                            error_message = result_data.get('error_message')
                            
                            # Update step log
                            cursor.execute(
                                """
                                INSERT INTO step_logs 
                                (batch_item_id, step_id, started_at, completed_at, status, output, metrics, error_message)
                                SELECT 
                                    bi.batch_item_id, 
                                    ps.step_id, 
                                    %s, 
                                    %s, 
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
                                    result_data.get('started_at'),
                                    result_data.get('completed_at'),
                                    status,
                                    json.dumps(output_files),
                                    json.dumps(metrics),
                                    error_message,
                                    structure_id,
                                    batch_id,
                                    step_name
                                )
                            )
                            
                            # Update structure record
                            if status == "COMPLETED":
                                cursor.execute(
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
                                        json.dumps(output_files),
                                        structure_id
                                    )
                                )
                                
                            conn.commit()
                            
            # Set checkpoint
            cursor.execute(
                """
                INSERT INTO batch_checkpoints (batch_id, step_name, status, created_at)
                VALUES (%s, %s, 'COMPLETED', NOW())
                """,
                (batch_id, step_name)
            )
            
            conn.commit()
            
        finally:
            cursor.close()
            conn.close()
    
    def _create_job_script(self, batch_id, step_name, manifest_path):
        """Create job script for OpenGrid"""
        # Template for OpenGrid job script
        script_template = f"""#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -o {self.grid_config['log_dir']}/dpam_{batch_id}_{step_name}.$SGE_TASK_ID.log
#$ -l h_rt={self.grid_config.get('max_runtime', '24:00:00')}
#$ -l h_vmem={self.grid_config.get('memory', '8G')}
{"#$ -pe threaded " + str(self.grid_config.get('threads', 4)) if step_name in ['run_hhsearch', 'run_foldseek', 'run_iterative_dali'] else ""}

# Load modules
module load python/3.9
{self._get_step_specific_modules(step_name)}

# Activate environment if needed
source {self.grid_config.get('env_path', '/path/to/env/bin/activate')}

# Set temporary directory
export TMPDIR=${{TMPDIR:-/tmp}}

# Run the task
python {self.grid_config['script_path']} \\
    --batch-id {batch_id} \\
    --step {step_name} \\
    --task-id $SGE_TASK_ID \\
    --manifest {manifest_path} \\
    --data-dir {self.data_dir} \\
    --output-dir {self.grid_config['output_dir']} \\
    --threads {self.grid_config.get('threads', 4) if step_name in ['run_hhsearch', 'run_foldseek', 'run_iterative_dali'] else 1}

exit $?
"""
        return script_template
    
    def _get_step_specific_modules(self, step_name):
        """Get step-specific module loads"""
        # Define modules needed for each step
        step_modules = {
            'run_hhsearch': "module load hhsuite/3.3.0",
            'run_foldseek': "module load foldseek/2.1.0",
            'get_sse': "module load dssp/3.0.0",
            'run_iterative_dali': "module load dali/5.0"
        }
        
        return step_modules.get(step_name, "")