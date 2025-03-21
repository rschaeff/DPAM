## dpam/pipeline/quality.py

class DPAMQualityControl:
    """Quality control and analytics for DPAM pipeline"""
    
    def __init__(self, db_config):
        self.db_config = db_config
    
    def get_db_connection(self):
        """Get connection to PostgreSQL database"""
        return psycopg2.connect(**self.db_config)
    
    def analyze_batch_results(self, batch_id):
        """Analyze results for a completed batch"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get batch information
            cursor.execute(
                """
                SELECT b.name, b.status, b.created_at, b.completed_at, 
                       COUNT(bi.batch_item_id) as total_structures
                FROM batches b
                JOIN batch_items bi ON b.batch_id = bi.batch_id
                WHERE b.batch_id = %s
                GROUP BY b.batch_id
                """,
                (batch_id,)
            )
            batch_info = cursor.fetchone()
            
            if not batch_info:
                return {
                    'success': False,
                    'message': f"Batch {batch_id} not found"
                }
                
            name, status, created_at, completed_at, total_structures = batch_info
            
            # Check if batch is completed
            if not status.startswith(('PIPELINE_COMPLETED', 'COMPLETED')):
                return {
                    'success': False,
                    'message': f"Batch not completed (status: {status})"
                }
            
            # Get structure success counts by step
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
            step_stats = cursor.fetchall()
            
            # Get domain statistics
            cursor.execute(
                """
                SELECT COUNT(s.structure_id) as with_domains,
                       AVG(JSONB_ARRAY_LENGTH(s.parameters->'step_outputs'->'parse_domains'->'domains')) as avg_domains
                FROM structures s
                JOIN batch_items bi ON s.structure_id = bi.structure_id
                WHERE bi.batch_id = %s
                AND s.parameters->'step_outputs'->'parse_domains'->'domains' IS NOT NULL
                """,
                (batch_id,)
            )
            domain_stats = cursor.fetchone()
            structures_with_domains, avg_domains_per_structure = domain_stats
            
            # Format step statistics
            step_summary = {}
            for step, status, count in step_stats:
                if step not in step_summary:
                    step_summary[step] = {}
                step_summary[step][status] = count
            
            # Calculate processing time
            processing_time = None
            if created_at and completed_at:
                processing_time = (completed_at - created_at).total_seconds() / 3600  # hours
            
            # Prepare summary
            summary = {
                'batch_id': batch_id,
                'name': name,
                'status': status,
                'total_structures': total_structures,
                'structures_with_domains': structures_with_domains,
                'avg_domains_per_structure': avg_domains_per_structure,
                'processing_time_hours': processing_time,
                'step_summary': step_summary,
                'created_at': created_at.isoformat() if created_at else None,
                'completed_at': completed_at.isoformat() if completed_at else None
            }
            
            # Store summary in database
            cursor.execute(
                """
                UPDATE batches 
                SET parameters = jsonb_set(
                    COALESCE(parameters, '{}'::jsonb),
                    '{analysis_summary}',
                    %s::jsonb
                )
                WHERE batch_id = %s
                """,
                (
                    json.dumps(summary),
                    batch_id
                )
            )
            
            conn.commit()
            
            return {
                'success': True,
                'summary': summary
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def generate_batch_report(self, batch_id):
        """Generate a comprehensive report for a batch"""
        # Get analysis summary
        analysis_result = self.analyze_batch_results(batch_id)
        
        if not analysis_result['success']:
            return analysis_result
        
        summary = analysis_result['summary']
        
        # Get batch path
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT parameters->>'batch_path' FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            batch_path = cursor.fetchone()[0]
            
            # Create report directory
            report_dir = f"{batch_path}/reports"
            os.makedirs(report_dir, exist_ok=True)
            
            # Generate HTML report
            report_path = f"{report_dir}/batch_{batch_id}_report.html"
            
            with open(report_path, 'w') as f:
                f.write(self._generate_html_report(summary))
            
            # Generate domain statistics report
            cursor.execute(
                """
                SELECT s.pdb_id, s.parameters->'step_outputs'->'parse_domains'->'domains'
                FROM structures s
                JOIN batch_items bi ON s.structure_id = bi.structure_id
                WHERE bi.batch_id = %s
                AND s.parameters->'step_outputs'->'parse_domains'->'domains' IS NOT NULL
                """,
                (batch_id,)
            )
            domain_data = cursor.fetchall()
            
            domains_report_path = f"{report_dir}/batch_{batch_id}_domains.csv"
            
            with open(domains_report_path, 'w') as f:
                f.write("pdb_id,domain_id,domain_range\n")
                for pdb_id, domains_json in domain_data:
                    domains = json.loads(domains_json)
                    for domain in domains:
                        f.write(f"{pdb_id},{domain['domain_id']},{domain['domain_range']}\n")
            
            return {
                'success': True,
                'report_path': report_path,
                'domains_report_path': domains_report_path,
                'summary': summary
            }
            
        finally:
            cursor.close()
            conn.close()
    
    def _generate_html_report(self, summary):
        """Generate HTML report from summary data"""
        # Simple HTML template for demonstration
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DPAM Batch {summary['batch_id']} Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                .stats {{ display: flex; flex-wrap: wrap; }}
                .stat-box {{ 
                    background-color: #f8f9fa; 
                    border-radius: 5px; 
                    padding: 15px; 
                    margin: 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    width: 200px;
                }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .stat-label {{ color: #7f8c8d; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>DPAM Batch Report</h1>
            <p><strong>Batch ID:</strong> {summary['batch_id']}</p>
            <p><strong>Name:</strong> {summary['name']}</p>
            <p><strong>Status:</strong> {summary['status']}</p>
            <p><strong>Created:</strong> {summary['created_at']}</p>
            <p><strong>Completed:</strong> {summary['completed_at']}</p>
            <p><strong>Processing Time:</strong> {summary['processing_time_hours']:.2f} hours</p>
            
            <h2>Summary Statistics</h2>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{summary['total_structures']}</div>
                    <div class="stat-label">Total Structures</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{summary['structures_with_domains']}</div>
                    <div class="stat-label">Structures with Domains</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{summary['avg_domains_per_structure']:.2f}</div>
                    <div class="stat-label">Avg Domains/Structure</div>
                </div>
            </div>
            
            <h2>Step Statistics</h2>
            <table>
                <tr>
                    <th>Step</th>
                    <th>Completed</th>
                    <th>Failed</th>
                    <th>Skipped</th>
                    <th>Success Rate</th>
                </tr>
        """
        
        # Add step statistics rows
        for step, stats in summary['step_summary'].items():
            completed = stats.get('COMPLETED', 0)
            failed = stats.get('FAILED', 0)
            skipped = stats.get('SKIPPED', 0)
            success_rate = completed / (completed + failed + skipped) * 100 if (completed + failed + skipped) > 0 else 0
            
            html += f"""
                <tr>
                    <td>{step}</td>
                    <td>{completed}</td>
                    <td>{failed}</td>
                    <td>{skipped}</td>
                    <td>{success_rate:.1f}%</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html