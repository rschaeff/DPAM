#!/usr/bin/env python3
"""
API routes and handlers for DPAM pipeline.

This module defines the routes and handlers for the DPAM REST API.
"""

import os
import logging
import json
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Union, Set
from enum import Enum
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile, Query, Path, Body, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger("dpam.api.routes")

# Define API models
class BatchStatus(str, Enum):
    """Possible batch statuses"""
    INITIALIZED = "INITIALIZED"
    PREPARING = "PREPARING"
    READY_FOR_DOWNLOAD = "READY_FOR_DOWNLOAD"
    DOWNLOADING = "DOWNLOADING"
    DOWNLOADING_PAE = "DOWNLOADING_PAE"
    READY_FOR_PROCESSING = "READY_FOR_PROCESSING"
    PIPELINE_READY = "PIPELINE_READY"
    PIPELINE_RUNNING = "PIPELINE_RUNNING"
    PIPELINE_COMPLETED = "PIPELINE_COMPLETED"
    PIPELINE_FAILED = "PIPELINE_FAILED"
    PIPELINE_RECOVERING = "PIPELINE_RECOVERING"
    PIPELINE_COMPLETED_WITH_FAILURES = "PIPELINE_COMPLETED_WITH_FAILURES"

class BatchCreate(BaseModel):
    """Model for batch creation"""
    name: str
    description: Optional[str] = None
    accessions: List[str]

class BatchResponse(BaseModel):
    """Model for batch response"""
    batch_id: int
    name: str
    description: Optional[str] = None
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    structure_count: int
    
    class Config:
        orm_mode = True

class StructureResponse(BaseModel):
    """Model for structure response"""
    structure_id: int
    pdb_id: str
    processing_status: str
    
    class Config:
        orm_mode = True

class DomainResponse(BaseModel):
    """Model for domain response"""
    domain_id: str
    start: int
    end: int
    size: int
    confidence_level: str
    ecod_mapping: Optional[Dict[str, Any]] = None
    
    class Config:
        orm_mode = True

class PipelineStart(BaseModel):
    """Model for pipeline start request"""
    step: Optional[str] = None
    restart: bool = False

class PipelineResponse(BaseModel):
    """Model for pipeline response"""
    status: str
    current_step: Optional[str] = None
    progress_percent: float
    
    class Config:
        orm_mode = True

def setup_routes(app):
    """
    Set up API routes
    
    Args:
        app: FastAPI application
    """
    router = APIRouter(prefix="/api/v1")
    
    # Dependency to get controllers
    def get_controllers(request: Request):
        """Get controllers from app state"""
        return {
            "pipeline_controller": request.app.state.pipeline_controller,
            "batch_manager": request.app.state.batch_manager,
            "batch_preparation": request.app.state.batch_preparation,
            "db_manager": request.app.state.db_manager
        }
    
    # Health check endpoint
    @router.get("/health", tags=["System"])
    async def health_check():
        """Check API health"""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    # Batch endpoints
    @router.post("/batches", response_model=BatchResponse, tags=["Batches"])
    async def create_batch(
        batch_data: BatchCreate,
        controllers: Dict[str, Any] = Depends(get_controllers)
    ):
        """Create a new batch from accessions"""
        try:
            batch_id = controllers["batch_manager"].create_batch_from_accessions(
                batch_data.accessions,
                batch_name=batch_data.name,
                description=batch_data.description
            )
            
            # Get batch details
            batch = controllers["db_manager"].fetchone(
                """
                SELECT b.batch_id, b.name, b.description, b.status, b.created_at, b.completed_at,
                       COUNT(bi.batch_item_id) as structure_count
                FROM batches b
                LEFT JOIN batch_items bi ON b.batch_id = bi.batch_id
                WHERE b.batch_id = %s
                GROUP BY b.batch_id
                """,
                (batch_id,)
            )
            
            return batch
            
        except Exception as e:
            logger.error(f"Error creating batch: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/batches", tags=["Batches"])
    async def list_batches(
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        controllers: Dict[str, Any] = Depends(get_controllers)
    ):
        """List all batches"""
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
            if status:
                query += " WHERE b.status = %s"
                params.append(status)
            
            # Add group by, order by, and pagination
            query += """
                GROUP BY b.batch_id, b.name, b.status, b.created_at, b.completed_at
                ORDER BY b.created_at DESC
                LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])
            
            # Execute query
            batches = controllers["db_manager"].fetchall(query, tuple(params))
            
            # Get total count
            count_query = "SELECT COUNT(*) as total FROM batches"
            if status:
                count_query += " WHERE status = %s"
                count_params = (status,)
            else:
                count_params = ()
                
            count_result = controllers["db_manager"].fetchone(count_query, count_params)
            total = count_result['total'] if count_result else 0
            
            return {
                "batches": batches,
                "total": total,
                "limit": limit,
                "offset": offset
            }
            
        except Exception as e:
            logger.error(f"Error listing batches: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/batches/{batch_id}", response_model=BatchResponse, tags=["Batches"])
    async def get_batch(
        batch_id: int = Path(..., description="Batch ID"),
        controllers: Dict[str, Any] = Depends(get_controllers)
    ):
        """Get batch details"""
        try:
            batch = controllers["db_manager"].fetchone(
                """
                SELECT b.batch_id, b.name, b.description, b.status, b.created_at, b.completed_at,
                       COUNT(bi.batch_item_id) as structure_count
                FROM batches b
                LEFT JOIN batch_items bi ON b.batch_id = bi.batch_id
                WHERE b.batch_id = %s
                GROUP BY b.batch_id
                """,
                (batch_id,)
            )
            
            if not batch:
                raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
                
            return batch
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting batch {batch_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/batches/{batch_id}/prepare", tags=["Batches"])
    async def prepare_batch(
        batch_id: int = Path(..., description="Batch ID"),
        background_tasks: BackgroundTasks = None,
        controllers: Dict[str, Any] = Depends(get_controllers)
    ):
        """Prepare batch for processing"""
        try:
            # Check if batch exists
            batch = controllers["db_manager"].fetchone(
                "SELECT status FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            
            if not batch:
                raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
                
            # Check if batch is ready for preparation
            if batch['status'] != "READY_FOR_DOWNLOAD":
                raise HTTPException(
                    status_code=400, 
                    detail=f"Batch not ready for preparation (status: {batch['status']})"
                )
            
            # Start preparation in background
            if background_tasks:
                background_tasks.add_task(
                    controllers["batch_preparation"].prepare_batch_directory,
                    batch_id
                )
                
                return {
                    "message": f"Batch {batch_id} preparation started in background",
                    "batch_id": batch_id
                }
            else:
                # Run synchronously
                result = controllers["batch_preparation"].prepare_batch_directory(batch_id)
                
                return {
                    "message": f"Batch {batch_id} preparation completed",
                    "batch_id": batch_id,
                    "status": result['status'],
                    "batch_path": result.get('batch_path'),
                    "metrics": result.get('metrics')
                }
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error preparing batch {batch_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/batches/{batch_id}/structures", tags=["Batches"])
    async def list_batch_structures(
        batch_id: int = Path(..., description="Batch ID"),
        status: Optional[str] = None,
        controllers: Dict[str, Any] = Depends(get_controllers)
    ):
        """List structures in a batch"""
        try:
            # Build query
            query = """
                SELECT s.structure_id, s.pdb_id, p.uniprot_id, s.processing_status, bi.status as batch_status,
                       bi.error_message
                FROM batch_items bi
                JOIN structures s ON bi.structure_id = s.structure_id
                JOIN proteins p ON s.protein_id = p.protein_id
                WHERE bi.batch_id = %s
            """
            params = [batch_id]
            
            # Add status filter if specified
            if status:
                query += " AND bi.status = %s"
                params.append(status)
            
            # Add order by
            query += " ORDER BY s.pdb_id"
            
            # Execute query
            structures = controllers["db_manager"].fetchall(query, tuple(params))
            
            if not structures and not controllers["db_manager"].fetchone(
                "SELECT 1 FROM batches WHERE batch_id = %s", (batch_id,)
            ):
                raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
                
            return {
                "batch_id": batch_id,
                "structures": structures,
                "total": len(structures)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error listing structures for batch {batch_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Pipeline endpoints
    @router.post("/batches/{batch_id}/pipeline/prepare", tags=["Pipeline"])
    async def prepare_pipeline(
        batch_id: int = Path(..., description="Batch ID"),
        controllers: Dict[str, Any] = Depends(get_controllers)
    ):
        """Prepare pipeline for batch execution"""
        try:
            result = controllers["pipeline_controller"].prepare_batch_pipeline(batch_id)
            
            if not result['success']:
                raise HTTPException(status_code=400, detail=result['message'])
                
            return {
                "message": f"Pipeline prepared for batch {batch_id}",
                "batch_id": batch_id,
                "grid_config": result['grid_config'],
                "steps": result['steps']
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error preparing pipeline for batch {batch_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/batches/{batch_id}/pipeline/start", tags=["Pipeline"])
    async def start_pipeline(
        batch_id: int = Path(..., description="Batch ID"),
        pipeline_data: PipelineStart = Body(...),
        controllers: Dict[str, Any] = Depends(get_controllers)
    ):
        """Start pipeline execution"""
        try:
            result = controllers["pipeline_controller"].start_pipeline(
                batch_id,
                start_from=pipeline_data.step,
                restart=pipeline_data.restart
            )
            
            if not result['success']:
                raise HTTPException(status_code=400, detail=result['message'])
                
            return {
                "message": f"Pipeline started for batch {batch_id}",
                "batch_id": batch_id,
                "job_id": result['job_id'],
                "grid_job_id": result['grid_job_id'],
                "step": result['step'],
                "step_index": result.get('step_index', 0)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error starting pipeline for batch {batch_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/batches/{batch_id}/pipeline/status", tags=["Pipeline"])
    async def get_pipeline_status(
        batch_id: int = Path(..., description="Batch ID"),
        controllers: Dict[str, Any] = Depends(get_controllers)
    ):
        """Get pipeline status"""
        try:
            result = controllers["pipeline_controller"].get_pipeline_status(batch_id)
            
            if not result['success']:
                raise HTTPException(status_code=400, detail=result['message'])
                
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting pipeline status for batch {batch_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/batches/{batch_id}/pipeline/recover", tags=["Pipeline"])
    async def recover_pipeline(
        batch_id: int = Path(..., description="Batch ID"),
        controllers: Dict[str, Any] = Depends(get_controllers)
    ):
        """Recover failed pipeline"""
        try:
            result = controllers["pipeline_controller"].recover_failed_pipeline(batch_id)
            
            if not result['success']:
                raise HTTPException(status_code=400, detail=result['message'])
                
            return {
                "message": f"Pipeline recovery initiated for batch {batch_id}",
                "batch_id": batch_id,
                "recovery_plans": result.get('recovery_plans', []),
                "status": result.get('status')
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error recovering pipeline for batch {batch_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Domains endpoints
    @router.get("/batches/{batch_id}/domains", tags=["Domains"])
    async def list_batch_domains(
        batch_id: int = Path(..., description="Batch ID"),
        confidence: Optional[str] = None,
        controllers: Dict[str, Any] = Depends(get_controllers)
    ):
        """List domains in a batch"""
        try:
            # First check if batch has completed domain detection
            batch = controllers["db_manager"].fetchone(
                "SELECT status FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            
            if not batch:
                raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
                
            if not batch['status'].startswith(("PIPELINE_COMPLETED", "COMPLETED")):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Batch processing not completed (status: {batch['status']})"
                )
            
            # Get batch path
            batch_path = controllers["db_manager"].fetchone(
                "SELECT parameters->>'batch_path' FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            
            if not batch_path or not batch_path.get('parameters', {}).get('batch_path'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch path not found for batch {batch_id}"
                )
                
            batch_path = batch_path.get('parameters', {}).get('batch_path')
            
            # Get structure IDs in batch
            structures = controllers["db_manager"].fetchall(
                """
                SELECT s.structure_id, s.pdb_id
                FROM batch_items bi
                JOIN structures s ON bi.structure_id = s.structure_id
                WHERE bi.batch_id = %s AND bi.status = 'COMPLETED'
                """,
                (batch_id,)
            )
            
            # Collect domains from all structures
            all_domains = []
            
            for structure in structures:
                structure_id = structure['structure_id']
                pdb_id = structure['pdb_id']
                
                # Look for domain file
                domains_file = os.path.join(
                    batch_path, "results", "parse_domains", f"struct_{structure_id}", 
                    f"struct_{structure_id}_domains.json"
                )
                
                if not os.path.exists(domains_file):
                    continue
                
                # Load domains
                with open(domains_file, 'r') as f:
                    domains_data = json.load(f)
                
                domains = domains_data.get('domains', [])
                
                # Filter by confidence if specified
                if confidence:
                    domains = [d for d in domains if d.get('confidence_level') == confidence]
                
                # Add structure info to domains
                for domain in domains:
                    domain['structure_id'] = structure_id
                    domain['pdb_id'] = pdb_id
                    all_domains.append(domain)
            
            return {
                "batch_id": batch_id,
                "domains": all_domains,
                "total": len(all_domains)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error listing domains for batch {batch_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/structures/{structure_id}/domains", tags=["Domains"])
    async def get_structure_domains(
        structure_id: int = Path(..., description="Structure ID"),
        controllers: Dict[str, Any] = Depends(get_controllers)
    ):
        """Get domains for a structure"""
        try:
            # Get batch and structure information
            structure_info = controllers["db_manager"].fetchone(
                """
                SELECT s.pdb_id, s.processing_status, bi.batch_id, bi.status,
                       b.parameters->>'batch_path' as batch_path
                FROM structures s
                JOIN batch_items bi ON s.structure_id = bi.structure_id
                JOIN batches b ON bi.batch_id = b.batch_id
                WHERE s.structure_id = %s
                """,
                (structure_id,)
            )
            
            if not structure_info:
                raise HTTPException(status_code=404, detail=f"Structure {structure_id} not found")
                
            if structure_info['status'] != 'COMPLETED':
                raise HTTPException(
                    status_code=400,
                    detail=f"Structure processing not completed (status: {structure_info['status']})"
                )
                
            batch_path = structure_info['batch_path']
            if not batch_path:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch path not found for structure {structure_id}"
                )
            
            # Look for domain file
            domains_file = os.path.join(
                batch_path, "results", "parse_domains", f"struct_{structure_id}", 
                f"struct_{structure_id}_domains.json"
            )
            
            if not os.path.exists(domains_file):
                raise HTTPException(
                    status_code=404,
                    detail=f"Domains not found for structure {structure_id}"
                )
            
            # Load domains
            with open(domains_file, 'r') as f:
                domains_data = json.load(f)
            
            domains = domains_data.get('domains', [])
            
            # Add structure info to domains
            for domain in domains:
                domain['structure_id'] = structure_id
                domain['pdb_id'] = structure_info['pdb_id']
            
            return {
                "structure_id": structure_id,
                "pdb_id": structure_info['pdb_id'],
                "domains": domains,
                "total": len(domains)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting domains for structure {structure_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Download endpoints
    @router.get("/structures/{structure_id}/file", tags=["Downloads"])
    async def download_structure_file(
        structure_id: int = Path(..., description="Structure ID"),
        controllers: Dict[str, Any] = Depends(get_controllers)
    ):
        """Download structure file"""
        try:
            # Get structure path
            structure = controllers["db_manager"].fetchone(
                "SELECT structure_path, pdb_id FROM structures WHERE structure_id = %s",
                (structure_id,)
            )
            
            if not structure or not structure['structure_path']:
                raise HTTPException(status_code=404, detail=f"Structure {structure_id} not found")
                
            structure_path = structure['structure_path']
            
            if not os.path.exists(structure_path):
                raise HTTPException(status_code=404, detail=f"Structure file not found")
                
            return FileResponse(
                structure_path,
                filename=f"{structure['pdb_id']}.cif.gz",
                media_type="application/gzip"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error downloading structure file {structure_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/batches/{batch_id}/report", tags=["Downloads"])
    async def download_batch_report(
        batch_id: int = Path(..., description="Batch ID"),
        format: str = Query("json", description="Report format (json or tsv)"),
        controllers: Dict[str, Any] = Depends(get_controllers)
    ):
        """Download batch report"""
        try:
            # Get batch path
            batch_path_result = controllers["db_manager"].fetchone(
                "SELECT parameters->>'batch_path' FROM batches WHERE batch_id = %s",
                (batch_id,)
            )
            
            if not batch_path_result:
                raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
                
            batch_path = batch_path_result.get('parameters', {}).get('batch_path')
            if not batch_path:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch path not found for batch {batch_id}"
                )
            
            # Look for report file
            report_dir = os.path.join(batch_path, "reports")
            
            if format.lower() == "json":
                report_file = os.path.join(report_dir, f"batch_{batch_id}_report.json")
                media_type = "application/json"
                filename = f"batch_{batch_id}_report.json"
            elif format.lower() == "tsv":
                report_file = os.path.join(report_dir, f"batch_{batch_id}_domains.tsv")
                media_type = "text/tab-separated-values"
                filename = f"batch_{batch_id}_domains.tsv"
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported format: {format}. Use 'json' or 'tsv'."
                )
            
            if not os.path.exists(report_file):
                raise HTTPException(
                    status_code=404,
                    detail=f"Report not found for batch {batch_id}"
                )
                
            return FileResponse(
                report_file,
                filename=filename,
                media_type=media_type
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error downloading batch report {batch_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Add routes to app
    app.include_router(router)