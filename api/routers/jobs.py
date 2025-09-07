"""
Router para manejo de trabajos/jobs
REFACTORIZADO: Usa jobs_service + Cola Compartida para jobs asíncronos
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging
import sys
import uuid
from datetime import datetime

# Importar modelos Pydantic desde la carpeta models
from models.schemas import (
    JobRequest, 
    TrainingRequest, 
    SyntheticDataRequest, 
    RestorationRequest
)

# Asegurar imports
sys.path.append('/app/api')

# Importar cola compartida
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from shared_job_queue import create_shared_queue

try:
    from services.jobs_service import jobs_service
    from config.constants import RESPONSE_MESSAGES
    SERVICES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Error importando servicios en jobs router: {e}")
    jobs_service = None
    SERVICES_AVAILABLE = False
    RESPONSE_MESSAGES = {}

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["Trabajos"])

# Inicializar cola compartida
shared_queue = create_shared_queue()

def create_job_id(job_type: str) -> str:
    """Crear ID único para job"""
    timestamp = int(datetime.now().timestamp() * 1000) % 100000000
    random_suffix = str(uuid.uuid4())[:8]
    return f"{job_type}_{random_suffix}"

@router.get("/info")
async def get_jobs_info():
    """Obtener información del servicio de trabajos"""
    try:
        if not SERVICES_AVAILABLE:
            return JSONResponse({
                "status": "error",
                "message": "Servicio de trabajos no disponible"
            })
        
        return JSONResponse({
            "status": "active",
            "service": "jobs_service",
            "configuration": {
                "max_jobs": jobs_service.max_jobs,
                "job_timeout": jobs_service.job_timeout,
                "cleanup_interval": jobs_service.cleanup_interval
            },
            "available_operations": [
                "list_jobs",
                "get_job_status", 
                "delete_job",
                "create_job",
                "update_job_status",
                "cleanup_old_jobs"
            ]
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo información de trabajos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/summary")
async def get_jobs_summary():
    """Obtener resumen estadístico de trabajos usando jobs_service"""
    try:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicio de trabajos no disponible")
        
        result = jobs_service.list_jobs()
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        # Extraer solo las estadísticas
        return JSONResponse(result.get("statistics", {}))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo resumen de trabajos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def list_jobs():
    """Listar todos los trabajos usando jobs_service"""
    try:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicio de trabajos no disponible")
        
        result = jobs_service.list_jobs()
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listando trabajos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """Obtener estado de trabajo específico usando jobs_service"""
    try:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicio de trabajos no disponible")
        
        result = jobs_service.get_job_status(job_id)
        
        if result["status"] == "error":
            if result.get("error_code") == "JOB_NOT_FOUND":
                raise HTTPException(status_code=404, detail=result["message"])
            else:
                raise HTTPException(status_code=500, detail=result["message"])
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo trabajo {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """Eliminar trabajo específico usando jobs_service"""
    try:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicio de trabajos no disponible")
        
        result = jobs_service.delete_job(job_id)
        
        if result["status"] == "error":
            if result.get("error_code") == "JOB_NOT_FOUND":
                raise HTTPException(status_code=404, detail=result["message"])
            else:
                raise HTTPException(status_code=500, detail=result["message"])
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando trabajo {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
async def clear_completed_jobs():
    """Limpiar trabajos completados usando jobs_service"""
    try:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicio de trabajos no disponible")
        
        result = jobs_service.cleanup_old_jobs(max_age_seconds=0)  # Limpiar inmediatamente
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error limpiando trabajos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/summary")
async def get_jobs_summary():
    """Obtener resumen estadístico de trabajos usando jobs_service"""
    try:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicio de trabajos no disponible")
        
        result = jobs_service.list_jobs()
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        # Extraer solo las estadísticas
        return JSONResponse(result.get("statistics", {}))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo resumen de trabajos: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# ================================
# ENDPOINTS DE COLA COMPARTIDA
# ================================

@router.post("/queue")
async def create_shared_job(request: JobRequest):
    """Crear job genérico en cola compartida"""
    job_id = create_job_id(request.job_type)
    
    job_data = {
        "job_id": job_id,
        "job_type": request.job_type,
        "parameters": request.parameters,
        "priority": request.priority,
        "created_at": datetime.now().isoformat(),
        "status": "queued"
    }
    
    shared_queue.enqueue_job(job_data)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Job {request.job_type} creado exitosamente",
        "check_status_url": f"/jobs/queue/{job_id}",
        "created_at": job_data["created_at"]
    }

@router.get("/queue/{job_id}")
async def get_shared_job_status(job_id: str):
    """Obtener status de job específico en cola compartida"""
    job_status = shared_queue.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    
    return {
        "job_id": job_id,
        **job_status
    }

@router.get("/queue")
async def list_shared_jobs():
    """Listar todos los jobs de cola compartida"""
    return shared_queue.get_all_jobs()

@router.delete("/queue/{job_id}")
async def cancel_shared_job(job_id: str):
    """Cancelar job en cola compartida"""
    job_status = shared_queue.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    
    if job_status["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="No se puede cancelar job terminado")
    
    shared_queue.update_job_status(job_id, "cancelled", cancelled_at=datetime.now().isoformat())
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job cancelado exitosamente"
    }

@router.post("/queue/training/layer2")
async def create_training_shared_job(request: TrainingRequest):
    """Crear job de entrenamiento Layer 2 en cola compartida"""
    job_id = create_job_id("layer2_training")
    
    job_data = {
        "job_id": job_id,
        "job_type": "layer2_training",
        "parameters": {
            "num_epochs": request.num_epochs,
            "batch_size": request.batch_size,
            "max_pairs": request.max_pairs,
            "use_training_bucket": request.use_training_bucket
        },
        "created_at": datetime.now().isoformat(),
        "status": "queued"
    }
    
    shared_queue.enqueue_job(job_data)
    logger.info(f"🧠 Training job creado: {job_id} ({request.num_epochs} épocas)")
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Training Layer 2 iniciado ({request.num_epochs} épocas)",
        "check_status_url": f"/jobs/queue/{job_id}",
        "parameters": job_data["parameters"]
    }

@router.post("/queue/synthetic-data/generate")
async def create_synthetic_data_shared_job(request: SyntheticDataRequest):
    """Crear job de generación de datos sintéticos en cola compartida"""
    job_id = create_job_id("synthetic_data")
    
    job_data = {
        "job_id": job_id,
        "job_type": "synthetic_data_generation",
        "parameters": {
            "count": request.count,
            "bucket": request.bucket,
            "augmentation_types": request.augmentation_types
        },
        "created_at": datetime.now().isoformat(),
        "status": "queued"
    }
    
    shared_queue.enqueue_job(job_data)
    logger.info(f"🎨 Synthetic data job creado: {job_id} ({request.count} imágenes)")
    
    return {
        "job_id": job_id,
        "status": "queued", 
        "message": f"Generación de {request.count} imágenes sintéticas iniciada",
        "check_status_url": f"/jobs/queue/{job_id}",
        "parameters": job_data["parameters"]
    }

@router.post("/queue/restoration/batch")
async def create_restoration_shared_job(request: RestorationRequest):
    """Crear job de restauración por lotes en cola compartida"""
    job_id = create_job_id("restoration")
    
    job_data = {
        "job_id": job_id,
        "job_type": "batch_restoration",
        "parameters": {
            "file_count": request.file_count,
            "model_type": request.model_type,
            "bucket": request.bucket
        },
        "created_at": datetime.now().isoformat(),
        "status": "queued"
    }
    
    shared_queue.enqueue_job(job_data)
    logger.info(f"🔧 Restoration job creado: {job_id} ({request.file_count} archivos)")
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Restauración de {request.file_count} archivos iniciada",
        "check_status_url": f"/jobs/queue/{job_id}",
        "parameters": job_data["parameters"]
    }
