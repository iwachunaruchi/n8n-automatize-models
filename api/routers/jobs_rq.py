"""
🚀 ROUTER DE JOBS CON RQ - SISTEMA PROFESIONAL
==============================================
Router refactorizado para usar Redis Queue (RQ) en lugar de archivos JSON.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import logging
import sys
import uuid
from datetime import datetime

# Asegurar imports
sys.path.append('/app/api')
sys.path.append('/app')
sys.path.append('/app/project_root')

# Importar modelos Pydantic
from models.schemas import (
    JobRequest, 
    TrainingRequest, 
    SyntheticDataRequest, 
    RestorationRequest
)

try:
    from rq_job_system import get_job_queue_manager
    RQ_AVAILABLE = True
except ImportError as e:
    logging.error(f"RQ no disponible: {e}")
    RQ_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["Jobs RQ"])

def create_job_id(job_type: str) -> str:
    """Crear ID único para job"""
    timestamp = int(datetime.now().timestamp() * 1000) % 100000000
    random_suffix = str(uuid.uuid4())[:8]
    return f"{job_type}_{random_suffix}"

# ================================
# ENDPOINTS RQ JOBS
# ================================

@router.post("/rq/test")
async def create_test_rq_job(message: str = "Test RQ Job", duration: int = 5):
    """Crear job de prueba en RQ"""
    if not RQ_AVAILABLE:
        raise HTTPException(status_code=503, detail="RQ no está disponible")
    
    try:
        manager = get_job_queue_manager()
        
        job_id = create_job_id("test_rq")
        
        rq_job_id = manager.enqueue_job(
            job_function='workers.rq_tasks.test_job',
            job_kwargs={'message': message, 'duration': duration},
            priority='default',
            job_id=job_id
        )
        
        return {
            "job_id": rq_job_id,
            "status": "queued",
            "message": f"Test RQ job creado: {message}",
            "check_status_url": f"/jobs/rq/status/{rq_job_id}",
            "parameters": {"message": message, "duration": duration}
        }
        
    except Exception as e:
        logger.error(f"Error creando test RQ job: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.post("/rq/training/layer2")
async def create_layer2_training_rq_job(request: TrainingRequest):
    """Crear job de entrenamiento Layer 2 en RQ"""
    if not RQ_AVAILABLE:
        raise HTTPException(status_code=503, detail="RQ no está disponible")
    
    try:
        manager = get_job_queue_manager()
        
        job_id = create_job_id("layer2_training_rq")
        
        rq_job_id = manager.enqueue_job(
            job_function='workers.rq_tasks.layer2_training_job',
            job_kwargs={
                'num_epochs': request.num_epochs,
                'batch_size': request.batch_size,
                'max_pairs': request.max_pairs,
                'use_training_bucket': request.use_training_bucket
            },
            priority='high',  # Training tiene alta prioridad
            timeout=3600,     # 1 hora timeout
            job_id=job_id
        )
        
        logger.info(f"🧠 Training RQ job creado: {rq_job_id} ({request.num_epochs} épocas)")
        
        return {
            "job_id": rq_job_id,
            "status": "queued",
            "message": f"Training Layer 2 iniciado con RQ ({request.num_epochs} épocas)",
            "check_status_url": f"/jobs/rq/status/{rq_job_id}",
            "parameters": {
                "num_epochs": request.num_epochs,
                "batch_size": request.batch_size,
                "max_pairs": request.max_pairs,
                "use_training_bucket": request.use_training_bucket
            }
        }
        
    except Exception as e:
        logger.error(f"Error creando training RQ job: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.post("/rq/synthetic-data/generate")
async def create_synthetic_data_rq_job(request: SyntheticDataRequest):
    """Crear job de generación de datos sintéticos en RQ"""
    if not RQ_AVAILABLE:
        raise HTTPException(status_code=503, detail="RQ no está disponible")
    
    try:
        manager = get_job_queue_manager()
        
        job_id = create_job_id("synthetic_data_rq")
        
        rq_job_id = manager.enqueue_job(
            job_function='workers.rq_tasks.synthetic_data_generation_job',
            job_kwargs={
                'count': request.count,
                'bucket': request.bucket,
                'augmentation_types': request.augmentation_types
            },
            priority='default',
            timeout=1800,  # 30 min timeout
            job_id=job_id
        )
        
        logger.info(f"🎨 Synthetic data RQ job creado: {rq_job_id} ({request.count} imágenes)")
        
        return {
            "job_id": rq_job_id,
            "status": "queued",
            "message": f"Generación de {request.count} imágenes sintéticas iniciada con RQ",
            "check_status_url": f"/jobs/rq/status/{rq_job_id}",
            "parameters": {
                "count": request.count,
                "bucket": request.bucket,
                "augmentation_types": request.augmentation_types
            }
        }
        
    except Exception as e:
        logger.error(f"Error creando synthetic data RQ job: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.post("/rq/restoration/batch")
async def create_restoration_rq_job(request: RestorationRequest):
    """Crear job de restauración por lotes en RQ"""
    if not RQ_AVAILABLE:
        raise HTTPException(status_code=503, detail="RQ no está disponible")
    
    try:
        manager = get_job_queue_manager()
        
        job_id = create_job_id("restoration_rq")
        
        rq_job_id = manager.enqueue_job(
            job_function='workers.rq_tasks.batch_restoration_job',
            job_kwargs={
                'file_count': request.file_count,
                'model_type': request.model_type,
                'bucket': request.bucket
            },
            priority='default',
            timeout=1800,  # 30 min timeout
            job_id=job_id
        )
        
        logger.info(f"🔧 Restoration RQ job creado: {rq_job_id} ({request.file_count} archivos)")
        
        return {
            "job_id": rq_job_id,
            "status": "queued",
            "message": f"Restauración de {request.file_count} archivos iniciada con RQ",
            "check_status_url": f"/jobs/rq/status/{rq_job_id}",
            "parameters": {
                "file_count": request.file_count,
                "model_type": request.model_type,
                "bucket": request.bucket
            }
        }
        
    except Exception as e:
        logger.error(f"Error creando restoration RQ job: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# ================================
# ENDPOINTS DE MONITOREO RQ
# ================================

@router.get("/rq/status/{job_id}")
async def get_rq_job_status(job_id: str):
    """Obtener estado de job RQ específico"""
    if not RQ_AVAILABLE:
        raise HTTPException(status_code=503, detail="RQ no está disponible")
    
    try:
        manager = get_job_queue_manager()
        job_status = manager.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail="Job no encontrado")
        
        return {
            "job_id": job_id,
            **job_status
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estado RQ job: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.get("/rq/stats")
async def get_rq_stats():
    """Obtener estadísticas de RQ"""
    if not RQ_AVAILABLE:
        raise HTTPException(status_code=503, detail="RQ no está disponible")
    
    try:
        manager = get_job_queue_manager()
        stats = manager.get_queue_stats()
        
        return {
            "system": "Redis Queue (RQ)",
            "timestamp": datetime.now().isoformat(),
            **stats
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas RQ: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.get("/rq/jobs")
async def list_rq_jobs(queue: str = "default", status: str = "all", limit: int = 50):
    """Listar jobs de RQ"""
    if not RQ_AVAILABLE:
        raise HTTPException(status_code=503, detail="RQ no está disponible")
    
    try:
        manager = get_job_queue_manager()
        jobs = manager.list_jobs(queue_name=queue, status=status)
        
        # Limitar resultados
        if limit > 0:
            jobs = jobs[:limit]
        
        return {
            "queue": queue,
            "status_filter": status,
            "total_jobs": len(jobs),
            "jobs": jobs,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listando jobs RQ: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.delete("/rq/jobs/{job_id}")
async def cancel_rq_job(job_id: str):
    """Cancelar job RQ"""
    if not RQ_AVAILABLE:
        raise HTTPException(status_code=503, detail="RQ no está disponible")
    
    try:
        manager = get_job_queue_manager()
        
        # Verificar que el job existe
        job_status = manager.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail="Job no encontrado")
        
        # Verificar que se puede cancelar
        if job_status['status'] in ['finished', 'failed']:
            raise HTTPException(status_code=400, detail=f"No se puede cancelar job en estado: {job_status['status']}")
        
        # Cancelar job
        success = manager.cancel_job(job_id)
        
        if success:
            return {
                "job_id": job_id,
                "status": "cancelled",
                "message": "Job cancelado exitosamente"
            }
        else:
            raise HTTPException(status_code=500, detail="Error cancelando job")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelando job RQ: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.post("/rq/cleanup")
async def cleanup_old_rq_jobs(max_age_hours: int = 24):
    """Limpiar jobs antiguos de RQ"""
    if not RQ_AVAILABLE:
        raise HTTPException(status_code=503, detail="RQ no está disponible")
    
    try:
        manager = get_job_queue_manager()
        cleaned_count = manager.cleanup_old_jobs(max_age_hours=max_age_hours)
        
        return {
            "message": f"Limpieza completada",
            "cleaned_jobs": cleaned_count,
            "max_age_hours": max_age_hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en limpieza RQ: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# ================================
# ENDPOINT DE SISTEMA
# ================================

@router.get("/rq/health")
async def rq_health_check():
    """Health check del sistema RQ"""
    try:
        if not RQ_AVAILABLE:
            return {
                "status": "unhealthy",
                "message": "RQ no está disponible",
                "timestamp": datetime.now().isoformat()
            }
        
        manager = get_job_queue_manager()
        
        # Intentar obtener estadísticas para verificar conexión
        stats = manager.get_queue_stats()
        
        return {
            "status": "healthy",
            "message": "RQ funcionando correctamente",
            "redis_connected": True,
            "workers_active": stats.get('workers', {}).get('active', 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Error en RQ: {str(e)}",
            "redis_connected": False,
            "timestamp": datetime.now().isoformat()
        }
