"""
Router para manejo de trabajos/jobs
REFACTORIZADO: Usa jobs_service, sin dependencias externas hardcodeadas
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import logging
import sys

# Asegurar imports
sys.path.append('/app/api')

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
