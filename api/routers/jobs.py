"""
Router para manejo de trabajos/jobs
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from models.schemas import ProcessingJob
from config import jobs_state
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["Trabajos"])

@router.get("/")
async def list_jobs():
    """Listar todos los trabajos"""
    try:
        jobs = []
        for job_id, job in jobs_state.items():
            job_dict = job.dict() if hasattr(job, 'dict') else job.__dict__
            jobs.append(job_dict)
        
        return JSONResponse({
            "jobs": jobs,
            "total": len(jobs)
        })
        
    except Exception as e:
        logger.error(f"Error listando trabajos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """Obtener estado de trabajo específico"""
    try:
        if job_id not in jobs_state:
            raise HTTPException(status_code=404, detail="Trabajo no encontrado")
        
        job = jobs_state[job_id]
        job_dict = job.dict() if hasattr(job, 'dict') else job.__dict__
        
        return JSONResponse(job_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo trabajo {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """Eliminar trabajo específico"""
    try:
        if job_id not in jobs_state:
            raise HTTPException(status_code=404, detail="Trabajo no encontrado")
        
        del jobs_state[job_id]
        
        return JSONResponse({
            "status": "success",
            "message": f"Trabajo {job_id} eliminado"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando trabajo {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
async def clear_completed_jobs():
    """Limpiar trabajos completados"""
    try:
        completed_jobs = []
        for job_id, job in list(jobs_state.items()):
            job_status = job.status if hasattr(job, 'status') else job.get('status')
            if job_status in ['completed', 'failed']:
                completed_jobs.append(job_id)
                del jobs_state[job_id]
        
        return JSONResponse({
            "status": "success",
            "cleared_jobs": completed_jobs,
            "count": len(completed_jobs)
        })
        
    except Exception as e:
        logger.error(f"Error limpiando trabajos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/summary")
async def get_jobs_summary():
    """Obtener resumen estadístico de trabajos"""
    try:
        stats = {
            "total": len(jobs_state),
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0
        }
        
        for job in jobs_state.values():
            status = job.status if hasattr(job, 'status') else job.get('status', 'unknown')
            if status in stats:
                stats[status] += 1
        
        return JSONResponse(stats)
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))
