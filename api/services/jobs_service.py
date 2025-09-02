"""
Servicio especializado para manejo de trabajos/jobs
"""
import logging
import sys
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

# Asegurar que podemos importar desde el directorio api
sys.path.append('/app/api')

# Importar constantes y configuraciones
try:
    from config.constants import (
        PROCESSING_CONFIG, 
        RESPONSE_MESSAGES
    )
    from config.settings import jobs_state
    from models.schemas import ProcessingJob
    logging.info("Constantes de jobs importadas exitosamente")
except ImportError as e:
    logging.error(f"Error importando constantes de jobs: {e}")
    # Fallback con valores por defecto
    PROCESSING_CONFIG = {
        "MAX_JOBS": 100,
        "JOB_TIMEOUT": 3600,
        "CLEANUP_INTERVAL": 86400
    }
    RESPONSE_MESSAGES = {"job_created": "Trabajo creado exitosamente"}
    jobs_state = {}
    ProcessingJob = None

logger = logging.getLogger(__name__)

class JobsService:
    """Servicio centralizado para manejo de trabajos"""
    
    def __init__(self):
        self.max_jobs = PROCESSING_CONFIG.get("MAX_JOBS", 100)
        self.job_timeout = PROCESSING_CONFIG.get("JOB_TIMEOUT", 3600)
        self.cleanup_interval = PROCESSING_CONFIG.get("CLEANUP_INTERVAL", 86400)
        
    def list_jobs(self) -> Dict[str, Any]:
        """
        Listar todos los trabajos
        
        Returns:
            Dict con lista de trabajos y estadísticas
        """
        try:
            jobs = []
            
            for job_id, job in jobs_state.items():
                try:
                    # Convertir job a dict de forma robusta
                    if hasattr(job, 'dict'):
                        job_dict = job.dict()
                    elif hasattr(job, '__dict__'):
                        job_dict = job.__dict__.copy()
                    else:
                        job_dict = {
                            "job_id": job_id,
                            "status": "unknown",
                            "created_at": datetime.now().isoformat(),
                            "error": "Job format not recognized"
                        }
                    
                    # Asegurar que tenga job_id
                    job_dict["job_id"] = job_id
                    jobs.append(job_dict)
                    
                except Exception as e:
                    logger.warning(f"Error procesando job {job_id}: {e}")
                    jobs.append({
                        "job_id": job_id,
                        "status": "error",
                        "error": f"Processing error: {str(e)}"
                    })
            
            # Estadísticas
            status_counts = {}
            for job in jobs:
                status = job.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "status": "success",
                "jobs": jobs,
                "total": len(jobs),
                "statistics": {
                    "total_jobs": len(jobs),
                    "status_breakdown": status_counts,
                    "max_jobs_allowed": self.max_jobs
                }
            }
            
        except Exception as e:
            logger.error(f"Error listando trabajos: {e}")
            return {
                "status": "error",
                "message": f"Error obteniendo lista de trabajos: {str(e)}",
                "jobs": [],
                "total": 0
            }
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Obtener estado de trabajo específico
        
        Args:
            job_id: ID del trabajo
            
        Returns:
            Dict con información del trabajo
        """
        try:
            if not job_id:
                raise ValueError("ID de trabajo requerido")
            
            if job_id not in jobs_state:
                return {
                    "status": "error",
                    "message": "Trabajo no encontrado",
                    "job_id": job_id,
                    "error_code": "JOB_NOT_FOUND"
                }
            
            job = jobs_state[job_id]
            
            # Convertir job a dict de forma robusta
            if hasattr(job, 'dict'):
                job_dict = job.dict()
            elif hasattr(job, '__dict__'):
                job_dict = job.__dict__.copy()
            else:
                job_dict = {
                    "job_id": job_id,
                    "status": "unknown",
                    "created_at": datetime.now().isoformat(),
                    "message": "Job format not recognized"
                }
            
            # Asegurar que tenga job_id
            job_dict["job_id"] = job_id
            job_dict["status"] = job_dict.get("status", "unknown")
            
            return {
                "status": "success",
                "job": job_dict
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo trabajo {job_id}: {e}")
            return {
                "status": "error",
                "message": f"Error obteniendo trabajo: {str(e)}",
                "job_id": job_id,
                "error_type": type(e).__name__
            }
    
    def delete_job(self, job_id: str) -> Dict[str, Any]:
        """
        Eliminar trabajo específico
        
        Args:
            job_id: ID del trabajo a eliminar
            
        Returns:
            Dict con resultado de la operación
        """
        try:
            if not job_id:
                raise ValueError("ID de trabajo requerido")
            
            if job_id not in jobs_state:
                return {
                    "status": "error",
                    "message": "Trabajo no encontrado",
                    "job_id": job_id,
                    "error_code": "JOB_NOT_FOUND"
                }
            
            # Obtener información del job antes de eliminarlo
            job = jobs_state[job_id]
            job_info = {}
            
            try:
                if hasattr(job, 'dict'):
                    job_info = job.dict()
                elif hasattr(job, '__dict__'):
                    job_info = job.__dict__.copy()
            except Exception as e:
                logger.warning(f"No se pudo obtener info del job {job_id}: {e}")
            
            # Eliminar trabajo
            del jobs_state[job_id]
            
            logger.info(f"Trabajo {job_id} eliminado exitosamente")
            
            return {
                "status": "success",
                "message": "Trabajo eliminado exitosamente",
                "job_id": job_id,
                "deleted_job_info": job_info
            }
            
        except Exception as e:
            logger.error(f"Error eliminando trabajo {job_id}: {e}")
            return {
                "status": "error",
                "message": f"Error eliminando trabajo: {str(e)}",
                "job_id": job_id,
                "error_type": type(e).__name__
            }
    
    def create_job(self, job_type: str, **kwargs) -> Dict[str, Any]:
        """
        Crear nuevo trabajo
        
        Args:
            job_type: Tipo de trabajo
            **kwargs: Parámetros adicionales del trabajo
            
        Returns:
            Dict con información del trabajo creado
        """
        try:
            job_id = str(uuid.uuid4())
            
            # Verificar límite de trabajos
            if len(jobs_state) >= self.max_jobs:
                return {
                    "status": "error",
                    "message": f"Límite máximo de trabajos alcanzado ({self.max_jobs})",
                    "error_code": "MAX_JOBS_EXCEEDED"
                }
            
            # Crear job básico
            job_data = {
                "job_id": job_id,
                "job_type": job_type,
                "status": "created",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "progress": 0,
                "parameters": kwargs
            }
            
            # Crear instancia de ProcessingJob si está disponible
            if ProcessingJob:
                try:
                    job = ProcessingJob(**job_data)
                except Exception as e:
                    logger.warning(f"No se pudo crear ProcessingJob, usando dict: {e}")
                    job = job_data
            else:
                job = job_data
            
            # Almacenar trabajo
            jobs_state[job_id] = job
            
            logger.info(f"Trabajo {job_id} de tipo {job_type} creado exitosamente")
            
            return {
                "status": "success",
                "message": RESPONSE_MESSAGES.get("job_created", "Trabajo creado exitosamente"),
                "job_id": job_id,
                "job_type": job_type
            }
            
        except Exception as e:
            logger.error(f"Error creando trabajo tipo {job_type}: {e}")
            return {
                "status": "error",
                "message": f"Error creando trabajo: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def update_job_status(self, job_id: str, status: str, progress: int = None, **kwargs) -> Dict[str, Any]:
        """
        Actualizar estado de trabajo
        
        Args:
            job_id: ID del trabajo
            status: Nuevo estado
            progress: Progreso del trabajo (0-100)
            **kwargs: Datos adicionales para actualizar
            
        Returns:
            Dict con resultado de la actualización
        """
        try:
            if job_id not in jobs_state:
                return {
                    "status": "error",
                    "message": "Trabajo no encontrado",
                    "job_id": job_id,
                    "error_code": "JOB_NOT_FOUND"
                }
            
            job = jobs_state[job_id]
            
            # Actualizar campos
            if hasattr(job, '__dict__'):
                job.__dict__["status"] = status
                job.__dict__["updated_at"] = datetime.now().isoformat()
                if progress is not None:
                    job.__dict__["progress"] = progress
                for key, value in kwargs.items():
                    job.__dict__[key] = value
            elif isinstance(job, dict):
                job["status"] = status
                job["updated_at"] = datetime.now().isoformat()
                if progress is not None:
                    job["progress"] = progress
                job.update(kwargs)
            
            logger.info(f"Trabajo {job_id} actualizado a estado {status}")
            
            return {
                "status": "success",
                "message": "Trabajo actualizado exitosamente",
                "job_id": job_id,
                "new_status": status
            }
            
        except Exception as e:
            logger.error(f"Error actualizando trabajo {job_id}: {e}")
            return {
                "status": "error",
                "message": f"Error actualizando trabajo: {str(e)}",
                "job_id": job_id,
                "error_type": type(e).__name__
            }
    
    def cleanup_old_jobs(self, max_age_seconds: int = None) -> Dict[str, Any]:
        """
        Limpiar trabajos antiguos
        
        Args:
            max_age_seconds: Edad máxima en segundos (por defecto usa cleanup_interval)
            
        Returns:
            Dict con resultado de la limpieza
        """
        try:
            max_age = max_age_seconds or self.cleanup_interval
            current_time = datetime.now()
            jobs_to_delete = []
            
            for job_id, job in jobs_state.items():
                try:
                    # Obtener fecha de creación
                    created_at = None
                    if hasattr(job, '__dict__') and 'created_at' in job.__dict__:
                        created_at = job.__dict__.get('created_at')
                    elif isinstance(job, dict) and 'created_at' in job:
                        created_at = job.get('created_at')
                    
                    if created_at:
                        job_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        age_seconds = (current_time - job_time).total_seconds()
                        
                        if age_seconds > max_age:
                            jobs_to_delete.append(job_id)
                            
                except Exception as e:
                    logger.warning(f"Error verificando edad del job {job_id}: {e}")
            
            # Eliminar trabajos antiguos
            deleted_count = 0
            for job_id in jobs_to_delete:
                try:
                    del jobs_state[job_id]
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error eliminando job antiguo {job_id}: {e}")
            
            logger.info(f"Limpieza completada: {deleted_count} trabajos eliminados")
            
            return {
                "status": "success",
                "message": f"Limpieza completada",
                "deleted_jobs": deleted_count,
                "remaining_jobs": len(jobs_state)
            }
            
        except Exception as e:
            logger.error(f"Error en limpieza de trabajos: {e}")
            return {
                "status": "error",
                "message": f"Error en limpieza: {str(e)}",
                "error_type": type(e).__name__
            }

# Instancia global del servicio
jobs_service = JobsService()
