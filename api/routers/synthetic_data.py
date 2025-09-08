"""
Router para generación de datos sintéticos
REFACTORIZADO: Usa synthetic_data_service y jobs_service, sin HTTP requests internos
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import sys
import io

# Asegurar imports
sys.path.append('/app/api')

try:
    from services.synthetic_data_service import synthetic_data_service
    from services.jobs_service import jobs_service
    from config.constants import SYNTHETIC_DATA_CONFIG, BUCKETS, RESPONSE_MESSAGES, FILE_CONFIG
    SERVICES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Error importando servicios en synthetic_data router: {e}")
    synthetic_data_service = None
    jobs_service = None
    SERVICES_AVAILABLE = False
    SYNTHETIC_DATA_CONFIG = {}
    BUCKETS = {}
    RESPONSE_MESSAGES = {}
    FILE_CONFIG = {"MAX_SIZE": 50 * 1024 * 1024}

logger = logging.getLogger(__name__)

# Importar sistema RQ
try:
    from rq_job_system import get_job_queue_manager
    RQ_AVAILABLE = True
    job_manager = get_job_queue_manager()
except ImportError as e:
    logger.error(f"Error importando RQ system: {e}")
    RQ_AVAILABLE = False
    job_manager = None

router = APIRouter(prefix="/synthetic", tags=["Datos Sintéticos"])

@router.post("/noise")
async def add_noise_to_image(
    noise_type: str = "gaussian",
    intensity: float = 0.1,
    file: UploadFile = File(...)
):
    """Agregar ruido a una imagen usando synthetic_data_service"""
    try:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicios de datos sintéticos no disponibles")
        
        # Validar archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten imágenes")
        
        # Validar tamaño
        file_data = await file.read()
        if len(file_data) > FILE_CONFIG["MAX_SIZE"]:
            raise HTTPException(status_code=413, detail="Archivo demasiado grande")
        
        # Aplicar ruido usando el servicio
        result = synthetic_data_service.add_noise(file_data, noise_type, intensity)
        
        if result["status"] == "error":
            if result.get("error_code") in ["INVALID_NOISE_TYPE", "INVALID_INTENSITY"]:
                raise HTTPException(status_code=400, detail=result["message"])
            else:
                raise HTTPException(status_code=500, detail=result["message"])
        
        # Retornar imagen con ruido
        return StreamingResponse(
            io.BytesIO(result["data"]),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=noisy_{file.filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error agregando ruido: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/degrade")
async def degrade_image(
    degradation_type: str = "mixed",
    file: UploadFile = File(...)
):
    """Degradar imagen limpia usando synthetic_data_service"""
    try:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicios de datos sintéticos no disponibles")
        
        # Validar archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten imágenes")
        
        # Leer archivo
        file_data = await file.read()
        
        if len(file_data) > FILE_CONFIG["MAX_SIZE"]:
            raise HTTPException(status_code=413, detail="Archivo demasiado grande")
        
        # Degradar imagen usando el servicio
        result = synthetic_data_service.generate_degraded_version(file_data, degradation_type)
        
        if result["status"] == "error":
            if result.get("error_code") in ["INVALID_DEGRADATION_TYPE", "DECODE_ERROR"]:
                raise HTTPException(status_code=400, detail=result["message"])
            else:
                raise HTTPException(status_code=500, detail=result["message"])
        
        return JSONResponse({
            "status": "success",
            "message": result["message"],
            "result": result["result"]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error degradando imagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training-pairs")
async def generate_training_pairs(
    clean_bucket: str,
    count: int = 10
):
    """Generar pares de entrenamiento - MIGRADO A RQ"""
    try:
        if not RQ_AVAILABLE:
            raise HTTPException(status_code=503, detail="Sistema RQ no disponible")
        
        # Validar parámetros usando constantes
        count_limits = SYNTHETIC_DATA_CONFIG.get("COUNT_LIMITS", {"min": 1, "max": 1000})
        if count < count_limits["min"] or count > count_limits["max"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Count debe estar entre {count_limits['min']} y {count_limits['max']}"
            )
        
        # Validar bucket
        if clean_bucket not in BUCKETS.values():
            raise HTTPException(status_code=400, detail="Bucket no válido")
        
        # Enviar job a RQ
        job_data = {
            "clean_bucket": clean_bucket,
            "requested_count": count,
            "job_type": "training_pairs_generation"
        }
        
        job_id = job_manager.enqueue_job(
            'workers.tasks.synthetic_data_tasks.generate_synthetic_data_job',
            job_kwargs=job_data,
            priority='default'
        )
        
        return JSONResponse({
            "job_id": job_id,
            "status": "queued",
            "requested_count": count,
            "source_bucket": clean_bucket,
            "message": "Trabajo encolado en RQ exitosamente",
            "system": "rq"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error iniciando generación de pares: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/augment")
async def augment_dataset(
    bucket: str,
    target_count: int = 100
):
    """Aumentar dataset mediante augmentación - MIGRADO A RQ"""
    try:
        if not RQ_AVAILABLE:
            raise HTTPException(status_code=503, detail="Sistema RQ no disponible")
        
        # Validar bucket
        if bucket not in BUCKETS.values():
            raise HTTPException(status_code=400, detail="Bucket no válido")
        
        # Validar parámetros usando constantes
        count_limits = SYNTHETIC_DATA_CONFIG.get("COUNT_LIMITS", {"augment_min": 10, "augment_max": 10000})
        if target_count < count_limits["augment_min"] or target_count > count_limits["augment_max"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Target count debe estar entre {count_limits['augment_min']} y {count_limits['augment_max']}"
            )
        
        # Enviar job a RQ
        job_data = {
            "bucket": bucket,
            "target_count": target_count,
            "job_type": "dataset_augmentation"
        }
        
        job_id = job_manager.enqueue_job(
            'workers.tasks.synthetic_data_tasks.augment_dataset_job',
            job_kwargs=job_data,
            priority='default'
        )
        
        return JSONResponse({
            "job_id": job_id,
            "status": "queued",
            "target_count": target_count,
            "bucket": bucket,
            "message": "Trabajo de augmentación encolado en RQ exitosamente",
            "system": "rq"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error iniciando augmentación: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/{bucket}")
async def get_dataset_stats(bucket: str):
    """Obtener estadísticas del dataset usando synthetic_data_service"""
    try:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicios de datos sintéticos no disponibles")
        
        result = synthetic_data_service.get_dataset_stats(bucket)
        
        if result["status"] == "error":
            if result.get("error_code") == "INVALID_BUCKET":
                raise HTTPException(status_code=400, detail=result["message"])
            elif result.get("error_code") == "MINIO_UNAVAILABLE":
                raise HTTPException(status_code=503, detail=result["message"])
            else:
                raise HTTPException(status_code=500, detail=result["message"])
        
        return JSONResponse(result["statistics"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de {bucket}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info")
async def get_synthetic_data_info():
    """Obtener información del servicio de datos sintéticos"""
    try:
        if not SERVICES_AVAILABLE:
            return JSONResponse({
                "status": "error",
                "message": "Servicios de datos sintéticos no disponibles"
            })
        
        return JSONResponse(synthetic_data_service.get_service_info())
        
    except Exception as e:
        logger.error(f"Error obteniendo información de datos sintéticos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===============================================
# FUNCIONES DE PROCESAMIENTO EN BACKGROUND
# ===============================================

async def process_training_pairs_with_service(job_id: str, clean_bucket: str, count: int):
    """Procesar generación de pares de entrenamiento usando synthetic_data_service"""
    try:
        # Actualizar estado del trabajo
        jobs_service.update_job_status(job_id, "processing", progress=10)
        
        # Generar pares usando el servicio
        result = synthetic_data_service.generate_training_pairs(clean_bucket, count)
        
        if result["status"] == "success":
            # Actualizar estado final exitoso
            jobs_service.update_job_status(
                job_id, 
                "completed", 
                progress=100,
                result=result,
                generated_count=result["generated_count"]
            )
        else:
            # Actualizar estado de error
            jobs_service.update_job_status(
                job_id, 
                "failed", 
                progress=0,
                error=result["message"]
            )
        
    except Exception as e:
        logger.error(f"Error procesando pares {job_id}: {e}")
        jobs_service.update_job_status(
            job_id, 
            "failed", 
            progress=0,
            error=str(e)
        )

async def process_augmentation_with_service(job_id: str, bucket: str, target_count: int):
    """Procesar augmentación de dataset usando synthetic_data_service"""
    try:
        # Actualizar estado del trabajo
        jobs_service.update_job_status(job_id, "processing", progress=10)
        
        # Realizar augmentación usando el servicio
        result = synthetic_data_service.augment_dataset(bucket, target_count)
        
        if result["status"] == "success":
            # Actualizar estado final exitoso
            jobs_service.update_job_status(
                job_id, 
                "completed", 
                progress=100,
                result=result,
                generated_count=result["generated_count"]
            )
        else:
            # Actualizar estado de error
            jobs_service.update_job_status(
                job_id, 
                "failed", 
                progress=0,
                error=result["message"]
            )
        
    except Exception as e:
        logger.error(f"Error procesando augmentación {job_id}: {e}")
        jobs_service.update_job_status(
            job_id, 
            "failed", 
            progress=0,
            error=str(e)
        )
