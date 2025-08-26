"""
Router para generación de datos sintéticos
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from services import synthetic_data_service, minio_service
from models.schemas import ProcessingJob
from config import BUCKETS, jobs_state
import uuid
import io
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/synthetic", tags=["Datos Sintéticos"])

@router.post("/noise")
async def add_noise_to_image(
    noise_type: str = "gaussian",
    intensity: float = 0.1,
    file: UploadFile = File(...)
):
    """Agregar ruido a una imagen"""
    try:
        # Validar archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten imágenes")
        
        # Validar parámetros
        if noise_type not in ["gaussian", "salt_pepper", "blur"]:
            raise HTTPException(status_code=400, detail="Tipo de ruido no válido")
        
        if not 0.01 <= intensity <= 1.0:
            raise HTTPException(status_code=400, detail="Intensidad debe estar entre 0.01 y 1.0")
        
        # Leer archivo
        file_data = await file.read()
        
        # Aplicar ruido
        noisy_data = synthetic_data_service.add_noise(file_data, noise_type, intensity)
        
        # Retornar imagen con ruido
        return StreamingResponse(
            io.BytesIO(noisy_data),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=noisy_{file.filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error agregando ruido: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/degrade")
async def degrade_image(
    degradation_type: str = "mixed",
    file: UploadFile = File(...)
):
    """Degradar imagen limpia"""
    try:
        # Validar archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten imágenes")
        
        # Leer archivo
        file_data = await file.read()
        
        # Degradar imagen
        degraded_data = synthetic_data_service.generate_degraded_version(file_data, degradation_type)
        
        # Subir original y degradada
        original_filename = f"clean_{uuid.uuid4()}.png"
        degraded_filename = f"degraded_{uuid.uuid4()}.png"
        
        minio_service.upload_file(file_data, BUCKETS['clean'], original_filename)
        minio_service.upload_file(degraded_data, BUCKETS['degraded'], degraded_filename)
        
        return JSONResponse({
            "status": "success",
            "original_file": original_filename,
            "degraded_file": degraded_filename,
            "degradation_type": degradation_type
        })
        
    except Exception as e:
        logger.error(f"Error degradando imagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training-pairs")
async def generate_training_pairs(
    background_tasks: BackgroundTasks,
    clean_bucket: str,
    count: int = 10
):
    """Generar pares de entrenamiento"""
    try:
        # Validar parámetros
        if count < 1 or count > 1000:
            raise HTTPException(status_code=400, detail="Count debe estar entre 1 y 1000")
        
        # Crear job
        job_id = str(uuid.uuid4())
        job = ProcessingJob(
            job_id=job_id,
            status="pending",
            created_at=datetime.now(),
            input_file=clean_bucket,
            generation_type="training_pairs",
            generated_count=0
        )
        jobs_state[job_id] = job
        
        # Procesar en background
        background_tasks.add_task(process_training_pairs, job_id, clean_bucket, count)
        
        return JSONResponse({
            "job_id": job_id,
            "status": "pending",
            "requested_count": count,
            "source_bucket": clean_bucket
        })
        
    except Exception as e:
        logger.error(f"Error iniciando generación de pares: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/augment")
async def augment_dataset(
    background_tasks: BackgroundTasks,
    bucket: str,
    target_count: int = 100
):
    """Aumentar dataset mediante augmentación"""
    try:
        # Validar bucket
        if bucket not in BUCKETS.values():
            raise HTTPException(status_code=400, detail="Bucket no válido")
        
        # Validar parámetros
        if target_count < 10 or target_count > 10000:
            raise HTTPException(status_code=400, detail="Target count debe estar entre 10 y 10000")
        
        # Crear job
        job_id = str(uuid.uuid4())
        job = ProcessingJob(
            job_id=job_id,
            status="pending",
            created_at=datetime.now(),
            input_file=bucket,
            generation_type="augmentation",
            generated_count=0
        )
        jobs_state[job_id] = job
        
        # Procesar en background
        background_tasks.add_task(process_augmentation, job_id, bucket, target_count)
        
        return JSONResponse({
            "job_id": job_id,
            "status": "pending",
            "target_count": target_count,
            "bucket": bucket
        })
        
    except Exception as e:
        logger.error(f"Error iniciando augmentación: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/{bucket}")
async def get_dataset_stats(bucket: str):
    """Obtener estadísticas del dataset"""
    try:
        # Validar bucket
        if bucket not in BUCKETS.values():
            raise HTTPException(status_code=400, detail="Bucket no válido")
        
        # Contar archivos
        files = minio_service.list_files(bucket)
        
        # Estadísticas básicas
        stats = {
            "bucket": bucket,
            "total_files": len(files),
            "file_types": {}
        }
        
        # Contar por tipo de archivo
        for filename in files:
            if filename.startswith('clean_'):
                stats["file_types"]["clean"] = stats["file_types"].get("clean", 0) + 1
            elif filename.startswith('degraded_'):
                stats["file_types"]["degraded"] = stats["file_types"].get("degraded", 0) + 1
            elif filename.startswith('aug_'):
                stats["file_types"]["augmented"] = stats["file_types"].get("augmented", 0) + 1
            elif filename.startswith('restored_'):
                stats["file_types"]["restored"] = stats["file_types"].get("restored", 0) + 1
            else:
                stats["file_types"]["other"] = stats["file_types"].get("other", 0) + 1
        
        return JSONResponse(stats)
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_training_pairs(job_id: str, clean_bucket: str, count: int):
    """Procesar generación de pares de entrenamiento"""
    try:
        # Actualizar estado
        jobs_state[job_id].status = "processing"
        
        # Generar pares
        result = synthetic_data_service.generate_training_pairs(clean_bucket, count)
        
        # Actualizar estado final
        jobs_state[job_id].status = "completed"
        jobs_state[job_id].completed_at = datetime.now()
        jobs_state[job_id].generated_count = result["generated_count"]
        
    except Exception as e:
        logger.error(f"Error procesando pares {job_id}: {e}")
        jobs_state[job_id].status = "failed"
        jobs_state[job_id].error = str(e)

async def process_augmentation(job_id: str, bucket: str, target_count: int):
    """Procesar augmentación de dataset"""
    try:
        # Actualizar estado
        jobs_state[job_id].status = "processing"
        
        # Realizar augmentación
        result = synthetic_data_service.augment_dataset(bucket, target_count)
        
        # Actualizar estado final
        jobs_state[job_id].status = "completed"
        jobs_state[job_id].completed_at = datetime.now()
        jobs_state[job_id].generated_count = result["generated_count"]
        
    except Exception as e:
        logger.error(f"Error procesando augmentación {job_id}: {e}")
        jobs_state[job_id].status = "failed"
        jobs_state[job_id].error = str(e)
