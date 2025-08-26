"""
Router para endpoints de restauración de documentos
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uuid
import io
import logging
from datetime import datetime
import sys
import os

# Agregar path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from services.image_analysis_service import image_analysis_service
    from services.minio_service import minio_service
    from models.schemas import ProcessingJob
    from config.settings import BUCKETS, jobs_state
except ImportError as e:
    logging.warning(f"Error importando dependencias en restoration router: {e}")
    image_analysis_service = None
    minio_service = None
    ProcessingJob = None
    BUCKETS = {'restored': 'document-restored', 'degraded': 'document-degraded'}
    jobs_state = {}

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/restore", tags=["Restauración"])

@router.post("/document")
async def restore_document(file: UploadFile = File(...)):
    """Restaurar un documento individual"""
    try:
        # Validar archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten imágenes")
        
        # Leer archivo
        file_data = await file.read()
        
        # Restaurar imagen
        restored_data = image_analysis_service.restore_image(file_data)
        
        # Generar filename único para archivo restaurado
        filename = f"restored_{uuid.uuid4()}.png"
        
        # Subir archivo restaurado
        minio_service.upload_file(restored_data, BUCKETS['restored'], filename)
        
        # Retornar imagen restaurada
        return StreamingResponse(
            io.BytesIO(restored_data),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error restaurando documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/document/async")
async def restore_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Restaurar documento de forma asíncrona"""
    try:
        # Validar archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten imágenes")
        
        # Crear job
        job_id = str(uuid.uuid4())
        input_filename = f"input_{job_id}.png"
        
        # Leer y subir archivo original
        file_data = await file.read()
        minio_service.upload_file(file_data, BUCKETS['degraded'], input_filename)
        
        # Crear job en estado
        job = ProcessingJob(
            job_id=job_id,
            status="pending",
            created_at=datetime.now(),
            input_file=input_filename
        )
        jobs_state[job_id] = job
        
        # Agregar tarea en background
        background_tasks.add_task(process_restoration_job, job_id, file_data)
        
        return JSONResponse({
            "job_id": job_id,
            "status": "pending",
            "message": "Trabajo de restauración iniciado"
        })
        
    except Exception as e:
        logger.error(f"Error iniciando restauración asíncrona: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def restore_batch(
    background_tasks: BackgroundTasks,
    bucket_name: str,
    output_bucket: str = "document-restored"
):
    """Restaurar documentos en lote"""
    try:
        # Listar archivos en bucket
        files = minio_service.list_files(bucket_name)
        
        if not files:
            raise HTTPException(status_code=404, detail="No se encontraron archivos en el bucket")
        
        # Crear job para el lote
        job_id = str(uuid.uuid4())
        job = ProcessingJob(
            job_id=job_id,
            status="pending",
            created_at=datetime.now(),
            input_file=f"batch_{bucket_name}",
            generated_count=len(files)
        )
        jobs_state[job_id] = job
        
        # Procesar en background
        background_tasks.add_task(process_batch_restoration, job_id, bucket_name, output_bucket, files)
        
        return JSONResponse({
            "job_id": job_id,
            "status": "pending",
            "total_files": len(files),
            "message": "Trabajo de restauración en lote iniciado"
        })
        
    except Exception as e:
        logger.error(f"Error iniciando restauración en lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_restoration_job(job_id: str, file_data: bytes):
    """Procesar trabajo de restauración"""
    try:
        # Actualizar estado
        jobs_state[job_id].status = "processing"
        
        # Restaurar imagen
        restored_data = image_analysis_service.restore_image(file_data)
        
        # Subir resultado
        output_filename = f"restored_{job_id}.png"
        minio_service.upload_file(restored_data, BUCKETS['restored'], output_filename)
        
        # Actualizar estado final
        jobs_state[job_id].status = "completed"
        jobs_state[job_id].completed_at = datetime.now()
        jobs_state[job_id].output_file = output_filename
        
    except Exception as e:
        logger.error(f"Error procesando trabajo {job_id}: {e}")
        jobs_state[job_id].status = "failed"
        jobs_state[job_id].error = str(e)

async def process_batch_restoration(job_id: str, bucket_name: str, output_bucket: str, files: list):
    """Procesar restauración en lote"""
    try:
        # Actualizar estado
        jobs_state[job_id].status = "processing"
        
        processed = 0
        for filename in files:
            try:
                # Descargar archivo
                file_data = minio_service.download_file(bucket_name, filename)
                
                # Restaurar
                restored_data = image_analysis_service.restore_image(file_data)
                
                # Subir resultado
                output_filename = f"restored_{filename}"
                minio_service.upload_file(restored_data, output_bucket, output_filename)
                
                processed += 1
                
            except Exception as e:
                logger.error(f"Error procesando {filename}: {e}")
        
        # Actualizar estado final
        jobs_state[job_id].status = "completed"
        jobs_state[job_id].completed_at = datetime.now()
        jobs_state[job_id].generated_count = processed
        
    except Exception as e:
        logger.error(f"Error procesando lote {job_id}: {e}")
        jobs_state[job_id].status = "failed"
        jobs_state[job_id].error = str(e)
