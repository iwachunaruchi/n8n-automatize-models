"""
Router para endpoints de restauración de documentos
REFACTORIZADO: Usa restoration_service, sin HTTP requests internos
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import logging
import sys

# Asegurar imports
sys.path.append('/app/api')

try:
    from services.restoration_service import restoration_service
    from services.jobs_service import jobs_service
    from config.constants import RESPONSE_MESSAGES, FILE_CONFIG
    SERVICES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Error importando servicios en restoration router: {e}")
    restoration_service = None
    jobs_service = None
    SERVICES_AVAILABLE = False
    RESPONSE_MESSAGES = {}
    FILE_CONFIG = {"MAX_SIZE": 50 * 1024 * 1024}

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/restore", tags=["Restauración"])

@router.post("/document")
async def restore_document(file: UploadFile = File(...)):
    """Restaurar un documento individual usando restoration_service"""
    try:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicios de restauración no disponibles")
        
        # Validar archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten imágenes")
        
        # Validar tamaño
        file_size = 0
        file_data = await file.read()
        file_size = len(file_data)
        
        if file_size > FILE_CONFIG["MAX_SIZE"]:
            raise HTTPException(status_code=413, detail="Archivo demasiado grande")
        
        # Restaurar usando el servicio
        result = restoration_service.restore_document(file_data, file.filename)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restaurando documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/document/async")
async def restore_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Restaurar documento de forma asíncrona usando servicios"""
    try:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicios de restauración no disponibles")
        
        # Validar archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten imágenes")
        
        file_data = await file.read()
        
        if len(file_data) > FILE_CONFIG["MAX_SIZE"]:
            raise HTTPException(status_code=413, detail="Archivo demasiado grande")
        
        # Crear trabajo usando jobs_service
        job_result = jobs_service.create_job(
            "document_restoration",
            filename=file.filename,
            file_size=len(file_data)
        )
        
        if job_result["status"] == "error":
            raise HTTPException(status_code=500, detail=job_result["message"])
        
        job_id = job_result["job_id"]
        
        # Agregar tarea en background
        background_tasks.add_task(process_restoration_job_with_service, job_id, file_data, file.filename)
        
        return JSONResponse({
            "job_id": job_id,
            "status": "pending",
            "message": RESPONSE_MESSAGES.get("job_created", "Trabajo de restauración iniciado")
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error iniciando restauración asíncrona: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def restore_batch(
    background_tasks: BackgroundTasks,
    bucket_name: str,
    output_bucket: str = "document-restored"
):
    """Restaurar documentos en lote usando servicios"""
    try:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicios de restauración no disponibles")
        
        # Crear trabajo usando jobs_service
        job_result = jobs_service.create_job(
            "batch_restoration",
            source_bucket=bucket_name,
            output_bucket=output_bucket
        )
        
        if job_result["status"] == "error":
            raise HTTPException(status_code=500, detail=job_result["message"])
        
        job_id = job_result["job_id"]
        
        # Procesar en background
        background_tasks.add_task(process_batch_restoration_with_service, job_id, bucket_name, output_bucket)
        
        return JSONResponse({
            "job_id": job_id,
            "status": "pending",
            "message": RESPONSE_MESSAGES.get("job_created", "Trabajo de restauración en lote iniciado")
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error iniciando restauración en lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info")
async def get_restoration_info():
    """Obtener información del servicio de restauración"""
    try:
        if not SERVICES_AVAILABLE:
            return JSONResponse({
                "status": "error",
                "message": "Servicios de restauración no disponibles"
            })
        
        return JSONResponse(restoration_service.get_restoration_stats())
        
    except Exception as e:
        logger.error(f"Error obteniendo información de restauración: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===============================================
# FUNCIONES DE PROCESAMIENTO EN BACKGROUND
# ===============================================

async def process_restoration_job_with_service(job_id: str, file_data: bytes, filename: str = None):
    """Procesar trabajo de restauración usando restoration_service"""
    try:
        # Actualizar estado del trabajo
        jobs_service.update_job_status(job_id, "processing", progress=10)
        
        # Restaurar usando el servicio
        result = restoration_service.restore_document(file_data, filename)
        
        if result["status"] == "success":
            # Actualizar estado final exitoso
            jobs_service.update_job_status(
                job_id, 
                "completed", 
                progress=100,
                result=result["result"],
                output_file=result["result"]["restored_filename"]
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
        logger.error(f"Error procesando trabajo {job_id}: {e}")
        jobs_service.update_job_status(
            job_id, 
            "failed", 
            progress=0,
            error=str(e)
        )

async def process_batch_restoration_with_service(job_id: str, bucket_name: str, output_bucket: str):
    """Procesar restauración en lote usando servicios"""
    try:
        from services.minio_service import minio_service
        
        # Actualizar estado inicial
        jobs_service.update_job_status(job_id, "processing", progress=5)
        
        # Obtener lista de archivos
        files = minio_service.list_files(bucket_name)
        
        if not files:
            jobs_service.update_job_status(
                job_id, 
                "failed", 
                progress=0,
                error="No se encontraron archivos en el bucket"
            )
            return
        
        # Procesar archivos por lotes
        files_data = []
        total_files = len(files)
        
        for i, filename in enumerate(files):
            try:
                # Descargar archivo
                file_data = minio_service.download_file(bucket_name, filename)
                files_data.append(file_data)
                
                # Actualizar progreso
                progress = int(((i + 1) / total_files) * 50)  # 50% para descarga
                jobs_service.update_job_status(job_id, "processing", progress=progress)
                
            except Exception as e:
                logger.error(f"Error descargando {filename}: {e}")
        
        # Restaurar en lote
        batch_result = restoration_service.restore_batch(files_data, job_id)
        
        if batch_result["status"] == "completed":
            # Actualizar estado final exitoso
            jobs_service.update_job_status(
                job_id, 
                "completed", 
                progress=100,
                result=batch_result,
                batch_summary=batch_result["summary"]
            )
        else:
            # Actualizar estado de error
            jobs_service.update_job_status(
                job_id, 
                "failed", 
                progress=50,
                error=batch_result.get("message", "Error en procesamiento por lotes")
            )
        
    except Exception as e:
        logger.error(f"Error procesando lote {job_id}: {e}")
        jobs_service.update_job_status(
            job_id, 
            "failed", 
            progress=0,
            error=str(e)
        )
