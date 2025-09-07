"""
Router para operaciones con archivos
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
import io
import logging
from typing import Optional

try:
    from services.file_management_service import file_management_service
    from config.constants import BUCKETS, RESPONSE_MESSAGES
except ImportError as e:
    logging.warning(f"Error importando dependencias en files router: {e}")
    file_management_service = None
    BUCKETS = {
        'degraded': 'document-degraded',
        'clean': 'document-clean', 
        'restored': 'document-restored',
        'training': 'document-training'
    }
    RESPONSE_MESSAGES = {"service_unavailable": "Servicio no disponible"}

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/files", tags=["Archivos"])

@router.post("/upload")
async def upload_file(
    bucket: str,
    file: UploadFile = File(...)
):
    """Subir archivo a bucket específico"""
    try:
        # Verificar disponibilidad del servicio
        if file_management_service is None:
            raise HTTPException(
                status_code=503, 
                detail=RESPONSE_MESSAGES.get("service_unavailable", "Servicio no disponible")
            )
        
        # Validar bucket
        if bucket not in BUCKETS.values():
            available_buckets = ", ".join(BUCKETS.values())
            raise HTTPException(
                status_code=400, 
                detail=f"Bucket no válido. Disponibles: {available_buckets}"
            )
        
        # Leer archivo
        file_data = await file.read()
        
        # Subir usando el servicio
        result = file_management_service.upload_file(
            file_data=file_data,
            bucket=bucket,
            filename=file.filename,
            validate_type=True
        )
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error subiendo archivo: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.get("/download/{bucket}/{filename}")
async def download_file(bucket: str, filename: str):
    """Descargar archivo específico"""
    try:
        # Verificar disponibilidad del servicio
        if file_management_service is None:
            raise HTTPException(
                status_code=503, 
                detail=RESPONSE_MESSAGES.get("service_unavailable", "Servicio no disponible")
            )
        
        # Validar bucket
        if bucket not in BUCKETS.values():
            raise HTTPException(status_code=400, detail="Bucket no válido")
        
        # Descargar usando el servicio
        file_data, content_type = file_management_service.download_file(bucket, filename)
        
        return StreamingResponse(
            io.BytesIO(file_data),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error descargando archivo: {e}")
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

@router.get("/list/{bucket}")
async def list_files(
    bucket: str, 
    prefix: Optional[str] = Query("", description="Prefijo para filtrar archivos"),
    limit: Optional[int] = Query(None, description="Límite de archivos a retornar")
):
    """Listar archivos en bucket"""
    try:
        # Verificar disponibilidad del servicio
        if file_management_service is None:
            raise HTTPException(
                status_code=503, 
                detail=RESPONSE_MESSAGES.get("service_unavailable", "Servicio no disponible")
            )
        
        # Validar bucket
        if bucket not in BUCKETS.values():
            raise HTTPException(status_code=400, detail="Bucket no válido")
        
        # Listar usando el servicio
        result = file_management_service.list_files(
            bucket=bucket,
            prefix=prefix,
            limit=limit
        )
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listando archivos: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """Analizar archivo sin subirlo"""
    try:
        # Verificar disponibilidad del servicio
        if file_management_service is None:
            raise HTTPException(
                status_code=503, 
                detail=RESPONSE_MESSAGES.get("service_unavailable", "Servicio no disponible")
            )
        
        # Leer archivo
        file_data = await file.read()
        
        # Analizar usando el servicio
        result = file_management_service.analyze_file(file_data, file.filename)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analizando archivo: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.delete("/delete/{bucket}/{filename}")
async def delete_file(bucket: str, filename: str):
    """Eliminar archivo específico"""
    try:
        # Verificar disponibilidad del servicio
        if file_management_service is None:
            raise HTTPException(
                status_code=503, 
                detail=RESPONSE_MESSAGES.get("service_unavailable", "Servicio no disponible")
            )
        
        # Validar bucket
        if bucket not in BUCKETS.values():
            raise HTTPException(status_code=400, detail="Bucket no válido")
        
        # Eliminar usando el servicio
        result = file_management_service.delete_file(bucket, filename)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando archivo: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.get("/stats")
async def get_storage_stats():
    """Obtener estadísticas de almacenamiento"""
    try:
        # Verificar disponibilidad del servicio
        if file_management_service is None:
            raise HTTPException(
                status_code=503, 
                detail=RESPONSE_MESSAGES.get("service_unavailable", "Servicio no disponible")
            )
        
        # Obtener estadísticas del servicio
        stats = file_management_service.get_storage_stats()
        
        return JSONResponse(content=stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.get("/info")
async def get_files_info():
    """Obtener información de configuración de archivos"""
    try:
        from config.constants import FILE_CONFIG
        
        info = {
            "status": "active",
            "service": "file_management_service",
            "configuration": {
                "max_file_size": FILE_CONFIG["MAX_SIZE"],
                "allowed_extensions": FILE_CONFIG["ALLOWED_EXTENSIONS"],
                "allowed_mime_types": FILE_CONFIG["ALLOWED_MIME_TYPES"],
                "upload_timeout": FILE_CONFIG["UPLOAD_TIMEOUT"],
                "available_buckets": BUCKETS
            },
            "endpoints": [
                {"path": "/files/upload", "method": "POST", "description": "Subir archivo a bucket"},
                {"path": "/files/download/{bucket}/{filename}", "method": "GET", "description": "Descargar archivo"},
                {"path": "/files/list/{bucket}", "method": "GET", "description": "Listar archivos en bucket"},
                {"path": "/files/analyze", "method": "POST", "description": "Analizar archivo sin subirlo"},
                {"path": "/files/delete/{bucket}/{filename}", "method": "DELETE", "description": "Eliminar archivo"},
                {"path": "/files/stats", "method": "GET", "description": "Estadísticas de almacenamiento"},
                {"path": "/files/info", "method": "GET", "description": "Información de configuración"}
            ]
        }
        
        return JSONResponse(content=info)
        
    except Exception as e:
        logger.error(f"Error obteniendo información: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
