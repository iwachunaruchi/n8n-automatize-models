"""
Router para operaciones con archivos
"""
from fastapi import APIRouter, HTTPException, Response, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import uuid
import io
import logging
import sys
import os

# Agregar path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from services.image_analysis_service import image_analysis_service
    from services.minio_service import minio_service
    from config.settings import BUCKETS
except ImportError as e:
    logging.warning(f"Error importando dependencias en files router: {e}")
    image_analysis_service = None
    minio_service = None
    BUCKETS = {
        'degraded': 'document-degraded',
        'clean': 'document-clean', 
        'restored': 'document-restored',
        'training': 'document-training'
    }

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/files", tags=["Archivos"])

@router.post("/upload")
async def upload_file(
    bucket: str,
    file: UploadFile = File(...)
):
    """Subir archivo a bucket específico"""
    try:
        # Validar bucket
        if bucket not in BUCKETS.values():
            raise HTTPException(status_code=400, detail="Bucket no válido")
        
        # Leer archivo
        file_data = await file.read()
        
        # Generar filename único
        filename = f"{uuid.uuid4()}_{file.filename}"
        
        # Subir archivo
        uploaded_filename = minio_service.upload_file(file_data, bucket, filename)
        
        return JSONResponse({
            "status": "success",
            "bucket": bucket,
            "filename": uploaded_filename,
            "original_filename": file.filename,
            "size": len(file_data)
        })
        
    except Exception as e:
        logger.error(f"Error subiendo archivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{bucket}/{filename}")
async def download_file(bucket: str, filename: str):
    """Descargar archivo específico"""
    try:
        # Validar bucket
        if bucket not in BUCKETS.values():
            raise HTTPException(status_code=400, detail="Bucket no válido")
        
        # Descargar archivo
        file_data = minio_service.download_file(bucket, filename)
        
        return StreamingResponse(
            io.BytesIO(file_data),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error descargando archivo: {e}")
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

@router.get("/list/{bucket}")
async def list_files(bucket: str, prefix: str = ""):
    """Listar archivos en bucket"""
    try:
        # Validar bucket
        if bucket not in BUCKETS.values():
            raise HTTPException(status_code=400, detail="Bucket no válido")
        
        # Listar archivos
        files = minio_service.list_files(bucket, prefix)
        
        return JSONResponse({
            "bucket": bucket,
            "prefix": prefix,
            "files": files,
            "count": len(files)
        })
        
    except Exception as e:
        logger.error(f"Error listando archivos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete/{bucket}/{filename}")
async def delete_file(bucket: str, filename: str):
    """Eliminar archivo específico"""
    try:
        # Validar bucket
        if bucket not in BUCKETS.values():
            raise HTTPException(status_code=400, detail="Bucket no válido")
        
        # Eliminar archivo
        success = minio_service.delete_file(bucket, filename)
        
        if success:
            return JSONResponse({
                "status": "success",
                "message": f"Archivo {filename} eliminado de {bucket}"
            })
        else:
            raise HTTPException(status_code=500, detail="Error eliminando archivo")
        
    except Exception as e:
        logger.error(f"Error eliminando archivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """Analizar calidad de imagen SIN subirla automáticamente"""
    try:
        if image_analysis_service is None:
            raise HTTPException(status_code=503, detail="Servicio de análisis no disponible")
            
        # Validar archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten imágenes")
        
        # Leer archivo
        file_data = await file.read()
        
        # Analizar imagen con nuevo formato
        analysis = image_analysis_service.analyze_image_quality(file_data)
        classification_result = image_analysis_service.classify_document_type(file_data)
        
        # Extraer información de clasificación
        document_type = classification_result.get("type", "unknown")
        confidence = classification_result.get("confidence", 0.0)
        details = classification_result.get("details", {})
        
        # Determinar bucket recomendado según clasificación
        if document_type == "degraded":
            bucket = BUCKETS['degraded']
        elif document_type == "clean":
            bucket = BUCKETS['clean']
        else:
            bucket = BUCKETS['degraded']  # fallback para unknown
        
        return JSONResponse({
            "status": "success" if "error" not in details else "partial",
            "filename": file.filename,
            "classification": {
                "type": document_type,
                "confidence": confidence,
                "details": details
            },
            "analysis": analysis,
            "recommended_bucket": bucket,
            "message": f"Documento clasificado como {document_type} (confianza: {confidence:.2f})"
        })
        
    except MemoryError as e:
        logger.error(f"Error de memoria analizando archivo: {e}")
        return JSONResponse(
            status_code=507,
            content={
                "status": "error",
                "error": "Memoria insuficiente", 
                "message": "La imagen es demasiado grande para procesar. Intente con una imagen de menor resolución.",
                "filename": file.filename,
                "details": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Error analizando archivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/view/{bucket}/{filename}")
async def view_file(bucket: str, filename: str):
    """Ver archivo directamente en el navegador (sin descarga)"""
    try:
        # Validar bucket
        if bucket not in BUCKETS.values():
            raise HTTPException(status_code=400, detail=f"Bucket inválido: {bucket}")
        
        # Descargar archivo desde MinIO
        file_data = minio_service.download_file(bucket, filename)
        
        # Detectar tipo de contenido basado en extensión
        content_type = "image/png"
        if filename.lower().endswith(('.jpg', '.jpeg')):
            content_type = "image/jpeg"
        elif filename.lower().endswith('.gif'):
            content_type = "image/gif"
        elif filename.lower().endswith('.webp'):
            content_type = "image/webp"
        
        # Retornar con headers para visualización directa
        return Response(
            content=file_data,
            media_type=content_type,
            headers={
                "Content-Disposition": "inline; filename=" + filename,  # ✅ 'inline' = visualizar
                "Cache-Control": "public, max-age=3600"  # Cache por 1 hora
            }
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    except Exception as e:
        logger.error(f"Error visualizando archivo {bucket}/{filename}: {e}")
        raise HTTPException(status_code=500, detail="Error accediendo al archivo")


@router.get("/download/{bucket}/{filename}")
async def download_file(bucket: str, filename: str):
    """Descargar archivo específico (fuerza descarga)"""
    try:
        # Validar bucket
        if bucket not in BUCKETS.values():
            raise HTTPException(status_code=400, detail=f"Bucket inválido: {bucket}")
        
        # Descargar archivo desde MinIO
        file_data = minio_service.download_file(bucket, filename)
        
        # Retornar con headers para descarga forzada
        return Response(
            content=file_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=" + filename  # ✅ 'attachment' = descargar
            }
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    except Exception as e:
        logger.error(f"Error descargando archivo {bucket}/{filename}: {e}")
        raise HTTPException(status_code=500, detail="Error accediendo al archivo")

