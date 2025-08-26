"""
Router para endpoints de clasificación de documentos
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uuid
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
    logging.warning(f"Error importando dependencias en classification router: {e}")
    image_analysis_service = None
    minio_service = None
    BUCKETS = {'degraded': 'document-degraded', 'clean': 'document-clean'}

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/classify", tags=["Clasificación"])

@router.post("/document")
async def classify_document(file: UploadFile = File(...)):
    """Clasificar tipo de documento"""
    try:
        if image_analysis_service is None or minio_service is None:
            raise HTTPException(status_code=503, detail="Servicios no disponibles")
            
        # Validar archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten imágenes")
        
        # Leer archivo
        file_data = await file.read()
        
        # Analizar imagen con el nuevo formato
        analysis = image_analysis_service.analyze_image_quality(file_data)
        classification_result = image_analysis_service.classify_document_type(file_data)
        
        # Extraer tipo de documento del resultado
        document_type = classification_result.get("type", "unknown")
        confidence = classification_result.get("confidence", 0.0)
        details = classification_result.get("details", {})
        
        # Generar filename único
        filename = f"{uuid.uuid4()}.png"
        
        # Subir a bucket correspondiente según clasificación
        if document_type == "degraded":
            bucket = BUCKETS['degraded']
        elif document_type == "clean":
            bucket = BUCKETS['clean']
        else:
            # Para casos unknown, usar degraded como fallback
            bucket = BUCKETS['degraded']
        
        # Subir archivo solo si no hubo errores de memoria
        uploaded_filename = None
        upload_error = None
        file_url = None
        download_endpoint = None
        
        if "error" not in details:
            try:
                uploaded_filename = minio_service.upload_file(file_data, bucket, filename)
                
                # Generar URLs optimizadas para visualización
                # URL directa MinIO con parámetros para visualización
                file_url = f"http://localhost:9000/{bucket}/{uploaded_filename}?response-content-disposition=inline&response-content-type=image/png"
                
                # URL alternativa a través de la API con visualización
                view_endpoint = f"/files/view/{bucket}/{uploaded_filename}"
                download_endpoint = f"/files/download/{bucket}/{uploaded_filename}"
                
            except Exception as upload_e:
                upload_error = str(upload_e)
                logger.error(f"Error subiendo archivo: {upload_e}")
        
        return JSONResponse({
            "status": "success" if "error" not in details else "partial",
            "classification": {
                "type": document_type,
                "confidence": confidence,
                "details": details
            },
            "analysis": analysis,
            "upload": {
                "bucket": bucket if uploaded_filename else None,
                "filename": uploaded_filename,
                "view_endpoint": view_endpoint, 
                "file_url": file_url,                    # ✅ NUEVA: URL directa al archivo
                "download_endpoint": download_endpoint,   # ✅ NUEVA: Endpoint de la API para descargar
                "error": upload_error
            },
            "message": f"Documento clasificado como {document_type} (confianza: {confidence:.2f})"
        })
        
    except MemoryError as e:
        logger.error(f"Error de memoria clasificando documento: {e}")
        return JSONResponse(
            status_code=507,
            content={
                "status": "error",
                "error": "Memoria insuficiente",
                "message": "La imagen es demasiado grande para procesar. Intente con una imagen de menor resolución.",
                "details": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Error clasificando documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def classify_batch(bucket_name: str):
    """Clasificar documentos en lote"""
    try:
        # Listar archivos en bucket
        files = minio_service.list_files(bucket_name)
        
        results = []
        for filename in files:
            try:
                # Descargar archivo
                file_data = minio_service.download_file(bucket_name, filename)
                
                # Clasificar con nuevo formato
                classification_result = image_analysis_service.classify_document_type(file_data)
                analysis = image_analysis_service.analyze_image_quality(file_data)
                
                results.append({
                    "filename": filename,
                    "classification": classification_result,
                    "analysis": analysis,
                    "status": "success" if "error" not in classification_result.get("details", {}) else "error"
                })

                
                
            except MemoryError as e:
                logger.error(f"Error de memoria procesando {filename}: {e}")
                results.append({
                    "filename": filename,
                    "status": "memory_error",
                    "error": "Imagen demasiado grande para procesar"
                })
            except Exception as e:
                logger.error(f"Error procesando {filename}: {e}")
                results.append({
                    "filename": filename,
                    "status": "error",
                    "error": str(e)
                })
        
        # Contar resultados exitosos y con errores
        successful = len([r for r in results if r.get("status") == "success"])
        memory_errors = len([r for r in results if r.get("status") == "memory_error"])
        other_errors = len([r for r in results if r.get("status") == "error"])
        
        return JSONResponse({
            "status": "completed",
            "bucket": bucket_name,
            "summary": {
                "total_files": len(files),
                "successful": successful,
                "memory_errors": memory_errors,
                "other_errors": other_errors
            },
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error en clasificación en lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))
