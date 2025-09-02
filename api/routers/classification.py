"""
Router para endpoints de clasificación de documentos
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import logging
from typing import List

# Importaciones con fallbacks mejorados
try:
    from services.classification_service import classification_service
    from config.constants import RESPONSE_MESSAGES, FILE_CONFIG
except ImportError:
    try:
        # Importación alternativa 
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from services.classification_service import classification_service
        from config.constants import RESPONSE_MESSAGES, FILE_CONFIG
    except ImportError as e:
        logging.warning(f"Error importando dependencias en classification router: {e}")
        classification_service = None
        RESPONSE_MESSAGES = {"service_unavailable": "Servicio no disponible"}
        FILE_CONFIG = {"MAX_SIZE": 50 * 1024 * 1024}

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/classify", tags=["Clasificación"])

@router.post("/document")
async def classify_document(file: UploadFile = File(...)):
    """Clasificar tipo de documento y subirlo al bucket apropiado"""
    try:
        # Verificar disponibilidad del servicio
        if classification_service is None:
            raise HTTPException(
                status_code=503, 
                detail=RESPONSE_MESSAGES.get("service_unavailable", "Servicio no disponible")
            )
            
        # Validar archivo
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten imágenes")
        
        # Validar tamaño
        file_data = await file.read()
        if len(file_data) > FILE_CONFIG["MAX_SIZE"]:
            raise HTTPException(status_code=413, detail="Archivo demasiado grande")
        
        # Clasificar documento usando el servicio
        result = classification_service.classify_document(file_data, file.filename)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clasificando documento: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.post("/batch")
async def classify_batch(files: List[UploadFile] = File(...)):
    """Clasificar múltiples documentos en lote"""
    try:
        # Verificar disponibilidad del servicio
        if classification_service is None:
            raise HTTPException(
                status_code=503, 
                detail=RESPONSE_MESSAGES.get("service_unavailable", "Servicio no disponible")
            )
        
        # Validar que hay archivos
        if not files:
            raise HTTPException(status_code=400, detail="No se proporcionaron archivos")
        
        # Preparar datos para clasificación batch
        files_data = []
        for file in files:
            # Validar tipo
            if not file.content_type or not file.content_type.startswith('image/'):
                logger.warning(f"Archivo {file.filename} no es imagen, saltando...")
                continue
                
            # Leer datos
            file_data = await file.read()
            
            # Validar tamaño
            if len(file_data) > FILE_CONFIG["MAX_SIZE"]:
                logger.warning(f"Archivo {file.filename} demasiado grande, saltando...")
                continue
                
            files_data.append((file_data, file.filename))
        
        if not files_data:
            raise HTTPException(status_code=400, detail="No hay archivos válidos para procesar")
        
        # Procesar batch usando el servicio
        result = classification_service.classify_batch(files_data)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en clasificación batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.get("/stats")
async def get_classification_stats():
    """Obtener estadísticas de clasificación"""
    try:
        # Verificar disponibilidad del servicio
        if classification_service is None:
            raise HTTPException(
                status_code=503, 
                detail=RESPONSE_MESSAGES.get("service_unavailable", "Servicio no disponible")
            )
        
        # Obtener estadísticas del servicio
        stats = classification_service.get_classification_stats()
        
        return JSONResponse(content=stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.get("/info")
async def get_classification_info():
    """Obtener información de configuración de clasificación"""
    try:
        from config.constants import CLASSIFICATION_CONFIG, BUCKETS
        
        info = {
            "status": "active",
            "service": "classification_service",
            "configuration": {
                "confidence_threshold": CLASSIFICATION_CONFIG["CONFIDENCE_THRESHOLD"],
                "quality_thresholds": CLASSIFICATION_CONFIG["QUALITY_THRESHOLDS"],
                "document_types": CLASSIFICATION_CONFIG["DOCUMENT_TYPES"],
                "available_buckets": BUCKETS
            },
            "endpoints": [
                {"path": "/classify/document", "method": "POST", "description": "Clasificar documento individual"},
                {"path": "/classify/batch", "method": "POST", "description": "Clasificar múltiples documentos"},
                {"path": "/classify/stats", "method": "GET", "description": "Estadísticas de clasificación"},
                {"path": "/classify/info", "method": "GET", "description": "Información de configuración"}
            ]
        }
        
        return JSONResponse(content=info)
        
    except Exception as e:
        logger.error(f"Error obteniendo información: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
