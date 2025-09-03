"""
Router para gestión de modelos entrenados
"""

from fastapi import APIRouter, HTTPException, Response
from typing import List, Optional
import logging
import sys
import os

# Agregar path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar servicios
try:
    from services.minio_service import minio_service
except ImportError:
    # Fallback import
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from services.minio_service import minio_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

@router.get("/list")
async def list_models(layer: Optional[str] = None):
    """
    Listar modelos disponibles
    
    Args:
        layer: Filtrar por capa específica (opcional)
    
    Returns:
        Lista de modelos con información detallada
    """
    try:
        models = minio_service.list_models(layer)
        
        return {
            "status": "success",
            "total_models": len(models),
            "filter_layer": layer,
            "models": models
        }
        
    except Exception as e:
        logger.error(f"Error listando modelos: {e}")
        raise HTTPException(status_code=500, detail=f"Error listando modelos: {str(e)}")

@router.get("/list/{layer}")
async def list_models_by_layer(layer: str):
    """
    Listar modelos de una capa específica
    
    Args:
        layer: Número de capa (1, 2, etc.)
    
    Returns:
        Lista de modelos de la capa especificada
    """
    try:
        models = minio_service.list_models(layer)
        
        return {
            "status": "success",
            "layer": layer,
            "total_models": len(models),
            "models": models
        }
        
    except Exception as e:
        logger.error(f"Error listando modelos de capa {layer}: {e}")
        raise HTTPException(status_code=500, detail=f"Error listando modelos: {str(e)}")

@router.get("/download/{layer}/{model_name}")
async def download_model(layer: str, model_name: str):
    """
    Descargar un modelo específico
    
    Args:
        layer: Número de capa
        model_name: Nombre del archivo del modelo
    
    Returns:
        Archivo del modelo para descarga
    """
    try:
        model_data = minio_service.download_model(layer, model_name)
        
        return Response(
            content=model_data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={model_name}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error descargando modelo {layer}/{model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error descargando modelo: {str(e)}")

@router.get("/info/{layer}/{model_name}")
async def get_model_info(layer: str, model_name: str):
    """
    Obtener información de un modelo específico
    
    Args:
        layer: Número de capa
        model_name: Nombre del archivo del modelo
    
    Returns:
        Información detallada del modelo
    """
    try:
        # Buscar el modelo en la lista
        models = minio_service.list_models(layer)
        model_info = None
        
        for model in models:
            if model['filename'] == model_name:
                model_info = model
                break
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {layer}/{model_name}")
        
        return {
            "status": "success",
            "model": model_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo info del modelo {layer}/{model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo información: {str(e)}")

@router.get("/stats")
async def get_models_stats():
    """
    Obtener estadísticas generales de modelos
    
    Returns:
        Estadísticas de modelos por capa
    """
    try:
        all_models = minio_service.list_models()
        
        # Agrupar por capa
        stats_by_layer = {}
        total_size = 0
        
        for model in all_models:
            layer = model['layer']
            if layer not in stats_by_layer:
                stats_by_layer[layer] = {
                    "count": 0,
                    "total_size": 0,
                    "models": []
                }
            
            stats_by_layer[layer]["count"] += 1
            stats_by_layer[layer]["total_size"] += model['size']
            stats_by_layer[layer]["models"].append(model['filename'])
            total_size += model['size']
        
        return {
            "status": "success",
            "total_models": len(all_models),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "layers": stats_by_layer
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de modelos: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")
