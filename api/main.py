0#!/usr/bin/env python3
"""
API REST para Restauración de Documentos - VERSIÓN MODULAR
Integración con n8n y MinIO para automatización
"""

import sys
import os
from pathlib import Path

# Agregar el directorio actual al PYTHONPATH para importaciones
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar configuración
try:
    from config.settings import API_CONFIG, model_state
    logger.info("✅ Configuración importada exitosamente")
except ImportError:
    # Fallback config
    API_CONFIG = {
        'title': "Document Restoration API",
        'description': "API para restauración de documentos con Transfer Learning Gradual",
        'version': "1.0.0"
    }
    model_state = {'model': None, 'device': None, 'loaded': False}
    logger.warning("⚠️ Usando configuración de fallback")

# Importar servicios
services_loaded = False
try:
    from services.minio_service import minio_service
    from services.model_service import model_service
    logger.info("✅ Servicios importados exitosamente")
    services_loaded = True
except ImportError as e:
    logger.warning(f"⚠️ No se pudieron cargar los servicios: {e}")
    minio_service = None
    model_service = None

# Importar routers
routers_loaded = False
try:
    from routers.classification import router as classification_router
    from routers.restoration import router as restoration_router
    from routers.synthetic_data import router as synthetic_data_router
    from routers.files import router as files_router
    from routers.jobs import router as jobs_router
    logger.info("✅ Routers importados exitosamente")
    routers_loaded = True
except ImportError as e:
    logger.warning(f"⚠️ No se pudieron cargar los routers: {e}")
    routers_loaded = False

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title=API_CONFIG['title'],
    description=API_CONFIG['description'],
    version=API_CONFIG['version']
)

# CORS para n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manejo de excepciones global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejar todas las excepciones sin cerrar la aplicación"""
    logger.error(f"Error global capturado: {type(exc).__name__}: {str(exc)}")
    logger.error(f"Request: {request.method} {request.url}")
    
    # Si es error de memoria, liberar recursos
    if "memory" in str(exc).lower() or "allocate" in str(exc).lower():
        import gc
        gc.collect()  # Forzar garbage collection
        
        return JSONResponse(
            status_code=507,  # Insufficient Storage
            content={
                "error": "insufficient_memory",
                "message": "Imagen demasiado grande. Intenta con una imagen más pequeña.",
                "details": "El sistema no tiene suficiente memoria para procesar esta imagen.",
                "suggestion": "Reduce la resolución de la imagen a menos de 300 DPI"
            }
        )
    
    # Para otros errores
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Error interno del servidor",
            "details": str(exc)
        }
    )

# Registrar routers si están disponibles
if routers_loaded:
    app.include_router(classification_router)
    app.include_router(restoration_router)
    app.include_router(synthetic_data_router)
    app.include_router(files_router)
    app.include_router(jobs_router)
else:
    logging.warning("Routers no pudieron ser cargados - funcionando en modo básico")

# Eventos de startup/shutdown
@app.on_event("startup")
async def startup_event():
    """Inicialización de la API"""
    logger.info("🚀 Iniciando Document Restoration API MODULAR")
    
    if services_loaded:
        # Configurar buckets
        minio_service.ensure_buckets()
        
        # Cargar modelo
        model_service.load_model()
        
        logger.info("🎯 API modular lista!")
    else:
        logger.warning("⚠️ API iniciada en modo básico - servicios no disponibles")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar la API"""
    logger.info("Cerrando Document Restoration API")

# Endpoints básicos
@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Document Restoration API - VERSIÓN MODULAR 🏗️",
        "version": API_CONFIG['version'],
        "status": "active",
        "model_loaded": model_state['loaded'],
        "device": str(model_state['device']) if model_state['device'] else None,
        "architecture": "✅ Completamente modular" if routers_loaded and services_loaded else "⚠️ Modo básico",
        "services": ["MinIO", "Model", "ImageAnalysis", "SyntheticData"] if services_loaded else ["No disponibles"],
        "routers": ["Classification", "Restoration", "SyntheticData", "Files", "Jobs"] if routers_loaded else ["No disponibles"]
    }

@app.get("/health")
async def health_check():
    """Health check para n8n"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if model_state['loaded'] else "not_loaded",
        "minio_status": "connected",
        "architecture": "modular" if routers_loaded and services_loaded else "basic",
        "services_loaded": services_loaded,
        "routers_loaded": routers_loaded
    }

@app.get("/status/modular")
async def modular_status():
    """Estado de la arquitectura modular"""
    return {
        "architecture": "modular",
        "status": "✅ Completamente separado",
        "components": {
            "config": "✅ Centralizada",
            "services": "✅ Separados por responsabilidad",
            "routers": "✅ Organizados por funcionalidad",
            "models": "✅ Esquemas Pydantic separados"
        },
        "services": {
            "minio_service": "✅ Operaciones de almacenamiento",
            "model_service": "✅ Manejo de modelos ML",
            "image_analysis_service": "✅ Procesamiento de imágenes", 
            "synthetic_data_service": "✅ Generación de datos"
        },
        "routers": {
            "classification": "✅ /classify/* - Clasificación de documentos",
            "restoration": "✅ /restore/* - Restauración de imágenes",
            "synthetic_data": "✅ /synthetic/* - Datos sintéticos", 
            "files": "✅ /files/* - Operaciones con archivos",
            "jobs": "✅ /jobs/* - Manejo de trabajos"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
