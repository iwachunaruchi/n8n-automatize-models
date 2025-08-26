#!/usr/bin/env python3
"""
API REST para Restauración de Documentos - VERSIÓN MODULAR
Integración con n8n y MinIO para automatización
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime

# Importar configuración
from config import API_CONFIG, model_state

# Importar servicios
from services import minio_service, model_service

# Importar routers
from routers import (
    classification_router,
    restoration_router, 
    synthetic_data_router,
    files_router,
    jobs_router
)

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

# Registrar routers
app.include_router(classification_router)
app.include_router(restoration_router)
app.include_router(synthetic_data_router)
app.include_router(files_router)
app.include_router(jobs_router)

# Eventos de startup/shutdown
@app.on_event("startup")
async def startup_event():
    """Inicialización de la API"""
    logger.info("🚀 Iniciando Document Restoration API MODULAR")
    
    # Configurar buckets
    minio_service.ensure_buckets()
    
    # Cargar modelo
    model_service.load_model()
    
    logger.info("🎯 API modular lista!")

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
        "architecture": "✅ Completamente modular",
        "services": ["MinIO", "Model", "ImageAnalysis", "SyntheticData"],
        "routers": ["Classification", "Restoration", "SyntheticData", "Files", "Jobs"]
    }

@app.get("/health")
async def health_check():
    """Health check para n8n"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if model_state['loaded'] else "not_loaded",
        "minio_status": "connected",
        "architecture": "modular",
        "services_loaded": True
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
