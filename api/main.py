#!/usr/bin/env python3
"""
API REST para Restauración de Documentos - VERSIÓN MODULAR
Integración con n8n y MinIO para automatización + Redis Queue (RQ) para Jobs
"""

import sys
import os
from pathlib import Path

# Agregar el directorio actual al PYTHONPATH para importaciones
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

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
    from routers.jobs_rq import router as jobs_router
    from routers.training import router as training_router
    from routers.models import router as models_router
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
    app.include_router(training_router)
    app.include_router(models_router)
else:
    logging.warning("Routers no pudieron ser cargados - funcionando en modo básico")

# Función para mostrar todas las rutas
def show_all_routes():
    """Mostrar todas las rutas disponibles en la API"""
    print("\n" + "="*80)
    print("🚀 DOCUMENT RESTORATION API - RUTAS DISPONIBLES")
    print("="*80)
    
    routes_by_category = {}
    
    # Organizar rutas por categorías
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            methods = list(route.methods)
            if 'HEAD' in methods:
                methods.remove('HEAD')
            if 'OPTIONS' in methods:
                methods.remove('OPTIONS')
                
            if methods:  # Solo mostrar si tiene métodos HTTP válidos
                path = route.path
                
                # Categorizar rutas
                if path.startswith('/training'):
                    category = "🎯 ENTRENAMIENTO (TRAINING)"
                elif path.startswith('/synthetic'):
                    category = "🔄 DATOS SINTÉTICOS (SYNTHETIC)"
                elif path.startswith('/restore'):
                    category = "🛠️ RESTAURACIÓN (RESTORATION)"
                elif path.startswith('/classify'):
                    category = "📊 CLASIFICACIÓN (CLASSIFICATION)"
                elif path.startswith('/files'):
                    category = "📁 ARCHIVOS (FILES)"
                elif path.startswith('/jobs'):
                    category = "⚙️ TRABAJOS (JOBS)"
                elif path in ['/', '/health', '/status/modular']:
                    category = "🏠 BÁSICOS (CORE)"
                else:
                    category = "🔧 OTROS"
                
                if category not in routes_by_category:
                    routes_by_category[category] = []
                
                routes_by_category[category].append({
                    'path': path,
                    'methods': methods,
                    'name': getattr(route, 'name', 'unnamed')
                })
    
    # Mostrar rutas organizadas por categoría
    for category, routes in sorted(routes_by_category.items()):
        print(f"\n{category}")
        print("-" * len(category))
        
        for route in sorted(routes, key=lambda x: x['path']):
            methods_str = ', '.join(sorted(route['methods']))
            print(f"  {methods_str:12} {route['path']}")
    
    print("\n" + "="*80)
    print(f"📊 TOTAL DE ENDPOINTS: {sum(len(routes) for routes in routes_by_category.values())}")
    print("🌐 Base URL: http://localhost:8000")
    print("📚 Documentación: http://localhost:8000/docs")
    print("🔄 Redoc: http://localhost:8000/redoc")
    print("="*80 + "\n")

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
    
    # Mostrar todas las rutas disponibles
    show_all_routes()

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
        "routers": ["Classification", "Restoration", "SyntheticData", "Files", "Jobs", "Training"] if routers_loaded else ["No disponibles"]
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
            "jobs": "✅ /jobs/* - Manejo de trabajos",
            "training": "✅ /training/* - Entrenamiento de capas"
        }
    }

@app.get("/routes")
async def get_all_routes():
    """Obtener todas las rutas disponibles en la API"""
    routes_by_category = {}
    
    # Organizar rutas por categorías
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            methods = list(route.methods)
            if 'HEAD' in methods:
                methods.remove('HEAD')
            if 'OPTIONS' in methods:
                methods.remove('OPTIONS')
                
            if methods:  # Solo mostrar si tiene métodos HTTP válidos
                path = route.path
                
                # Categorizar rutas
                if path.startswith('/training'):
                    category = "training"
                elif path.startswith('/synthetic'):
                    category = "synthetic_data"
                elif path.startswith('/restore'):
                    category = "restoration"
                elif path.startswith('/classify'):
                    category = "classification"
                elif path.startswith('/files'):
                    category = "files"
                elif path.startswith('/jobs'):
                    category = "jobs"
                elif path in ['/', '/health', '/status/modular', '/routes']:
                    category = "core"
                else:
                    category = "others"
                
                if category not in routes_by_category:
                    routes_by_category[category] = []
                
                routes_by_category[category].append({
                    'path': path,
                    'methods': sorted(methods),
                    'name': getattr(route, 'name', 'unnamed'),
                    'summary': getattr(route, 'summary', None)
                })
    
    # Ordenar rutas dentro de cada categoría
    for category in routes_by_category:
        routes_by_category[category] = sorted(routes_by_category[category], key=lambda x: x['path'])
    
    return {
        "api_info": {
            "title": API_CONFIG['title'],
            "version": API_CONFIG['version'],
            "base_url": "http://localhost:8000",
            "documentation": "http://localhost:8000/docs",
            "redoc": "http://localhost:8000/redoc"
        },
        "total_endpoints": sum(len(routes) for routes in routes_by_category.values()),
        "categories": {
            "training": "🎯 Entrenamiento de capas (Layer 1 y Layer 2)",
            "synthetic_data": "🔄 Generación de datos sintéticos",
            "restoration": "🛠️ Restauración de documentos",
            "classification": "📊 Clasificación de documentos",
            "files": "📁 Gestión de archivos en MinIO",
            "jobs": "⚙️ Manejo de trabajos asíncronos",
            "core": "🏠 Endpoints básicos del sistema"
        },
        "routes": routes_by_category,
        "status": {
            "services_loaded": services_loaded,
            "routers_loaded": routers_loaded,
            "model_loaded": model_state['loaded'],
            "architecture": "modular" if routers_loaded and services_loaded else "basic"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
