#!/usr/bin/env python3
"""
🌐 API SERVER INTEGRADO - Con Cola Compartida
=============================================
Este servidor usa cola compartida para comunicarse con el worker.
"""

import sys
import os
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

# Configurar paths para importar módulos existentes
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

# Importar cola compartida
from shared_job_queue import create_shared_queue

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegratedAPIServer")

# Cola de jobs compartida
shared_queue = create_shared_queue()
logger.info("🔄 Cola compartida inicializada")

# Importar servicios existentes
try:
    from api.services.training_service import TrainingService
    from api.services.synthetic_data_service import SyntheticDataService
    from api.services.restoration_service import RestorationService  
    from api.services.model_service import ModelService
    from api.services.file_management_service import FileManagementService
    from api.config.settings import jobs_state
    
    SERVICES_AVAILABLE = True
    logger.info("✅ Servicios existentes importados correctamente")
except ImportError as e:
    logger.warning(f"⚠️  Error importando servicios existentes: {e}")
    logger.info("🔄 Modo standalone activado")
    SERVICES_AVAILABLE = False

# Crear app FastAPI
app = FastAPI(
    title="Sistema Integrado - API + Job Queue Compartida",
    description="API que integra jobs asíncronos con servicios existentes usando cola compartida",
    version="2.1.0"
)

# Modelos Pydantic
class JobRequest(BaseModel):
    job_type: str
    parameters: Dict[str, Any] = {}
    priority: int = 1

class TrainingRequest(BaseModel):
    num_epochs: int = 10
    batch_size: int = 2
    max_pairs: int = 100
    use_training_bucket: bool = True

class SyntheticDataRequest(BaseModel):
    count: int = 50
    bucket: str = "document-clean"
    augmentation_types: str = "noise blur"

class RestorationRequest(BaseModel):
    file_count: int = 10
    model_type: str = "layer2"
    bucket: str = "document-degraded"

@app.get("/")
async def root():
    """Endpoint raíz con información del sistema"""
    return {
        "message": "API Server Integrado con Cola Compartida",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0",
        "features": {
            "shared_queue": True,
            "async_processing": True,
            "services_integration": SERVICES_AVAILABLE,
            "real_time_monitoring": True
        }
    }

@app.get("/health")
async def health_check():
    """Health check detallado"""
    queue_stats = shared_queue.get_all_jobs()
    
    return {
        "status": "healthy",
        "service": "Integrated API Server",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "existing_services": SERVICES_AVAILABLE,
            "shared_queue": True,
            "integration": True,
            "training_service": SERVICES_AVAILABLE,
            "jobs_service": True
        },
        "queue_stats": queue_stats["statistics"],
        "total_jobs": queue_stats["total_jobs"]
    }

def create_job_id(job_type: str) -> str:
    """Crear ID único para job"""
    timestamp = int(datetime.now().timestamp() * 1000) % 100000000  # 8 dígitos
    random_suffix = str(uuid.uuid4())[:8]
    return f"{job_type}_{random_suffix}"

@app.post("/jobs")
async def create_job(request: JobRequest):
    """Crear job genérico"""
    job_id = create_job_id(request.job_type)
    
    job_data = {
        "job_id": job_id,
        "job_type": request.job_type,
        "parameters": request.parameters,
        "priority": request.priority,
        "created_at": datetime.now().isoformat(),
        "status": "queued"
    }
    
    # Encolar en cola compartida
    shared_queue.enqueue_job(job_data)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Job {request.job_type} creado exitosamente",
        "check_status_url": f"/jobs/{job_id}",
        "created_at": job_data["created_at"]
    }

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Obtener status de job específico"""
    job_status = shared_queue.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    
    return {
        "job_id": job_id,
        **job_status
    }

@app.get("/jobs")
async def list_jobs():
    """Listar todos los jobs"""
    return shared_queue.get_all_jobs()

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancelar job (actualizar status)"""
    job_status = shared_queue.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    
    if job_status["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="No se puede cancelar job terminado")
    
    shared_queue.update_job_status(job_id, "cancelled", cancelled_at=datetime.now().isoformat())
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job cancelado exitosamente"
    }

# ================================
# ENDPOINTS ESPECÍFICOS INTEGRADOS
# ================================

@app.post("/training/layer2")
async def create_training_job(request: TrainingRequest):
    """Crear job de entrenamiento Layer 2"""
    job_id = create_job_id("layer2_training")
    
    job_data = {
        "job_id": job_id,
        "job_type": "layer2_training",
        "parameters": {
            "num_epochs": request.num_epochs,
            "batch_size": request.batch_size,
            "max_pairs": request.max_pairs,
            "use_training_bucket": request.use_training_bucket
        },
        "created_at": datetime.now().isoformat(),
        "status": "queued"
    }
    
    # Encolar en cola compartida
    shared_queue.enqueue_job(job_data)
    
    logger.info(f"🧠 Training job creado: {job_id} ({request.num_epochs} épocas)")
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Training Layer 2 iniciado ({request.num_epochs} épocas)",
        "check_status_url": f"/jobs/{job_id}",
        "parameters": job_data["parameters"]
    }

@app.post("/synthetic-data/generate")
async def create_synthetic_data_job(request: SyntheticDataRequest):
    """Crear job de generación de datos sintéticos"""
    job_id = create_job_id("synthetic_data")
    
    job_data = {
        "job_id": job_id,
        "job_type": "synthetic_data_generation",
        "parameters": {
            "count": request.count,
            "bucket": request.bucket,
            "augmentation_types": request.augmentation_types
        },
        "created_at": datetime.now().isoformat(),
        "status": "queued"
    }
    
    # Encolar en cola compartida
    shared_queue.enqueue_job(job_data)
    
    logger.info(f"🎨 Synthetic data job creado: {job_id} ({request.count} imágenes)")
    
    return {
        "job_id": job_id,
        "status": "queued", 
        "message": f"Generación de {request.count} imágenes sintéticas iniciada",
        "check_status_url": f"/jobs/{job_id}",
        "parameters": job_data["parameters"]
    }

@app.post("/restoration/batch")
async def create_restoration_job(request: RestorationRequest):
    """Crear job de restauración por lotes"""
    job_id = create_job_id("restoration")
    
    job_data = {
        "job_id": job_id,
        "job_type": "batch_restoration",
        "parameters": {
            "file_count": request.file_count,
            "model_type": request.model_type,
            "bucket": request.bucket
        },
        "created_at": datetime.now().isoformat(),
        "status": "queued"
    }
    
    # Encolar en cola compartida
    shared_queue.enqueue_job(job_data)
    
    logger.info(f"🔧 Restoration job creado: {job_id} ({request.file_count} archivos)")
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Restauración de {request.file_count} archivos iniciada",
        "check_status_url": f"/jobs/{job_id}",
        "parameters": job_data["parameters"]
    }

# ================================
# COMPATIBILIDAD CON SISTEMA EXISTENTE
# ================================

@app.get("/training/jobs")
async def get_training_jobs():
    """Obtener jobs de training (compatibilidad)"""
    all_jobs = shared_queue.get_all_jobs()
    training_jobs = {
        job_id: job_info 
        for job_id, job_info in all_jobs["jobs"].items() 
        if job_info.get("job_type") == "layer2_training"
    }
    
    return {
        "training_jobs": training_jobs,
        "total": len(training_jobs)
    }

@app.get("/synthetic-data/jobs")
async def get_synthetic_data_jobs():
    """Obtener jobs de synthetic data (compatibilidad)"""
    all_jobs = shared_queue.get_all_jobs()
    synthetic_jobs = {
        job_id: job_info 
        for job_id, job_info in all_jobs["jobs"].items() 
        if job_info.get("job_type") == "synthetic_data_generation"
    }
    
    return {
        "synthetic_data_jobs": synthetic_jobs,
        "total": len(synthetic_jobs)
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🌐 API SERVER INTEGRADO CON COLA COMPARTIDA")
    print("=" * 60)
    print("🔄 Cola compartida inicializada")
    print("🔧 Integrado con servicios existentes")
    print("🎯 Procesamiento asíncrono")
    print("📊 Monitoreo en tiempo real")
    print("=" * 60)
    print()
    print("💡 CARACTERÍSTICAS:")
    print("• Cola compartida con worker")
    print("• API no bloqueante")
    print("• Integración con servicios existentes")
    print("• Monitoreo de jobs en tiempo real")
    print()
    print("🚀 Iniciando servidor en http://localhost:8000")
    print("📖 Documentación en http://localhost:8000/docs")
    print()
    
    # Iniciar servidor
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,  # Evitar problemas con reload
        log_level="info"
    )
