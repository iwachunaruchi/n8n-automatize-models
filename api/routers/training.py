"""
Router adicional para entrenamiento de capas
Endpoints para n8n para entrenar las diferentes capas del pipeline
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import subprocess
import asyncio
import requests
from typing import Optional

# Configurar logger
logger = logging.getLogger(__name__)

# Agregar path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.settings import jobs_state
except ImportError:
    jobs_state = {}

# Importar funciones de entrenamiento
try:
    sys.path.append(str(Path(__file__).parent.parent.parent))  # Agregar root del proyecto
    sys.path.append(str(Path(__file__).parent.parent.parent / "layers" / "train-layers"))  # Path específico
    sys.path.append(str(Path(__file__).parent.parent.parent / "layers" / "layer-1"))  # Path específico
    
    from train_layer_2 import create_layer2_trainer, validate_training_parameters
    from layer_1 import PreprocessingPipeline
except ImportError as e:
    print(f"Error importando módulos de entrenamiento: {e}")
    create_layer2_trainer = None
    validate_training_parameters = None
    PreprocessingPipeline = None

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/training", tags=["Entrenamiento"])

# ============================================================================
# ENDPOINTS PARA ENTRENAMIENTO DE CAPAS
# ============================================================================

@router.post("/layer1/evaluate")
async def start_layer1_evaluation(
    background_tasks: BackgroundTasks,
    max_images: int = 30
):
    """Iniciar evaluación de Capa 1 (Pipeline de preprocesamiento)"""
    try:
        job_id = str(uuid.uuid4())
        
        # Configurar trabajo
        job_config = {
            "job_id": job_id,
            "type": "layer1_evaluation",
            "status": "started",
            "start_time": datetime.now().isoformat(),
            "progress": 0,
            "max_images": max_images,
            "results": None,
            "error": None
        }
        
        jobs_state[job_id] = job_config
        
        # Ejecutar en background
        background_tasks.add_task(run_layer1_evaluation, job_id, max_images)
        
        return JSONResponse({
            "status": "success",
            "message": "Evaluación de Capa 1 iniciada",
            "job_id": job_id,
            "type": "layer1_evaluation",
            "max_images": max_images,
            "check_status_url": f"/training/status/{job_id}"
        })
        
    except Exception as e:
        logger.error(f"Error iniciando evaluación de Capa 1: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/layer2/train")
async def start_layer2_training(
    background_tasks: BackgroundTasks,
    num_epochs: int = 10,
    max_pairs: int = 100,
    batch_size: int = 2,
    use_training_bucket: bool = True
):
    """
    Iniciar entrenamiento de Capa 2 (NAFNet + DocUNet)
    
    Args:
        use_training_bucket: Si usar bucket 'document-training' con pares sintéticos (recomendado)
                           False: usar buckets separados 'document-degraded' y 'document-clean'
    """
    try:
        job_id = str(uuid.uuid4())
        
        # Configurar trabajo
        job_config = {
            "job_id": job_id,
            "type": "layer2_training",
            "status": "started",
            "start_time": datetime.now().isoformat(),
            "progress": 0,
            "num_epochs": num_epochs,
            "max_pairs": max_pairs,
            "batch_size": batch_size,
            "use_training_bucket": use_training_bucket,
            "current_epoch": 0,
            "results": None,
            "error": None
        }
        
        jobs_state[job_id] = job_config
        
        # Ejecutar en background
        background_tasks.add_task(run_layer2_training, job_id, num_epochs, max_pairs, 
                                batch_size, use_training_bucket)
        
        return JSONResponse({
            "status": "success",
            "message": "Entrenamiento de Capa 2 iniciado",
            "job_id": job_id,
            "type": "layer2_training",
            "parameters": {
                "num_epochs": num_epochs,
                "max_pairs": max_pairs,
                "batch_size": batch_size,
                "use_training_bucket": use_training_bucket,
                "data_source": "document-training bucket" if use_training_bucket else "separate buckets"
            },
            "check_status_url": f"/training/status/{job_id}"
        })
        
    except Exception as e:
        logger.error(f"Error iniciando entrenamiento de Capa 2: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}")
async def get_training_status(job_id: str):
    """Obtener estado de un trabajo de entrenamiento"""
    try:
        if job_id not in jobs_state:
            raise HTTPException(status_code=404, detail="Trabajo no encontrado")
        
        job = jobs_state[job_id]
        
        # Calcular duración si está en progreso
        duration = None
        if job["status"] in ["running", "completed", "failed"]:
            start_time = datetime.fromisoformat(job["start_time"])
            current_time = datetime.now()
            duration = str(current_time - start_time)
        
        response = {
            "job_id": job_id,
            "type": job["type"],
            "status": job["status"],
            "progress": job["progress"],
            "start_time": job["start_time"],
            "duration": duration,
            "error": job.get("error"),
            "results": job.get("results")
        }
        
        # Agregar información específica según el tipo
        if job["type"] == "layer1_evaluation":
            response["max_images"] = job.get("max_images")
        elif job["type"] == "layer2_training":
            response["training_info"] = {
                "num_epochs": job.get("num_epochs"),
                "current_epoch": job.get("current_epoch", 0),
                "max_pairs": job.get("max_pairs"),
                "batch_size": job.get("batch_size")
            }
        
        return JSONResponse(response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo estado del trabajo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs")
async def list_training_jobs():
    """Listar todos los trabajos de entrenamiento"""
    try:
        jobs_list = []
        
        for job_id, job in jobs_state.items():
            if job.get("type", "").startswith(("layer1", "layer2")):
                jobs_list.append({
                    "job_id": job_id,
                    "type": job["type"],
                    "status": job["status"],
                    "start_time": job["start_time"],
                    "progress": job["progress"]
                })
        
        return JSONResponse({
            "total_jobs": len(jobs_list),
            "jobs": sorted(jobs_list, key=lambda x: x["start_time"], reverse=True)
        })
        
    except Exception as e:
        logger.error(f"Error listando trabajos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/layer2/prepare-data")
async def prepare_layer2_data(
    background_tasks: BackgroundTasks,
    target_pairs: int = 100,
    source_bucket: str = "document-clean"
):
    """
    Preparar datos para entrenamiento de Capa 2
    Genera pares sintéticos adicionales si es necesario
    """
    try:
        # Verificar pares existentes en bucket de entrenamiento
        response = requests.get(f"http://localhost:8000/files/list/document-training")
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Error accediendo al bucket de entrenamiento")
        
        files = response.json().get('files', [])
        clean_files = [f for f in files if f.startswith('clean_')]
        degraded_files = [f for f in files if f.startswith('degraded_')]
        
        current_pairs = min(len(clean_files), len(degraded_files))
        
        if current_pairs >= target_pairs:
            return JSONResponse({
                "status": "success",
                "message": f"Ya hay suficientes pares ({current_pairs}/{target_pairs})",
                "current_pairs": current_pairs,
                "target_pairs": target_pairs,
                "action": "none_needed"
            })
        
        # Generar pares adicionales
        needed_pairs = target_pairs - current_pairs
        
        # Usar servicio de datos sintéticos para generar más pares
        from services.synthetic_data_service import synthetic_data_service
        
        job_id = str(uuid.uuid4())
        
        job_config = {
            "job_id": job_id,
            "type": "data_preparation",
            "status": "started",
            "start_time": datetime.now().isoformat(),
            "progress": 0,
            "target_pairs": target_pairs,
            "current_pairs": current_pairs,
            "needed_pairs": needed_pairs,
            "source_bucket": source_bucket,
            "results": None,
            "error": None
        }
        
        jobs_state[job_id] = job_config
        
        # Ejecutar generación en background
        background_tasks.add_task(generate_additional_pairs, job_id, needed_pairs, source_bucket)
        
        return JSONResponse({
            "status": "success",
            "message": f"Generando {needed_pairs} pares adicionales",
            "job_id": job_id,
            "current_pairs": current_pairs,
            "target_pairs": target_pairs,
            "needed_pairs": needed_pairs,
            "check_status_url": f"/training/status/{job_id}"
        })
        
    except Exception as e:
        logger.error(f"Error preparando datos para Capa 2: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_additional_pairs(job_id: str, needed_pairs: int, source_bucket: str):
    """Generar pares adicionales en background"""
    try:
        job = jobs_state[job_id]
        job["status"] = "running"
        job["progress"] = 20
        
        from services.synthetic_data_service import synthetic_data_service
        
        # Generar pares adicionales
        result = synthetic_data_service.generate_training_pairs(source_bucket, needed_pairs)
        
        job["status"] = "completed"
        job["progress"] = 100
        job["results"] = result
        job["end_time"] = datetime.now().isoformat()
        
        logger.info(f"Generación de pares completada: {job_id}")
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["end_time"] = datetime.now().isoformat()
        logger.error(f"Error generando pares adicionales: {e}")

@router.get("/layer2/data-status")
async def get_layer2_data_status():
    """Verificar estado de datos para Capa 2"""
    try:
        # Verificar bucket de entrenamiento
        response = requests.get("http://localhost:8000/files/list/document-training")
        
        if response.status_code != 200:
            return JSONResponse({
                "status": "error",
                "message": "No se puede acceder al bucket de entrenamiento"
            })
        
        files = response.json().get('files', [])
        clean_files = [f for f in files if f.startswith('clean_')]
        degraded_files = [f for f in files if f.startswith('degraded_')]
        
        # Verificar pares válidos
        valid_pairs = 0
        for clean_file in clean_files:
            if '_' in clean_file and '.' in clean_file:
                uuid_part = clean_file.split('_', 1)[1].rsplit('.', 1)[0]
                degraded_match = f"degraded_{uuid_part}.png"
                if degraded_match in degraded_files:
                    valid_pairs += 1
        
        # Estadísticas adicionales
        other_files = [f for f in files if not (f.startswith('clean_') or f.startswith('degraded_'))]
        
        return JSONResponse({
            "status": "success",
            "bucket": "document-training",
            "statistics": {
                "total_files": len(files),
                "clean_files": len(clean_files),
                "degraded_files": len(degraded_files),
                "valid_pairs": valid_pairs,
                "other_files": len(other_files)
            },
            "ready_for_training": valid_pairs > 0,
            "recommendations": {
                "minimum_pairs": 50,
                "recommended_pairs": 200,
                "current_status": "sufficient" if valid_pairs >= 50 else "needs_more" if valid_pairs > 0 else "empty"
            }
        })
        
    except Exception as e:
        logger.error(f"Error verificando estado de datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancelar trabajo de entrenamiento"""
    try:
        if job_id not in jobs_state:
            raise HTTPException(status_code=404, detail="Trabajo no encontrado")
        
        job = jobs_state[job_id]
        
        if job["status"] == "completed":
            return JSONResponse({
                "status": "info",
                "message": "El trabajo ya está completado",
                "job_id": job_id
            })
        
        # Marcar como cancelado
        job["status"] = "cancelled"
        job["end_time"] = datetime.now().isoformat()
        
        return JSONResponse({
            "status": "success",
            "message": "Trabajo cancelado",
            "job_id": job_id
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelando trabajo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{job_id}")
async def get_training_results(job_id: str):
    """Obtener resultados detallados de un entrenamiento"""
    try:
        if job_id not in jobs_state:
            raise HTTPException(status_code=404, detail="Trabajo no encontrado")
        
        job = jobs_state[job_id]
        
        if job["status"] != "completed":
            return JSONResponse({
                "status": "pending",
                "message": "El trabajo aún no está completado",
                "current_status": job["status"]
            })
        
        # Buscar archivos de resultados
        results_dir = Path("outputs") / f"{job['type']}_results"
        
        results = {
            "job_id": job_id,
            "type": job["type"],
            "status": job["status"],
            "results": job.get("results", {}),
            "output_files": []
        }
        
        # Listar archivos de salida si existen
        if results_dir.exists():
            for file_path in results_dir.glob("*"):
                if file_path.is_file():
                    results["output_files"].append({
                        "filename": file_path.name,
                        "size": file_path.stat().st_size,
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
        
        return JSONResponse(results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo resultados: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# FUNCIONES DE BACKGROUND PARA EJECUTAR ENTRENAMIENTO
# ============================================================================

async def run_layer1_evaluation(job_id: str, max_images: int):
    """Ejecutar evaluación de Capa 1 en background"""
    try:
        job = jobs_state[job_id]
        job["status"] = "running"
        job["progress"] = 10
        
        # Ejecutar script de evaluación
        script_path = Path(__file__).parent.parent / "train-layers" / "train_layer_1.py"
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script no encontrado: {script_path}")
        
        # Ejecutar como subprocess
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        job["progress"] = 50
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            job["status"] = "completed"
            job["progress"] = 100
            job["results"] = {
                "success": True,
                "output": stdout.decode('utf-8'),
                "max_images_processed": max_images
            }
            logger.info(f"Evaluación de Capa 1 completada: {job_id}")
        else:
            job["status"] = "failed"
            job["error"] = stderr.decode('utf-8')
            logger.error(f"Error en evaluación de Capa 1: {stderr.decode('utf-8')}")
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.error(f"Error ejecutando evaluación de Capa 1: {e}")
    
    finally:
        job["end_time"] = datetime.now().isoformat()

async def run_layer2_training(job_id: str, num_epochs: int, max_pairs: int, 
                              batch_size: int, use_training_bucket: bool = True):
    """Ejecutar entrenamiento de Capa 2 en background usando funciones importadas"""
    try:
        job = jobs_state[job_id]
        job["status"] = "running"
        job["progress"] = 10
        
        # Verificar que las funciones estén disponibles
        if create_layer2_trainer is None:
            raise ImportError("Módulo de entrenamiento Layer 2 no disponible")
        
        # Validar parámetros
        validation_errors = validate_training_parameters(num_epochs, max_pairs, batch_size)
        if validation_errors:
            raise ValueError(f"Parámetros inválidos: {validation_errors}")
        
        job["progress"] = 20
        
        # Crear trainer
        trainer = create_layer2_trainer("http://localhost:8000")
        
        job["progress"] = 30
        
        # Ejecutar entrenamiento en un hilo separado para no bloquear
        import asyncio
        import threading
        
        def run_training():
            try:
                trainer.train(
                    num_epochs=num_epochs,
                    max_pairs=max_pairs,
                    batch_size=batch_size,
                    use_training_bucket=use_training_bucket
                )
                return True
            except Exception as e:
                raise e
        
        # Ejecutar en thread separado
        loop = asyncio.get_event_loop()
        
        def update_progress():
            for epoch in range(1, num_epochs + 1):
                import time
                time.sleep(30)  # Simular tiempo por época
                if job["status"] == "running":
                    job["current_epoch"] = epoch
                    job["progress"] = 30 + (epoch / num_epochs) * 60
        
        # Ejecutar entrenamiento
        training_thread = threading.Thread(target=run_training)
        progress_thread = threading.Thread(target=update_progress)
        
        training_thread.start()
        progress_thread.start()
        
        # Esperar a que termine el entrenamiento
        training_thread.join()
        
        job["status"] = "completed"
        job["progress"] = 100
        job["current_epoch"] = num_epochs
        job["results"] = {
            "success": True,
            "epochs_completed": num_epochs,
            "pairs_used": max_pairs,
            "batch_size": batch_size,
            "use_training_bucket": use_training_bucket
        }
        logger.info(f"Entrenamiento de Capa 2 completado: {job_id}")
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        print(f"Error ejecutando entrenamiento de Capa 2: {e}")
    
    finally:
        job["end_time"] = datetime.now().isoformat()

@router.get("/info")
async def get_training_info():
    """Información sobre las capacidades de entrenamiento"""
    return JSONResponse({
        "available_layers": {
            "layer1": {
                "name": "Pipeline de Preprocesamiento",
                "description": "Otsu + CLAHE + Deskew por Hough",
                "type": "evaluation",
                "requires_gpu": False,
                "endpoints": ["/training/layer1/evaluate"]
            },
            "layer2": {
                "name": "NAFNet + DocUNet",
                "description": "Denoising/Deblurring + Dewarping",
                "type": "training",
                "requires_gpu": True,
                "endpoints": ["/training/layer2/train"],
                "data_sources": {
                    "recommended": "document-training bucket (pares sintéticos)",
                    "alternative": "document-degraded + document-clean buckets"
                },
                "parameters": {
                    "use_training_bucket": "true/false - usar bucket de entrenamiento con pares sintéticos",
                    "num_epochs": "número de épocas de entrenamiento",
                    "max_pairs": "máximo número de pares a usar",
                    "batch_size": "tamaño del batch"
                }
            }
        },
        "data_source": "MinIO buckets via API",
        "buckets_used": [
            "document-degraded (imágenes dañadas)",
            "document-clean (imágenes limpias)",
            "document-training (resultados de entrenamiento)"
        ],
        "output_location": "outputs/ directory",
        "job_management": [
            "GET /training/jobs - Listar trabajos",
            "GET /training/status/{job_id} - Estado de trabajo",
            "GET /training/results/{job_id} - Resultados",
            "DELETE /training/jobs/{job_id} - Cancelar trabajo"
        ]
    })
