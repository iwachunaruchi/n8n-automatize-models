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
from typing import Optional

# Agregar path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.settings import jobs_state
except ImportError:
    jobs_state = {}

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
    batch_size: int = 2
):
    """Iniciar entrenamiento de Capa 2 (NAFNet + DocUNet)"""
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
            "current_epoch": 0,
            "results": None,
            "error": None
        }
        
        jobs_state[job_id] = job_config
        
        # Ejecutar en background
        background_tasks.add_task(run_layer2_training, job_id, num_epochs, max_pairs, batch_size)
        
        return JSONResponse({
            "status": "success",
            "message": "Entrenamiento de Capa 2 iniciado",
            "job_id": job_id,
            "type": "layer2_training",
            "parameters": {
                "num_epochs": num_epochs,
                "max_pairs": max_pairs,
                "batch_size": batch_size
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

async def run_layer2_training(job_id: str, num_epochs: int, max_pairs: int, batch_size: int):
    """Ejecutar entrenamiento de Capa 2 en background"""
    try:
        job = jobs_state[job_id]
        job["status"] = "running"
        job["progress"] = 10
        
        # Ejecutar script de entrenamiento
        script_path = Path(__file__).parent.parent / "train-layers" / "train_layer_2.py"
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script no encontrado: {script_path}")
        
        # Configurar variables de entorno para el script
        env = os.environ.copy()
        env["TRAINING_EPOCHS"] = str(num_epochs)
        env["TRAINING_MAX_PAIRS"] = str(max_pairs)
        env["TRAINING_BATCH_SIZE"] = str(batch_size)
        env["TRAINING_JOB_ID"] = job_id
        
        # Ejecutar como subprocess
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        job["progress"] = 30
        
        # Simular progreso por épocas (esto se puede mejorar con comunicación IPC)
        async def update_progress():
            for epoch in range(1, num_epochs + 1):
                await asyncio.sleep(30)  # Simular tiempo por época
                if job["status"] == "running":
                    job["current_epoch"] = epoch
                    job["progress"] = 30 + (epoch / num_epochs) * 60
        
        # Ejecutar actualización de progreso en paralelo
        progress_task = asyncio.create_task(update_progress())
        
        stdout, stderr = await process.communicate()
        
        # Cancelar task de progreso
        progress_task.cancel()
        
        if process.returncode == 0:
            job["status"] = "completed"
            job["progress"] = 100
            job["current_epoch"] = num_epochs
            job["results"] = {
                "success": True,
                "output": stdout.decode('utf-8'),
                "epochs_completed": num_epochs,
                "pairs_used": max_pairs,
                "batch_size": batch_size
            }
            logger.info(f"Entrenamiento de Capa 2 completado: {job_id}")
        else:
            job["status"] = "failed"
            job["error"] = stderr.decode('utf-8')
            logger.error(f"Error en entrenamiento de Capa 2: {stderr.decode('utf-8')}")
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.error(f"Error ejecutando entrenamiento de Capa 2: {e}")
    
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
                "endpoints": ["/training/layer2/train"]
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
