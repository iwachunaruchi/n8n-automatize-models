"""
Router adicional para entrenamiento de capas
Endpoints para n8n para entrenar las diferentes capas del pipeline
REFACTORIZADO: Usa servicios directamente, no HTTP requests
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
import sys
import os
from typing import Optional

# Configurar logger
logger = logging.getLogger(__name__)

# Agregar path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar servicio de entrenamiento
try:
    from services.training_service import training_service
    TRAINING_SERVICE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importando training_service: {e}")
    training_service = None
    TRAINING_SERVICE_AVAILABLE = False

router = APIRouter(prefix="/training", tags=["Entrenamiento"])

# ============================================================================
# ENDPOINTS REFACTORIZADOS - USAN TRAINING_SERVICE DIRECTAMENTE
# ============================================================================

@router.post("/layer1/evaluate")
async def start_layer1_evaluation(
    background_tasks: BackgroundTasks,
    max_images: int = 30
):
    """Iniciar evaluación de Capa 1 (Pipeline de preprocesamiento)"""
    try:
        if not TRAINING_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicio de entrenamiento no disponible")
        
        # Crear trabajo usando el servicio
        job_id = training_service.create_job("layer1_evaluation", max_images=max_images)
        
        # Ejecutar en background usando el servicio
        background_tasks.add_task(training_service.start_layer1_evaluation, job_id, max_images)
        
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
    use_training_bucket: bool = True,
    use_finetuning: bool = True,
    freeze_backbone: bool = False,
    finetuning_lr_factor: float = 0.1
):
    """
    Iniciar entrenamiento de Capa 2 (NAFNet + DocUNet) con fine-tuning
    
    Args:
        num_epochs: Número de épocas de entrenamiento
        max_pairs: Máximo número de pares de entrenamiento a usar
        batch_size: Tamaño del batch
        use_training_bucket: Si usar bucket 'document-training' con pares sintéticos (recomendado)
                           False: usar buckets separados 'document-degraded' y 'document-clean'
        use_finetuning: Si usar fine-tuning con modelo preentrenado NAFNet-SIDD-width64
        freeze_backbone: Si congelar las capas del backbone preentrenado
        finetuning_lr_factor: Factor de reducción del learning rate para capas preentrenadas (0.1 = 10% del LR base)
    """
    try:
        if not TRAINING_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicio de entrenamiento no disponible")
        
        # Validar parámetros usando el servicio
        validation_errors = training_service.validate_training_parameters(num_epochs, max_pairs, batch_size)
        if validation_errors:
            raise HTTPException(status_code=400, detail=f"Parámetros inválidos: {', '.join(validation_errors)}")
        
        # Crear trabajo usando el servicio
        job_id = training_service.create_job(
            "layer2_training",
            num_epochs=num_epochs,
            max_pairs=max_pairs,
            batch_size=batch_size,
            use_training_bucket=use_training_bucket,
            use_finetuning=use_finetuning,
            freeze_backbone=freeze_backbone,
            finetuning_lr_factor=finetuning_lr_factor
        )
        
        # Ejecutar en background usando el servicio
        background_tasks.add_task(
            training_service.start_layer2_training, 
            job_id, num_epochs, max_pairs, batch_size, use_training_bucket,
            use_finetuning, freeze_backbone, finetuning_lr_factor
        )
        
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
                "use_finetuning": use_finetuning,
                "freeze_backbone": freeze_backbone,
                "finetuning_lr_factor": finetuning_lr_factor,
                "data_source": "document-training bucket" if use_training_bucket else "separate buckets",
                "pretrained_model": "NAFNet-SIDD-width64" if use_finetuning else "None"
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
        if not TRAINING_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicio de entrenamiento no disponible")
        
        job = training_service.get_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Trabajo no encontrado")
        
        # Calcular duración si está en progreso
        duration = None
        if job["status"] in ["running", "completed", "failed"]:
            from datetime import datetime
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
            response["max_images"] = job.get("parameters", {}).get("max_images")
        elif job["type"] == "layer2_training":
            params = job.get("parameters", {})
            response["training_info"] = {
                "num_epochs": params.get("num_epochs"),
                "current_epoch": job.get("current_epoch", 0),
                "max_pairs": params.get("max_pairs"),
                "batch_size": params.get("batch_size")
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
        if not TRAINING_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicio de entrenamiento no disponible")
        
        jobs_list = training_service.list_training_jobs()
        
        return JSONResponse({
            "total_jobs": len(jobs_list),
            "jobs": jobs_list
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
        if not TRAINING_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicio de entrenamiento no disponible")
        
        # Usar servicio directamente (no HTTP request)
        result = await training_service.prepare_layer2_data(target_pairs, source_bucket)
        
        if result["success"]:
            response_data = {
                "status": "success",
                "message": result["message"],
                "current_pairs": result.get("current_pairs", 0),
                "target_pairs": target_pairs,
                "action": result.get("action", "completed")
            }
            
            # Agregar información adicional si se generaron pares
            if "generated_count" in result:
                response_data["generated_count"] = result["generated_count"]
            if "total_files_created" in result:
                response_data["total_files_created"] = result["total_files_created"]
            if "needed_pairs" in result:
                response_data["needed_pairs"] = result["needed_pairs"]
            
            return JSONResponse(response_data)
        else:
            raise HTTPException(status_code=500, detail=result["error"])
        
    except Exception as e:
        logger.error(f"Error preparando datos para Capa 2: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/layer2/data-status")
async def get_layer2_data_status():
    """Verificar estado de datos para Capa 2"""
    try:
        if not TRAINING_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicio de entrenamiento no disponible")
        
        # Usar servicio directamente (no HTTP request)
        data_status = training_service.check_layer2_data_status()
        
        if data_status["success"]:
            return JSONResponse(data_status)
        else:
            return JSONResponse({
                "status": "error",
                "message": "No se puede acceder al bucket de entrenamiento",
                "error": data_status.get("error", "Error desconocido")
            })
        
    except Exception as e:
        logger.error(f"Error verificando estado de datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancelar trabajo de entrenamiento"""
    try:
        if not TRAINING_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicio de entrenamiento no disponible")
        
        job = training_service.get_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Trabajo no encontrado")
        
        if job["status"] == "completed":
            return JSONResponse({
                "status": "info",
                "message": "El trabajo ya está completado",
                "job_id": job_id
            })
        
        # Marcar como cancelado
        from datetime import datetime
        training_service.update_job_status(
            job_id, 
            "cancelled",
            end_time=datetime.now().isoformat()
        )
        
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
        if not TRAINING_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Servicio de entrenamiento no disponible")
        
        job = training_service.get_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Trabajo no encontrado")
        
        if job["status"] != "completed":
            return JSONResponse({
                "status": "pending",
                "message": "El trabajo aún no está completado",
                "current_status": job["status"]
            })
        
        # Buscar archivos de resultados
        from pathlib import Path
        from datetime import datetime
        
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
# ENDPOINT DE INFORMACIÓN
# ============================================================================

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
        "data_source": "MinIO buckets via direct service access",
        "buckets_used": [
            "document-degraded (imágenes dañadas)",
            "document-clean (imágenes limpias)",
            "document-training (pares de entrenamiento)"
        ],
        "output_location": "outputs/ directory",
        "job_management": [
            "GET /training/jobs - Listar trabajos",
            "GET /training/status/{job_id} - Estado de trabajo",
            "GET /training/results/{job_id} - Resultados",
            "DELETE /training/jobs/{job_id} - Cancelar trabajo"
        ],
        "service_architecture": "✅ Refactorizado - Uso directo de servicios (no HTTP requests)",
        "training_service_available": TRAINING_SERVICE_AVAILABLE
    })

# ============================================================================
# ENDPOINTS DE REPORTES DE ENTRENAMIENTO
# ============================================================================

@router.get("/reports")
async def list_training_reports(layer: Optional[str] = None):
    """
    Listar reportes de entrenamiento disponibles
    
    Args:
        layer: Filtrar por capa específica (opcional)
    
    Returns:
        Lista de reportes de entrenamiento generados
    """
    if not TRAINING_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Servicio de entrenamiento no disponible")
    
    try:
        # Importar servicio de reportes
        from services.training_report_service import training_report_service
        
        reports = training_report_service.list_training_reports(layer)
        
        return {
            "status": "success", 
            "filter_layer": layer,
            "total_reports": len(reports),
            "reports": reports
        }
        
    except Exception as e:
        logger.error(f"Error listando reportes: {e}")
        raise HTTPException(status_code=500, detail=f"Error listando reportes: {str(e)}")

@router.get("/reports/{report_filename}")
async def download_training_report(report_filename: str):
    """
    Descargar reporte de entrenamiento específico
    
    Args:
        report_filename: Nombre del archivo de reporte
    
    Returns:
        Contenido del reporte en formato texto
    """
    if not TRAINING_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Servicio de entrenamiento no disponible")
    
    try:
        from services.training_report_service import training_report_service
        
        report_path = f"reports/{report_filename}"
        report_content = training_report_service.download_report(report_path)
        
        if not report_content:
            raise HTTPException(status_code=404, detail="Reporte no encontrado")
        
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content=report_content,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={report_filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error descargando reporte {report_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error descargando reporte: {str(e)}")

@router.get("/reports/job/{job_id}")
async def get_job_training_report(job_id: str):
    """
    Obtener reporte de entrenamiento asociado a un job específico
    
    Args:
        job_id: ID del trabajo de entrenamiento
    
    Returns:
        Información del reporte asociado al job
    """
    if not TRAINING_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Servicio de entrenamiento no disponible")
    
    try:
        # Obtener información del job
        job_status = training_service.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail="Job no encontrado")
        
        # Verificar si tiene reporte asociado
        results = job_status.get("results", {})
        report_path = results.get("training_report")
        
        if not report_path:
            return {
                "status": "no_report",
                "message": "No hay reporte asociado a este job",
                "job_id": job_id,
                "job_status": job_status.get("status")
            }
        
        # Obtener información del reporte
        from services.training_report_service import training_report_service
        
        return {
            "status": "success",
            "job_id": job_id,
            "report_path": report_path,
            "report_filename": report_path.split("/")[-1],
            "download_url": f"/training/reports/{report_path.split('/')[-1]}",
            "job_completed": job_status.get("status") == "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo reporte del job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo reporte: {str(e)}")
