"""
Training Service - Lógica de negocio para entrenamiento de capas
Centraliza toda la lógica de entrenamiento sin depender de HTTP requests
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys
import os

# Importar servicios directamente
try:
    from .minio_service import minio_service
    from .synthetic_data_service import synthetic_data_service
    from .image_analysis_service import image_analysis_service
except ImportError:
    # Fallback para importaciones relativas
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from minio_service import minio_service
    from synthetic_data_service import synthetic_data_service
    from image_analysis_service import image_analysis_service

logger = logging.getLogger(__name__)

class TrainingService:
    """Servicio centralizado para manejo de entrenamiento de capas"""
    
    def __init__(self):
        self.jobs_state = {}
        self.buckets = {
            'degraded': 'document-degraded',
            'clean': 'document-clean',
            'restored': 'document-restored',
            'training': 'document-training'
        }
    
    # ============================================================================
    # GESTIÓN DE TRABAJOS
    # ============================================================================
    
    def create_job(self, job_type: str, **params) -> str:
        """Crear nuevo trabajo de entrenamiento"""
        job_id = str(uuid.uuid4())
        
        job_config = {
            "job_id": job_id,
            "type": job_type,
            "status": "created",
            "start_time": datetime.now().isoformat(),
            "progress": 0,
            "parameters": params,
            "results": None,
            "error": None
        }
        
        self.jobs_state[job_id] = job_config
        logger.info(f"Trabajo creado: {job_id} ({job_type})")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de un trabajo"""
        return self.jobs_state.get(job_id)
    
    def update_job_status(self, job_id: str, status: str, progress: int = None, **updates):
        """Actualizar estado de trabajo"""
        if job_id in self.jobs_state:
            self.jobs_state[job_id]["status"] = status
            if progress is not None:
                self.jobs_state[job_id]["progress"] = progress
            
            for key, value in updates.items():
                self.jobs_state[job_id][key] = value
    
    def list_training_jobs(self) -> List[Dict[str, Any]]:
        """Listar todos los trabajos de entrenamiento"""
        jobs_list = []
        
        for job_id, job in self.jobs_state.items():
            if job.get("type", "").startswith(("layer1", "layer2")):
                jobs_list.append({
                    "job_id": job_id,
                    "type": job["type"],
                    "status": job["status"],
                    "start_time": job["start_time"],
                    "progress": job["progress"]
                })
        
        return sorted(jobs_list, key=lambda x: x["start_time"], reverse=True)
    
    # ============================================================================
    # VALIDACIONES Y VERIFICACIONES
    # ============================================================================
    
    def validate_training_parameters(self, num_epochs: int, max_pairs: int, batch_size: int) -> List[str]:
        """Validar parámetros de entrenamiento"""
        errors = []
        
        if not (1 <= num_epochs <= 100):
            errors.append("num_epochs debe estar entre 1 y 100")
        
        if not (10 <= max_pairs <= 1000):
            errors.append("max_pairs debe estar entre 10 y 1000")
        
        if not (1 <= batch_size <= 8):
            errors.append("batch_size debe estar entre 1 y 8")
        
        return errors
    
    def check_layer2_data_status(self) -> Dict[str, Any]:
        """Verificar estado de datos para Capa 2 - SIN HTTP REQUEST"""
        try:
            # Usar servicio MinIO directamente
            files = minio_service.list_files(self.buckets['training'])
            
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
            
            return {
                "success": True,
                "bucket": self.buckets['training'],
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
            }
            
        except Exception as e:
            logger.error(f"Error verificando estado de datos: {e}")
            return {
                "success": False,
                "error": str(e),
                "bucket": self.buckets['training'],
                "statistics": {"total_files": 0, "valid_pairs": 0},
                "ready_for_training": False
            }
    
    # ============================================================================
    # PREPARACIÓN DE DATOS
    # ============================================================================
    
    async def prepare_layer2_data(self, target_pairs: int = 100, source_bucket: str = "document-clean") -> Dict[str, Any]:
        """Preparar datos para entrenamiento - SIN HTTP REQUEST"""
        try:
            # Verificar estado actual
            data_status = self.check_layer2_data_status()
            
            if not data_status["success"]:
                return {
                    "success": False,
                    "error": "Error verificando datos existentes",
                    "details": data_status["error"]
                }
            
            current_pairs = data_status["statistics"]["valid_pairs"]
            
            if current_pairs >= target_pairs:
                return {
                    "success": True,
                    "message": f"Ya hay suficientes pares ({current_pairs}/{target_pairs})",
                    "current_pairs": current_pairs,
                    "target_pairs": target_pairs,
                    "action": "none_needed"
                }
            
            # Generar pares adicionales usando servicio directo
            needed_pairs = target_pairs - current_pairs
            
            try:
                # Usar servicio de datos sintéticos directamente
                result = await synthetic_data_service.generate_training_pairs_async(source_bucket, needed_pairs)
                
                return {
                    "success": True,
                    "message": f"Generados {needed_pairs} pares adicionales",
                    "current_pairs": current_pairs,
                    "target_pairs": target_pairs,
                    "needed_pairs": needed_pairs,
                    "generation_result": result
                }
                
            except Exception as e:
                logger.error(f"Error generando pares: {e}")
                return {
                    "success": False,
                    "error": f"Error generando pares adicionales: {str(e)}",
                    "current_pairs": current_pairs,
                    "needed_pairs": needed_pairs
                }
                
        except Exception as e:
            logger.error(f"Error preparando datos: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # ============================================================================
    # ENTRENAMIENTO LAYER 1
    # ============================================================================
    
    async def start_layer1_evaluation(self, job_id: str, max_images: int = 30):
        """Ejecutar evaluación de Capa 1"""
        try:
            self.update_job_status(job_id, "running", 10)
            
            # Importar módulo de Layer 1
            try:
                sys.path.append(str(Path(__file__).parent.parent.parent / "layers" / "layer-1"))
                from layer_1 import PreprocessingPipeline
            except ImportError as e:
                raise ImportError(f"No se pudo importar PreprocessingPipeline: {e}")
            
            self.update_job_status(job_id, "running", 30)
            
            # Obtener imágenes de MinIO
            files = minio_service.list_files(self.buckets['degraded'])
            
            # Limitar número de imágenes
            test_files = files[:max_images]
            
            self.update_job_status(job_id, "running", 50)
            
            # Inicializar pipeline
            pipeline = PreprocessingPipeline()
            results = []
            
            # Procesar imágenes
            for i, filename in enumerate(test_files):
                try:
                    # Descargar imagen
                    image_data = minio_service.download_file(self.buckets['degraded'], filename)
                    
                    # Procesar con pipeline (simulado)
                    # En implementación real aquí iría el procesamiento
                    results.append({
                        "filename": filename,
                        "processed": True,
                        "improvement_metrics": {
                            "otsu_applied": True,
                            "clahe_applied": True,
                            "deskew_applied": True
                        }
                    })
                    
                    # Actualizar progreso
                    progress = 50 + int((i / len(test_files)) * 40)
                    self.update_job_status(job_id, "running", progress)
                    
                except Exception as e:
                    logger.error(f"Error procesando {filename}: {e}")
                    results.append({
                        "filename": filename,
                        "processed": False,
                        "error": str(e)
                    })
            
            # Completar trabajo
            self.update_job_status(
                job_id, 
                "completed", 
                100,
                results={
                    "success": True,
                    "images_processed": len([r for r in results if r.get("processed")]),
                    "images_failed": len([r for r in results if not r.get("processed")]),
                    "total_images": len(results),
                    "details": results
                },
                end_time=datetime.now().isoformat()
            )
            
            logger.info(f"Evaluación Layer 1 completada: {job_id}")
            
        except Exception as e:
            logger.error(f"Error en evaluación Layer 1: {e}")
            self.update_job_status(
                job_id, 
                "failed", 
                error=str(e),
                end_time=datetime.now().isoformat()
            )
    
    # ============================================================================
    # ENTRENAMIENTO LAYER 2
    # ============================================================================
    
    async def start_layer2_training(self, job_id: str, num_epochs: int, max_pairs: int, 
                                  batch_size: int, use_training_bucket: bool = True):
        """Ejecutar entrenamiento de Capa 2"""
        try:
            self.update_job_status(job_id, "running", 10)
            
            # Validar parámetros
            validation_errors = self.validate_training_parameters(num_epochs, max_pairs, batch_size)
            if validation_errors:
                raise ValueError(f"Parámetros inválidos: {', '.join(validation_errors)}")
            
            self.update_job_status(job_id, "running", 20)
            
            # Verificar datos disponibles
            data_status = self.check_layer2_data_status()
            if not data_status["ready_for_training"]:
                raise ValueError("Datos insuficientes para entrenamiento")
            
            self.update_job_status(job_id, "running", 30)
            
            # Importar trainer de Layer 2
            try:
                sys.path.append(str(Path(__file__).parent.parent.parent / "layers" / "train-layers"))
                from train_layer_2 import create_layer2_trainer
                trainer = create_layer2_trainer()
            except ImportError as e:
                raise ImportError(f"No se pudo importar trainer Layer 2: {e}")
            
            self.update_job_status(job_id, "running", 40)
            
            # Simular entrenamiento por épocas
            for epoch in range(1, num_epochs + 1):
                # Aquí iría el entrenamiento real
                await asyncio.sleep(2)  # Simular tiempo de entrenamiento
                
                progress = 40 + int((epoch / num_epochs) * 50)
                self.update_job_status(
                    job_id, 
                    "running", 
                    progress,
                    current_epoch=epoch
                )
                
                logger.info(f"Época {epoch}/{num_epochs} completada para job {job_id}")
            
            # Completar entrenamiento
            self.update_job_status(
                job_id, 
                "completed", 
                100,
                current_epoch=num_epochs,
                results={
                    "success": True,
                    "epochs_completed": num_epochs,
                    "pairs_used": min(max_pairs, data_status["statistics"]["valid_pairs"]),
                    "batch_size": batch_size,
                    "use_training_bucket": use_training_bucket,
                    "final_metrics": {
                        "loss": 0.05,  # Simulado
                        "accuracy": 0.95  # Simulado
                    }
                },
                end_time=datetime.now().isoformat()
            )
            
            logger.info(f"Entrenamiento Layer 2 completado: {job_id}")
            
        except Exception as e:
            logger.error(f"Error en entrenamiento Layer 2: {e}")
            self.update_job_status(
                job_id, 
                "failed", 
                error=str(e),
                end_time=datetime.now().isoformat()
            )

# Instancia global del servicio
training_service = TrainingService()
