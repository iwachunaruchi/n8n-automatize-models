#!/usr/bin/env python3
"""
🧠 TRAINING HANDLER
==================
Maneja exclusivamente jobs de entrenamiento Layer 2
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("TrainingHandler")

class TrainingHandler:
    """Handler especializado para jobs de entrenamiento y evaluación"""

    def __init__(self, training_service=None, minio_service=None):
        self.training_service = training_service
        self.minio_service = minio_service
        self.service_available = training_service is not None
    
    async def process_training_job(self, job: Dict[str, Any], shared_queue) -> None:
        """Procesar job de entrenamiento Layer 2"""
        job_id = job["job_id"]
        params = job["parameters"]
        epochs = params.get("num_epochs", 10)
        batch_size = params.get("batch_size", 2)
        max_pairs = params.get("max_pairs", 100)
        
        logger.info(f"🧠 Iniciando entrenamiento Layer 2: {job_id}")
        shared_queue.update_job_status(job_id, "running", started_at=datetime.now().isoformat())
        
        try:
            if not self.service_available:
                raise Exception("Servicio de entrenamiento no disponible")
            
            # Ejecutar entrenamiento real usando el servicio
            result = await self.training_service.start_layer2_training(
                job_id=job_id,
                num_epochs=epochs,
                max_pairs=max_pairs,
                batch_size=batch_size,
                use_training_bucket=params.get("use_training_bucket", True),
                use_finetuning=params.get("use_finetuning", True),
                freeze_backbone=params.get("freeze_backbone", False),
                finetuning_lr_factor=params.get("finetuning_lr_factor", 0.1)
            )
            
            shared_queue.update_job_status(
                job_id, "completed", 
                completed_at=datetime.now().isoformat(),
                results=result,
                progress=100
            )
            logger.info(f"✅ Entrenamiento Layer 2 completado: {job_id}")
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento Layer 2 {job_id}: {e}")
            shared_queue.update_job_status(
                job_id, "failed",
                completed_at=datetime.now().isoformat(), 
                error=str(e)
            )
    
    async def process_evaluation_job(self, job: Dict[str, Any], shared_queue) -> None:
        """Procesar job de evaluación Layer 1"""
        job_id = job["job_id"]
        params = job["parameters"]
        max_images = params.get("max_images", 30)
        
        logger.info(f"🔍 Iniciando evaluación Layer 1: {job_id}")
        shared_queue.update_job_status(job_id, "running", started_at=datetime.now().isoformat())
        
        try:
            if not self.service_available:
                raise Exception("Servicio de entrenamiento no disponible")
            
            # Ejecutar evaluación real usando el servicio
            result = await self.training_service.start_layer1_evaluation(job_id, max_images)
            
            shared_queue.update_job_status(
                job_id, "completed",
                completed_at=datetime.now().isoformat(),
                results=result,
                progress=100
            )
            logger.info(f"✅ Evaluación Layer 1 completada: {job_id}")
            
        except Exception as e:
            logger.error(f"❌ Error en evaluación Layer 1 {job_id}: {e}")
            shared_queue.update_job_status(
                job_id, "failed",
                completed_at=datetime.now().isoformat(),
                error=str(e)
            )
