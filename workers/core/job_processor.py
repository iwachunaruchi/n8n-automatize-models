#!/usr/bin/env python3
"""
⚙️ JOB PROCESSOR
===============
Coordina el procesamiento de jobs usando los handlers especializados
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("JobProcessor")

class JobProcessor:
    """Procesador central que coordina los handlers"""
    
    def __init__(self, training_handler, synthetic_handler, restoration_handler):
        self.training_handler = training_handler
        self.synthetic_handler = synthetic_handler
        self.restoration_handler = restoration_handler
    
    async def process_job(self, job: Dict[str, Any], shared_queue) -> None:
        """Procesar job usando el handler apropiado"""
        job_id = job["job_id"]
        job_type = job["job_type"]
        
        logger.info(f"🔄 Procesando job: {job_id} ({job_type})")
        
        # Marcar como running
        shared_queue.update_job_status(
            job_id, 
            "running", 
            progress=0, 
            start_time=datetime.now().isoformat()
        )
        
        try:
            # Dispatch al handler apropiado
            if job_type == "layer2_training":
                await self.training_handler.process_training_job(job, shared_queue)
            elif job_type == "synthetic_data_generation":
                await self.synthetic_handler.process_synthetic_job(job, shared_queue)
            elif job_type == "batch_restoration":
                await self.restoration_handler.process_restoration_job(job, shared_queue)
            else:
                raise ValueError(f"Tipo de job desconocido: {job_type}")
            
            # Marcar como completado
            shared_queue.update_job_status(
                job_id, 
                "completed", 
                progress=100,
                completed_at=datetime.now().isoformat()
            )
            
            logger.info(f"✅ Job {job_id} completado exitosamente")
            
        except Exception as e:
            # Marcar como fallido
            shared_queue.update_job_status(
                job_id,
                "failed",
                error=str(e),
                failed_at=datetime.now().isoformat()
            )
            
            logger.error(f"❌ Job {job_id} falló: {e}")
            raise
