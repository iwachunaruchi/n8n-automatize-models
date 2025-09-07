#!/usr/bin/env python3
"""
🎨 SYNTHETIC DATA HANDLER
========================
Maneja exclusivamente jobs de generación de datos sintéticos
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("SyntheticHandler")

class SyntheticDataHandler:
    """Handler especializado para jobs de datos sintéticos"""
    
    def __init__(self, synthetic_service=None):
        self.synthetic_service = synthetic_service
        self.service_available = synthetic_service is not None
    
    async def process_synthetic_job(self, job: Dict[str, Any], shared_queue) -> None:
        """Procesar job de generación de datos sintéticos"""
        job_id = job["job_id"]
        params = job["parameters"]
        count = params.get("count", 50)
        bucket = params.get("bucket", "document-clean")
        
        logger.info(f"🎨 Generando {count} imágenes sintéticas - Job: {job_id}")
        
        if not self.service_available:
            raise Exception("SyntheticDataService no disponible")
        
        try:
            logger.info("🔧 Usando SyntheticDataService")
            
            # Llamada real al SyntheticDataService
            result = await self.synthetic_service.generate_training_pairs_async(
                source_bucket=bucket,
                count=count
            )
            
            logger.info("✅ Generación completada")
            
        except Exception as e:
            logger.error(f"❌ Error en SyntheticDataService: {e}")
            raise
