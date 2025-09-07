#!/usr/bin/env python3
"""
⚙️ SHARED JOB WORKER - ARQUITECTURA MODULAR
==========================================
Worker principal que coordina handlers especializados
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Configurar paths para importar módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(current_dir)

# Importar cola compartida
from shared_job_queue import create_shared_queue

# Importar componentes modulares
from core.worker_base import WorkerStats, import_services
from core.job_processor import JobProcessor
from handlers.training_handler import TrainingHandler
from handlers.synthetic_handler import SyntheticDataHandler
from handlers.restoration_handler import RestorationHandler

logger = logging.getLogger("SharedJobWorker")

class SharedJobWorker:
    """Worker principal con arquitectura modular"""
    
    def __init__(self):
        """Inicialización del worker y todos sus componentes"""
        self.running = False
        self.stats = WorkerStats()
        
        # Inicializar cola compartida
        self.shared_queue = create_shared_queue()
        logger.info("🔄 Cola compartida conectada")
        
        # Importar servicios existentes
        services, services_available = import_services()
        self.services_available = services_available
        
        # Inicializar handlers especializados
        if services_available:
            self.training_handler = TrainingHandler(
                services.get('training'), 
                services.get('minio')
            )
            self.synthetic_handler = SyntheticDataHandler(
                services.get('synthetic')
            )
            self.restoration_handler = RestorationHandler(
                services.get('restoration')
            )
            self.jobs_state = services.get('jobs_state', {})
        else:
            # Handlers sin servicios (modo standalone)
            self.training_handler = TrainingHandler()
            self.synthetic_handler = SyntheticDataHandler()
            self.restoration_handler = RestorationHandler()
            self.jobs_state = {}
        
        # Inicializar procesador central
        self.job_processor = JobProcessor(
            self.training_handler,
            self.synthetic_handler,
            self.restoration_handler
        )
        
        logger.info("🔧 Worker modular inicializado")
    
    async def start(self):
        """Iniciar el worker"""
        self.running = True
        self.stats.set_start_time()
        
        logger.info("🚀 Job Worker Modular iniciado")
        logger.info("=" * 50)
        logger.info("🔄 Cola compartida conectada")
        logger.info("⚙️  Handlers especializados activos")
        logger.info("🎯 Procesamiento asíncrono")
        logger.info("=" * 50)
        
        # Loop principal
        consecutive_empty_polls = 0
        max_empty_polls = 30
        
        while self.running:
            try:
                # Buscar siguiente job
                job = self.shared_queue.dequeue_job()
                
                if job:
                    consecutive_empty_polls = 0
                    await self._process_job(job)
                else:
                    consecutive_empty_polls += 1
                    
                    if consecutive_empty_polls <= 5:
                        logger.debug("⏳ Esperando jobs...")
                    elif consecutive_empty_polls == max_empty_polls:
                        logger.info("😴 Worker en standby...")
                    
                    await asyncio.sleep(2)
                    
            except KeyboardInterrupt:
                logger.info("🛑 Worker detenido por usuario")
                break
            except Exception as e:
                logger.error(f"❌ Error en worker: {e}")
                await asyncio.sleep(5)
        
        self.stats.print_summary()
    
    async def _process_job(self, job):
        """Procesar job usando el procesador modular"""
        try:
            await self.job_processor.process_job(job, self.shared_queue)
            self.stats.increment_processed()
        except Exception as e:
            self.stats.increment_failed()
            logger.error(f"❌ Error procesando job: {e}")

async def main():
    """Función principal del worker"""
    logger.info("⚙️ INICIANDO JOB WORKER MODULAR")
    
    worker = SharedJobWorker()
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("🛑 Worker detenido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error fatal en worker: {e}")

if __name__ == "__main__":
    print("⚙️ JOB WORKER - ARQUITECTURA MODULAR")
    print("=" * 50)
    print("🔧 Handlers especializados por responsabilidad")
    print("📁 Código organizado en módulos")
    print("🎯 Fácil mantenimiento y extensión")
    print("=" * 50)
    print()
    
    # Ejecutar el worker
    asyncio.run(main())
