#!/usr/bin/env python3
"""
🔧 WORKER BASE
=============
Configuración base y utilidades comunes para el worker
"""

import os
import sys
import logging
from datetime import datetime

# Configurar paths para importar módulos desde la raíz del proyecto
current_dir = os.path.dirname(os.path.abspath(__file__))
workers_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(workers_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'api'))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class WorkerStats:
    """Maneja estadísticas del worker"""
    
    def __init__(self):
        self.processed_jobs = 0
        self.failed_jobs = 0
        self.start_time = None
    
    def increment_processed(self):
        """Incrementar contador de jobs procesados"""
        self.processed_jobs += 1
    
    def increment_failed(self):
        """Incrementar contador de jobs fallidos"""
        self.failed_jobs += 1
    
    def set_start_time(self):
        """Establecer tiempo de inicio"""
        self.start_time = datetime.now()
    
    def print_summary(self):
        """Imprimir resumen de estadísticas"""
        if self.start_time:
            runtime = datetime.now() - self.start_time
            logger = logging.getLogger("WorkerStats")
            
            logger.info("=" * 50)
            logger.info("📊 RESUMEN DEL WORKER")
            logger.info(f"⏱️  Tiempo de ejecución: {runtime}")
            logger.info(f"✅ Jobs procesados: {self.processed_jobs}")
            logger.info(f"❌ Jobs fallidos: {self.failed_jobs}")
            
            if (self.processed_jobs + self.failed_jobs) > 0:
                success_rate = self.processed_jobs/(self.processed_jobs + self.failed_jobs)*100
                logger.info(f"📈 Tasa de éxito: {success_rate:.1f}%")
            
            logger.info("=" * 50)

def import_services():
    """Importar servicios existentes de forma segura"""
    services = {}
    services_available = False
    
    try:
        from api.services.training_service import TrainingService
        from api.services.synthetic_data_service import SyntheticDataService  
        from api.services.restoration_service import RestorationService
        from api.services.minio_service import MinIOService  # Nombre correcto
        from api.config.settings import jobs_state
        
        services = {
            'training': TrainingService(),
            'synthetic': SyntheticDataService(),
            'restoration': RestorationService(),
            'minio': MinIOService(),  # Nombre correcto
            'jobs_state': jobs_state
        }
        services_available = True
        
        logging.getLogger("WorkerBase").info("✅ Servicios existentes importados correctamente")
        
    except ImportError as e:
        logging.getLogger("WorkerBase").warning(f"⚠️  Error importando servicios: {e}")
        logging.getLogger("WorkerBase").info("🔄 Modo standalone activado")
    
    return services, services_available
