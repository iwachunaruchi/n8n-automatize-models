#!/usr/bin/env python3
"""
🛠️ UTILIDADES COMUNES PARA RQ WORKERS
=====================================
Funciones y clases de utilidad compartidas entre todos los workers RQ.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from rq import get_current_job

logger = logging.getLogger(__name__)

class RQJobProgressTracker:
    """Tracker de progreso para jobs RQ"""
    
    def __init__(self):
        self.current_job = get_current_job()
        self.start_time = datetime.now()
    
    def update_progress(self, progress: int, message: str = None, extra_data: dict = None):
        """Actualizar progreso del job actual"""
        try:
            if self.current_job:
                self.current_job.meta['progress'] = progress
                self.current_job.meta['updated_at'] = datetime.now().isoformat()
                
                if message:
                    self.current_job.meta['message'] = message
                
                if extra_data:
                    self.current_job.meta.update(extra_data)
                
                self.current_job.save_meta()
                logger.info(f"📈 Progreso actualizado: {progress}% - {message}")
                
        except Exception as e:
            logger.error(f"❌ Error actualizando progreso: {e}")
    
    def set_result(self, result: Any):
        """Establecer resultado final del job"""
        try:
            if self.current_job:
                elapsed_time = (datetime.now() - self.start_time).total_seconds()
                self.current_job.meta['completed_at'] = datetime.now().isoformat()
                self.current_job.meta['elapsed_time'] = elapsed_time
                self.current_job.meta['final_result'] = result
                self.current_job.save_meta()
                
        except Exception as e:
            logger.error(f"❌ Error estableciendo resultado: {e}")
    
    def log_error(self, error: Exception, context: str = None):
        """Registrar error en el job"""
        try:
            if self.current_job:
                error_info = {
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                    'error_context': context,
                    'error_time': datetime.now().isoformat()
                }
                self.current_job.meta['error'] = error_info
                self.current_job.save_meta()
                
        except Exception as e:
            logger.error(f"❌ Error registrando error: {e}")

def setup_job_environment():
    """Configurar entorno común para todos los jobs"""
    import sys
    import os
    
    # Configurar paths
    paths_to_add = [
        '/app',
        '/app/api', 
        '/app/project_root',
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.append(path)
    
    # Configurar logging para el job
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def execute_with_progress(func: Callable, 
                         total_steps: int,
                         step_description: str = "Procesando",
                         *args, **kwargs) -> Any:
    """Ejecutar función con tracking automático de progreso"""
    tracker = RQJobProgressTracker()
    
    try:
        tracker.update_progress(0, f"Iniciando {step_description}...")
        
        # Si la función acepta un callback de progreso, pasárselo
        if 'progress_callback' in kwargs:
            kwargs['progress_callback'] = tracker.update_progress
        
        result = func(*args, **kwargs)
        
        tracker.update_progress(100, f"{step_description} completado")
        tracker.set_result(result)
        
        return result
        
    except Exception as e:
        tracker.log_error(e, step_description)
        logger.error(f"❌ Error en {step_description}: {e}")
        raise

def simulate_work_with_progress(duration: int, description: str = "Trabajo"):
    """Simular trabajo con progreso para testing"""
    tracker = RQJobProgressTracker()
    
    for i in range(duration):
        progress = int((i + 1) / duration * 100)
        tracker.update_progress(progress, f"{description} - paso {i+1}/{duration}")
        time.sleep(1)
    
    return {
        'status': 'completed',
        'description': description,
        'duration': duration,
        'completed_at': datetime.now().isoformat()
    }
