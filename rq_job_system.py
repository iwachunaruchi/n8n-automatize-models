#!/usr/bin/env python3
"""
🚀 REDIS QUEUE (RQ) JOB SYSTEM
==============================
Sistema profesional de jobs usando Redis Queue en lugar de archivos JSON.

Ventajas:
- Sistema robusto y probado en producción
- Persistencia en Redis
- Monitoreo con RQ Dashboard
- Retry automático
- Failover
- Múltiples workers
- Priority queues
"""
import os
import logging
from redis import Redis
from rq import Queue, Worker, get_current_job
from rq.job import Job
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de Redis
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

class JobQueueManager:
    """Manager profesional para cola de jobs usando RQ"""
    
    def __init__(self, redis_url: str = None):
        """Inicializar manager con conexión Redis"""
        self.redis_url = redis_url or REDIS_URL
        self.redis_conn = Redis.from_url(self.redis_url)
        
        # Crear colas con prioridades
        self.queues = {
            'high': Queue('high', connection=self.redis_conn),
            'default': Queue('default', connection=self.redis_conn),
            'low': Queue('low', connection=self.redis_conn)
        }
        self.default_queue = self.queues['default']
        
        logger.info(f"🚀 JobQueueManager inicializado - Redis: {self.redis_url}")
    
    def enqueue_job(self, 
                   job_function: str,
                   job_args: tuple = (),
                   job_kwargs: dict = None,
                   priority: str = 'default',
                   timeout: int = 300,
                   retry_attempts: int = 3,
                   job_id: str = None) -> str:
        """
        Encolar un job para procesamiento
        
        Args:
            job_function: Nombre del módulo.función (ej: 'workers.rq_tasks.training_job')
            job_args: Argumentos posicionales
            job_kwargs: Argumentos nombrados
            priority: 'high', 'default', 'low'
            timeout: Timeout en segundos
            retry_attempts: Intentos de retry
            job_id: ID personalizado del job
            
        Returns:
            Job ID de RQ
        """
        try:
            job_kwargs = job_kwargs or {}
            queue = self.queues.get(priority, self.default_queue)
            
            # Importar función del job dinámicamente
            module_path, function_name = job_function.rsplit('.', 1)
            module = __import__(module_path, fromlist=[function_name])
            func = getattr(module, function_name)
            
            # Encolar job
            rq_job = queue.enqueue(
                func,
                *job_args,
                **job_kwargs,
                timeout=timeout,
                job_id=job_id
            )
            
            logger.info(f"✅ Job encolado - ID: {rq_job.id}, Cola: {priority}, Función: {job_function}")
            return rq_job.id
            
        except Exception as e:
            logger.error(f"❌ Error encolando job: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de un job"""
        try:
            job = Job.fetch(job_id, connection=self.redis_conn)
            
            return {
                'job_id': job.id,
                'status': job.get_status(),
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'ended_at': job.ended_at.isoformat() if job.ended_at else None,
                'result': job.result,
                'meta': job.meta,
                'exc_info': job.exc_info,
                'timeout': job.timeout,
                'origin': job.origin
            }
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo status del job {job_id}: {e}")
            return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancelar un job"""
        try:
            job = Job.fetch(job_id, connection=self.redis_conn)
            job.cancel()
            logger.info(f"🚫 Job cancelado: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cancelando job {job_id}: {e}")
            return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de las colas"""
        try:
            stats = {}
            
            for name, queue in self.queues.items():
                stats[name] = {
                    'pending': len(queue),
                    'started': queue.started_job_registry.count,
                    'finished': queue.finished_job_registry.count,
                    'failed': queue.failed_job_registry.count,
                    'deferred': queue.deferred_job_registry.count
                }
            
            # Estadísticas globales
            stats['total'] = {
                'pending': sum(s['pending'] for s in stats.values()),
                'started': sum(s['started'] for s in stats.values()),
                'finished': sum(s['finished'] for s in stats.values()),
                'failed': sum(s['failed'] for s in stats.values()),
                'deferred': sum(s['deferred'] for s in stats.values())
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {e}")
            return {}
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Limpiar jobs antiguos"""
        try:
            cleaned = 0
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for queue in self.queues.values():
                # Limpiar registry de jobs terminados
                finished_jobs = queue.finished_job_registry.get_job_ids()
                for job_id in finished_jobs:
                    try:
                        job = Job.fetch(job_id, connection=self.redis_conn)
                        if job.ended_at and job.ended_at < cutoff_time:
                            job.delete()
                            cleaned += 1
                    except:
                        continue
                        
                # Limpiar registry de jobs fallidos
                failed_jobs = queue.failed_job_registry.get_job_ids()
                for job_id in failed_jobs:
                    try:
                        job = Job.fetch(job_id, connection=self.redis_conn)
                        if job.ended_at and job.ended_at < cutoff_time:
                            job.delete()
                            cleaned += 1
                    except:
                        continue
            
            logger.info(f"🧹 Limpieza completada: {cleaned} jobs eliminados")
            return cleaned
            
        except Exception as e:
            logger.error(f"❌ Error en limpieza: {e}")
            return 0
    
    def list_jobs(self, queue_name: str = 'default', status: str = 'all') -> List[Dict[str, Any]]:
        """Listar jobs de una cola"""
        try:
            queue = self.queues.get(queue_name, self.default_queue)
            jobs = []
            
            if status == 'all' or status == 'pending':
                for job in queue.get_jobs():
                    jobs.append(self._job_to_dict(job))
            
            if status == 'all' or status == 'started':
                for job_id in queue.started_job_registry.get_job_ids():
                    try:
                        job = Job.fetch(job_id, connection=self.redis_conn)
                        jobs.append(self._job_to_dict(job))
                    except:
                        continue
            
            if status == 'all' or status == 'finished':
                for job_id in queue.finished_job_registry.get_job_ids():
                    try:
                        job = Job.fetch(job_id, connection=self.redis_conn)
                        jobs.append(self._job_to_dict(job))
                    except:
                        continue
            
            if status == 'all' or status == 'failed':
                for job_id in queue.failed_job_registry.get_job_ids():
                    try:
                        job = Job.fetch(job_id, connection=self.redis_conn)
                        jobs.append(self._job_to_dict(job))
                    except:
                        continue
            
            return jobs
            
        except Exception as e:
            logger.error(f"❌ Error listando jobs: {e}")
            return []
    
    def _job_to_dict(self, job) -> Dict[str, Any]:
        """Convertir job RQ a diccionario"""
        return {
            'job_id': job.id,
            'status': job.get_status(),
            'created_at': job.created_at.isoformat() if job.created_at else None,
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'ended_at': job.ended_at.isoformat() if job.ended_at else None,
            'timeout': job.timeout,
            'origin': job.origin,
            'meta': job.meta,
            'description': job.description
        }

# Instancia global del manager
job_queue_manager = None

def get_job_queue_manager() -> JobQueueManager:
    """Factory function para obtener instancia del manager"""
    global job_queue_manager
    if job_queue_manager is None:
        job_queue_manager = JobQueueManager()
    return job_queue_manager

if __name__ == "__main__":
    # Test del sistema RQ
    print("🧪 Testing JobQueueManager...")
    
    manager = get_job_queue_manager()
    
    # Test 1: Obtener estadísticas
    print("\n📊 Estadísticas de colas:")
    stats = manager.get_queue_stats()
    for queue_name, queue_stats in stats.items():
        print(f"  {queue_name}: {queue_stats}")
    
    # Test 2: Encolar job simple
    print("\n🚀 Encolando job de prueba...")
    try:
        job_id = manager.enqueue_job(
            'test_jobs.simple_test_job',
            job_kwargs={'message': 'Test desde JobQueueManager', 'duration': 2},
            priority='default'
        )
        print(f"✅ Job encolado: {job_id}")
        
        # Verificar status
        status = manager.get_job_status(job_id)
        print(f"📋 Status: {status['status'] if status else 'No encontrado'}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✅ Test completado")
