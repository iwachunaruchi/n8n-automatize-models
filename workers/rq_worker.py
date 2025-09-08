#!/usr/bin/env python3
"""
🏭 RQ WORKER - WORKER PROFESIONAL
=================================
Worker usando Redis Queue (RQ) en lugar del sistema de archivos.
"""

import os
import sys
import logging
import signal
from typing import List

# Configurar paths
sys.path.append('/app')
sys.path.append('/app/api')
sys.path.append('/app/project_root')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_worker():
    """Configurar worker RQ"""
    try:
        from redis import Redis
        from rq import Worker, Queue, Connection
        
        # Configuración Redis
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        redis_conn = Redis.from_url(redis_url)
        
        # Configuración del worker
        worker_name = os.getenv('RQ_WORKER_NAME', 'doc-restoration-worker')
        queue_names = os.getenv('RQ_QUEUE_NAMES', 'high,default,low').split(',')
        
        logger.info(f"🚀 Iniciando RQ Worker: {worker_name}")
        logger.info(f"📋 Colas a procesar: {queue_names}")
        logger.info(f"🔗 Redis URL: {redis_url}")
        
        # Crear colas
        queues = [Queue(name.strip(), connection=redis_conn) for name in queue_names]
        
        # Crear worker
        worker = Worker(
            queues,
            connection=redis_conn,
            name=worker_name,
            default_result_ttl=3600  # Mantener resultados 1 hora
        )
        
        # Configurar handlers de señales para shutdown graceful
        def signal_handler(sig, frame):
            logger.info(f"🛑 Recibida señal {sig}, deteniendo worker...")
            worker.request_stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        return worker
        
    except Exception as e:
        logger.error(f"❌ Error configurando worker: {e}")
        raise

def main():
    """Función principal del worker"""
    try:
        logger.info("🏭 INICIANDO RQ WORKER")
        logger.info("=" * 50)
        
        # Verificar conexión Redis
        try:
            from redis import Redis
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            redis_conn = Redis.from_url(redis_url)
            redis_conn.ping()
            logger.info("✅ Conexión Redis establecida")
        except Exception as e:
            logger.error(f"❌ Error conectando a Redis: {e}")
            return 1
        
        # Configurar worker
        worker = setup_worker()
        
        # Mostrar información del worker
        logger.info(f"🔧 Worker configurado: {worker.name}")
        logger.info(f"📋 Colas asignadas: {[q.name for q in worker.queues]}")
        logger.info(f"⏰ Resultado TTL: {worker.default_result_ttl}s")
        
        # Mostrar estadísticas iniciales
        from rq_job_system import get_job_queue_manager
        manager = get_job_queue_manager()
        stats = manager.get_queue_stats()
        logger.info(f"📊 Estadísticas iniciales: {stats}")
        
        # Iniciar worker (esto bloquea hasta recibir señal de parada)
        logger.info("🚀 Worker iniciado, esperando jobs...")
        logger.info("=" * 50)
        
        worker.work(with_scheduler=True)  # with_scheduler permite jobs programados
        
        logger.info("🛑 Worker detenido")
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Worker interrumpido por usuario")
        return 0
    except Exception as e:
        logger.error(f"❌ Error en worker: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
