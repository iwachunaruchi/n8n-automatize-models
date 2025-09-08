#!/usr/bin/env python3
"""
🧪 TEST LOCAL DEL SISTEMA RQ
============================
Prueba básica del sistema Redis Queue a nivel de file system.
"""

import sys
import os
import time
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_redis_connection():
    """Test 1: Verificar conexión a Redis"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        logger.info("✅ Test 1: Redis conectado correctamente")
        return True
    except Exception as e:
        logger.error(f"❌ Test 1: Error conectando a Redis: {e}")
        return False

def test_rq_basic():
    """Test 2: Probar funcionalidad básica de RQ"""
    try:
        from redis import Redis
        from rq import Queue
        
        # Conectar a Redis
        redis_conn = Redis(host='localhost', port=6379, db=0)
        
        # Crear cola de prueba
        test_queue = Queue('test', connection=redis_conn)
        
        logger.info("✅ Test 2: RQ inicializado correctamente")
        logger.info(f"📊 Jobs en cola: {len(test_queue)}")
        
        return True, test_queue
    except Exception as e:
        logger.error(f"❌ Test 2: Error con RQ: {e}")
        return False, None

def simple_job_function(message, duration=3):
    """Función de job simple para pruebas"""
    logger.info(f"🚀 Job iniciado: {message}")
    
    for i in range(duration):
        logger.info(f"📈 Progreso: {i+1}/{duration}")
        time.sleep(1)
    
    result = {
        'message': message,
        'duration': duration,
        'completed_at': datetime.now().isoformat(),
        'status': 'completed'
    }
    
    logger.info(f"✅ Job completado: {result}")
    return result

def test_job_enqueue():
    """Test 3: Encolar un job"""
    try:
        from redis import Redis
        from rq import Queue
        from test_jobs import simple_test_job
        
        redis_conn = Redis(host='localhost', port=6379, db=0)
        test_queue = Queue('test', connection=redis_conn)
        
        # Encolar job
        job = test_queue.enqueue(
            simple_test_job,
            message='Test job desde file system',
            duration=3
        )
        
        logger.info(f"✅ Test 3: Job encolado - ID: {job.id}")
        logger.info(f"📋 Estado inicial: {job.get_status()}")
        
        return True, job
    except Exception as e:
        logger.error(f"❌ Test 3: Error encolando job: {e}")
        return False, None

def test_job_status_monitoring(job):
    """Test 4: Monitorear estado del job"""
    try:
        logger.info("🔍 Test 4: Monitoreando job...")
        
        for i in range(10):  # Monitorear por 10 segundos máximo
            status = job.get_status()
            logger.info(f"📊 Estado: {status}")
            
            if status == 'finished':
                result = job.result
                logger.info(f"✅ Job terminado: {result}")
                return True, result
            elif status == 'failed':
                logger.error(f"❌ Job falló: {job.exc_info}")
                return False, None
            
            time.sleep(1)
        
        logger.warning("⏰ Timeout monitoreando job")
        return False, None
        
    except Exception as e:
        logger.error(f"❌ Test 4: Error monitoreando job: {e}")
        return False, None

def test_rq_job_system():
    """Test 5: Probar nuestro JobQueueManager"""
    try:
        # Importar nuestro sistema
        from rq_job_system import JobQueueManager
        
        manager = JobQueueManager(redis_url='redis://localhost:6379/0')
        
        # Obtener estadísticas
        stats = manager.get_queue_stats()
        logger.info(f"✅ Test 5: JobQueueManager funcionando")
        logger.info(f"📊 Estadísticas: {stats}")
        
        return True, manager
        
    except Exception as e:
        logger.error(f"❌ Test 5: Error con JobQueueManager: {e}")
        return False, None

def main():
    """Función principal de tests"""
    logger.info("🧪 INICIANDO TESTS RQ LOCAL")
    logger.info("=" * 50)
    
    # Test 1: Redis
    if not test_redis_connection():
        logger.error("💥 Redis no disponible, deteniendo tests")
        return
    
    # Test 2: RQ básico
    success, queue = test_rq_basic()
    if not success:
        logger.error("💥 RQ no disponible, deteniendo tests")
        return
    
    # Test 3: Encolar job
    success, job = test_job_enqueue()
    if not success:
        logger.error("💥 Error encolando job")
        return
    
    # Nota: Para test 4 necesitaríamos un worker corriendo
    logger.info("⚠️  Para ejecutar el job necesitas un worker corriendo")
    logger.info("🏭 Ejecuta en otra terminal: python test_rq_local.py --worker")
    
    # Test 5: Nuestro sistema
    success, manager = test_rq_job_system()
    if success:
        logger.info("🎉 TODOS LOS TESTS BÁSICOS PASARON")
    
    # Mostrar información del job
    logger.info(f"📝 Job ID creado: {job.id}")
    logger.info(f"📊 Estado actual: {job.get_status()}")
    
    logger.info("=" * 50)
    logger.info("✅ Tests completados")

def run_worker():
    """Ejecutar worker simple para procesar jobs"""
    try:
        from redis import Redis
        from rq import Worker, Queue
        import os
        
        logger.info("🏭 INICIANDO WORKER LOCAL")
        logger.info("=" * 30)
        
        redis_conn = Redis(host='localhost', port=6379, db=0)
        queue = Queue('test', connection=redis_conn)
        
        # En Windows, RQ no puede usar fork, así que usamos SimpleWorker
        from rq import SimpleWorker
        worker = SimpleWorker([queue], connection=redis_conn)
        logger.info("🚀 Worker iniciado (SimpleWorker para Windows), procesando jobs...")
        
        worker.work()
        
    except KeyboardInterrupt:
        logger.info("🛑 Worker detenido por usuario")
    except Exception as e:
        logger.error(f"❌ Error en worker: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        run_worker()
    else:
        main()
