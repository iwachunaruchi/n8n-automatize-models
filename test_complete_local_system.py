#!/usr/bin/env python3
"""
🧪 TESTING COMPLETO DEL SISTEMA RQ MODULAR LOCAL
===============================================
Script para probar todo el sistema RQ con estructura modular en local.
"""

import time
import sys
import os

# Configurar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_redis_connection():
    """Test 1: Verificar conexión a Redis"""
    print("🔍 Test 1: Verificando conexión a Redis...")
    try:
        from redis import Redis
        redis_conn = Redis(host='localhost', port=6379, db=0)
        redis_conn.ping()
        print("✅ Redis conectado correctamente")
        return True, redis_conn
    except Exception as e:
        print(f"❌ Error conectando a Redis: {e}")
        return False, None

def test_rq_job_manager():
    """Test 2: Probar RQ Job Manager"""
    print("\n🔍 Test 2: Probando RQ Job Manager...")
    try:
        from rq_job_system import get_job_queue_manager
        
        manager = get_job_queue_manager()
        
        # Obtener estadísticas
        stats = manager.get_queue_stats()
        print(f"📊 Estadísticas de colas:")
        for queue_name, queue_stats in stats.items():
            if queue_name != 'total':
                print(f"  • {queue_name}: {queue_stats['pending']} pending, {queue_stats['finished']} finished")
        
        print("✅ RQ Job Manager funcionando")
        return True, manager
    except Exception as e:
        print(f"❌ Error con RQ Job Manager: {e}")
        return False, None

def test_modular_tasks():
    """Test 3: Probar tasks modulares"""
    print("\n🔍 Test 3: Probando tasks modulares...")
    try:
        from workers.tasks.test_tasks import simple_test_job, math_calculation_job
        from workers.tasks.training_tasks import layer2_training_job
        
        # Test job simple
        result1 = simple_test_job(message="Test modular local", duration=1)
        print(f"✅ Simple test: {result1['status']}")
        
        # Test job matemático
        result2 = math_calculation_job(operation="multiply", a=7, b=3, complexity=1)
        print(f"✅ Math test: {result2['operation']} = {result2['result']}")
        
        print("✅ Tasks modulares funcionando directamente")
        return True
    except Exception as e:
        print(f"❌ Error con tasks modulares: {e}")
        return False

def test_rq_with_modular_tasks(manager):
    """Test 4: Probar RQ con tasks modulares"""
    print("\n🔍 Test 4: Probando RQ con tasks modulares...")
    try:
        # Encolar job simple usando el manager
        job_id1 = manager.enqueue_job(
            'workers.tasks.test_tasks.simple_test_job',
            job_kwargs={'message': 'Test RQ modular', 'duration': 2},
            priority='default'
        )
        print(f"✅ Job simple encolado: {job_id1}")
        
        # Encolar job matemático
        job_id2 = manager.enqueue_job(
            'workers.tasks.test_tasks.math_calculation_job',
            job_kwargs={'operation': 'power', 'a': 2, 'b': 3, 'complexity': 1},
            priority='default'
        )
        print(f"✅ Job matemático encolado: {job_id2}")
        
        # Verificar status
        status1 = manager.get_job_status(job_id1)
        status2 = manager.get_job_status(job_id2)
        
        print(f"📋 Status job 1: {status1['status'] if status1 else 'No encontrado'}")
        print(f"📋 Status job 2: {status2['status'] if status2 else 'No encontrado'}")
        
        print("✅ RQ con tasks modulares funcionando")
        return True, [job_id1, job_id2]
    except Exception as e:
        print(f"❌ Error con RQ + tasks modulares: {e}")
        return False, []

def test_worker_simulation():
    """Test 5: Simular worker processing"""
    print("\n🔍 Test 5: Simulando worker processing...")
    try:
        from redis import Redis
        from rq import Queue, SimpleWorker
        from workers.tasks.test_tasks import simple_test_job
        
        # Configurar worker
        redis_conn = Redis(host='localhost', port=6379, db=0)
        queue = Queue('default', connection=redis_conn)
        
        # Encolar job
        job = queue.enqueue(simple_test_job, message='Worker simulation', duration=2)
        print(f"✅ Job encolado para worker: {job.id}")
        
        # Simular worker (procesar 1 job)
        worker = SimpleWorker([queue], connection=redis_conn)
        print("🔄 Simulando worker processing...")
        
        # Procesar jobs (esto ejecutará realmente el job)
        worker.work(burst=True, with_scheduler=False)
        
        # Verificar resultado
        job.refresh()
        print(f"📋 Job status después del worker: {job.get_status()}")
        if job.result:
            print(f"📊 Resultado: {job.result}")
        
        print("✅ Worker simulation completada")
        return True
    except Exception as e:
        print(f"❌ Error en worker simulation: {e}")
        return False

def main():
    """Función principal de testing"""
    print("🚀 TESTING COMPLETO DEL SISTEMA RQ MODULAR LOCAL")
    print("=" * 60)
    
    # Test 1: Redis
    redis_ok, redis_conn = test_redis_connection()
    if not redis_ok:
        print("❌ Redis no disponible. Instalar y iniciar Redis primero.")
        return False
    
    # Test 2: RQ Job Manager
    manager_ok, manager = test_rq_job_manager()
    if not manager_ok:
        print("❌ RQ Job Manager no funciona.")
        return False
    
    # Test 3: Tasks modulares directas
    tasks_ok = test_modular_tasks()
    if not tasks_ok:
        print("❌ Tasks modulares no funcionan.")
        return False
    
    # Test 4: RQ + Tasks modulares
    rq_tasks_ok, job_ids = test_rq_with_modular_tasks(manager)
    if not rq_tasks_ok:
        print("❌ RQ con tasks modulares no funciona.")
        return False
    
    # Test 5: Worker simulation
    worker_ok = test_worker_simulation()
    if not worker_ok:
        print("❌ Worker simulation falló.")
        return False
    
    print("\n🎉 TODOS LOS TESTS PASARON!")
    print("✅ Sistema RQ modular listo para usar")
    
    # Mostrar estadísticas finales
    print("\n📊 ESTADÍSTICAS FINALES:")
    final_stats = manager.get_queue_stats()
    print(f"📋 Total jobs procesados: {final_stats.get('total', {}).get('finished', 0)}")
    print(f"📋 Jobs pendientes: {final_stats.get('total', {}).get('pending', 0)}")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 COMANDOS PARA USAR EL SISTEMA:")
        print("-" * 40)
        print("1. 💻 Iniciar Worker:")
        print("   python workers/rq_worker.py")
        print("\n2. 📊 Crear Jobs:")
        print("   python -c \"from rq_job_system import get_job_queue_manager; manager = get_job_queue_manager(); print('Job ID:', manager.enqueue_job('workers.tasks.test_tasks.simple_test_job', job_kwargs={'message': 'Mi job'}))\"")
        print("\n3. 🔍 Ver Status:")
        print("   python -c \"from rq_job_system import get_job_queue_manager; manager = get_job_queue_manager(); print(manager.get_queue_stats())\"")
    else:
        print("\n❌ SISTEMA NO LISTO - Revisar errores arriba")
        sys.exit(1)
