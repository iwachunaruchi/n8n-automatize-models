#!/usr/bin/env python3
"""
🧪 TEST TASKS - RQ SPECIALIZED
=============================
Tasks de prueba y utilidades para validar el sistema RQ.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

# Importar utilidades RQ
from ..utils.rq_utils import RQJobProgressTracker, setup_job_environment, simulate_work_with_progress

logger = logging.getLogger(__name__)

def simple_test_job(message: str = "Test job",
                   duration: int = 5,
                   **kwargs) -> Dict[str, Any]:
    """
    🧪 Job de prueba simple
    
    Args:
        message: Mensaje del test
        duration: Duración en segundos
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultado del test
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    logger.info(f"🧪 Iniciando test simple: {message}")
    
    try:
        # Usar la función de utilidad para simular trabajo
        result = simulate_work_with_progress(duration, f"Test: {message}")
        
        final_result = {
            'status': 'completed',
            'message': message,
            'duration': duration,
            'result': result,
            'completed_at': datetime.now().isoformat(),
            'job_type': 'simple_test'
        }
        
        tracker.set_result(final_result)
        logger.info(f"✅ Test simple completado: {message}")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "Simple Test")
        logger.error(f"❌ Error en test simple: {e}")
        raise

def math_calculation_job(operation: str = "add",
                        a: float = 10,
                        b: float = 5,
                        complexity: int = 3,
                        **kwargs) -> Dict[str, Any]:
    """
    🔢 Job de cálculo matemático
    
    Args:
        operation: Operación a realizar (add, multiply, power, etc.)
        a: Primer operando
        b: Segundo operando
        complexity: Complejidad del cálculo (tiempo de simulación)
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultado del cálculo
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    logger.info(f"🔢 Iniciando cálculo: {operation}({a}, {b})")
    
    try:
        tracker.update_progress(10, f"Preparando operación {operation}...")
        
        # Simular complejidad computacional
        for i in range(complexity):
            progress = int(10 + ((i / complexity) * 70))
            tracker.update_progress(progress, f"Procesando paso {i+1}/{complexity}")
            time.sleep(1)
        
        # Realizar cálculo
        tracker.update_progress(80, "Realizando cálculo...")
        
        if operation == "add":
            result = a + b
        elif operation == "multiply":
            result = a * b
        elif operation == "power":
            result = a ** b
        elif operation == "divide":
            if b != 0:
                result = a / b
            else:
                raise ValueError("División por cero")
        elif operation == "factorial":
            import math
            result = math.factorial(int(a))
        else:
            raise ValueError(f"Operación no soportada: {operation}")
        
        tracker.update_progress(90, "Finalizando cálculo...")
        time.sleep(0.5)
        
        final_result = {
            'status': 'completed',
            'operation': operation,
            'operands': {'a': a, 'b': b},
            'result': result,
            'complexity': complexity,
            'completed_at': datetime.now().isoformat(),
            'job_type': 'math_calculation'
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, f"Cálculo completado: {result}")
        
        logger.info(f"✅ Cálculo completado: {operation}({a}, {b}) = {result}")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "Math Calculation")
        logger.error(f"❌ Error en cálculo matemático: {e}")
        raise

def system_health_check_job(**kwargs) -> Dict[str, Any]:
    """
    🏥 Job de verificación de salud del sistema
    
    Args:
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con estado de salud del sistema
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    logger.info("🏥 Iniciando verificación de salud del sistema")
    
    try:
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'services': {},
            'overall_status': 'healthy'
        }
        
        # Verificar Redis
        tracker.update_progress(20, "Verificando conexión a Redis...")
        try:
            from redis import Redis
            redis_conn = Redis(host='localhost', port=6379, db=0)
            redis_conn.ping()
            health_status['services']['redis'] = {'status': 'healthy', 'response_time': 'fast'}
        except Exception as redis_error:
            health_status['services']['redis'] = {'status': 'unhealthy', 'error': str(redis_error)}
            health_status['overall_status'] = 'degraded'
        
        # Verificar MinIO
        tracker.update_progress(40, "Verificando conexión a MinIO...")
        try:
            from api.services.minio_service import minio_service
            buckets = minio_service.list_buckets()
            health_status['services']['minio'] = {
                'status': 'healthy', 
                'buckets_count': len(buckets),
                'buckets': buckets
            }
        except Exception as minio_error:
            health_status['services']['minio'] = {'status': 'unhealthy', 'error': str(minio_error)}
            health_status['overall_status'] = 'degraded'
        
        # Verificar sistema de archivos
        tracker.update_progress(60, "Verificando sistema de archivos...")
        try:
            import os
            import psutil
            
            disk_usage = psutil.disk_usage('/')
            memory_info = psutil.virtual_memory()
            
            health_status['services']['filesystem'] = {
                'status': 'healthy',
                'disk_usage_percent': (disk_usage.used / disk_usage.total) * 100,
                'memory_usage_percent': memory_info.percent,
                'available_memory_gb': memory_info.available / (1024**3)
            }
            
            # Alertas si los recursos son bajos
            if memory_info.percent > 90:
                health_status['services']['filesystem']['warning'] = 'High memory usage'
                health_status['overall_status'] = 'warning'
                
        except Exception as fs_error:
            health_status['services']['filesystem'] = {'status': 'unhealthy', 'error': str(fs_error)}
        
        # Verificar modelos de ML
        tracker.update_progress(80, "Verificando modelos de ML...")
        try:
            # Verificar que los modelos están disponibles
            model_paths = [
                '/app/models/NAFnet/NAFNet-SIDD-width64.pth',
                '/app/temp_models/nafnet_sidd_width64.pth'
            ]
            
            available_models = []
            for model_path in model_paths:
                if os.path.exists(model_path):
                    available_models.append(model_path)
            
            health_status['services']['ml_models'] = {
                'status': 'healthy' if available_models else 'degraded',
                'available_models': len(available_models),
                'model_paths': available_models
            }
            
        except Exception as ml_error:
            health_status['services']['ml_models'] = {'status': 'unhealthy', 'error': str(ml_error)}
        
        final_result = {
            'status': 'completed',
            'health_status': health_status,
            'completed_at': datetime.now().isoformat(),
            'job_type': 'system_health_check'
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, f"Verificación completada: {health_status['overall_status']}")
        
        logger.info(f"✅ Verificación de salud completada: {health_status['overall_status']}")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "System Health Check")
        logger.error(f"❌ Error en verificación de salud: {e}")
        raise

def stress_test_job(duration: int = 30,
                   cpu_intensive: bool = True,
                   memory_test: bool = True,
                   **kwargs) -> Dict[str, Any]:
    """
    💪 Job de prueba de estrés del sistema
    
    Args:
        duration: Duración del test en segundos
        cpu_intensive: Realizar pruebas intensivas de CPU
        memory_test: Realizar pruebas de memoria
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultados del stress test
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    logger.info(f"💪 Iniciando stress test: {duration}s")
    
    try:
        import psutil
        import math
        
        stress_results = {
            'duration': duration,
            'cpu_intensive': cpu_intensive,
            'memory_test': memory_test,
            'start_time': datetime.now().isoformat(),
            'performance_metrics': []
        }
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            progress = int((elapsed / duration) * 90)
            
            tracker.update_progress(
                progress, 
                f"Stress test: {elapsed:.1f}s/{duration}s"
            )
            
            # Test CPU intensivo
            if cpu_intensive:
                # Cálculos matemáticos intensivos
                for _ in range(10000):
                    math.sqrt(math.factorial(10))
            
            # Test de memoria
            if memory_test:
                # Crear y liberar memoria
                temp_data = [i for i in range(100000)]
                del temp_data
            
            # Registrar métricas cada 5 segundos
            if int(elapsed) % 5 == 0:
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                stress_results['performance_metrics'].append({
                    'elapsed_time': elapsed,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent
                })
            
            time.sleep(0.1)
        
        # Métricas finales
        final_cpu = psutil.cpu_percent(interval=1)
        final_memory = psutil.virtual_memory().percent
        
        stress_results['end_time'] = datetime.now().isoformat()
        stress_results['final_metrics'] = {
            'cpu_percent': final_cpu,
            'memory_percent': final_memory
        }
        
        final_result = {
            'status': 'completed',
            'stress_results': stress_results,
            'completed_at': datetime.now().isoformat(),
            'job_type': 'stress_test'
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, f"Stress test completado: CPU {final_cpu}%, RAM {final_memory}%")
        
        logger.info(f"✅ Stress test completado: {duration}s")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "Stress Test")
        logger.error(f"❌ Error en stress test: {e}")
        raise
