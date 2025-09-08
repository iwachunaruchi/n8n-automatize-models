#!/usr/bin/env python3
"""
🎯 RQ TASKS REGISTRY - MAIN ENTRY POINT
=======================================
Punto de entrada principal para todas las tasks RQ.
Importa y expone todas las tasks especializadas de forma organizada.
"""

import logging
from typing import Dict, Any

# Importar todas las tasks especializadas
from .tasks import AVAILABLE_TASKS, get_task_function, list_available_tasks, get_task_info

# Importar tasks específicas para compatibilidad hacia atrás
from .tasks.training_tasks import layer2_training_job, layer1_training_job, fine_tuning_job
from .tasks.synthetic_data_tasks import generate_synthetic_data_job, augment_dataset_job, validate_synthetic_data_job
from .tasks.restoration_tasks import single_document_restoration_job, batch_restoration_job, quality_assessment_job
from .tasks.test_tasks import simple_test_job, math_calculation_job, system_health_check_job, stress_test_job

# Importar utilidades
from .utils.rq_utils import RQJobProgressTracker, setup_job_environment

logger = logging.getLogger(__name__)

# ================================
# COMPATIBILITY ALIASES
# ================================
# Mantener nombres originales para compatibilidad

def test_job(message: str = "Test job", duration: int = 5, **kwargs) -> Dict[str, Any]:
    """Alias para simple_test_job - compatibilidad hacia atrás"""
    return simple_test_job(message=message, duration=duration, **kwargs)

# Alias para jobs principales
training_job = layer2_training_job
synthetic_data_generation_job = generate_synthetic_data_job
document_restoration_job = single_document_restoration_job

# ================================
# REGISTRY FUNCTIONS
# ================================

def get_all_tasks():
    """Obtener todas las tasks disponibles"""
    return AVAILABLE_TASKS

def execute_task(task_name: str, **kwargs) -> Any:
    """
    Ejecutar una task por nombre
    
    Args:
        task_name: Nombre de la task a ejecutar
        **kwargs: Argumentos para la task
        
    Returns:
        Resultado de la task
        
    Raises:
        ValueError: Si la task no existe
    """
    task_function = get_task_function(task_name)
    
    if task_function is None:
        available_tasks = list_available_tasks()
        raise ValueError(f"Task '{task_name}' no encontrada. Disponibles: {available_tasks}")
    
    logger.info(f"🚀 Ejecutando task: {task_name}")
    return task_function(**kwargs)

def get_task_registry_info() -> Dict[str, Any]:
    """
    Obtener información completa del registro de tasks
    
    Returns:
        Dict con información del registro
    """
    return {
        'total_tasks': len(AVAILABLE_TASKS),
        'task_categories': get_task_info(),
        'available_tasks': list_available_tasks(),
        'registry_version': '2.0.0',
        'system': 'RQ Specialized Tasks'
    }

# ================================
# EXPORTED FUNCTIONS
# ================================

__all__ = [
    # Task execution
    'execute_task',
    'get_all_tasks',
    'get_task_registry_info',
    
    # Training tasks
    'layer2_training_job',
    'layer1_training_job', 
    'fine_tuning_job',
    'training_job',  # alias
    
    # Synthetic data tasks
    'generate_synthetic_data_job',
    'augment_dataset_job',
    'validate_synthetic_data_job',
    'synthetic_data_generation_job',  # alias
    
    # Restoration tasks
    'single_document_restoration_job',
    'batch_restoration_job',
    'quality_assessment_job',
    'document_restoration_job',  # alias
    
    # Test tasks
    'simple_test_job',
    'math_calculation_job',
    'system_health_check_job',
    'stress_test_job',
    'test_job',  # alias
    
    # Utils
    'RQJobProgressTracker',
    'setup_job_environment',
    
    # Registry
    'AVAILABLE_TASKS',
    'get_task_function',
    'list_available_tasks',
    'get_task_info'
]

if __name__ == "__main__":
    # Mostrar información del registro cuando se ejecuta directamente
    print("🎯 RQ TASKS REGISTRY")
    print("=" * 50)
    
    registry_info = get_task_registry_info()
    print(f"📊 Total tasks: {registry_info['total_tasks']}")
    print(f"🏷️ Versión: {registry_info['registry_version']}")
    
    print("\n📋 Tasks por categoría:")
    for category, tasks in registry_info['task_categories'].items():
        print(f"\n🔸 {category.upper()}:")
        for task_name, description in tasks.items():
            print(f"  • {task_name}: {description}")
    
    print(f"\n✅ Sistema de tasks modular listo para usar")
