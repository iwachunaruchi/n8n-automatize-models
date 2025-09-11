#!/usr/bin/env python3
"""
🎯 TASKS PACKAGE - RQ SPECIALIZED
================================
Paquete de tasks especializadas para el sistema RQ.

Estructura modular:
- training_tasks: Jobs de entrenamiento de modelos
- synthetic_data_tasks: Jobs de generación de datos sintéticos  
- restoration_tasks: Jobs de restauración de documentos
- test_tasks: Jobs de prueba y utilidades
"""

# Importar todas las tasks especializadas
from .training_tasks import (
    layer2_training_job,
    layer1_training_job,
    fine_tuning_job
)

from .synthetic_data_tasks import (
    generate_synthetic_data_job,
    augment_dataset_job,
    validate_synthetic_data_job,
    generate_nafnet_dataset_job,
    validate_nafnet_dataset_job
)

from .restoration_tasks import (
    single_document_restoration_job,
    batch_restoration_job,
    quality_assessment_job
)

from .test_tasks import (
    simple_test_job,
    math_calculation_job,
    system_health_check_job,
    stress_test_job
)

# Registro de todas las tasks disponibles
AVAILABLE_TASKS = {
    # Training Tasks
    'layer2_training': layer2_training_job,
    'layer1_training': layer1_training_job,
    'fine_tuning': fine_tuning_job,
    
    # Synthetic Data Tasks
    'generate_synthetic_data': generate_synthetic_data_job,
    'augment_dataset': augment_dataset_job,
    'validate_synthetic_data': validate_synthetic_data_job,
    'generate_nafnet_dataset': generate_nafnet_dataset_job,
    'validate_nafnet_dataset': validate_nafnet_dataset_job,
    
    # Restoration Tasks
    'restore_document': single_document_restoration_job,
    'batch_restoration': batch_restoration_job,
    'quality_assessment': quality_assessment_job,
    
    # Test & Utility Tasks
    'simple_test': simple_test_job,
    'math_calculation': math_calculation_job,
    'system_health_check': system_health_check_job,
    'stress_test': stress_test_job
}

def get_task_function(task_name: str):
    """
    Obtener función de task por nombre
    
    Args:
        task_name: Nombre de la task
        
    Returns:
        Función de la task o None si no existe
    """
    return AVAILABLE_TASKS.get(task_name)

def list_available_tasks():
    """
    Listar todas las tasks disponibles
    
    Returns:
        Lista de nombres de tasks disponibles
    """
    return list(AVAILABLE_TASKS.keys())

def get_task_info():
    """
    Obtener información detallada de todas las tasks
    
    Returns:
        Dict con información de cada task
    """
    return {
        'training_tasks': {
            'layer2_training': 'Entrenamiento de modelo Layer 2',
            'layer1_training': 'Entrenamiento de modelo Layer 1', 
            'fine_tuning': 'Fine-tuning de modelos existentes'
        },
        'synthetic_data_tasks': {
            'generate_synthetic_data': 'Generación de datos sintéticos',
            'augment_dataset': 'Aumento de dataset existente',
            'validate_synthetic_data': 'Validación de calidad de datos sintéticos',
            'generate_nafnet_dataset': 'Generación de dataset NAFNet estructurado',
            'validate_nafnet_dataset': 'Validación de dataset NAFNet'
        },
        'restoration_tasks': {
            'restore_document': 'Restauración de documento individual',
            'batch_restoration': 'Restauración en lotes',
            'quality_assessment': 'Evaluación de calidad de restauraciones'
        },
        'test_tasks': {
            'simple_test': 'Test simple del sistema',
            'math_calculation': 'Cálculo matemático de prueba',
            'system_health_check': 'Verificación de salud del sistema',
            'stress_test': 'Prueba de estrés del sistema'
        }
    }
