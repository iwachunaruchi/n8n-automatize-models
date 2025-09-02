"""
Configuración del paquete config
"""
from .settings import (
    MINIO_CONFIG,
    BUCKETS,
    model_state,
    jobs_state,
    API_CONFIG
)

# Importar constantes globales
from .constants import *

__all__ = [
    # Settings existentes
    'MINIO_CONFIG',
    'BUCKETS', 
    'model_state',
    'jobs_state',
    'API_CONFIG',
    
    # Constantes nuevas
    'MINIO_LOCAL_URL',
    'API_LOCAL_URL',
    'FILE_CONFIG',
    'PROCESSING_CONFIG', 
    'TRAINING_CONFIG',
    'CLASSIFICATION_CONFIG',
    'MODEL_PATHS',
    'LOGGING_CONFIG',
    'RESPONSE_MESSAGES',
    'BACKGROUND_TASKS_CONFIG',
    'CACHE_CONFIG'
]
