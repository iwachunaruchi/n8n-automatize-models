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

__all__ = [
    'MINIO_CONFIG',
    'BUCKETS', 
    'model_state',
    'jobs_state',
    'API_CONFIG'
]
