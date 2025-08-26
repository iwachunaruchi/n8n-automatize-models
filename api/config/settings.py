"""
Configuración centralizada de la aplicación
"""
import os
from typing import Dict, Any

# Configuración de MinIO
MINIO_CONFIG = {
    'endpoint': os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
    'access_key': os.getenv('MINIO_ACCESS_KEY', 'minio'),
    'secret_key': os.getenv('MINIO_SECRET_KEY', 'minio123'),
    'secure': os.getenv('MINIO_SECURE', 'false').lower() == 'true'
}

# Buckets de MinIO
BUCKETS = {
    'degraded': 'document-degraded',
    'clean': 'document-clean',
    'restored': 'document-restored',
    'training': 'document-training'
}

# Estado global del modelo
model_state = {
    'model': None,
    'device': None,
    'loaded': False
}

# Estado de trabajos
jobs_state: Dict[str, Any] = {}

# Configuración de la API
API_CONFIG = {
    'title': "Document Restoration API",
    'description': "API para restauración de documentos con Transfer Learning Gradual",
    'version': "1.0.0"
}
