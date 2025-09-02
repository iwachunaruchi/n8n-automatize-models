"""
Servicios de la aplicación
"""
from .minio_service import minio_service
from .model_service import model_service  
from .image_analysis_service import image_analysis_service
from .synthetic_data_service import synthetic_data_service
from .training_service import training_service
from .classification_service import classification_service
from .file_management_service import file_management_service

__all__ = [
    'minio_service',
    'model_service',
    'image_analysis_service', 
    'synthetic_data_service',
    'training_service',
    'classification_service',
    'file_management_service'
]