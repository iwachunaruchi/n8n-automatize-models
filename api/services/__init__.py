"""
Servicios de la aplicación
"""
from .minio_service import minio_service
from .model_service import model_service  
from .image_analysis_service import image_analysis_service
from .synthetic_data_service import synthetic_data_service

__all__ = [
    'minio_service',
    'model_service',
    'image_analysis_service', 
    'synthetic_data_service'
]