"""
Doc Restormer - Pipeline de restauración de documentos
Combina Restormer + ESRGAN para restauración completa
"""

__version__ = "1.0.0"
__author__ = "Document Restoration Team"

from .pipeline import DocumentRestorationPipeline
from .models.restormer import Restormer
from .models.esrgan import RealESRGAN

__all__ = [
    'DocumentRestorationPipeline',
    'Restormer',
    'RealESRGAN'
]
