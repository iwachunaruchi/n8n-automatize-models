"""
Routers de la aplicación
"""
from .classification import router as classification_router
from .restoration import router as restoration_router
from .synthetic_data import router as synthetic_data_router
from .files import router as files_router
from .jobs_rq import router as jobs_router  # Corregido: jobs_rq
from .training import router as training_router
from .models import router as models_router

__all__ = [
    'classification_router',
    'restoration_router',
    'synthetic_data_router',
    'files_router',
    'jobs_router',
    'training_router',
    'models_router'
]