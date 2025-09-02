"""
Constantes globales para la API de Restauración de Documentos
"""

# ===== CONFIGURACIÓN DE SERVICIOS =====
MINIO_CONFIG = {
    "ENDPOINT": "minio:9000",
    "ACCESS_KEY": "minio",
    "SECRET_KEY": "minio123",
    "SECURE": False
}

# URLs para desarrollo local
MINIO_LOCAL_URL = "http://localhost:9000"
API_LOCAL_URL = "http://localhost:8000"

# ===== BUCKETS DE ALMACENAMIENTO =====
BUCKETS = {
    "degraded": "document-degraded",
    "clean": "document-clean", 
    "restored": "document-restored",
    "training": "document-training"
}

# ===== CONFIGURACIÓN DE ARCHIVOS =====
FILE_CONFIG = {
    "MAX_SIZE": 50 * 1024 * 1024,  # 50MB
    "ALLOWED_EXTENSIONS": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
    "ALLOWED_MIME_TYPES": ["image/jpeg", "image/png", "image/tiff", "image/bmp"],
    "UPLOAD_TIMEOUT": 300  # 5 minutos
}

# ===== CONFIGURACIÓN DE PROCESAMIENTO =====
PROCESSING_CONFIG = {
    "MAX_IMAGE_SIZE": 2048,  # Tamaño máximo para procesamiento
    "ANALYSIS_SIZE": 1024,   # Tamaño para análisis
    "SAMPLE_SIZE": 512,      # Tamaño para muestras
    "BATCH_SIZE": 4,         # Batch size para entrenamiento
    "MAX_WORKERS": 4,        # Workers para procesamiento paralelo
    "MAX_PROCESSING_TIME": 300,  # Tiempo máximo de procesamiento en segundos
    "SUPPORTED_FORMATS": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
    "OUTPUT_FORMAT": "png",
    "QUALITY_SETTINGS": {"compression": 0.9, "dpi": 300},
    "MAX_JOBS": 100,         # Máximo número de trabajos simultáneos
    "JOB_TIMEOUT": 3600,     # Timeout para trabajos en segundos
    "CLEANUP_INTERVAL": 86400  # Intervalo de limpieza en segundos (24h)
}

# ===== CONFIGURACIÓN DE ENTRENAMIENTO =====
TRAINING_CONFIG = {
    "MIN_EPOCHS": 1,
    "MAX_EPOCHS": 100,
    "DEFAULT_EPOCHS": 10,
    "MIN_BATCH_SIZE": 1,
    "MAX_BATCH_SIZE": 32,
    "DEFAULT_BATCH_SIZE": 4,
    "LEARNING_RATES": [0.0001, 0.0005, 0.001, 0.005],
    "DEFAULT_LEARNING_RATE": 0.0001,
    "CHECKPOINT_FREQUENCY": 5,  # Cada cuántas épocas guardar checkpoint
    "VALIDATION_SPLIT": 0.2
}

# ===== CONFIGURACIÓN DE CLASIFICACIÓN =====
CLASSIFICATION_CONFIG = {
    "CONFIDENCE_THRESHOLD": 0.7,
    "QUALITY_THRESHOLDS": {
        "excellent": 0.9,
        "good": 0.7,
        "fair": 0.5,
        "poor": 0.3
    },
    "DOCUMENT_TYPES": {
        "clean": "documento_limpio",
        "degraded": "documento_degradado", 
        "handwritten": "documento_manuscrito",
        "printed": "documento_impreso",
        "mixed": "documento_mixto",
        "unknown": "documento_desconocido"
    }
}

# ===== RUTAS DE MODELOS =====
MODEL_PATHS = {
    "restormer_pretrained": "models/pretrained/restormer_denoising.pth",
    "layer1_checkpoint": "outputs/checkpoints/layer1_best.pth", 
    "layer2_checkpoint": "outputs/checkpoints/layer2_best.pth",
    "gradual_transfer": "outputs/checkpoints/gradual_transfer_final.pth",
    "temp_dir": "/tmp/training"
}

# ===== CONFIGURACIÓN DE LOGGING =====
LOGGING_CONFIG = {
    "LEVEL": "INFO",
    "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "DATE_FORMAT": "%Y-%m-%d %H:%M:%S",
    "MAX_BYTES": 10 * 1024 * 1024,  # 10MB
    "BACKUP_COUNT": 5
}

# ===== RESPUESTAS ESTÁNDAR =====
RESPONSE_MESSAGES = {
    "success": "Operación completada exitosamente",
    "error": "Error en la operación",
    "not_found": "Recurso no encontrado",
    "invalid_input": "Datos de entrada inválidos",
    "service_unavailable": "Servicio no disponible",
    "unauthorized": "No autorizado",
    "forbidden": "Acceso denegado",
    "timeout": "Tiempo de espera agotado",
    "file_too_large": "Archivo demasiado grande",
    "invalid_file_type": "Tipo de archivo no válido",
    "processing_error": "Error durante el procesamiento",
    "upload_success": "Archivo subido exitosamente",
    "training_started": "Entrenamiento iniciado",
    "training_completed": "Entrenamiento completado",
    "model_loaded": "Modelo cargado exitosamente",
    "restore_success": "Documento restaurado exitosamente",
    "job_created": "Trabajo creado exitosamente",
    "job_updated": "Trabajo actualizado exitosamente",
    "job_deleted": "Trabajo eliminado exitosamente",
    "batch_restore_completed": "Restauración por lotes completada",
    "synthetic_data_generated": "Datos sintéticos generados exitosamente",
    "noise_applied": "Ruido aplicado exitosamente",
    "degradation_completed": "Degradación completada exitosamente",
    "training_pairs_generated": "Pares de entrenamiento generados exitosamente",
    "augmentation_completed": "Augmentación del dataset completada"
}

# ===== CONFIGURACIÓN DE DATOS SINTÉTICOS =====
SYNTHETIC_DATA_CONFIG = {
    "NOISE_TYPES": ["gaussian", "salt_pepper", "blur", "speckle", "poisson"],
    "DEGRADATION_TYPES": ["mixed", "blur", "noise", "compression", "distortion", "aging"],
    "INTENSITY_RANGE": {"min": 0.01, "max": 1.0},
    "COUNT_LIMITS": {"min": 1, "max": 1000, "augment_min": 10, "augment_max": 10000},
    "SUPPORTED_FORMATS": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
    "OUTPUT_FORMAT": "png",
    "AUGMENTATION_TECHNIQUES": ["rotation", "flip", "brightness", "contrast", "noise", "blur"],
    "QUALITY_LEVELS": ["low", "medium", "high", "ultra"],
    "BATCH_SIZE": 8
}

# ===== CONFIGURACIÓN DE API =====
API_CONFIG = {
    "TITLE": "Document Restoration API",
    "VERSION": "1.0.0", 
    "DESCRIPTION": "API para restauración y procesamiento de documentos con ML",
    "DOCS_URL": "/docs",
    "REDOC_URL": "/redoc",
    "CORS_ORIGINS": ["*"],
    "CORS_METHODS": ["GET", "POST", "PUT", "DELETE"],
    "CORS_HEADERS": ["*"]
}

# ===== CONFIGURACIÓN DE TAREAS EN BACKGROUND =====
BACKGROUND_TASKS_CONFIG = {
    "MAX_CONCURRENT_TASKS": 5,
    "TASK_TIMEOUT": 3600,  # 1 hora
    "CLEANUP_INTERVAL": 300,  # 5 minutos
    "RETRY_ATTEMPTS": 3,
    "RETRY_DELAY": 60  # 1 minuto
}

# ===== CONFIGURACIÓN DE CACHE =====
CACHE_CONFIG = {
    "ENABLED": True,
    "TTL": 3600,  # 1 hora
    "MAX_SIZE": 1000,
    "CLEANUP_THRESHOLD": 0.8
}
