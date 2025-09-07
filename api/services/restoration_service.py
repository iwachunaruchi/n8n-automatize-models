"""
Servicio especializado para restauración de documentos
"""
import logging
import sys
import uuid
import io
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Asegurar que podemos importar desde el directorio api
sys.path.append('/app/api')

# Importaciones más robustas
image_analysis_service = None
minio_service = None

try:
    from services.image_analysis_service import image_analysis_service
    from services.minio_service import minio_service
    logging.info("Servicios de restauración importados exitosamente")
except ImportError as e:
    logging.error(f"Error importando servicios de restauración: {e}")
    image_analysis_service = None
    minio_service = None

# Importar constantes
try:
    from config.constants import (
        BUCKETS, 
        PROCESSING_CONFIG, 
        RESPONSE_MESSAGES,
        MINIO_LOCAL_URL
    )
    logging.info("Constantes de restauración importadas exitosamente")
except ImportError as e:
    logging.error(f"Error importando constantes de restauración: {e}")
    # Fallback con valores por defecto
    BUCKETS = {
        "degraded": "document-degraded",
        "clean": "document-clean", 
        "restored": "document-restored",
        "training": "document-training"
    }
    PROCESSING_CONFIG = {
        "BATCH_SIZE": 4,
        "MAX_PROCESSING_TIME": 300,
        "SUPPORTED_FORMATS": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
        "OUTPUT_FORMAT": "png",
        "QUALITY_SETTINGS": {"compression": 0.9, "dpi": 300}
    }
    RESPONSE_MESSAGES = {"restore_success": "Documento restaurado exitosamente"}
    MINIO_LOCAL_URL = "http://localhost:9000"

logger = logging.getLogger(__name__)

class RestorationService:
    """Servicio centralizado para restauración de documentos"""
    
    def __init__(self):
        self.batch_size = PROCESSING_CONFIG["BATCH_SIZE"]
        self.max_processing_time = PROCESSING_CONFIG["MAX_PROCESSING_TIME"]
        self.supported_formats = PROCESSING_CONFIG["SUPPORTED_FORMATS"]
        self.output_format = PROCESSING_CONFIG["OUTPUT_FORMAT"]
        
    def restore_document(self, file_data: bytes, filename: str = None) -> Dict[str, Any]:
        """
        Restaurar un documento individual
        
        Args:
            file_data: Datos del archivo
            filename: Nombre del archivo (opcional)
            
        Returns:
            Dict con información de la restauración
        """
        try:
            if not image_analysis_service:
                raise Exception("Servicio de análisis de imagen no disponible")
                
            if not minio_service:
                raise Exception("Servicio MinIO no disponible")
            
            # Validar entrada
            if not file_data:
                raise ValueError("No se proporcionaron datos de archivo")
            
            # Generar ID único para este trabajo
            job_id = str(uuid.uuid4())
            start_time = datetime.now()
            
            logger.info(f"Iniciando restauración de documento {job_id}")
            
            # Restaurar imagen usando el servicio de análisis
            restored_data = image_analysis_service.restore_image(file_data)
            
            if not restored_data:
                raise Exception("Error en la restauración - datos vacíos")
            
            # Generar nombre único para archivo restaurado
            restored_filename = f"restored_{uuid.uuid4()}.{self.output_format}"
            
            # Subir archivo restaurado a MinIO
            upload_result = minio_service.upload_file(
                restored_data, 
                BUCKETS['restored'], 
                restored_filename
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "success",
                "job_id": job_id,
                "message": RESPONSE_MESSAGES.get("restore_success", "Documento restaurado exitosamente"),
                "result": {
                    "original_filename": filename or "unknown",
                    "restored_filename": restored_filename,
                    "bucket": BUCKETS['restored'],
                    "download_url": f"{MINIO_LOCAL_URL}/{BUCKETS['restored']}/{restored_filename}",
                    "processing_time_seconds": processing_time,
                    "file_size_bytes": len(restored_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error en restauración de documento: {e}")
            return {
                "status": "error",
                "message": f"Error restaurando documento: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def restore_batch(self, files_data: list, background_job_id: str = None) -> Dict[str, Any]:
        """
        Restaurar múltiples documentos en lote
        
        Args:
            files_data: Lista de datos de archivos
            background_job_id: ID del trabajo en background (opcional)
            
        Returns:
            Dict con información del procesamiento por lotes
        """
        try:
            if not files_data:
                raise ValueError("No se proporcionaron archivos para restaurar")
            
            batch_id = background_job_id or str(uuid.uuid4())
            start_time = datetime.now()
            results = []
            successful_restorations = 0
            failed_restorations = 0
            
            logger.info(f"Iniciando restauración por lotes {batch_id} de {len(files_data)} archivos")
            
            for i, file_data in enumerate(files_data):
                try:
                    # Restaurar archivo individual
                    result = self.restore_document(file_data, f"batch_file_{i+1}")
                    
                    if result["status"] == "success":
                        successful_restorations += 1
                    else:
                        failed_restorations += 1
                        
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error restaurando archivo {i+1} en lote: {e}")
                    failed_restorations += 1
                    results.append({
                        "status": "error",
                        "file_index": i+1,
                        "message": f"Error: {str(e)}"
                    })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "completed",
                "batch_id": batch_id,
                "summary": {
                    "total_files": len(files_data),
                    "successful_restorations": successful_restorations,
                    "failed_restorations": failed_restorations,
                    "processing_time_seconds": processing_time
                },
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error en restauración por lotes: {e}")
            return {
                "status": "error",
                "message": f"Error en procesamiento por lotes: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def get_restoration_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de restauraciones
        
        Returns:
            Dict con estadísticas del servicio
        """
        try:
            stats = {
                "service": "restoration_service",
                "status": "active",
                "configuration": {
                    "batch_size": self.batch_size,
                    "max_processing_time": self.max_processing_time,
                    "supported_formats": self.supported_formats,
                    "output_format": self.output_format
                },
                "buckets": {
                    "input_bucket": BUCKETS.get('degraded', 'document-degraded'),
                    "output_bucket": BUCKETS.get('restored', 'document-restored')
                }
            }
            
            # Intentar obtener estadísticas de archivos restaurados
            if minio_service:
                try:
                    restored_files = minio_service.list_files(BUCKETS['restored'])
                    stats["statistics"] = {
                        "total_restored_files": len(restored_files),
                        "latest_files": restored_files[:5] if restored_files else []
                    }
                except Exception as e:
                    logger.warning(f"No se pudieron obtener estadísticas de archivos: {e}")
                    stats["statistics"] = {"error": "No se pudieron obtener estadísticas"}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas de restauración: {e}")
            return {
                "service": "restoration_service",
                "status": "error",
                "message": f"Error: {str(e)}"
            }

# Instancia global del servicio
restoration_service = RestorationService()
