"""
Servicio especializado para clasificación de documentos
"""
import logging
import sys
from typing import Dict, Any, Optional, Tuple
import uuid
from datetime import datetime

# Asegurar que podemos importar desde el directorio api
sys.path.append('/app/api')

# Importaciones más robustas
image_analysis_service = None
minio_service = None

try:
    from services.image_analysis_service import image_analysis_service
    from services.minio_service import minio_service
    logging.info("Servicios de clasificación importados exitosamente")
except ImportError as e:
    logging.error(f"Error importando servicios de clasificación: {e}")
    image_analysis_service = None
    minio_service = None

# Importar constantes
try:
    from config.constants import (
        BUCKETS, 
        CLASSIFICATION_CONFIG, 
        RESPONSE_MESSAGES,
        MINIO_LOCAL_URL
    )
    logging.info("Constantes de clasificación importadas exitosamente")
except ImportError as e:
    logging.error(f"Error importando constantes de clasificación: {e}")
    # Fallback con valores por defecto
    BUCKETS = {
        "degraded": "document-degraded",
        "clean": "document-clean", 
        "restored": "document-restored",
        "training": "document-training"
    }
    CLASSIFICATION_CONFIG = {
        "CONFIDENCE_THRESHOLD": 0.7,
        "QUALITY_THRESHOLDS": {"excellent": 0.9, "good": 0.7, "fair": 0.5, "poor": 0.3},
        "DOCUMENT_TYPES": {"clean": "documento_limpio", "degraded": "documento_degradado"}
    }
    RESPONSE_MESSAGES = {"upload_success": "Archivo subido exitosamente"}
    MINIO_LOCAL_URL = "http://localhost:9000"

logger = logging.getLogger(__name__)

class ClassificationService:
    """Servicio centralizado para clasificación de documentos"""
    
    def __init__(self):
        self.confidence_threshold = CLASSIFICATION_CONFIG["CONFIDENCE_THRESHOLD"]
        self.quality_thresholds = CLASSIFICATION_CONFIG["QUALITY_THRESHOLDS"]
        self.document_types = CLASSIFICATION_CONFIG["DOCUMENT_TYPES"]
        
    def classify_document(self, file_data: bytes, filename: str = None) -> Dict[str, Any]:
        """
        Clasificar documento y determinar bucket de destino
        
        Args:
            file_data: Datos binarios del archivo
            filename: Nombre del archivo (opcional)
            
        Returns:
            Dict con resultado de clasificación
        """
        try:
            logger.info(f"Iniciando clasificación de documento: {filename}")
            
            # Análisis de calidad de imagen
            quality_analysis = image_analysis_service.analyze_image_quality(file_data)
            
            # Clasificación de tipo de documento
            type_analysis = image_analysis_service.classify_document_type(file_data)
            
            # Extraer información del análisis
            document_type = type_analysis.get("type", "unknown")
            confidence = type_analysis.get("confidence", 0.0)
            details = type_analysis.get("details", {})
            
            # Determinar bucket de destino
            target_bucket = self._determine_target_bucket(document_type, confidence)
            
            # Generar filename único si no se proporciona
            if not filename:
                filename = f"classified_{uuid.uuid4().hex[:8]}.png"
            else:
                # Agregar timestamp para evitar colisiones
                base_name = filename.rsplit('.', 1)[0]
                extension = filename.rsplit('.', 1)[1] if '.' in filename else 'png'
                filename = f"{base_name}_{uuid.uuid4().hex[:8]}.{extension}"
            
            # Subir archivo al bucket determinado
            uploaded_filename = minio_service.upload_file(file_data, target_bucket, filename)
            
            # Generar URL de acceso
            file_url = f"{MINIO_LOCAL_URL}/{target_bucket}/{uploaded_filename}"
            
            # Construir respuesta completa
            result = {
                "status": "success",
                "message": RESPONSE_MESSAGES["upload_success"],
                "classification": {
                    "document_type": document_type,
                    "mapped_type": self.document_types.get(document_type, document_type),
                    "confidence": confidence,
                    "quality_score": quality_analysis.get("overall_quality", 0.0),
                    "quality_level": self._get_quality_level(quality_analysis.get("overall_quality", 0.0)),
                    "details": details
                },
                "storage": {
                    "bucket": target_bucket,
                    "filename": uploaded_filename,
                    "file_url": file_url,
                    "size": len(file_data)
                },
                "analysis": quality_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Clasificación completada: {document_type} -> {target_bucket}")
            return result
            
        except Exception as e:
            logger.error(f"Error en clasificación de documento: {e}")
            raise Exception(f"Error en clasificación: {str(e)}")
    
    def classify_batch(self, files_data: list) -> Dict[str, Any]:
        """
        Clasificar múltiples documentos en lote
        
        Args:
            files_data: Lista de tuplas (file_data, filename)
            
        Returns:
            Dict con resultados de clasificación batch
        """
        try:
            logger.info(f"Iniciando clasificación batch de {len(files_data)} archivos")
            
            results = []
            errors = []
            stats = {
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "buckets": {}
            }
            
            for i, (file_data, filename) in enumerate(files_data):
                try:
                    result = self.classify_document(file_data, filename)
                    results.append(result)
                    
                    # Actualizar estadísticas
                    bucket = result["storage"]["bucket"]
                    stats["buckets"][bucket] = stats["buckets"].get(bucket, 0) + 1
                    stats["successful"] += 1
                    
                except Exception as e:
                    error_info = {
                        "filename": filename,
                        "error": str(e),
                        "index": i
                    }
                    errors.append(error_info)
                    stats["failed"] += 1
                
                stats["processed"] += 1
            
            batch_result = {
                "status": "completed",
                "message": f"Clasificación batch completada: {stats['successful']}/{stats['processed']} exitosos",
                "statistics": stats,
                "results": results,
                "errors": errors,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Clasificación batch completada: {stats}")
            return batch_result
            
        except Exception as e:
            logger.error(f"Error en clasificación batch: {e}")
            raise Exception(f"Error en clasificación batch: {str(e)}")
    
    def _determine_target_bucket(self, document_type: str, confidence: float) -> str:
        """Determinar bucket de destino basado en clasificación"""
        
        # Si la confianza es baja, usar bucket degraded por defecto
        if confidence < self.confidence_threshold:
            return BUCKETS["degraded"]
        
        # Mapear tipos de documento a buckets
        if document_type in ["clean", "documento_limpio"]:
            return BUCKETS["clean"]
        elif document_type in ["degraded", "documento_degradado"]:
            return BUCKETS["degraded"]
        else:
            # Para tipos desconocidos o mixtos, usar degraded como fallback
            return BUCKETS["degraded"]
    
    def _get_quality_level(self, quality_score: float) -> str:
        """Determinar nivel de calidad basado en score"""
        for level, threshold in sorted(self.quality_thresholds.items(), key=lambda x: x[1], reverse=True):
            if quality_score >= threshold:
                return level
        return "poor"
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de clasificación"""
        try:
            stats = {}
            
            # Obtener estadísticas de cada bucket
            for bucket_name, bucket_id in BUCKETS.items():
                try:
                    files = minio_service.list_files(bucket_id)
                    stats[bucket_name] = {
                        "bucket_id": bucket_id,
                        "file_count": len(files),
                        "files": files[:10]  # Primeros 10 archivos como muestra
                    }
                except Exception as e:
                    logger.warning(f"Error obteniendo stats de bucket {bucket_name}: {e}")
                    stats[bucket_name] = {
                        "bucket_id": bucket_id,
                        "file_count": 0,
                        "error": str(e)
                    }
            
            return {
                "status": "success",
                "buckets": stats,
                "total_files": sum(bucket.get("file_count", 0) for bucket in stats.values()),
                "classification_config": {
                    "confidence_threshold": self.confidence_threshold,
                    "quality_thresholds": self.quality_thresholds,
                    "document_types": self.document_types
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas de clasificación: {e}")
            raise Exception(f"Error en estadísticas: {str(e)}")

# Instancia global del servicio
classification_service = ClassificationService()
