"""
Servicio especializado para manejo de archivos
"""
import logging
import sys
from typing import Dict, Any, Optional, List, Tuple
import uuid
import io
from datetime import datetime

# Asegurar que podemos importar desde el directorio api
sys.path.append('/app/api')

# Importaciones más robustas
minio_service = None
image_analysis_service = None

try:
    from services.minio_service import minio_service
    from services.image_analysis_service import image_analysis_service
    logging.info("Servicios importados exitosamente")
except ImportError as e:
    logging.error(f"Error importando servicios: {e}")
    minio_service = None
    image_analysis_service = None

# Importar constantes
try:
    from config.constants import (
        BUCKETS, 
        FILE_CONFIG, 
        RESPONSE_MESSAGES,
        MINIO_LOCAL_URL
    )
    logging.info("Constantes importadas exitosamente")
except ImportError as e:
    logging.error(f"Error importando constantes: {e}")
    # Fallback con valores por defecto
    BUCKETS = {
        "degraded": "document-degraded",
        "clean": "document-clean", 
        "restored": "document-restored",
        "training": "document-training"
    }
    FILE_CONFIG = {
        "MAX_SIZE": 50 * 1024 * 1024,
        "ALLOWED_EXTENSIONS": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
        "ALLOWED_MIME_TYPES": ["image/jpeg", "image/png", "image/tiff", "image/bmp"],
        "UPLOAD_TIMEOUT": 300
    }
    RESPONSE_MESSAGES = {"upload_success": "Archivo subido exitosamente"}
    MINIO_LOCAL_URL = "http://localhost:9000"

logger = logging.getLogger(__name__)

class FileManagementService:
    """Servicio centralizado para manejo de archivos"""
    
    def __init__(self):
        self.max_size = FILE_CONFIG["MAX_SIZE"]
        self.allowed_extensions = FILE_CONFIG["ALLOWED_EXTENSIONS"]
        self.allowed_mime_types = FILE_CONFIG["ALLOWED_MIME_TYPES"]
        
    def upload_file(self, file_data: bytes, bucket: str, filename: str = None, 
                   validate_type: bool = True) -> Dict[str, Any]:
        """
        Subir archivo a bucket específico con validaciones
        
        Args:
            file_data: Datos binarios del archivo
            bucket: Bucket de destino
            filename: Nombre del archivo (opcional)
            validate_type: Si validar tipo de archivo
            
        Returns:
            Dict con información del archivo subido
        """
        try:
            logger.info(f"Subiendo archivo a bucket: {bucket}")
            
            # Validar bucket
            if bucket not in BUCKETS.values():
                raise ValueError(f"Bucket no válido: {bucket}")
            
            # Validar tamaño
            if len(file_data) > self.max_size:
                raise ValueError(f"Archivo demasiado grande: {len(file_data)} bytes")
            
            # Validar tipo si se requiere
            if validate_type:
                self._validate_file_type(file_data)
            
            # Generar filename único si no se proporciona
            if not filename:
                filename = f"upload_{uuid.uuid4().hex[:12]}.png"
            else:
                # Asegurar filename único
                base_name = filename.rsplit('.', 1)[0]
                extension = filename.rsplit('.', 1)[1] if '.' in filename else 'png'
                filename = f"{base_name}_{uuid.uuid4().hex[:8]}.{extension}"
            
            # Subir archivo
            uploaded_filename = minio_service.upload_file(file_data, bucket, filename)
            
            # Generar URL de acceso
            file_url = f"{MINIO_LOCAL_URL}/{bucket}/{uploaded_filename}"
            
            result = {
                "status": "success",
                "message": RESPONSE_MESSAGES["upload_success"],
                "bucket": bucket,
                "filename": uploaded_filename,
                "original_filename": filename,
                "file_url": file_url,
                "size": len(file_data),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Archivo subido exitosamente: {uploaded_filename}")
            return result
            
        except Exception as e:
            logger.error(f"Error subiendo archivo: {e}")
            raise Exception(f"Error en upload: {str(e)}")
    
    def download_file(self, bucket: str, filename: str) -> Tuple[bytes, str]:
        """
        Descargar archivo específico
        
        Args:
            bucket: Bucket origen
            filename: Nombre del archivo
            
        Returns:
            Tupla (file_data, content_type)
        """
        try:
            logger.info(f"Descargando archivo: {bucket}/{filename}")
            
            # Validar bucket
            if bucket not in BUCKETS.values():
                raise ValueError(f"Bucket no válido: {bucket}")
            
            # Descargar archivo
            file_data = minio_service.download_file(bucket, filename)
            
            # Determinar content type basado en extensión
            content_type = self._get_content_type(filename)
            
            logger.info(f"Archivo descargado exitosamente: {len(file_data)} bytes")
            return file_data, content_type
            
        except Exception as e:
            logger.error(f"Error descargando archivo: {e}")
            raise Exception(f"Error en download: {str(e)}")
    
    def list_files(self, bucket: str, prefix: str = "", limit: int = None) -> Dict[str, Any]:
        """
        Listar archivos en bucket
        
        Args:
            bucket: Bucket a listar
            prefix: Prefijo para filtrar archivos
            limit: Límite de archivos a retornar
            
        Returns:
            Dict con lista de archivos
        """
        try:
            logger.info(f"Listando archivos en bucket: {bucket}")
            
            # Validar bucket
            if bucket not in BUCKETS.values():
                raise ValueError(f"Bucket no válido: {bucket}")
            
            # Listar archivos
            files = minio_service.list_files(bucket, prefix)
            
            # Aplicar límite si se especifica
            if limit and len(files) > limit:
                files = files[:limit]
                limited = True
            else:
                limited = False
            
            # Agregar URLs a los archivos
            files_with_urls = []
            for file_info in files:
                file_name = file_info.get("name", file_info) if isinstance(file_info, dict) else file_info
                file_with_url = {
                    "name": file_name,
                    "url": f"{MINIO_LOCAL_URL}/{bucket}/{file_name}",
                    "bucket": bucket
                }
                
                # Agregar información adicional si está disponible
                if isinstance(file_info, dict):
                    file_with_url.update({k: v for k, v in file_info.items() if k != "name"})
                
                files_with_urls.append(file_with_url)
            
            result = {
                "status": "success",
                "bucket": bucket,
                "prefix": prefix,
                "total_files": len(files_with_urls),
                "limited": limited,
                "limit_applied": limit,
                "files": files_with_urls,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Archivos listados: {len(files_with_urls)} en {bucket}")
            return result
            
        except Exception as e:
            logger.error(f"Error listando archivos: {e}")
            raise Exception(f"Error en listado: {str(e)}")
    
    def analyze_file(self, file_data: bytes, filename: str = None) -> Dict[str, Any]:
        """
        Analizar archivo sin subirlo
        
        Args:
            file_data: Datos binarios del archivo
            filename: Nombre del archivo (opcional)
            
        Returns:
            Dict con análisis del archivo
        """
        try:
            logger.info(f"Analizando archivo: {filename}")
            
            # Validar tipo
            self._validate_file_type(file_data)
            
            # Realizar análisis de imagen
            quality_analysis = image_analysis_service.analyze_image_quality(file_data)
            type_analysis = image_analysis_service.classify_document_type(file_data)
            
            result = {
                "status": "success",
                "filename": filename,
                "size": len(file_data),
                "quality_analysis": quality_analysis,
                "type_analysis": type_analysis,
                "recommendations": self._get_recommendations(quality_analysis, type_analysis),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Análisis completado para: {filename}")
            return result
            
        except Exception as e:
            logger.error(f"Error analizando archivo: {e}")
            raise Exception(f"Error en análisis: {str(e)}")
    
    def delete_file(self, bucket: str, filename: str) -> Dict[str, Any]:
        """
        Eliminar archivo específico
        
        Args:
            bucket: Bucket origen
            filename: Nombre del archivo
            
        Returns:
            Dict con resultado de eliminación
        """
        try:
            logger.info(f"Eliminando archivo: {bucket}/{filename}")
            
            # Validar bucket
            if bucket not in BUCKETS.values():
                raise ValueError(f"Bucket no válido: {bucket}")
            
            # Eliminar archivo
            minio_service.delete_file(bucket, filename)
            
            result = {
                "status": "success",
                "message": "Archivo eliminado exitosamente",
                "bucket": bucket,
                "filename": filename,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Archivo eliminado exitosamente: {filename}")
            return result
            
        except Exception as e:
            logger.error(f"Error eliminando archivo: {e}")
            raise Exception(f"Error en eliminación: {str(e)}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de almacenamiento"""
        try:
            stats = {}
            total_files = 0
            
            for bucket_name, bucket_id in BUCKETS.items():
                try:
                    files = minio_service.list_files(bucket_id)
                    file_count = len(files)
                    total_files += file_count
                    
                    stats[bucket_name] = {
                        "bucket_id": bucket_id,
                        "file_count": file_count,
                        "sample_files": files[:5]  # Muestra de 5 archivos
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
                "total_files": total_files,
                "buckets": stats,
                "bucket_configuration": BUCKETS,
                "file_configuration": FILE_CONFIG,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas de almacenamiento: {e}")
            raise Exception(f"Error en estadísticas: {str(e)}")
    
    def _validate_file_type(self, file_data: bytes) -> None:
        """Validar tipo de archivo"""
        # Verificar magic bytes para imágenes
        if not file_data:
            raise ValueError("Archivo vacío")
        
        # Magic bytes para diferentes formatos de imagen
        image_signatures = [
            b'\xff\xd8\xff',  # JPEG
            b'\x89\x50\x4e\x47',  # PNG
            b'\x47\x49\x46',  # GIF
            b'\x42\x4d',  # BMP
            b'\x49\x49\x2a\x00',  # TIFF LE
            b'\x4d\x4d\x00\x2a',  # TIFF BE
        ]
        
        is_valid_image = any(file_data.startswith(sig) for sig in image_signatures)
        
        if not is_valid_image:
            raise ValueError("Tipo de archivo no válido: debe ser una imagen")
    
    def _get_content_type(self, filename: str) -> str:
        """Determinar content type basado en extensión"""
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        content_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'tiff': 'image/tiff',
            'tif': 'image/tiff'
        }
        
        return content_types.get(extension, 'application/octet-stream')
    
    def _get_recommendations(self, quality_analysis: Dict, type_analysis: Dict) -> List[str]:
        """Generar recomendaciones basadas en análisis"""
        recommendations = []
        
        quality_score = quality_analysis.get("overall_quality", 0.0)
        document_type = type_analysis.get("type", "unknown")
        
        if quality_score < 0.5:
            recommendations.append("El documento requiere restauración para mejorar la calidad")
        
        if document_type == "degraded":
            recommendations.append("Documento clasificado como degradado, considerar procesamiento")
        
        if quality_analysis.get("blur_detected", False):
            recommendations.append("Se detectó desenfoque, aplicar filtros de nitidez")
        
        if quality_analysis.get("noise_level", 0) > 0.3:
            recommendations.append("Alto nivel de ruido detectado, aplicar filtros de reducción de ruido")
        
        return recommendations

# Instancia global del servicio
file_management_service = FileManagementService()
