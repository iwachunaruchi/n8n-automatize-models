"""
Servicio para generación de datos sintéticos
REFACTORIZADO: Usa constantes centralizadas, manejo robusto de errores
"""
import cv2
import numpy as np
import logging
import uuid
import sys
import os
import asyncio
import io
from typing import Dict, Any, Optional, List
from datetime import datetime

# Asegurar que podemos importar desde el directorio api
sys.path.append('/app/api')

# Importaciones más robustas
minio_service = None

try:
    from services.minio_service import minio_service
    logging.info("MinIO service importado exitosamente en synthetic_data_service")
except ImportError as e:
    logging.error(f"Error importando MinIO service en synthetic_data_service: {e}")
    minio_service = None

# Importar constantes
try:
    from config.constants import (
        BUCKETS, 
        SYNTHETIC_DATA_CONFIG, 
        RESPONSE_MESSAGES,
        MINIO_LOCAL_URL
    )
    logging.info("Constantes de datos sintéticos importadas exitosamente")
except ImportError as e:
    logging.error(f"Error importando constantes de datos sintéticos: {e}")
    # Fallback con valores por defecto
    BUCKETS = {
        'degraded': 'document-degraded',
        'clean': 'document-clean', 
        'restored': 'document-restored',
        'training': 'document-training'
    }
    SYNTHETIC_DATA_CONFIG = {
        "NOISE_TYPES": ["gaussian", "salt_pepper", "blur"],
        "DEGRADATION_TYPES": ["mixed", "blur", "noise"],
        "INTENSITY_RANGE": {"min": 0.01, "max": 1.0},
        "COUNT_LIMITS": {"min": 1, "max": 1000, "augment_min": 10, "augment_max": 10000},
        "BATCH_SIZE": 8
    }
    RESPONSE_MESSAGES = {"synthetic_data_generated": "Datos sintéticos generados exitosamente"}
    MINIO_LOCAL_URL = "http://localhost:9000"

logger = logging.getLogger(__name__)

class SyntheticDataService:
    """Servicio centralizado para generación de datos sintéticos"""
    
    def __init__(self):
        self.noise_types = SYNTHETIC_DATA_CONFIG["NOISE_TYPES"]
        self.degradation_types = SYNTHETIC_DATA_CONFIG["DEGRADATION_TYPES"]
        self.intensity_range = SYNTHETIC_DATA_CONFIG["INTENSITY_RANGE"]
        self.count_limits = SYNTHETIC_DATA_CONFIG["COUNT_LIMITS"]
        self.batch_size = SYNTHETIC_DATA_CONFIG["BATCH_SIZE"]
    
    def add_noise(self, image_data: bytes, noise_type: str = "gaussian", intensity: float = 0.1) -> Dict[str, Any]:
        """
        Agregar ruido a una imagen
        
        Args:
            image_data: Datos de la imagen
            noise_type: Tipo de ruido a aplicar
            intensity: Intensidad del ruido (0.01-1.0)
            
        Returns:
            Dict con el resultado de la operación
        """
        try:
            # Validar parámetros
            if noise_type not in self.noise_types:
                return {
                    "status": "error",
                    "message": f"Tipo de ruido no válido. Tipos disponibles: {', '.join(self.noise_types)}",
                    "error_code": "INVALID_NOISE_TYPE"
                }
            
            if not (self.intensity_range["min"] <= intensity <= self.intensity_range["max"]):
                return {
                    "status": "error",
                    "message": f"Intensidad debe estar entre {self.intensity_range['min']} y {self.intensity_range['max']}",
                    "error_code": "INVALID_INTENSITY"
                }
            
            # Convertir bytes a imagen
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    "status": "error",
                    "message": "No se pudo decodificar la imagen",
                    "error_code": "DECODE_ERROR"
                }
            
            # Aplicar ruido según el tipo
            noisy_image = self._apply_noise(image, noise_type, intensity)
            
            # Convertir de vuelta a bytes
            _, buffer = cv2.imencode('.png', noisy_image)
            noisy_data = buffer.tobytes()
            
            return {
                "status": "success",
                "message": RESPONSE_MESSAGES.get("noise_applied", "Ruido aplicado exitosamente"),
                "result": {
                    "noise_type": noise_type,
                    "intensity": intensity,
                    "original_size": len(image_data),
                    "processed_size": len(noisy_data),
                    "output_format": "png"
                },
                "data": noisy_data
            }
            
        except Exception as e:
            logger.error(f"Error aplicando ruido {noise_type}: {e}")
            return {
                "status": "error",
                "message": f"Error aplicando ruido: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def generate_degraded_version(self, image_data: bytes, degradation_type: str = "mixed") -> Dict[str, Any]:
        """
        Degradar imagen limpia para crear versión degradada
        
        Args:
            image_data: Datos de la imagen limpia
            degradation_type: Tipo de degradación
            
        Returns:
            Dict con el resultado de la operación
        """
        try:
            # Validar parámetros
            if degradation_type not in self.degradation_types:
                return {
                    "status": "error",
                    "message": f"Tipo de degradación no válido. Tipos disponibles: {', '.join(self.degradation_types)}",
                    "error_code": "INVALID_DEGRADATION_TYPE"
                }
            
            # Convertir bytes a imagen
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    "status": "error",
                    "message": "No se pudo decodificar la imagen",
                    "error_code": "DECODE_ERROR"
                }
            
            # Aplicar degradación
            degraded_image = self._apply_degradation(image, degradation_type)
            
            # Convertir de vuelta a bytes
            _, buffer = cv2.imencode('.png', degraded_image)
            degraded_data = buffer.tobytes()
            
            # Generar nombres únicos
            clean_filename = f"clean_{uuid.uuid4()}.png"
            degraded_filename = f"degraded_{uuid.uuid4()}.png"
            
            # Subir archivos si MinIO está disponible
            upload_results = {}
            if minio_service:
                try:
                    clean_upload = minio_service.upload_file(image_data, BUCKETS['clean'], clean_filename)
                    degraded_upload = minio_service.upload_file(degraded_data, BUCKETS['degraded'], degraded_filename)
                    upload_results = {
                        "clean_uploaded": clean_upload is not None,
                        "degraded_uploaded": degraded_upload is not None
                    }
                except Exception as e:
                    logger.warning(f"Error subiendo archivos: {e}")
                    upload_results = {"upload_error": str(e)}
            
            return {
                "status": "success",
                "message": RESPONSE_MESSAGES.get("degradation_completed", "Degradación completada exitosamente"),
                "result": {
                    "degradation_type": degradation_type,
                    "clean_filename": clean_filename,
                    "degraded_filename": degraded_filename,
                    "clean_bucket": BUCKETS['clean'],
                    "degraded_bucket": BUCKETS['degraded'],
                    "original_size": len(image_data),
                    "degraded_size": len(degraded_data),
                    "upload_results": upload_results
                },
                "degraded_data": degraded_data
            }
            
        except Exception as e:
            logger.error(f"Error generando versión degradada: {e}")
            return {
                "status": "error",
                "message": f"Error en degradación: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def get_dataset_stats(self, bucket: str) -> Dict[str, Any]:
        """
        Obtener estadísticas del dataset
        
        Args:
            bucket: Nombre del bucket
            
        Returns:
            Dict con estadísticas del dataset
        """
        try:
            # Validar bucket
            if bucket not in BUCKETS.values():
                return {
                    "status": "error",
                    "message": f"Bucket no válido. Buckets disponibles: {', '.join(BUCKETS.values())}",
                    "error_code": "INVALID_BUCKET"
                }
            
            if not minio_service:
                return {
                    "status": "error",
                    "message": "Servicio MinIO no disponible",
                    "error_code": "MINIO_UNAVAILABLE"
                }
            
            # Obtener lista de archivos
            files = minio_service.list_files(bucket)
            
            # Analizar archivos
            stats = {
                "bucket": bucket,
                "total_files": len(files),
                "file_types": {},
                "size_distribution": {},
                "creation_patterns": {}
            }
            
            # Clasificar archivos por tipo
            for filename in files:
                file_type = self._classify_file_type(filename)
                stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
            
            return {
                "status": "success",
                "statistics": stats
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas de {bucket}: {e}")
            return {
                "status": "error",
                "message": f"Error obteniendo estadísticas: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Obtener información del servicio de datos sintéticos
        
        Returns:
            Dict con información del servicio
        """
        try:
            return {
                "status": "active",
                "service": "synthetic_data_service",
                "configuration": {
                    "noise_types": self.noise_types,
                    "degradation_types": self.degradation_types,
                    "intensity_range": self.intensity_range,
                    "count_limits": self.count_limits,
                    "batch_size": self.batch_size
                },
                "available_operations": [
                    "add_noise",
                    "generate_degraded_version",
                    "generate_training_pairs",
                    "augment_dataset",
                    "get_dataset_stats"
                ],
                "buckets": BUCKETS,
                "minio_available": minio_service is not None
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo información del servicio: {e}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def _apply_noise(self, image: np.ndarray, noise_type: str, intensity: float) -> np.ndarray:
        """Aplicar ruido específico a la imagen"""
        height, width, channels = image.shape
        
        if noise_type == "gaussian":
            noise = np.random.normal(0, intensity * 255, (height, width, channels))
            return np.clip(image + noise, 0, 255).astype(np.uint8)
        
        elif noise_type == "salt_pepper":
            noisy = image.copy()
            salt_prob = intensity / 2
            pepper_prob = intensity / 2
            
            # Salt noise
            salt_coords = np.random.random((height, width)) < salt_prob
            noisy[salt_coords] = 255
            
            # Pepper noise  
            pepper_coords = np.random.random((height, width)) < pepper_prob
            noisy[pepper_coords] = 0
            
            return noisy
        
        elif noise_type == "blur":
            kernel_size = max(3, int(intensity * 15))
            if kernel_size % 2 == 0:
                kernel_size += 1
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        elif noise_type == "speckle":
            noise = np.random.randn(*image.shape) * intensity * 255
            return np.clip(image + image * noise / 255.0, 0, 255).astype(np.uint8)
        
        elif noise_type == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return np.clip(noisy, 0, 255).astype(np.uint8)
        
        else:
            return image
    
    def _apply_degradation(self, image: np.ndarray, degradation_type: str) -> np.ndarray:
        """Aplicar degradación específica a la imagen"""
        if degradation_type == "mixed":
            # Aplicar múltiples degradaciones
            degraded = image.copy()
            degraded = self._apply_noise(degraded, "gaussian", 0.1)
            degraded = cv2.GaussianBlur(degraded, (5, 5), 0)
            return degraded
        
        elif degradation_type == "blur":
            return cv2.GaussianBlur(image, (9, 9), 0)
        
        elif degradation_type == "noise":
            return self._apply_noise(image, "gaussian", 0.2)
        
        elif degradation_type == "compression":
            # Simular compresión JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            _, encimg = cv2.imencode('.jpg', image, encode_param)
            return cv2.imdecode(encimg, 1)
        
        elif degradation_type == "distortion":
            # Aplicar distorsión simple
            rows, cols, _ = image.shape
            pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
            pts2 = np.float32([[0, 10], [cols-11, 0], [10, rows-11], [cols-1, rows-1]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            return cv2.warpPerspective(image, M, (cols, rows))
        
        elif degradation_type == "aging":
            # Simular envejecimiento del papel
            aged = image.copy().astype(np.float32)
            aged *= 0.8  # Reducir brillo
            aged += np.random.normal(0, 10, aged.shape)  # Agregar ruido
            return np.clip(aged, 0, 255).astype(np.uint8)
        
        else:
            return image
    
    def _classify_file_type(self, filename: str) -> str:
        """Clasificar tipo de archivo basado en el nombre"""
        filename_lower = filename.lower()
        
        if filename_lower.startswith('clean_'):
            return "clean"
        elif filename_lower.startswith('degraded_'):
            return "degraded"
        elif filename_lower.startswith('aug_'):
            return "augmented"
        elif filename_lower.startswith('restored_'):
            return "restored"
        elif filename_lower.startswith('synthetic_'):
            return "synthetic"
        else:
            return "other"
    
    # ===== MÉTODOS PARA COMPATIBILIDAD CON CÓDIGO EXISTENTE =====
    
    def apply_random_degradation(self, image: np.ndarray) -> np.ndarray:
        """Aplicar degradación aleatoria a una imagen"""
        import random
        
        # Seleccionar degradación aleatoria
        degradation_types = ["mixed", "blur", "noise", "compression", "distortion", "aging"]
        selected_degradation = random.choice(degradation_types)
        
        return self._apply_degradation(image, selected_degradation)
    
    async def generate_training_pairs_async(self, source_bucket: str, count: int) -> dict:
        """Generar pares de entrenamiento de forma asíncrona - IMPLEMENTACIÓN REAL"""
        try:
            logger.info(f"Generando {count} pares de entrenamiento desde {source_bucket}")
            
            # Validar bucket de origen
            if source_bucket not in BUCKETS.values():
                raise ValueError(f"Bucket de origen inválido: {source_bucket}")
            
            # Obtener archivos del bucket origen
            files = minio_service.list_files(source_bucket)
            if not files:
                raise ValueError(f"No hay archivos en el bucket {source_bucket}")
            
            # Seleccionar archivos aleatoriamente
            import random
            selected_files = random.sample(files, min(len(files), count))
            
            generated_pairs = []
            target_bucket = BUCKETS['training']
            
            for i, filename in enumerate(selected_files):
                try:
                    # Descargar imagen original
                    image_data = minio_service.download_file(source_bucket, filename)
                    
                    # Convertir a imagen
                    import cv2
                    import numpy as np
                    from io import BytesIO
                    
                    # Decodificar imagen
                    nparr = np.frombuffer(image_data, np.uint8)
                    clean_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if clean_image is None:
                        logger.warning(f"No se pudo decodificar {filename}")
                        continue
                    
                    # Generar versión degradada
                    degraded_image = self.apply_random_degradation(clean_image)
                    
                    # Generar UUID único
                    pair_uuid = str(uuid.uuid4())
                    
                    # Guardar imagen limpia
                    clean_filename = f"clean_{pair_uuid}.png"
                    clean_encoded = cv2.imencode('.png', clean_image)[1].tobytes()
                    minio_service.upload_file(clean_encoded, target_bucket, clean_filename)
                    
                    # Guardar imagen degradada  
                    degraded_filename = f"degraded_{pair_uuid}.png"
                    degraded_encoded = cv2.imencode('.png', degraded_image)[1].tobytes()
                    minio_service.upload_file(degraded_encoded, target_bucket, degraded_filename)
                    
                    generated_pairs.append({
                        "clean_file": clean_filename,
                        "degraded_file": degraded_filename,
                        "uuid": pair_uuid,
                        "source_file": filename
                    })
                    
                    # Simular pequeña pausa para no sobrecargar
                    await asyncio.sleep(0.1)
                    
                    logger.info(f"Par {i+1}/{len(selected_files)} generado: {pair_uuid}")
                    
                except Exception as e:
                    logger.error(f"Error procesando {filename}: {e}")
                    continue
            
            logger.info(f"Generación completada: {len(generated_pairs)} pares creados")
            
            return {
                "status": "success",
                "generated_count": len(generated_pairs),
                "total_files_created": len(generated_pairs) * 2,
                "source_bucket": source_bucket,
                "target_bucket": target_bucket,
                "pairs": generated_pairs
            }
            
        except Exception as e:
            logger.error(f"Error generando pares de entrenamiento: {e}")
            return {
                "status": "error",
                "message": str(e),
                "generated_count": 0
            }
    
    def generate_training_pairs(self, clean_bucket: str, count: int = 10) -> dict:
        """Generar pares de entrenamiento limpio/degradado"""
        try:
            if not minio_service:
                raise Exception("Servicio MinIO no disponible")
            
            # Listar archivos limpios en el bucket especificado
            clean_files = minio_service.list_files(clean_bucket)
            
            if not clean_files:
                raise ValueError(f"No hay archivos limpios para procesar en bucket: {clean_bucket}")
            
            generated_pairs = []
            
            for i in range(count):
                # Seleccionar archivo aleatorio
                clean_file = np.random.choice(clean_files)
                
                # Descargar imagen limpia
                clean_data = minio_service.download_file(clean_bucket, clean_file)
                
                # Generar versión degradada usando el nuevo método
                degraded_result = self.generate_degraded_version(clean_data, "mixed")
                
                if degraded_result["status"] != "success":
                    logger.warning(f"Error generando versión degradada: {degraded_result['message']}")
                    continue
                
                # Generar nombres únicos
                pair_id = str(uuid.uuid4())
                clean_filename = f"clean_{pair_id}.png"
                degraded_filename = f"degraded_{pair_id}.png"
                
                # Subir archivos al bucket de entrenamiento
                minio_service.upload_file(clean_data, BUCKETS['training'], clean_filename)
                minio_service.upload_file(degraded_result["degraded_data"], BUCKETS['training'], degraded_filename)
                
                generated_pairs.append({
                    "pair_id": pair_id,
                    "clean_file": clean_filename,
                    "degraded_file": degraded_filename,
                    "source_file": clean_file
                })
                
                logger.info(f"Generado par {i + 1}/{count}: {clean_filename} -> {degraded_filename}")
            
            return {
                "status": "success",
                "generated_count": len(generated_pairs),
                "pairs": generated_pairs,
                "source_bucket": clean_bucket,
                "total_files_created": len(generated_pairs) * 2
            }
            
        except Exception as e:
            logger.error(f"Error generando pares de entrenamiento: {e}")
            return {
                "status": "error",
                "message": str(e),
                "generated_count": 0
            }
    
    def augment_dataset(self, bucket: str, target_count: int = 100) -> dict:
        """Aumentar dataset mediante augmentación"""
        try:
            if not minio_service:
                raise Exception("Servicio MinIO no disponible")
            
            # Listar archivos existentes
            existing_files = minio_service.list_files(bucket)
            current_count = len(existing_files)
            
            if current_count >= target_count:
                return {
                    "status": "success",
                    "message": "Dataset ya tiene suficientes archivos",
                    "current_count": current_count,
                    "target_count": target_count,
                    "generated_count": 0
                }
            
            needed = target_count - current_count
            generated = []
            
            for i in range(needed):
                # Seleccionar archivo base aleatorio
                base_file = np.random.choice(existing_files)
                
                # Descargar archivo
                file_data = minio_service.download_file(bucket, base_file)
                
                # Aplicar augmentación aleatoria usando el nuevo método
                augmentation_type = np.random.choice(self.noise_types[:3])  # gaussian, salt_pepper, blur
                intensity = np.random.uniform(0.05, 0.15)
                
                noise_result = self.add_noise(file_data, augmentation_type, intensity)
                
                if noise_result["status"] != "success":
                    logger.warning(f"Error aplicando augmentación: {noise_result['message']}")
                    continue
                
                # Subir archivo augmentado
                aug_filename = f"aug_{uuid.uuid4()}.png"
                minio_service.upload_file(noise_result["data"], bucket, aug_filename)
                
                generated.append({
                    "filename": aug_filename,
                    "source": base_file,
                    "augmentation": augmentation_type,
                    "intensity": intensity
                })
            
            return {
                "status": "success",
                "generated_count": len(generated),
                "new_total": current_count + len(generated),
                "generated_files": generated
            }
            
        except Exception as e:
            logger.error(f"Error aumentando dataset: {e}")
            return {
                "status": "error",
                "message": str(e),
                "generated_count": 0
            }

# Instancia global del servicio
synthetic_data_service = SyntheticDataService()
