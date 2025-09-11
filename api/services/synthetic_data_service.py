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
        NAFNET_CONFIG,
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
    NAFNET_CONFIG = {
        "CORE_NAME": "NAFNet",
        "CURRENT_TASK": "SIDD-width64",
        "AVAILABLE_TASKS": {
            "SIDD-width64": {
                "degradation_types": ["gaussian_noise", "real_noise", "mixed_noise"],
                "recommended_intensity": {"min": 0.01, "max": 0.3}
            }
        }
    }
    RESPONSE_MESSAGES = {"synthetic_data_generated": "Datos sintéticos generados exitosamente"}
    MINIO_LOCAL_URL = "http://localhost:9000"

logger = logging.getLogger(__name__)

class SyntheticDataService:
    """Servicio centralizado para generación de datos sintéticos con estructura NAFNet"""
    
    def __init__(self):
        self.noise_types = SYNTHETIC_DATA_CONFIG["NOISE_TYPES"]
        self.degradation_types = SYNTHETIC_DATA_CONFIG["DEGRADATION_TYPES"]
        self.intensity_range = SYNTHETIC_DATA_CONFIG["INTENSITY_RANGE"]
        self.count_limits = SYNTHETIC_DATA_CONFIG["COUNT_LIMITS"]
        self.batch_size = SYNTHETIC_DATA_CONFIG["BATCH_SIZE"]
        
        # Configuración NAFNet
        self.nafnet_config = NAFNET_CONFIG
        self.core_name = NAFNET_CONFIG["CORE_NAME"]
        self.current_task = NAFNET_CONFIG["CURRENT_TASK"]
        self.available_tasks = NAFNET_CONFIG["AVAILABLE_TASKS"]
    
    def _build_nafnet_path(self, task: str, split: str, quality: str) -> str:
        """
        Construir path siguiendo estructura NAFNet recomendada
        
        Args:
            task: Tarea NAFNet (ej: "SIDD-width64")
            split: "train" o "val" 
            quality: "lq" (low-quality) o "gt" (ground-truth)
            
        Returns:
            String con el path completo
        """
        return f"datasets/{self.core_name}/{task}/{split}/{quality}/"
    
    def _get_nafnet_degradation_for_task(self, task: str) -> List[str]:
        """
        Obtener tipos de degradación específicos para una tarea NAFNet
        
        Args:
            task: Nombre de la tarea NAFNet
            
        Returns:
            Lista de tipos de degradación recomendados
        """
        if task in self.available_tasks:
            return self.available_tasks[task]["degradation_types"]
        else:
            # Fallback a degradaciones generales
            return ["gaussian_noise", "real_noise", "mixed_noise"]
    
    def _apply_nafnet_degradation(self, image: np.ndarray, task: str) -> np.ndarray:
        """
        Aplicar degradación específica para tarea NAFNet
        
        Args:
            image: Imagen numpy array
            task: Tarea NAFNet
            
        Returns:
            Imagen degradada según especificaciones de la tarea
        """
        degradation_types = self._get_nafnet_degradation_for_task(task)
        task_config = self.available_tasks.get(task, {})
        
        # Obtener intensidad recomendada para la tarea
        intensity_config = task_config.get("recommended_intensity", {"min": 0.01, "max": 0.3})
        intensity = np.random.uniform(intensity_config["min"], intensity_config["max"])
        
        # Seleccionar degradación aleatoria de las recomendadas para la tarea
        selected_degradation = np.random.choice(degradation_types)
        
        if selected_degradation == "gaussian_noise":
            return self._apply_noise(image, "gaussian", intensity)
        elif selected_degradation == "real_noise":
            return self._apply_noise(image, "speckle", intensity)
        elif selected_degradation == "mixed_noise":
            # Aplicar múltiples tipos de ruido
            noisy = self._apply_noise(image, "gaussian", intensity * 0.6)
            noisy = self._apply_noise(noisy, "salt_pepper", intensity * 0.4)
            return noisy
        elif selected_degradation == "motion_blur":
            return self._apply_motion_blur(image, intensity)
        elif selected_degradation == "gaussian_blur":
            return self._apply_noise(image, "blur", intensity)
        elif selected_degradation == "downsampling":
            return self._apply_downsampling(image, intensity)
        elif selected_degradation == "compression":
            return self._apply_degradation(image, "compression")
        else:
            # Fallback a ruido gaussiano
            return self._apply_noise(image, "gaussian", intensity)
    
    def _apply_motion_blur(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Aplicar motion blur específico para GoPro dataset"""
        # Crear kernel de motion blur
        size = max(3, int(intensity * 20))
        if size % 2 == 0:
            size += 1
            
        # Kernel lineal para simular motion blur
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        
        return cv2.filter2D(image, -1, kernel)
    
    def _apply_downsampling(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Aplicar downsampling para super resolution tasks"""
        h, w = image.shape[:2]
        
        # Factor de reducción basado en intensidad
        scale_factor = max(0.25, 1.0 - intensity)
        
        # Reducir resolución
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        downsampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Volver al tamaño original (simula upsampling de baja calidad)
        upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return upsampled
    
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
                "service": "synthetic_data_service_nafnet",
                "configuration": {
                    "noise_types": self.noise_types,
                    "degradation_types": self.degradation_types,
                    "intensity_range": self.intensity_range,
                    "count_limits": self.count_limits,
                    "batch_size": self.batch_size
                },
                "nafnet_configuration": {
                    "core_name": self.core_name,
                    "current_task": self.current_task,
                    "available_tasks": list(self.available_tasks.keys()),
                    "dataset_structure": "datasets/{core}/{task}/{split}/{quality}/",
                    "validation_split": NAFNET_CONFIG["VALIDATION_SPLIT"]
                },
                "available_operations": [
                    "add_noise",
                    "generate_degraded_version", 
                    "generate_training_pairs",
                    "generate_nafnet_training_dataset",
                    "get_nafnet_dataset_info",
                    "list_available_nafnet_tasks",
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
    
    def generate_nafnet_training_dataset(self, 
                                        source_bucket: str, 
                                        count: int,
                                        task: str = None,
                                        train_val_split: bool = True) -> Dict[str, Any]:
        """
        Generar dataset estructurado para entrenamiento NAFNet
        
        Args:
            source_bucket: Bucket con imágenes fuente
            count: Número total de pares a generar
            task: Tarea NAFNet específica (por defecto usa CURRENT_TASK)
            train_val_split: Si dividir en train/val automáticamente
            
        Returns:
            Dict con resultados de la generación
        """
        try:
            if not minio_service:
                raise Exception("Servicio MinIO no disponible")
            
            # Usar tarea actual si no se especifica
            if task is None:
                task = self.current_task
            
            # Validar que la tarea existe
            if task not in self.available_tasks:
                raise ValueError(f"Tarea '{task}' no disponible. Tareas disponibles: {list(self.available_tasks.keys())}")
            
            # Obtener archivos fuente
            source_files = minio_service.list_files(source_bucket)
            if not source_files:
                raise ValueError(f"No hay archivos en bucket fuente: {source_bucket}")
            
            # Calcular splits
            if train_val_split:
                train_count = int(count * NAFNET_CONFIG["TRAIN_VAL_RATIO"])
                val_count = count - train_count
            else:
                train_count = count
                val_count = 0
            
            logger.info(f"Generando dataset NAFNet para tarea '{task}': {train_count} train, {val_count} val")
            
            # Generar datos de entrenamiento
            train_result = self._generate_nafnet_split(
                source_files, source_bucket, task, "train", train_count
            )
            
            # Generar datos de validación si es necesario
            val_result = {"generated_count": 0, "pairs": []}
            if val_count > 0:
                val_result = self._generate_nafnet_split(
                    source_files, source_bucket, task, "val", val_count
                )
            
            total_generated = train_result["generated_count"] + val_result["generated_count"]
            
            return {
                "status": "success",
                "task": task,
                "dataset_structure": self._build_nafnet_path(task, "*", "*"),
                "total_generated": total_generated,
                "train": {
                    "count": train_result["generated_count"],
                    "pairs": train_result["pairs"]
                },
                "val": {
                    "count": val_result["generated_count"], 
                    "pairs": val_result["pairs"]
                },
                "paths": {
                    "train_lq": self._build_nafnet_path(task, "train", "lq"),
                    "train_gt": self._build_nafnet_path(task, "train", "gt"),
                    "val_lq": self._build_nafnet_path(task, "val", "lq"),
                    "val_gt": self._build_nafnet_path(task, "val", "gt")
                }
            }
            
        except Exception as e:
            logger.error(f"Error generando dataset NAFNet: {e}")
            return {
                "status": "error",
                "message": str(e),
                "generated_count": 0
            }
    
    def _generate_nafnet_split(self, 
                              source_files: List[str], 
                              source_bucket: str, 
                              task: str, 
                              split: str, 
                              count: int) -> Dict[str, Any]:
        """
        Generar split específico (train/val) para dataset NAFNet
        
        Args:
            source_files: Lista de archivos fuente
            source_bucket: Bucket fuente
            task: Tarea NAFNet
            split: "train" o "val"
            count: Número de pares a generar
            
        Returns:
            Dict con resultado de la generación del split
        """
        try:
            import random
            
            # Seleccionar archivos aleatoriamente
            selected_files = random.sample(source_files, min(len(source_files), count))
            
            generated_pairs = []
            training_bucket = BUCKETS['training']
            
            # Construir paths de destino
            lq_path = self._build_nafnet_path(task, split, "lq")
            gt_path = self._build_nafnet_path(task, split, "gt")
            
            for i, filename in enumerate(selected_files):
                try:
                    # Descargar imagen original (ground truth)
                    gt_data = minio_service.download_file(source_bucket, filename)
                    
                    # Decodificar imagen
                    nparr = np.frombuffer(gt_data, np.uint8)
                    gt_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if gt_image is None:
                        logger.warning(f"No se pudo decodificar {filename}")
                        continue
                    
                    # Aplicar degradación específica para la tarea
                    lq_image = self._apply_nafnet_degradation(gt_image, task)
                    
                    # Generar nombres de archivo únicos
                    pair_uuid = str(uuid.uuid4())
                    base_name = f"{split}_{pair_uuid}"
                    
                    # Nombres siguiendo convención NAFNet
                    gt_filename = f"{gt_path}{base_name}_gt.png"
                    lq_filename = f"{lq_path}{base_name}_lq.png"
                    
                    # Codificar imágenes
                    gt_encoded = cv2.imencode('.png', gt_image)[1].tobytes()
                    lq_encoded = cv2.imencode('.png', lq_image)[1].tobytes()
                    
                    # Subir a MinIO con estructura organizada
                    minio_service.upload_file(gt_encoded, training_bucket, gt_filename)
                    minio_service.upload_file(lq_encoded, training_bucket, lq_filename)
                    
                    generated_pairs.append({
                        "pair_id": pair_uuid,
                        "gt_file": gt_filename,
                        "lq_file": lq_filename,
                        "source_file": filename,
                        "task": task,
                        "split": split
                    })
                    
                    logger.info(f"Generado par {split} {i+1}/{len(selected_files)}: {base_name}")
                    
                except Exception as e:
                    logger.error(f"Error procesando {filename} para {split}: {e}")
                    continue
            
            logger.info(f"Split {split} completado: {len(generated_pairs)} pares generados")
            
            return {
                "generated_count": len(generated_pairs),
                "pairs": generated_pairs
            }
            
        except Exception as e:
            logger.error(f"Error generando split {split}: {e}")
            return {
                "generated_count": 0,
                "pairs": []
            }
    
    def get_nafnet_dataset_info(self, task: str = None) -> Dict[str, Any]:
        """
        Obtener información sobre el dataset NAFNet existente
        
        Args:
            task: Tarea específica (opcional)
            
        Returns:
            Dict con información del dataset
        """
        try:
            if task is None:
                task = self.current_task
            
            if not minio_service:
                return {
                    "status": "error",
                    "message": "Servicio MinIO no disponible"
                }
            
            training_bucket = BUCKETS['training']
            
            # Contar archivos en cada split y calidad
            stats = {
                "task": task,
                "train": {"lq": 0, "gt": 0},
                "val": {"lq": 0, "gt": 0},
                "total_pairs": 0,
                "structure": {}
            }
            
            # Obtener todos los archivos del bucket de entrenamiento
            all_files = minio_service.list_files(training_bucket)
            
            # Filtrar archivos que pertenecen a esta tarea
            task_prefix = f"datasets/{self.core_name}/{task}/"
            task_files = [f for f in all_files if f.startswith(task_prefix)]
            
            # Analizar estructura
            for file_path in task_files:
                # Extraer componentes del path
                path_parts = file_path.replace(task_prefix, "").split("/")
                
                if len(path_parts) >= 2:
                    split = path_parts[0]  # train o val
                    quality = path_parts[1]  # lq o gt
                    
                    if split in ["train", "val"] and quality in ["lq", "gt"]:
                        stats[split][quality] += 1
            
            # Calcular pares (el mínimo entre lq y gt por split)
            train_pairs = min(stats["train"]["lq"], stats["train"]["gt"])
            val_pairs = min(stats["val"]["lq"], stats["val"]["gt"])
            stats["total_pairs"] = train_pairs + val_pairs
            
            # Información de estructura
            stats["structure"] = {
                "train_lq_path": self._build_nafnet_path(task, "train", "lq"),
                "train_gt_path": self._build_nafnet_path(task, "train", "gt"),
                "val_lq_path": self._build_nafnet_path(task, "val", "lq"),
                "val_gt_path": self._build_nafnet_path(task, "val", "gt"),
                "complete_pairs": {
                    "train": train_pairs,
                    "val": val_pairs
                }
            }
            
            return {
                "status": "success",
                "dataset_info": stats
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo info dataset NAFNet: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def list_available_nafnet_tasks(self) -> Dict[str, Any]:
        """
        Listar todas las tareas NAFNet disponibles y sus configuraciones
        
        Returns:
            Dict con información de tareas disponibles
        """
        return {
            "core_name": self.core_name,
            "current_task": self.current_task,
            "available_tasks": self.available_tasks,
            "dataset_structure_template": "datasets/{core}/{task}/{split}/{quality}/",
            "example_paths": {
                "train_lq": self._build_nafnet_path(self.current_task, "train", "lq"),
                "train_gt": self._build_nafnet_path(self.current_task, "train", "gt"),
                "val_lq": self._build_nafnet_path(self.current_task, "val", "lq"),
                "val_gt": self._build_nafnet_path(self.current_task, "val", "gt")
            }
        }
    
    # ===== MÉTODOS PARA COMPATIBILIDAD CON CÓDIGO EXISTENTE =====
    
    def apply_random_degradation(self, image: np.ndarray, task: str = None) -> np.ndarray:
        """Aplicar degradación aleatoria específica para tarea NAFNet"""
        if task is None:
            task = self.current_task
        return self._apply_nafnet_degradation(image, task)
    
    async def generate_training_pairs_async(self, source_bucket: str, count: int, task: str = None) -> dict:
        """Generar pares de entrenamiento de forma asíncrona con estructura NAFNet"""
        try:
            if task is None:
                task = self.current_task
                
            logger.info(f"Generando {count} pares de entrenamiento NAFNet para tarea '{task}' desde {source_bucket}")
            
            # Usar el nuevo método de generación NAFNet
            result = self.generate_nafnet_training_dataset(
                source_bucket=source_bucket,
                count=count,
                task=task,
                train_val_split=True
            )
            
            if result["status"] == "success":
                logger.info(f"Generación NAFNet completada: {result['total_generated']} pares creados")
                
                return {
                    "status": "success",
                    "generated_count": result["total_generated"],
                    "total_files_created": result["total_generated"] * 2,
                    "source_bucket": source_bucket,
                    "task": task,
                    "structure": result["paths"],
                    "train_pairs": result["train"]["count"],
                    "val_pairs": result["val"]["count"]
                }
            else:
                return result
            
        except Exception as e:
            logger.error(f"Error generando pares de entrenamiento NAFNet: {e}")
            return {
                "status": "error",
                "message": str(e),
                "generated_count": 0
            }
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
    
    def generate_training_pairs(self, clean_bucket: str, count: int = 10, task: str = None, use_nafnet_structure: bool = True) -> dict:
        """
        Generar pares de entrenamiento limpio/degradado
        
        Args:
            clean_bucket: Bucket con imágenes limpias
            count: Número de pares a generar
            task: Tarea NAFNet específica (opcional)
            use_nafnet_structure: Si usar estructura NAFNet organizada
            
        Returns:
            Dict con resultado de la operación
        """
        try:
            if not minio_service:
                raise Exception("Servicio MinIO no disponible")
            
            # Si se usa estructura NAFNet, delegar al método especializado
            if use_nafnet_structure:
                return self.generate_nafnet_training_dataset(
                    source_bucket=clean_bucket,
                    count=count,
                    task=task,
                    train_val_split=True
                )
            
            # Método legacy para compatibilidad
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
                
                # Subir archivos al bucket de entrenamiento (método legacy)
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
                "total_files_created": len(generated_pairs) * 2,
                "structure_type": "legacy"
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
