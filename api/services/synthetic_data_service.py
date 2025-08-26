"""
Servicio para generación de datos sintéticos
"""
import cv2
import numpy as np
import logging
import uuid
import sys
import os

# Agregar path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .minio_service import minio_service
    from config.settings import BUCKETS
except ImportError:
    logging.warning("No se pudieron importar dependencias de synthetic_data_service")
    minio_service = None
    BUCKETS = {
        'degraded': 'document-degraded',
        'clean': 'document-clean', 
        'restored': 'document-restored',
        'training': 'document-training'
    }

logger = logging.getLogger(__name__)

class SyntheticDataService:
    """Servicio para generación de datos sintéticos"""
    
    def add_noise(self, image_data: bytes, noise_type: str = "gaussian", intensity: float = 0.1) -> bytes:
        """Agregar ruido a imagen"""
        try:
            # Convertir bytes a imagen OpenCV
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("No se pudo decodificar la imagen")
            
            # Normalizar imagen
            img_float = img.astype(np.float32) / 255.0
            
            if noise_type == "gaussian":
                # Ruido gaussiano
                noise = np.random.normal(0, intensity, img_float.shape)
                noisy_img = img_float + noise
            elif noise_type == "salt_pepper":
                # Ruido sal y pimienta
                noisy_img = img_float.copy()
                prob = intensity
                
                # Sal (blanco)
                coords = np.random.random(img_float.shape[:2]) < prob/2
                noisy_img[coords] = 1
                
                # Pimienta (negro)
                coords = np.random.random(img_float.shape[:2]) < prob/2
                noisy_img[coords] = 0
            elif noise_type == "blur":
                # Desenfoque
                kernel_size = max(3, int(intensity * 20))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                noisy_img = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), 0)
            else:
                raise ValueError(f"Tipo de ruido no soportado: {noise_type}")
            
            # Recortar valores y convertir de vuelta
            noisy_img = np.clip(noisy_img, 0, 1)
            noisy_img = (noisy_img * 255).astype(np.uint8)
            
            # Codificar de vuelta a bytes
            _, buffer = cv2.imencode('.png', noisy_img)
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Error agregando ruido: {e}")
            raise e
    
    def generate_degraded_version(self, clean_image_data: bytes, degradation_type: str = "mixed") -> bytes:
        """Generar versión degradada de imagen limpia"""
        try:
            if degradation_type == "mixed":
                # Aplicar múltiples degradaciones
                degraded = clean_image_data
                
                # Agregar ruido gaussiano ligero
                degraded = self.add_noise(degraded, "gaussian", 0.05)
                
                # Agregar desenfoque
                degraded = self.add_noise(degraded, "blur", 0.3)
                
                # Agregar ruido sal y pimienta ocasional
                if np.random.random() > 0.5:
                    degraded = self.add_noise(degraded, "salt_pepper", 0.02)
                
                return degraded
            else:
                return self.add_noise(clean_image_data, degradation_type, 0.1)
                
        except Exception as e:
            logger.error(f"Error generando versión degradada: {e}")
            raise e
    
    def generate_training_pairs(self, clean_bucket: str, count: int = 10) -> dict:
        """Generar pares de entrenamiento limpio/degradado"""
        try:
            # Listar archivos limpios en el bucket especificado
            clean_files = minio_service.list_files(clean_bucket)
            
            if not clean_files:
                raise ValueError(f"No hay archivos limpios para procesar en bucket: {clean_bucket}")
            
            generated_pairs = []
            
            # Si solo hay un archivo, generar múltiples versiones degradadas de ese archivo
            if len(clean_files) == 1:
                logger.info(f"Generando {count} versiones degradadas del archivo único: {clean_files[0]}")
                clean_file = clean_files[0]
                
                # Descargar imagen limpia una sola vez
                clean_data = minio_service.download_file(clean_bucket, clean_file)
                
                # Generar múltiples versiones degradadas
                for i in range(count):
                    # Generar versión degradada con variación aleatoria
                    degraded_data = self.generate_degraded_version(clean_data, "mixed")
                    
                    # Generar nombres únicos
                    pair_id = str(uuid.uuid4())
                    clean_filename = f"clean_{pair_id}.png"
                    degraded_filename = f"degraded_{pair_id}.png"
                    
                    # Subir archivos al bucket de entrenamiento
                    minio_service.upload_file(clean_data, BUCKETS['training'], clean_filename)
                    minio_service.upload_file(degraded_data, BUCKETS['training'], degraded_filename)
                    
                    generated_pairs.append({
                        "pair_id": pair_id,
                        "clean_file": clean_filename,
                        "degraded_file": degraded_filename,
                        "source_file": clean_file,
                        "variation": i + 1
                    })
                    
                    logger.info(f"Generado par {i + 1}/{count}: {clean_filename} -> {degraded_filename}")
            
            else:
                # Si hay múltiples archivos, usar la lógica original
                logger.info(f"Generando {count} pares de {len(clean_files)} archivos disponibles")
                
                for i in range(min(count, len(clean_files))):
                    # Seleccionar archivo aleatorio
                    clean_file = np.random.choice(clean_files)
                    
                    # Descargar imagen limpia
                    clean_data = minio_service.download_file(clean_bucket, clean_file)
                    
                    # Generar versión degradada
                    degraded_data = self.generate_degraded_version(clean_data, "mixed")
                    
                    # Generar nombres únicos
                    pair_id = str(uuid.uuid4())
                    clean_filename = f"clean_{pair_id}.png"
                    degraded_filename = f"degraded_{pair_id}.png"
                    
                    # Subir archivos al bucket de entrenamiento
                    minio_service.upload_file(clean_data, BUCKETS['training'], clean_filename)
                    minio_service.upload_file(degraded_data, BUCKETS['training'], degraded_filename)
                    
                    generated_pairs.append({
                        "pair_id": pair_id,
                        "clean_file": clean_filename,
                        "degraded_file": degraded_filename,
                        "source_file": clean_file
                    })
            
            return {
                "status": "success",
                "generated_count": len(generated_pairs),
                "pairs": generated_pairs,
                "source_bucket": clean_bucket,
                "total_files_created": len(generated_pairs) * 2  # clean + degraded
            }
            
        except Exception as e:
            logger.error(f"Error generando pares de entrenamiento: {e}")
            raise e
    
    def augment_dataset(self, bucket: str, target_count: int = 100) -> dict:
        """Aumentar dataset mediante augmentación"""
        try:
            # Listar archivos existentes
            existing_files = minio_service.list_files(bucket)
            current_count = len(existing_files)
            
            if current_count >= target_count:
                return {
                    "status": "success",
                    "message": "Dataset ya tiene suficientes archivos",
                    "current_count": current_count,
                    "target_count": target_count
                }
            
            needed = target_count - current_count
            generated = []
            
            for i in range(needed):
                # Seleccionar archivo base aleatorio
                base_file = np.random.choice(existing_files)
                
                # Descargar archivo
                file_data = minio_service.download_file(bucket, base_file)
                
                # Aplicar augmentación aleatoria
                augmentation_type = np.random.choice(["gaussian", "blur", "salt_pepper"])
                intensity = np.random.uniform(0.05, 0.15)
                
                augmented_data = self.add_noise(file_data, augmentation_type, intensity)
                
                # Subir archivo augmentado
                aug_filename = f"aug_{uuid.uuid4()}.png"
                minio_service.upload_file(augmented_data, bucket, aug_filename)
                
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
            raise e

# Instancia global del servicio
synthetic_data_service = SyntheticDataService()
