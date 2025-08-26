"""
Servicio para análisis y procesamiento de imágenes
"""
import cv2
import numpy as np
import torch
from PIL import Image
import io
import logging
import sys
import os

# Agregar path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .model_service import model_service
except ImportError:
    logging.warning("No se pudo importar model_service")
    model_service = None

logger = logging.getLogger(__name__)

class ImageAnalysisService:
    """Servicio para análisis y procesamiento de imágenes"""
    
    def __init__(self):
        """Inicializar servicio con configuración para imágenes grandes"""
        self.max_image_size = 2048  # Máximo tamaño para procesar
        self.analysis_size = 1024   # Tamaño para análisis (más pequeño)
    
    def _resize_for_analysis(self, img):
        """Redimensionar imagen para análisis sin perder calidad"""
        h, w = img.shape[:2]
        
        # Si la imagen es muy grande, redimensionar para análisis
        if max(h, w) > self.analysis_size:
            scale = self.analysis_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Usar interpolación de alta calidad
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            logger.info(f"Imagen redimensionada para análisis: {w}x{h} -> {new_w}x{new_h}")
            return img_resized
        
        return img
    
    def _check_memory_usage(self, image_data: bytes) -> bool:
        """Verificar si la imagen es demasiado grande para procesar"""
        import psutil
        
        # Obtener memoria disponible
        available_memory = psutil.virtual_memory().available
        image_size_mb = len(image_data) / (1024 * 1024)
        
        # Estimar memoria necesaria (imagen descomprimida + procesamiento)
        estimated_memory_needed = image_size_mb * 10  # Factor de seguridad
        
        logger.info(f"Imagen: {image_size_mb:.1f}MB, Memoria disponible: {available_memory/(1024*1024):.1f}MB")
        
        if estimated_memory_needed > (available_memory / (1024 * 1024)) * 0.8:  # 80% de memoria disponible
            return False
        
        return True
    
    def analyze_image_quality(self, image_data: bytes) -> dict:
        """Analizar calidad de imagen con manejo optimizado de memoria"""
        try:
            # Verificar memoria disponible
            if not self._check_memory_usage(image_data):
                raise MemoryError("Imagen demasiado grande para procesar con la memoria disponible")
            
            # Convertir bytes a imagen OpenCV
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("No se pudo decodificar la imagen")
            
            # Obtener dimensiones originales
            original_h, original_w = img.shape[:2]
            original_size = original_w * original_h
            
            # Redimensionar para análisis si es necesario
            img_analysis = self._resize_for_analysis(img)
            
            # Análisis básico de calidad
            gray = cv2.cvtColor(img_analysis, cv2.COLOR_BGR2GRAY)
            
            # Métricas existentes
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Detectar bordes para evaluar nitidez
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Detección de ruido optimizada
            kernel = np.ones((5,5), np.float32) / 25
            img_smooth = cv2.filter2D(gray, -1, kernel)
            noise_level = float(np.mean(np.abs(gray.astype(float) - img_smooth.astype(float))))
            
            # Análisis de frecuencias (solo en muestra pequeña para memoria)
            sample_size = min(512, gray.shape[0], gray.shape[1])
            if gray.shape[0] > sample_size or gray.shape[1] > sample_size:
                # Tomar muestra del centro
                center_y, center_x = gray.shape[0] // 2, gray.shape[1] // 2
                half_sample = sample_size // 2
                gray_sample = gray[center_y-half_sample:center_y+half_sample, 
                                 center_x-half_sample:center_x+half_sample]
            else:
                gray_sample = gray
            
            f_transform = np.fft.fft2(gray_sample)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            high_freq_energy = float(np.mean(magnitude_spectrum[magnitude_spectrum > np.percentile(magnitude_spectrum, 90)]))
            
            # Limpiar memoria
            del img, img_analysis, gray, edges, f_transform, f_shift, magnitude_spectrum
            import gc
            gc.collect()
            
            return {
                "blur_score": float(blur_score),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "edge_density": float(edge_density),
                "noise_level": noise_level,
                "high_freq_energy": high_freq_energy,
                "resolution": {"width": original_w, "height": original_h},
                "total_pixels": original_size,
                "channels": 3,
                "memory_optimized": original_size > (self.analysis_size ** 2)
            }
            
        except MemoryError as e:
            logger.error(f"Error de memoria analizando imagen: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error analizando imagen: {e}")
            raise e
    
    def preprocess_for_model(self, image_data: bytes) -> torch.Tensor:
        """Preprocesar imagen para el modelo"""
        try:
            # Convertir a PIL Image
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Redimensionar si es necesario (manteniendo proporción)
            max_size = 512
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convertir a tensor
            img_array = np.array(img) / 255.0
            img_tensor = torch.from_numpy(img_array).float()
            
            # Reorganizar dimensiones (H, W, C) -> (1, C, H, W)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            
            return img_tensor
            
        except Exception as e:
            logger.error(f"Error preprocesando imagen: {e}")
            raise e
    
    def postprocess_from_model(self, output_tensor: torch.Tensor) -> bytes:
        """Postprocesar salida del modelo a imagen"""
        try:
            # Convertir tensor a numpy (1, C, H, W) -> (H, W, C)
            output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Asegurar que los valores estén en [0, 1]
            output_np = np.clip(output_np, 0, 1)
            
            # Convertir a uint8
            output_np = (output_np * 255).astype(np.uint8)
            
            # Convertir a PIL Image
            img = Image.fromarray(output_np)
            
            # Convertir a bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            return img_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error postprocesando imagen: {e}")
            raise e
    
    def restore_image(self, image_data: bytes) -> bytes:
        """Restaurar imagen usando el modelo"""
        try:
            if model_service is None:
                raise Exception("Model service no disponible")
                
            # Preprocesar imagen
            input_tensor = self.preprocess_for_model(image_data)
            
            # Realizar predicción
            output_tensor = model_service.predict(input_tensor)
            
            # Postprocesar resultado
            restored_image = self.postprocess_from_model(output_tensor)
            
            return restored_image
            
        except Exception as e:
            logger.error(f"Error restaurando imagen: {e}")
            raise e
    
    def classify_document_type(self, image_data: bytes) -> dict:
        """Clasificar tipo de documento con análisis mejorado y manejo de memoria optimizado"""
        try:
            # Verificar memoria disponible
            if not self._check_memory_usage(image_data):
                # Intentar procesamiento básico con muy poca memoria
                return {
                    "type": "unknown",
                    "confidence": 0.0,
                    "details": {
                        "error": "Imagen demasiado grande para análisis completo",
                        "suggestion": "Reducir resolución de imagen"
                    }
                }
            
            # Convertir bytes a imagen OpenCV
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("No se pudo decodificar la imagen")
            
            # Obtener dimensiones originales para referencia
            original_h, original_w = img.shape[:2]
            
            # Redimensionar para análisis si es necesario
            img_analysis = self._resize_for_analysis(img)
            
            # Análizar calidad con imagen optimizada
            quality_metrics = self.analyze_image_quality(image_data)
            
            # Convertir a escala de grises para análisis
            gray = cv2.cvtColor(img_analysis, cv2.COLOR_BGR2GRAY)
            
            # ANÁLISIS MEJORADO DE CLASIFICACIÓN
            
            # 1. Detección de ruido (principal indicador de degradación)
            noise_level = quality_metrics.get("noise_level", 0)
            
            # 2. Análisis de nitidez
            blur_score = quality_metrics.get("blur_score", 0)
            
            # 3. Análisis de contraste
            contrast = quality_metrics.get("contrast", 0)
            
            # 4. Análisis de bordes (imágenes limpias tienen bordes más definidos)
            edge_density = quality_metrics.get("edge_density", 0)
            
            # 5. Análisis de frecuencias altas
            high_freq_energy = quality_metrics.get("high_freq_energy", 0)
            
            # LÓGICA DE CLASIFICACIÓN MEJORADA
            
            # Pesos para cada métrica
            weights = {
                "noise": 0.35,      # Mayor peso al ruido
                "blur": 0.25,       # Nitidez importante
                "contrast": 0.15,   # Contraste moderado
                "edges": 0.15,      # Definición de bordes
                "freq": 0.10        # Frecuencias altas
            }
            
            # Normalizar métricas (valores típicos basados en experiencia)
            noise_norm = min(noise_level / 20.0, 1.0)  # Ruido alto > 20
            blur_norm = min(blur_score / 1000.0, 1.0)  # Nitidez alta > 1000
            contrast_norm = min(contrast / 100.0, 1.0)  # Contraste alto > 100
            edge_norm = min(edge_density * 10.0, 1.0)   # Densidad de bordes
            freq_norm = min(high_freq_energy / 1000000.0, 1.0)  # Energía de freq altas
            
            # Calcular puntuación de calidad (0 = muy degradado, 1 = muy limpio)
            quality_score = (
                weights["blur"] * blur_norm +
                weights["contrast"] * contrast_norm +
                weights["edges"] * edge_norm +
                weights["freq"] * freq_norm -
                weights["noise"] * noise_norm  # El ruido reduce la calidad
            )
            
            # Limpiar memoria temprana
            del img, img_analysis, gray, nparr
            import gc
            gc.collect()
            
            # Clasificación basada en puntuación
            if quality_score > 0.7:
                doc_type = "clean"
                confidence = min(quality_score, 0.95)
                details = {
                    "reason": "Imagen de alta calidad detectada",
                    "quality_indicators": {
                        "low_noise": noise_level < 15,
                        "high_sharpness": blur_score > 500,
                        "good_contrast": contrast > 50,
                        "clear_edges": edge_density > 0.1
                    }
                }
            elif quality_score < 0.3:
                doc_type = "degraded"
                confidence = max(0.6, 1.0 - quality_score)
                details = {
                    "reason": "Imagen degradada detectada",
                    "degradation_indicators": {
                        "high_noise": noise_level > 20,
                        "low_sharpness": blur_score < 200,
                        "poor_contrast": contrast < 30,
                        "blurred_edges": edge_density < 0.05
                    }
                }
            else:
                # Zona intermedia - usar análisis adicional
                if noise_level > 15 and blur_score < 300:
                    doc_type = "degraded"
                    confidence = 0.65
                    details = {
                        "reason": "Degradación moderada detectada",
                        "mixed_indicators": True
                    }
                else:
                    doc_type = "clean"
                    confidence = 0.70
                    details = {
                        "reason": "Calidad aceptable detectada",
                        "mixed_indicators": True
                    }
            
            return {
                "type": doc_type,
                "confidence": float(confidence),
                "details": {
                    **details,
                    "quality_score": float(quality_score),
                    "metrics": {
                        "noise_level": float(noise_level),
                        "blur_score": float(blur_score),
                        "contrast": float(contrast),
                        "edge_density": float(edge_density),
                        "high_freq_energy": float(high_freq_energy)
                    },
                    "resolution": f"{original_w}x{original_h}",
                    "memory_optimized": original_w * original_h > (self.analysis_size ** 2)
                }
            }
            
        except MemoryError as e:
            logger.error(f"Error de memoria en clasificación: {e}")
            return {
                "type": "unknown",
                "confidence": 0.0,
                "details": {
                    "error": "Memoria insuficiente para análisis",
                    "suggestion": "Reducir resolución de imagen"
                }
            }
        except Exception as e:
            logger.error(f"Error clasificando documento: {e}")
            return {
                "type": "unknown",
                "confidence": 0.0,
                "details": {"error": str(e)}
            }

# Instancia global del servicio
image_analysis_service = ImageAnalysisService()
