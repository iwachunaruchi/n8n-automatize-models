"""
Pipeline de Preprocesamiento (Capa 1)
Otsu + CLAHE + Deskew por Hough
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import math

class PreprocessingPipeline:
    """Pipeline liviano para preprocesamiento de documentos"""
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def apply_otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Aplicar umbralización Otsu
        Funciona tanto para imágenes en color como escala de grises
        """
        # Si es imagen a color, convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Aplicar Otsu
        _, otsu_result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Si la imagen original era a color, mantener los canales
        if len(image.shape) == 3:
            # Crear máscara y aplicar a imagen original
            mask = otsu_result / 255.0
            result = image.copy().astype(np.float32)
            for c in range(3):
                result[:, :, c] *= mask
            return result.astype(np.uint8)
        else:
            return otsu_result
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Funciona para imágenes en color y escala de grises
        """
        if len(image.shape) == 3:
            # Para imágenes a color, convertir a LAB y aplicar CLAHE solo en el canal L
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Para escala de grises, aplicar CLAHE directamente
            return self.clahe.apply(image)
    
    def detect_skew_angle(self, image: np.ndarray) -> float:
        """
        Detectar ángulo de inclinación usando transformada de Hough
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detectar bordes
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Aplicar transformada de Hough para detectar líneas
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return 0.0
        
        # Calcular ángulos de las líneas detectadas
        angles = []
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            # Normalizar ángulo para que esté entre -45 y 45 grados
            if angle > 45:
                angle = angle - 90
            elif angle < -45:
                angle = angle + 90
            angles.append(angle)
        
        # Usar la mediana de los ángulos para mayor robustez
        if angles:
            return np.median(angles)
        else:
            return 0.0
    
    def deskew_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Corregir inclinación de la imagen
        """
        if abs(angle) < 0.1:  # Si el ángulo es muy pequeño, no hacer nada
            return image
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Crear matriz de rotación
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calcular nuevas dimensiones para evitar recorte
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        # Ajustar la matriz de rotación para centrar la imagen
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Aplicar rotación
        deskewed = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                 flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        
        return deskewed
    
    def process_document(self, image: np.ndarray, 
                        apply_otsu: bool = True,
                        apply_clahe: bool = True,
                        apply_deskew: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Procesar documento completo con el pipeline de Capa 1
        
        Args:
            image: Imagen de entrada
            apply_otsu: Si aplicar umbralización Otsu
            apply_clahe: Si aplicar CLAHE
            apply_deskew: Si aplicar corrección de inclinación
            
        Returns:
            Tuple de (imagen procesada, información del procesamiento)
        """
        result = image.copy()
        processing_info = {
            "original_shape": image.shape,
            "otsu_applied": False,
            "clahe_applied": False,
            "deskew_applied": False,
            "skew_angle": 0.0
        }
        
        # 1. Aplicar Otsu si está habilitado
        if apply_otsu:
            result = self.apply_otsu_threshold(result)
            processing_info["otsu_applied"] = True
            print("✅ Otsu aplicado")
        
        # 2. Aplicar CLAHE si está habilitado
        if apply_clahe:
            result = self.apply_clahe(result)
            processing_info["clahe_applied"] = True
            print("✅ CLAHE aplicado")
        
        # 3. Detectar y corregir inclinación si está habilitado
        if apply_deskew:
            skew_angle = self.detect_skew_angle(result)
            processing_info["skew_angle"] = skew_angle
            
            if abs(skew_angle) > 0.5:  # Solo corregir si hay inclinación significativa
                result = self.deskew_image(result, skew_angle)
                processing_info["deskew_applied"] = True
                print(f"✅ Deskew aplicado: {skew_angle:.2f}°")
            else:
                print(f"ℹ️ Inclinación mínima detectada: {skew_angle:.2f}°")
        
        processing_info["final_shape"] = result.shape
        
        return result, processing_info