"""
Utilidades para el entrenamiento y evaluación
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import logging
from typing import Dict, Any, Tuple

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Cargar archivo de configuración YAML
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Diccionario con la configuración
    """
    if not os.path.exists(config_path):
        # Configuración por defecto
        return {
            'model': {
                'dim': 48,
                'num_blocks': [4, 6, 6, 8],
                'num_refinement_blocks': 4,
                'heads': [1, 2, 4, 8],
                'ffn_expansion_factor': 2.66
            },
            'training': {
                'batch_size': 2,
                'learning_rate': 1e-4,
                'num_epochs': 10,
                'image_size': 128
            },
            'processing': {
                'use_esrgan': False,
                'tile_size': 512
            }
        }
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Error cargando configuración: {e}")
        return {}

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Preprocesar imagen para el modelo
    
    Args:
        image: Imagen de entrada (BGR, uint8)
        target_size: Tamaño objetivo (height, width)
        
    Returns:
        Imagen preprocesada
    """
    if target_size:
        image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Normalizar a [0, 1]
    processed = image.astype(np.float32) / 255.0
    
    return processed

def postprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Postprocesar imagen desde el modelo
    
    Args:
        image: Imagen del modelo (float32, [0, 1])
        
    Returns:
        Imagen final (BGR, uint8)
    """
    # Convertir a uint8
    processed = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    
    return processed

def calculate_psnr(img1, img2, max_value=255):
    """Calcular PSNR entre dos imágenes"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    return psnr(img1, img2, data_range=max_value)

def calculate_ssim(img1, img2):
    """Calcular SSIM entre dos imágenes"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Convertir a escala de grises si es necesario
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1_gray = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    return ssim(img1_gray, img2_gray)

def save_comparison_image(original, degraded, restored, save_path):
    """Guardar imagen de comparación"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convertir tensors a numpy si es necesario
    if isinstance(original, torch.Tensor):
        original = original.squeeze().permute(1, 2, 0).cpu().numpy()
    if isinstance(degraded, torch.Tensor):
        degraded = degraded.squeeze().permute(1, 2, 0).cpu().numpy()
    if isinstance(restored, torch.Tensor):
        restored = restored.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Normalizar a [0, 1] si es necesario
    if original.max() > 1:
        original = original / 255.0
    if degraded.max() > 1:
        degraded = degraded / 255.0
    if restored.max() > 1:
        restored = restored / 255.0
    
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(degraded)
    axes[1].set_title('Degradado')
    axes[1].axis('off')
    
    axes[2].imshow(restored)
    axes[2].set_title('Restaurado')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def count_parameters(model):
    """Contar parámetros del modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_logging(log_file=None):
    """Configurar logging"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

class AverageMeter:
    """Clase para calcular y almacenar promedios"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def tensor_to_image(tensor):
    """Convertir tensor a imagen numpy"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    image = tensor.permute(1, 2, 0).cpu().numpy()
    
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image

def image_to_tensor(image, device='cpu'):
    """Convertir imagen numpy a tensor"""
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    if len(image.shape) == 3:
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    else:
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    
    return tensor.to(device)

def create_degraded_image(clean_image, degradation_type='mixed'):
    """
    Crear imagen degradada sintéticamente para entrenamiento/testing
    
    Args:
        clean_image: Imagen limpia (numpy array)
        degradation_type: Tipo de degradación
    """
    degraded = clean_image.copy()
    
    if degradation_type == 'noise' or degradation_type == 'mixed':
        # Agregar ruido gaussiano
        noise = np.random.normal(0, np.random.randint(5, 25), degraded.shape)
        degraded = np.clip(degraded + noise, 0, 255).astype(np.uint8)
    
    if degradation_type == 'blur' or degradation_type == 'mixed':
        # Agregar blur
        kernel_size = np.random.choice([3, 5, 7])
        degraded = cv2.GaussianBlur(degraded, (kernel_size, kernel_size), 0)
    
    if degradation_type == 'compression' or degradation_type == 'mixed':
        # Simular compresión JPEG
        quality = np.random.randint(20, 70)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', degraded, encode_param)
        degraded = cv2.imdecode(encoded_img, 1)
    
    if degradation_type == 'resolution' or degradation_type == 'mixed':
        # Reducir resolución
        h, w = degraded.shape[:2]
        scale = np.random.uniform(0.5, 0.8)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Reducir y volver a aumentar
        small = cv2.resize(degraded, (new_w, new_h), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    
    return degraded

def prepare_data_directories(base_dir):
    """Preparar estructura de directorios para datos"""
    dirs = [
        os.path.join(base_dir, 'train', 'clean'),
        os.path.join(base_dir, 'train', 'degraded'),
        os.path.join(base_dir, 'val', 'clean'),
        os.path.join(base_dir, 'val', 'degraded'),
        os.path.join(base_dir, 'test', 'clean'),
        os.path.join(base_dir, 'test', 'degraded')
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Estructura de directorios creada en: {base_dir}")

def validate_image_pair(clean_path, degraded_path):
    """Validar que un par de imágenes sea válido"""
    try:
        # Verificar que ambos archivos existan
        if not os.path.exists(clean_path) or not os.path.exists(degraded_path):
            return False, "Archivo no encontrado"
        
        # Cargar imágenes
        clean = cv2.imread(clean_path)
        degraded = cv2.imread(degraded_path)
        
        if clean is None or degraded is None:
            return False, "No se puede cargar la imagen"
        
        # Verificar que tengan el mismo tamaño
        if clean.shape != degraded.shape:
            return False, f"Tamaños diferentes: {clean.shape} vs {degraded.shape}"
        
        return True, "Válido"
    
    except Exception as e:
        return False, str(e)

def analyze_dataset(data_dir):
    """Analizar dataset y mostrar estadísticas"""
    stats = {
        'train': {'clean': 0, 'degraded': 0, 'valid_pairs': 0},
        'val': {'clean': 0, 'degraded': 0, 'valid_pairs': 0},
        'test': {'clean': 0, 'degraded': 0, 'valid_pairs': 0}
    }
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
    
    for split in ['train', 'val', 'test']:
        clean_dir = os.path.join(data_dir, split, 'clean')
        degraded_dir = os.path.join(data_dir, split, 'degraded')
        
        if os.path.exists(clean_dir):
            clean_files = [f for f in os.listdir(clean_dir) 
                          if any(f.lower().endswith(ext) for ext in image_extensions)]
            stats[split]['clean'] = len(clean_files)
        
        if os.path.exists(degraded_dir):
            degraded_files = [f for f in os.listdir(degraded_dir) 
                            if any(f.lower().endswith(ext) for ext in image_extensions)]
            stats[split]['degraded'] = len(degraded_files)
            
            # Contar pares válidos
            if os.path.exists(clean_dir):
                for file in degraded_files:
                    clean_path = os.path.join(clean_dir, file)
                    degraded_path = os.path.join(degraded_dir, file)
                    
                    is_valid, _ = validate_image_pair(clean_path, degraded_path)
                    if is_valid:
                        stats[split]['valid_pairs'] += 1
    
    # Mostrar estadísticas
    print("Estadísticas del dataset:")
    print("=" * 40)
    for split in ['train', 'val', 'test']:
        print(f"{split.upper()}:")
        print(f"  Imágenes limpias: {stats[split]['clean']}")
        print(f"  Imágenes degradadas: {stats[split]['degraded']}")
        print(f"  Pares válidos: {stats[split]['valid_pairs']}")
        print()
    
    return stats
