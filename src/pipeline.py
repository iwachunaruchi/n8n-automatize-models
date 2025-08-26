"""
Pipeline completo de restauración de documentos con Restormer + ESRGAN
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import os
import yaml
from typing import Optional, Tuple, Union

from models.restormer import Restormer
from models.esrgan import RealESRGAN

class DocumentRestorationPipeline:
    """
    Pipeline completo para restauración de documentos que combina:
    1. Restormer para restauración inicial (denoise, deblur, etc.)
    2. ESRGAN opcional para super-resolución y afinamiento
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializar el pipeline
        
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar configuración
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        # Inicializar modelos
        self.restormer = None
        self.esrgan = None
        self.is_initialized = False
        
    def _get_default_config(self) -> dict:
        """Configuración por defecto"""
        return {
            'restormer': {
                'inp_channels': 3,
                'out_channels': 3,
                'dim': 32,  # Actualizado para coincidir con el modelo entrenado
                'num_blocks': [2, 4, 4, 6],  # Actualizado
                'num_refinement_blocks': 2,  # Actualizado
                'heads': [1, 2, 4, 8],
                'ffn_expansion_factor': 2.0,  # Actualizado
                'bias': False
            },
            'esrgan': {
                'num_in_ch': 3,
                'num_out_ch': 3,
                'scale': 4,
                'num_feat': 64,
                'num_block': 23,
                'num_grow_ch': 32
            },
            'processing': {
                'max_size': 1024,
                'tile_size': 512,
                'tile_overlap': 64,
                'use_esrgan': True,
                'esrgan_threshold': 0.7  # Umbral de calidad para usar ESRGAN
            }
        }
    
    def initialize_models(self, 
                         restormer_path: Optional[str] = None, 
                         esrgan_path: Optional[str] = None):
        """
        Inicializar y cargar los modelos preentrenados
        
        Args:
            restormer_path: Ruta al modelo Restormer preentrenado
            esrgan_path: Ruta al modelo ESRGAN preentrenado
        """
        print("Inicializando modelos...")
        
        # Inicializar Restormer
        self.restormer = Restormer(**self.config['restormer'])
        if restormer_path and os.path.exists(restormer_path):
            checkpoint = torch.load(restormer_path, map_location=self.device)
            self.restormer.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
            print(f"Restormer cargado desde: {restormer_path}")
        else:
            print("Restormer inicializado con pesos aleatorios")
            
        self.restormer.to(self.device)
        self.restormer.eval()
        
        # Inicializar ESRGAN si está habilitado
        if self.config['processing']['use_esrgan']:
            self.esrgan = RealESRGAN(**self.config['esrgan'])
            if esrgan_path and os.path.exists(esrgan_path):
                checkpoint = torch.load(esrgan_path, map_location=self.device)
                self.esrgan.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
                print(f"ESRGAN cargado desde: {esrgan_path}")
            else:
                print("ESRGAN inicializado con pesos aleatorios")
                
            self.esrgan.to(self.device)
            self.esrgan.eval()
        
        self.is_initialized = True
        print("Modelos inicializados correctamente")
    
    def preprocess_image(self, image: Union[np.ndarray, str]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocesar imagen para el modelo
        
        Args:
            image: Imagen como array numpy o ruta del archivo
            
        Returns:
            Tuple de (Tensor preprocessado, dimensiones originales)
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Guardar dimensiones originales
        original_h, original_w = image.shape[:2]
        
        # Normalizar a [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Redimensionar si es muy grande
        h, w = image.shape[:2]
        max_size = self.config['processing']['max_size']
        
        if max(h, w) > max_size:
            if h > w:
                new_h, new_w = max_size, int(w * max_size / h)
            else:
                new_h, new_w = int(h * max_size / w), max_size
            
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Asegurar que las dimensiones sean divisibles por 8 (para evitar errores de pixel_unshuffle)
        h, w = image.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # Convertir a tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image.to(self.device), (original_h, original_w)
    
    def postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Postprocesar tensor del modelo a imagen
        
        Args:
            tensor: Tensor de salida del modelo
            
        Returns:
            Imagen como array numpy
        """
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return image
    
    def tile_process(self, image: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        Procesamiento por tiles para imágenes grandes
        
        Args:
            image: Imagen tensor
            model: Modelo a usar
            
        Returns:
            Imagen procesada
        """
        _, _, h, w = image.shape
        tile_size = self.config['processing']['tile_size']
        tile_overlap = self.config['processing']['tile_overlap']
        
        if h <= tile_size and w <= tile_size:
            # Imagen pequeña, procesamiento directo
            with torch.no_grad():
                return model(image)
        
        # Procesamiento por tiles
        output = torch.zeros_like(image)
        count = torch.zeros((1, 1, h, w), device=image.device)
        
        step = tile_size - tile_overlap
        
        for y in range(0, h - tile_overlap, step):
            for x in range(0, w - tile_overlap, step):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                tile = image[:, :, y:y_end, x:x_end]
                
                with torch.no_grad():
                    tile_output = model(tile)
                
                output[:, :, y:y_end, x:x_end] += tile_output
                count[:, :, y:y_end, x:x_end] += 1
        
        output = output / count
        return output
    
    def assess_quality(self, image: torch.Tensor) -> float:
        """
        Evaluar la calidad de la imagen para decidir si usar ESRGAN
        
        Args:
            image: Tensor de imagen
            
        Returns:
            Score de calidad (0-1)
        """
        # Convertir a escala de grises
        if image.shape[1] == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image
        
        # Calcular varianza del Laplaciano (medida de nitidez)
        laplacian_kernel = torch.tensor([[[[-1, -1, -1], 
                                          [-1,  8, -1], 
                                          [-1, -1, -1]]]], 
                                      dtype=torch.float32, device=image.device)
        
        laplacian = torch.conv2d(gray, laplacian_kernel, padding=1)
        quality_score = torch.var(laplacian).item()
        
        # Normalizar el score
        quality_score = min(quality_score / 1000.0, 1.0)
        
        return quality_score
    
    def restore_document(self, 
                        image: Union[np.ndarray, str], 
                        save_path: Optional[str] = None,
                        return_intermediate: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Restaurar documento completo con el pipeline
        
        Args:
            image: Imagen de entrada
            save_path: Ruta para guardar el resultado
            return_intermediate: Si devolver también el resultado intermedio de Restormer
            
        Returns:
            Imagen restaurada, opcionalmente con resultado intermedio
        """
        if not self.is_initialized:
            raise RuntimeError("Los modelos no han sido inicializados. Llama a initialize_models() primero.")
        
        print("Iniciando restauración de documento...")
        
        # Preprocesar imagen
        input_tensor, original_dims = self.preprocess_image(image)
        
        # Paso 1: Restauración con Restormer
        print("Aplicando Restormer...")
        restored_tensor = self.tile_process(input_tensor, self.restormer)
        restored_image = self.postprocess_image(restored_tensor)
        
        final_image = restored_image.copy()
        
        # Paso 2: Afinamiento con ESRGAN (opcional)
        if self.config['processing']['use_esrgan'] and self.esrgan is not None:
            quality_score = self.assess_quality(restored_tensor)
            print(f"Score de calidad: {quality_score:.3f}")
            
            if quality_score < self.config['processing']['esrgan_threshold']:
                print("Aplicando ESRGAN para afinamiento...")
                
                # Preprocesar para ESRGAN
                esrgan_input = torch.from_numpy(restored_image.astype(np.float32) / 255.0)
                esrgan_input = esrgan_input.permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                # Aplicar ESRGAN
                esrgan_output = self.tile_process(esrgan_input, self.esrgan)
                final_image = self.postprocess_image(esrgan_output)
            else:
                print("Calidad suficiente, omitiendo ESRGAN")
        
        # Guardar resultado si se especifica
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
            print(f"Resultado guardado en: {save_path}")
        
        print("Restauración completada")
        
        if return_intermediate:
            return final_image, restored_image
        else:
            return final_image
    
    def batch_restore(self, 
                     input_dir: str, 
                     output_dir: str, 
                     extensions: list = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']):
        """
        Restauración en lote de múltiples documentos
        
        Args:
            input_dir: Directorio de imágenes de entrada
            output_dir: Directorio de salida
            extensions: Extensiones de archivo soportadas
        """
        if not self.is_initialized:
            raise RuntimeError("Los modelos no han sido inicializados. Llama a initialize_models() primero.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Obtener lista de archivos
        files = []
        for ext in extensions:
            files.extend([f for f in os.listdir(input_dir) 
                         if f.lower().endswith(ext.lower())])
        
        print(f"Procesando {len(files)} archivos...")
        
        for i, filename in enumerate(files):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"restored_{filename}")
            
            print(f"[{i+1}/{len(files)}] Procesando: {filename}")
            
            try:
                self.restore_document(input_path, output_path)
            except Exception as e:
                print(f"Error procesando {filename}: {e}")
                continue
        
        print("Procesamiento en lote completado")
    
    def save_config(self, path: str):
        """Guardar configuración actual"""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def load_config(self, path: str):
        """Cargar configuración desde archivo"""
        with open(path, 'r') as f:
            self.config = yaml.safe_load(f)
