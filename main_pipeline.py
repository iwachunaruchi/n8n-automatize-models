#!/usr/bin/env python3
"""
PIPELINE PRINCIPAL DE RESTAURACIÓN DE DOCUMENTOS
Transfer Learning Gradual con Restormer
"""

import os
import sys
import torch
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Añadir paths
sys.path.append('src')
sys.path.append('models')

from src.models.restormer import Restormer
from src.utils import load_config, preprocess_image, postprocess_image

class DocumentRestorationPipeline:
    """
    Pipeline principal para restauración de documentos usando Transfer Learning Gradual
    """
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Inicializar pipeline
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        
        print(f"🔧 Pipeline inicializado - Dispositivo: {self.device}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Cargar modelo entrenado con Transfer Learning Gradual
        
        Args:
            model_path: Ruta específica del modelo. Si None, usa el mejor disponible
            
        Returns:
            bool: True si se cargó exitosamente
        """
        # Prioridad de modelos (mejor a peor)
        model_priorities = [
            "outputs/checkpoints/gradual_transfer_final.pth",
            "outputs/checkpoints/optimized_restormer_final.pth", 
            "outputs/checkpoints/finetuned_restormer_final.pth",
            "models/pretrained/restormer_denoising.pth"
        ]
        
        # Usar modelo específico o buscar el mejor disponible
        if model_path:
            models_to_try = [model_path]
        else:
            models_to_try = model_priorities
        
        for model_path in models_to_try:
            if os.path.exists(model_path):
                try:
                    print(f"📥 Cargando modelo: {os.path.basename(model_path)}")
                    
                    # Crear modelo
                    self.model = Restormer(
                        inp_channels=3,
                        out_channels=3,
                        dim=48,
                        num_blocks=[4, 6, 6, 8],
                        num_refinement_blocks=4,
                        heads=[1, 2, 4, 8],
                        ffn_expansion_factor=2.66,
                        bias=False,
                        LayerNorm_type='WithBias',
                        dual_pixel_task=False
                    )
                    
                    # Cargar pesos
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        if 'epoch' in checkpoint:
                            print(f"   📊 Época: {checkpoint['epoch']}")
                        if 'loss' in checkpoint:
                            print(f"   📉 Loss: {checkpoint['loss']:.6f}")
                    else:
                        self.model.load_state_dict(checkpoint)
                    
                    self.model.to(self.device)
                    self.model.eval()
                    self.model_loaded = True
                    
                    print(f"✅ Modelo cargado exitosamente")
                    return True
                    
                except Exception as e:
                    print(f"❌ Error cargando {model_path}: {str(e)[:100]}...")
                    continue
        
        print("❌ No se pudo cargar ningún modelo")
        return False
    
    def pad_to_divisible(self, image: np.ndarray, factor: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Hacer que la imagen sea divisible por el factor (para el modelo)
        
        Args:
            image: Imagen de entrada
            factor: Factor de divisibilidad
            
        Returns:
            Imagen con padding y tamaño original
        """
        h, w = image.shape[:2]
        
        # Calcular padding necesario
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        
        # Aplicar padding simétrico
        if len(image.shape) == 3:
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        return padded, (h, w)
    
    def restore_document(self, image: np.ndarray) -> np.ndarray:
        """
        Restaurar documento usando el modelo cargado
        
        Args:
            image: Imagen degradada (BGR, uint8)
            
        Returns:
            Imagen restaurada (BGR, uint8)
        """
        if not self.model_loaded:
            raise RuntimeError("Modelo no cargado. Ejecuta load_model() primero.")
        
        try:
            # Preprocesar imagen
            original_shape = image.shape[:2]
            padded_image, original_size = self.pad_to_divisible(image, 8)
            
            # Convertir a tensor
            input_tensor = torch.from_numpy(padded_image).permute(2, 0, 1).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            # Inferencia
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            # Postprocesar
            output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
            
            # Recortar al tamaño original
            restored = output[:original_size[0], :original_size[1]]
            
            return restored
            
        except Exception as e:
            print(f"❌ Error en restauración: {e}")
            return image  # Retornar imagen original en caso de error
    
    def process_image_file(self, input_path: str, output_path: str) -> bool:
        """
        Procesar archivo de imagen completo
        
        Args:
            input_path: Ruta de imagen de entrada
            output_path: Ruta de imagen de salida
            
        Returns:
            bool: True si fue exitoso
        """
        try:
            # Cargar imagen
            image = cv2.imread(input_path)
            if image is None:
                print(f"❌ No se pudo cargar: {input_path}")
                return False
            
            print(f"📸 Procesando: {os.path.basename(input_path)} ({image.shape[1]}x{image.shape[0]})")
            
            # Restaurar
            restored = self.restore_document(image)
            
            # Crear directorio de salida si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Guardar
            cv2.imwrite(output_path, restored)
            print(f"✅ Guardado: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error procesando {input_path}: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información del modelo cargado
        
        Returns:
            Diccionario con información del modelo
        """
        if not self.model_loaded:
            return {"loaded": False}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "loaded": True,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": "Restormer-48-dim",
            "method": "Transfer Learning Gradual"
        }

def main():
    """Función principal de demostración"""
    print("🚀 PIPELINE DE RESTAURACIÓN DE DOCUMENTOS")
    print("🎯 Transfer Learning Gradual con Restormer")
    print("=" * 60)
    
    # Inicializar pipeline
    pipeline = DocumentRestorationPipeline()
    
    # Cargar modelo
    if not pipeline.load_model():
        print("❌ No se pudo inicializar el pipeline")
        return
    
    # Mostrar información del modelo
    info = pipeline.get_model_info()
    print(f"\n📊 INFORMACIÓN DEL MODELO:")
    print(f"   🔧 Dispositivo: {info['device']}")
    print(f"   📐 Arquitectura: {info['architecture']}")
    print(f"   🎓 Método: {info['method']}")
    print(f"   📊 Parámetros: {info['total_parameters']:,}")
    
    # Buscar imágenes de prueba
    test_dirs = ["data/val/degraded", "data/train/degraded"]
    test_images = []
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            test_images = [os.path.join(test_dir, f) for f in files[:3]]
            break
    
    if not test_images:
        print("\n❌ No se encontraron imágenes de prueba")
        return
    
    # Procesar imágenes
    print(f"\n📁 Procesando {len(test_images)} imágenes de prueba...")
    output_dir = "outputs/pipeline_results"
    
    success_count = 0
    for i, img_path in enumerate(test_images):
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"restored_{img_name}")
        
        if pipeline.process_image_file(img_path, output_path):
            success_count += 1
    
    print(f"\n🎉 Procesamiento completado: {success_count}/{len(test_images)} exitosos")
    print(f"📁 Resultados en: {output_dir}")

if __name__ == "__main__":
    main()
