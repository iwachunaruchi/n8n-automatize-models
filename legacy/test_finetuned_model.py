#!/usr/bin/env python3
"""
Script para probar el modelo con fine-tuning
"""

import sys
import os
sys.path.append('src')

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Importar nuestros modelos
from models.restormer import Restormer

def load_finetuned_model(device):
    """Cargar modelo con fine-tuning"""
    
    model_path = "outputs/checkpoints/finetuned_restormer_final.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo con fine-tuning: {model_path}")
        print("💡 Ejecuta primero: python finetune_pretrained.py")
        return None
    
    print(f"🔧 Cargando modelo con fine-tuning...")
    
    # Crear modelo con la misma arquitectura
    model = Restormer(
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
    
    try:
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Cargar pesos del modelo
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"✅ Modelo con fine-tuning cargado exitosamente")
        print(f"📉 Loss final del entrenamiento: {checkpoint.get('final_loss', 'N/A')}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None

def test_finetuned_model():
    """Probar el modelo con fine-tuning en imágenes de prueba"""
    
    print("🧪 PROBANDO MODELO CON FINE-TUNING")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    
    # Cargar modelo
    model = load_finetuned_model(device)
    if model is None:
        return
    
    # Directorio de imágenes de prueba
    test_dir = "data/train/degraded"
    output_dir = "outputs/samples/finetuned_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Imágenes de prueba
    test_images = ["1.png", "10.png", "11.png"]
    
    print(f"📁 Procesando imágenes de prueba...")
    
    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"⚠️ No se encontró: {img_name}")
            continue
        
        print(f"🖼️ Procesando: {img_name}")
        
        # Cargar imagen
        img = cv2.imread(img_path)
        original_shape = img.shape
        print(f"   📐 Tamaño original: {original_shape}")
        
        # Convertir a RGB y normalizar
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Redimensionar para el modelo (múltiplo de 8)
        h, w = img_normalized.shape[:2]
        new_h = ((h + 7) // 8) * 8
        new_w = ((w + 7) // 8) * 8
        
        img_resized = cv2.resize(img_normalized, (new_w, new_h))
        
        # Convertir a tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        
        # Procesar con el modelo
        with torch.no_grad():
            try:
                restored_tensor = model(img_tensor)
                
                # Convertir de vuelta a imagen
                restored = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                restored = np.clip(restored, 0, 1)
                
                # Redimensionar al tamaño original
                restored_resized = cv2.resize(restored, (original_shape[1], original_shape[0]))
                
                # Convertir a uint8
                restored_uint8 = (restored_resized * 255).astype(np.uint8)
                restored_bgr = cv2.cvtColor(restored_uint8, cv2.COLOR_RGB2BGR)
                
                # Guardar resultado
                output_path = os.path.join(output_dir, f"finetuned_{img_name}")
                cv2.imwrite(output_path, restored_bgr)
                
                print(f"   ✅ Resultado guardado: {output_path}")
                
                # Calcular métricas básicas
                gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_restored = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY)
                
                # Nitidez (Laplaciano)
                sharpness_orig = cv2.Laplacian(gray_original, cv2.CV_64F).var()
                sharpness_restored = cv2.Laplacian(gray_restored, cv2.CV_64F).var()
                
                print(f"   📈 Nitidez original: {sharpness_orig:.2f}")
                print(f"   📈 Nitidez restaurada: {sharpness_restored:.2f}")
                print(f"   📊 Mejora: {((sharpness_restored/sharpness_orig)-1)*100:.1f}%")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   ❌ CUDA out of memory para {img_name}")
                    torch.cuda.empty_cache()
                else:
                    print(f"   ❌ Error procesando {img_name}: {e}")
    
    print(f"\n🎯 RESULTADOS GUARDADOS EN: {output_dir}")
    print(f"💡 Ejecuta: python compare_all_models.py")
    print(f"   Para comparar todos los modelos")

if __name__ == "__main__":
    test_finetuned_model()
