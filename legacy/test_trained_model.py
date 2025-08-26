#!/usr/bin/env python3
"""
Script para probar el modelo Restormer ENTRENADO
"""

import sys
import os
sys.path.append('src')

from pipeline import DocumentRestorationPipeline
import torch
import cv2
import numpy as np

def test_trained_model():
    print("🎯 PROBANDO MODELO RESTORMER ENTRENADO")
    print("=" * 50)
    
    # Configurar el pipeline
    pipeline = DocumentRestorationPipeline()
    
    # Deshabilitar ESRGAN (solo Restormer entrenado)
    pipeline.config['processing']['use_esrgan'] = False
    
    # Inicializar modelos
    print("🔄 Inicializando pipeline...")
    pipeline.initialize_models()
    
    # CARGAR EL MODELO ENTRENADO
    checkpoint_path = "outputs/checkpoints/best_restormer.pth"
    if os.path.exists(checkpoint_path):
        print(f"📥 Cargando modelo entrenado: {checkpoint_path}")
        
        # Cargar checkpoint
        checkpoint = torch.load(checkpoint_path)
        pipeline.restormer.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Modelo entrenado cargado!")
        print(f"   📊 Epoch: {checkpoint['epoch']}")
        print(f"   📉 Loss: {checkpoint['loss']:.6f}")
    else:
        print(f"❌ No se encontró el checkpoint: {checkpoint_path}")
        return
    
    # Probar con datos de entrenamiento
    degraded_dir = "data/train/degraded"
    output_dir = "outputs/samples/trained_model"
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener lista de imágenes
    images = [f for f in os.listdir(degraded_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n📸 Probando con {min(3, len(images))} imágenes...")
    print("-" * 50)
    
    for i, image_name in enumerate(images[:3]):
        input_path = os.path.join(degraded_dir, image_name)
        output_path = os.path.join(output_dir, f"trained_{image_name}")
        
        print(f"🔍 Procesando {i+1}/3: {image_name}")
        
        try:
            # Cargar imagen
            image = cv2.imread(input_path)
            if image is None:
                print(f"   ❌ Error cargando imagen")
                continue
            
            h, w = image.shape[:2]
            print(f"   📐 Dimensiones: {w}x{h}")
            
            # Restaurar con modelo entrenado
            print("   🧠 Aplicando modelo entrenado...")
            restored = pipeline.restore_document(image)
            
            # Guardar resultado
            cv2.imwrite(output_path, restored)
            print(f"   ✅ Guardado: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    print("=" * 50)
    print("🎉 ¡PRUEBA COMPLETADA!")
    print(f"📁 Resultados en: {output_dir}")
    print("\n🔍 Compara los resultados:")
    print("   • Antes: outputs/samples/restormer_only_*.png (pesos aleatorios)")
    print("   • Después: outputs/samples/trained_model/trained_*.png (modelo entrenado)")

if __name__ == "__main__":
    test_trained_model()
