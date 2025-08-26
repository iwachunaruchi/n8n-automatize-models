#!/usr/bin/env python3
"""
Script simplificado para probar solo Restormer
"""

import sys
import os
sys.path.append('src')

from pipeline import DocumentRestorationPipeline
import cv2
import numpy as np

def test_restormer_only():
    # Configurar el pipeline
    pipeline = DocumentRestorationPipeline()
    
    # Deshabilitar ESRGAN manualmente
    pipeline.config['processing']['use_esrgan'] = False
    
    # Inicializar solo Restormer
    print("Inicializando solo Restormer...")
    pipeline.initialize_models()
    print("Restormer inicializado!")
    
    # Buscar imágenes degradadas para probar
    degraded_dir = "data/train/degraded"
    if os.path.exists(degraded_dir):
        images = [f for f in os.listdir(degraded_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            input_path = os.path.join(degraded_dir, images[0])
            output_path = f"outputs/samples/restormer_only_{images[0]}"
            
            print(f"Probando Restormer con: {input_path}")
            print(f"Salida en: {output_path}")
            
            # Crear directorio de salida si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            try:
                # Cargar imagen
                image = cv2.imread(input_path)
                if image is None:
                    print(f"Error: No se pudo cargar la imagen {input_path}")
                    return
                
                print(f"Imagen cargada: {image.shape}")
                
                # Restaurar imagen
                restored = pipeline.restore_document(image)
                
                # Guardar resultado
                cv2.imwrite(output_path, restored)
                print(f"¡Restauración completada! Resultado guardado en: {output_path}")
                
                return True
                
            except Exception as e:
                print(f"Error durante la restauración: {e}")
                import traceback
                traceback.print_exc()
                return False
    else:
        print(f"No se encontró el directorio: {degraded_dir}")
        return False

if __name__ == "__main__":
    test_restormer_only()
