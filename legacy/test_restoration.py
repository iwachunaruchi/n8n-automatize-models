#!/usr/bin/env python3
"""
Script simple para probar la restauración de documentos
"""

import sys
import os
sys.path.append('src')

from pipeline import DocumentRestorationPipeline
import cv2
import numpy as np

def test_restoration():
    # Configurar el pipeline
    pipeline = DocumentRestorationPipeline()
    
    # Inicializar los modelos
    print("Inicializando modelos...")
    pipeline.initialize_models()
    print("Modelos inicializados!")
    
    # Buscar imágenes degradadas para probar
    degraded_dir = "data/train/degraded"
    if os.path.exists(degraded_dir):
        images = [f for f in os.listdir(degraded_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            input_path = os.path.join(degraded_dir, images[0])
            output_path = f"outputs/samples/restored_{images[0]}"
            
            print(f"Probando restauración con: {input_path}")
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
                return False
    else:
        print(f"No se encontró el directorio: {degraded_dir}")
        return False

if __name__ == "__main__":
    test_restoration()
