#!/usr/bin/env python3
"""
Script completo de demostración del pipeline de restauración de documentos
"""

import sys
import os
import argparse
sys.path.append('src')

from pipeline import DocumentRestorationPipeline
import cv2
import numpy as np

def demo_restoration():
    """Función de demostración completa"""
    
    print("🔧 Inicializando Pipeline de Restauración de Documentos")
    print("=" * 60)
    
    # Configurar el pipeline
    pipeline = DocumentRestorationPipeline()
    
    # Deshabilitar ESRGAN para la demostración (solo usar Restormer)
    pipeline.config['processing']['use_esrgan'] = False
    
    # Inicializar modelos
    print("Inicializando modelos...")
    pipeline.initialize_models()
    print("✅ Modelos inicializados correctamente!\n")
    
    # Procesar todas las imágenes de prueba
    degraded_dir = "data/train/degraded"
    output_dir = "outputs/samples"
    
    if not os.path.exists(degraded_dir):
        print(f"❌ No se encontró el directorio: {degraded_dir}")
        return
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener lista de imágenes
    images = [f for f in os.listdir(degraded_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"❌ No se encontraron imágenes en: {degraded_dir}")
        return
    
    print(f"📸 Encontradas {len(images)} imágenes para procesar")
    print("-" * 60)
    
    success_count = 0
    
    for i, image_name in enumerate(images[:3]):  # Procesar solo las primeras 3 para la demo
        input_path = os.path.join(degraded_dir, image_name)
        output_path = os.path.join(output_dir, f"restored_{image_name}")
        
        print(f"📝 Procesando {i+1}/{min(3, len(images))}: {image_name}")
        
        try:
            # Cargar imagen
            image = cv2.imread(input_path)
            if image is None:
                print(f"   ❌ Error: No se pudo cargar {image_name}")
                continue
            
            h, w = image.shape[:2]
            print(f"   📐 Dimensiones: {w}x{h}")
            
            # Restaurar imagen
            print("   🔄 Aplicando restauración...")
            restored = pipeline.restore_document(image)
            
            # Guardar resultado
            cv2.imwrite(output_path, restored)
            print(f"   ✅ Guardado en: {output_path}")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Error durante la restauración: {e}")
        
        print()
    
    print("=" * 60)
    print(f"🎯 Resumen: {success_count}/{min(3, len(images))} imágenes procesadas exitosamente")
    print(f"📁 Resultados en: {output_dir}")
    print("=" * 60)

def process_single_image(input_path, output_path):
    """Procesar una sola imagen"""
    
    print(f"🔧 Procesando imagen individual: {input_path}")
    print("=" * 60)
    
    # Configurar el pipeline
    pipeline = DocumentRestorationPipeline()
    
    # Deshabilitar ESRGAN para la demostración (solo usar Restormer)
    pipeline.config['processing']['use_esrgan'] = False
    
    print("Inicializando modelos...")
    pipeline.initialize_models()
    print("✅ Modelos inicializados!\n")
    
    try:
        # Cargar imagen
        image = cv2.imread(input_path)
        if image is None:
            print(f"❌ Error: No se pudo cargar {input_path}")
            return False
        
        h, w = image.shape[:2]
        print(f"📐 Dimensiones originales: {w}x{h}")
        
        # Restaurar imagen
        print("🔄 Aplicando restauración...")
        restored = pipeline.restore_document(image)
        
        # Crear directorio de salida si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Guardar resultado
        cv2.imwrite(output_path, restored)
        print(f"✅ Imagen restaurada guardada en: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante la restauración: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Demostración del pipeline de restauración de documentos')
    parser.add_argument('--input', type=str,
                       help='Ruta de la imagen de entrada')
    parser.add_argument('--output', type=str,
                       help='Ruta de salida')
    parser.add_argument('--demo', action='store_true',
                       help='Ejecutar demostración con datos de prueba')
    
    args = parser.parse_args()
    
    if args.demo or (not args.input and not args.output):
        demo_restoration()
    elif args.input and args.output:
        process_single_image(args.input, args.output)
    else:
        print("❌ Error: Debes especificar --input y --output juntos, o usar --demo")
        parser.print_help()

if __name__ == "__main__":
    main()
