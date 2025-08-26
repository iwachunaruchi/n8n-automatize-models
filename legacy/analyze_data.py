#!/usr/bin/env python3
"""
Script para verificar y analizar los datos de entrenamiento
"""

import os
import cv2
import numpy as np

def analyze_dataset():
    """Analizar la estructura del dataset"""
    
    print("🔍 ANÁLISIS DE DATOS DE ENTRENAMIENTO")
    print("=" * 60)
    
    # Directorios
    train_clean = "data/train/clean"
    train_degraded = "data/train/degraded"
    val_clean = "data/val/clean"
    val_degraded = "data/val/degraded"
    
    # Analizar cada directorio
    for name, path in [("Train Clean", train_clean), 
                       ("Train Degraded", train_degraded),
                       ("Val Clean", val_clean),
                       ("Val Degraded", val_degraded)]:
        
        print(f"\n📁 {name}: {path}")
        print("-" * 40)
        
        if not os.path.exists(path):
            print("   ❌ Directorio no existe")
            continue
            
        files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"   📊 Total archivos: {len(files)}")
        
        if files:
            print("   📝 Archivos encontrados:")
            for i, file in enumerate(sorted(files)):
                if i >= 5:  # Mostrar solo los primeros 5
                    print(f"       ... y {len(files)-5} más")
                    break
                    
                filepath = os.path.join(path, file)
                try:
                    img = cv2.imread(filepath)
                    if img is not None:
                        h, w = img.shape[:2]
                        size_mb = os.path.getsize(filepath) / (1024*1024)
                        print(f"       ✅ {file}: {w}x{h} ({size_mb:.2f}MB)")
                    else:
                        print(f"       ❌ {file}: Error al cargar")
                except Exception as e:
                    print(f"       ❌ {file}: {e}")
    
    print("\n" + "=" * 60)
    print("📋 RECOMENDACIONES:")
    
    # Verificar correspondencia
    train_clean_files = set(os.listdir(train_clean)) if os.path.exists(train_clean) else set()
    train_degraded_files = set(os.listdir(train_degraded)) if os.path.exists(train_degraded) else set()
    
    # Buscar correspondencias
    matching_files = train_clean_files.intersection(train_degraded_files)
    
    if matching_files:
        print(f"✅ Archivos emparejados encontrados: {len(matching_files)}")
        for file in sorted(matching_files):
            print(f"   - {file}")
    else:
        print("⚠️  No se encontraron archivos emparejados entre clean y degraded")
        print("   Para entrenar necesitas:")
        print("   - data/train/clean/imagen1.png")
        print("   - data/train/degraded/imagen1.png (misma imagen degradada)")
    
    print("\n🎯 PRÓXIMOS PASOS:")
    if len(matching_files) > 0:
        print("✅ Puedes entrenar con los datos emparejados existentes")
        print("🚀 Ejecuta: python src/train.py --config config/train_config.yaml")
    else:
        print("🔧 Necesitas emparejar tus datos primero")
        print("📝 O crear datos sintéticos de degradación")

if __name__ == "__main__":
    analyze_dataset()
