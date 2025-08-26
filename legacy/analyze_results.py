#!/usr/bin/env python3
"""
Script para analizar y comparar resultados de diferentes modelos
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

def analyze_pretrained_results():
    """Analizar resultados del modelo preentrenado vs otros modelos"""
    
    print("📊 ANÁLISIS DETALLADO DE RESULTADOS")
    print("=" * 50)
    
    # Directorios de resultados
    dirs = {
        "Original degradada": "data/train/degraded",
        "Baseline (aleatorio)": "outputs/samples", 
        "Tu modelo entrenado": "outputs/samples/trained_model",
        "Modelo preentrenado": "outputs/samples/pretrained_model"
    }
    
    # Imágenes de prueba
    test_images = ["1.png", "10.png", "11.png"]
    
    print("🔍 Analizando imágenes de prueba...")
    print("-" * 40)
    
    for img_name in test_images:
        print(f"\n📸 Imagen: {img_name}")
        print("=" * 30)
        
        results = {}
        
        # Cargar imagen original degradada
        original_path = os.path.join("data/train/degraded", img_name)
        if os.path.exists(original_path):
            original_img = cv2.imread(original_path)
            if original_img is not None:
                results["Original"] = {
                    "path": original_path,
                    "image": original_img
                }
                print(f"✅ Original degradada: {original_img.shape}")
        
        # Buscar resultados de cada modelo
        for model_name, directory in dirs.items():
            if "Original" in model_name:
                continue
                
            found_file = None
            
            if "Baseline" in model_name:
                # Buscar archivos en el directorio principal
                for f in os.listdir(directory):
                    if f.endswith('.png') and ('1.png' in f or '10.png' in f or '11.png' in f):
                        if img_name.split('.')[0] in f and 'trained' not in f and 'pretrained' not in f:
                            found_file = f
                            break
            else:
                # Buscar en directorios específicos
                if os.path.exists(directory):
                    for f in os.listdir(directory):
                        if img_name in f:
                            found_file = f
                            break
            
            if found_file:
                file_path = os.path.join(directory, found_file)
                img = cv2.imread(file_path)
                if img is not None:
                    results[model_name] = {
                        "path": file_path,
                        "image": img,
                        "filename": found_file
                    }
                    
                    # Calcular métricas básicas
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Nitidez (Laplaciano)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    # Contraste
                    contrast = gray.std()
                    
                    # Brillo
                    brightness = gray.mean()
                    
                    # Tamaño de archivo
                    file_size = os.path.getsize(file_path) / 1024
                    
                    print(f"✅ {model_name}")
                    print(f"   📁 Archivo: {found_file}")
                    print(f"   📐 Dimensiones: {img.shape}")
                    print(f"   📈 Nitidez: {laplacian_var:.2f}")
                    print(f"   🔆 Contraste: {contrast:.2f}")
                    print(f"   💡 Brillo: {brightness:.1f}")
                    print(f"   💾 Tamaño: {file_size:.1f} KB")
                else:
                    print(f"❌ {model_name}: No se pudo cargar {found_file}")
            else:
                print(f"❌ {model_name}: No encontrado")
        
        # Crear comparación visual si tenemos resultados
        if len(results) > 1:
            create_visual_comparison(img_name, results)
    
    print(f"\n🎯 CONCLUSIONES")
    print("=" * 30)
    print("📁 Comparaciones visuales guardadas en: outputs/analysis/")
    print("\n💡 Para revisar los resultados:")
    print("   1. Abre las imágenes en outputs/analysis/")
    print("   2. Compara la calidad visual")
    print("   3. Observa la eliminación de ruido")
    print("   4. Evalúa la preservación de texto")

def create_visual_comparison(img_name, results):
    """Crear comparación visual de todos los modelos"""
    
    print(f"🖼️  Creando comparación para {img_name}...")
    
    # Crear directorio
    os.makedirs("outputs/analysis", exist_ok=True)
    
    # Configurar matplotlib
    plt.style.use('default')
    
    num_images = len(results)
    fig, axes = plt.subplots(1, num_images, figsize=(4*num_images, 4))
    
    if num_images == 1:
        axes = [axes]
    
    for i, (model_name, data) in enumerate(results.items()):
        img = data["image"]
        
        # Redimensionar para visualización
        img_resized = cv2.resize(img, (300, 300))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img_rgb)
        axes[i].set_title(model_name, fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Guardar
    output_path = f"outputs/analysis/comparison_{img_name.replace('.png', '.png')}"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Guardado: {output_path}")

def show_detailed_metrics():
    """Mostrar métricas detalladas de calidad"""
    
    print(f"\n📊 MÉTRICAS DETALLADAS DE CALIDAD")
    print("=" * 40)
    
    # Verificar qué archivos del modelo preentrenado existen
    pretrained_dir = "outputs/samples/pretrained_model"
    
    if not os.path.exists(pretrained_dir):
        print("❌ No se encontraron resultados del modelo preentrenado")
        return
    
    pretrained_files = os.listdir(pretrained_dir)
    print(f"📁 Archivos del modelo preentrenado:")
    
    for f in pretrained_files:
        if f.endswith('.png'):
            file_path = os.path.join(pretrained_dir, f)
            img = cv2.imread(file_path)
            
            if img is not None:
                # Métricas avanzadas
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Nitidez
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness = laplacian.var()
                
                # Entropía (información de la imagen)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = hist / hist.sum()
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                
                # Gradiente promedio
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
                avg_gradient = gradient_mag.mean()
                
                print(f"\n📄 {f}")
                print(f"   📐 Resolución: {img.shape[1]}x{img.shape[0]}")
                print(f"   📈 Nitidez: {sharpness:.2f}")
                print(f"   🔍 Entropía: {entropy:.2f}")
                print(f"   📊 Gradiente: {avg_gradient:.2f}")
                print(f"   💾 Tamaño: {os.path.getsize(file_path)/1024:.1f} KB")

if __name__ == "__main__":
    analyze_pretrained_results()
    show_detailed_metrics()
