#!/usr/bin/env python3
"""
GENERACIÓN MASIVA DE DATOS SINTÉTICOS
Versión optimizada para el Transfer Learning Gradual
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
import io
from tqdm import tqdm

def add_geometric_distortions(image):
    """Agregar distorsiones geométricas leves"""
    h, w = image.shape[:2]
    
    # Transformación de perspectiva sutil
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # Variaciones pequeñas para simular documentos escaneados
    offset = min(w, h) * 0.02  # 2% de distorsión máxima
    
    pts2 = np.float32([
        [random.uniform(-offset, offset), random.uniform(-offset, offset)],
        [w + random.uniform(-offset, offset), random.uniform(-offset, offset)], 
        [random.uniform(-offset, offset), h + random.uniform(-offset, offset)],
        [w + random.uniform(-offset, offset), h + random.uniform(-offset, offset)]
    ])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    distorted = cv2.warpPerspective(image, matrix, (w, h), 
                                   borderMode=cv2.BORDER_REFLECT_101)
    
    return distorted

def add_color_variations(image):
    """Agregar variaciones de color sutiles"""
    # Convertir a PIL para mejor control de color
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Ajustes sutiles de brillo y contraste
    brightness = random.uniform(0.85, 1.15)
    contrast = random.uniform(0.9, 1.1)
    saturation = random.uniform(0.95, 1.05)
    
    # Aplicar ajustes
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness)
    
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast)
    
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(saturation)
    
    # Volver a OpenCV
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def add_texture_noise(image):
    """Agregar ruido de textura (simula papel, scanner, etc.)"""
    h, w, c = image.shape
    
    # Ruido gaussiano sutil
    gaussian_noise = np.random.normal(0, random.uniform(3, 8), (h, w, c))
    
    # Ruido de papel (patrón granular)
    paper_noise = np.random.poisson(random.uniform(0.5, 2.0), (h, w, c))
    
    # Combinar ruidos
    combined_noise = gaussian_noise + paper_noise
    
    # Aplicar al imagen
    noisy = image.astype(np.float32) + combined_noise
    
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_random_patches(image):
    """Agregar manchas y artefactos aleatorios"""
    h, w = image.shape[:2]
    result = image.copy()
    
    # Número de patches
    num_patches = random.randint(1, 4)
    
    for _ in range(num_patches):
        # Tamaño del patch
        patch_size = random.randint(5, 25)
        
        # Posición aleatoria
        x = random.randint(0, max(1, w - patch_size))
        y = random.randint(0, max(1, h - patch_size))
        
        # Tipo de patch
        patch_type = random.choice(['dark', 'light', 'blur'])
        
        if patch_type == 'dark':
            # Mancha oscura
            intensity = random.uniform(0.3, 0.7)
            result[y:y+patch_size, x:x+patch_size] = \
                result[y:y+patch_size, x:x+patch_size] * intensity
        elif patch_type == 'light':
            # Mancha clara
            addition = random.uniform(20, 60)
            result[y:y+patch_size, x:x+patch_size] = \
                np.clip(result[y:y+patch_size, x:x+patch_size] + addition, 0, 255)
        else:  # blur
            # Área borrosa
            patch = result[y:y+patch_size, x:x+patch_size]
            blurred = cv2.GaussianBlur(patch, (5, 5), 2.0)
            result[y:y+patch_size, x:x+patch_size] = blurred
    
    return result

def add_compression_artifacts(image, quality=None):
    """Simular artefactos de compresión JPEG"""
    if quality is None:
        quality = random.randint(25, 60)
    
    # Convertir a PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Simular compresión JPEG
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    
    # Volver a OpenCV
    return cv2.cvtColor(np.array(compressed), cv2.COLOR_RGB2BGR)

def add_blur_effects(image):
    """Agregar efectos de desenfoque variados"""
    blur_type = random.choice(['gaussian', 'motion', 'defocus'])
    
    if blur_type == 'gaussian':
        # Desenfoque gaussiano
        kernel_size = random.choice([3, 5])
        sigma = random.uniform(0.5, 1.5)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
    elif blur_type == 'motion':
        # Motion blur
        size = random.randint(3, 7)
        angle = random.uniform(0, 180)
        
        # Crear kernel de motion blur
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        
        # Rotar kernel
        center = (size // 2, size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (size, size))
        
        return cv2.filter2D(image, -1, kernel)
        
    else:  # defocus
        # Defocus blur (desenfoque circular)
        radius = random.randint(2, 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
        kernel = kernel.astype(np.float32)
        kernel = kernel / np.sum(kernel)
        
        return cv2.filter2D(image, -1, kernel)

def degrade_document_advanced(image, degradation_params=None):
    """
    Aplicar degradación avanzada a documento
    
    Args:
        image: Imagen limpia
        degradation_params: Parámetros específicos de degradación
    
    Returns:
        Imagen degradada
    """
    if degradation_params is None:
        degradation_params = {
            'noise_prob': 0.7,
            'blur_prob': 0.6,
            'compression_prob': 0.5,
            'geometric_prob': 0.4,
            'color_prob': 0.8,
            'patches_prob': 0.3
        }
    
    result = image.copy()
    
    # Aplicar degradaciones de forma probabilística
    
    # 1. Variaciones de color (más común)
    if random.random() < degradation_params['color_prob']:
        result = add_color_variations(result)
    
    # 2. Ruido de textura
    if random.random() < degradation_params['noise_prob']:
        result = add_texture_noise(result)
    
    # 3. Efectos de desenfoque
    if random.random() < degradation_params['blur_prob']:
        result = add_blur_effects(result)
    
    # 4. Artefactos de compresión
    if random.random() < degradation_params['compression_prob']:
        result = add_compression_artifacts(result)
    
    # 5. Distorsiones geométricas
    if random.random() < degradation_params['geometric_prob']:
        result = add_geometric_distortions(result)
    
    # 6. Patches y manchas
    if random.random() < degradation_params['patches_prob']:
        result = add_random_patches(result)
    
    return result

def create_massive_training_pairs(target_count=500):
    """
    Crear massive dataset de pares clean/degraded
    
    Args:
        target_count: Número objetivo de imágenes degradadas a generar
    """
    print("🔥 GENERACIÓN MASIVA DE DATOS SINTÉTICOS")
    print("🎯 Transfer Learning Gradual Dataset Expansion")
    print("=" * 60)
    
    # Directorios
    clean_dir = "data/train/clean"
    degraded_dir = "data/train/degraded"
    
    # Verificar directorios
    if not os.path.exists(clean_dir):
        print(f"❌ No se encontró directorio clean: {clean_dir}")
        return
    
    os.makedirs(degraded_dir, exist_ok=True)
    
    # Obtener imágenes limpias base
    clean_files = [f for f in os.listdir(clean_dir) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(clean_files) == 0:
        print(f"❌ No se encontraron imágenes limpias en: {clean_dir}")
        return
    
    print(f"📁 Imágenes base encontradas: {len(clean_files)}")
    for f in clean_files:
        print(f"   📸 {f}")
    
    # Calcular imágenes por archivo base
    images_per_base = target_count // len(clean_files)
    remainder = target_count % len(clean_files)
    
    print(f"\n🎯 Objetivo: {target_count} imágenes degradadas")
    print(f"📊 {images_per_base} imágenes por archivo base")
    if remainder > 0:
        print(f"📊 +{remainder} imágenes adicionales en primeros archivos")
    
    # Contador global
    total_generated = 0
    
    # Generar para cada imagen base
    for base_idx, filename in enumerate(clean_files):
        clean_path = os.path.join(clean_dir, filename)
        
        print(f"\n📸 Procesando: {filename}")
        
        # Cargar imagen base
        clean_image = cv2.imread(clean_path)
        if clean_image is None:
            print(f"   ❌ Error cargando: {filename}")
            continue
        
        print(f"   📐 Dimensiones: {clean_image.shape}")
        
        # Calcular cuántas imágenes generar para este archivo
        images_to_generate = images_per_base
        if base_idx < remainder:
            images_to_generate += 1
        
        print(f"   🎯 Generando: {images_to_generate} variaciones")
        
        # Generar variaciones
        for i in tqdm(range(images_to_generate), desc=f"   Generando"):
            # Crear nombre único
            base_name = os.path.splitext(filename)[0]
            degraded_filename = f"{base_name}_synthetic_{i+1:03d}.png"
            degraded_path = os.path.join(degraded_dir, degraded_filename)
            
            # Configurar parámetros de degradación variables
            # Más variabilidad según el índice
            intensity = min(1.0, 0.3 + (i / images_to_generate) * 0.7)
            
            degradation_params = {
                'noise_prob': 0.5 + intensity * 0.4,
                'blur_prob': 0.4 + intensity * 0.4,
                'compression_prob': 0.3 + intensity * 0.5,
                'geometric_prob': 0.2 + intensity * 0.3,
                'color_prob': 0.6 + intensity * 0.3,
                'patches_prob': 0.1 + intensity * 0.4
            }
            
            # Aplicar degradación
            try:
                degraded_image = degrade_document_advanced(clean_image, degradation_params)
                
                # Guardar imagen degradada
                cv2.imwrite(degraded_path, degraded_image)
                total_generated += 1
                
            except Exception as e:
                print(f"   ❌ Error generando {degraded_filename}: {e}")
        
        print(f"   ✅ Completado: {images_to_generate} imágenes generadas")
    
    # Verificar resultados
    final_degraded_files = [f for f in os.listdir(degraded_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n🎉 GENERACIÓN COMPLETADA")
    print("=" * 60)
    print(f"📊 Total imágenes degradadas: {len(final_degraded_files)}")
    print(f"🎯 Objetivo cumplido: {total_generated}/{target_count}")
    print(f"📁 Ubicación: {degraded_dir}")
    
    # Mostrar estadísticas
    print(f"\n📈 ESTADÍSTICAS DEL DATASET:")
    print(f"   🔹 Imágenes limpias: {len(clean_files)}")
    print(f"   🔹 Imágenes degradadas: {len(final_degraded_files)}")
    print(f"   🔹 Factor de expansión: {len(final_degraded_files) / len(clean_files):.1f}x")
    
    print(f"\n🎯 PRÓXIMO PASO:")
    print(f"   ▶️  python training\\gradual_transfer_learning.py")
    print(f"   📊 El dataset expandido mejorará significativamente el Transfer Learning")

def verify_dataset():
    """Verificar la integridad del dataset generado"""
    print("\n🔍 VERIFICACIÓN DEL DATASET")
    print("-" * 40)
    
    clean_dir = "data/train/clean"
    degraded_dir = "data/train/degraded"
    
    # Verificar directorios
    for name, path in [("Clean", clean_dir), ("Degraded", degraded_dir)]:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            print(f"📁 {name}: {len(files)} archivos")
            
            # Verificar integridad de algunos archivos
            valid_count = 0
            for file in files[:5]:  # Verificar primeros 5
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    valid_count += 1
            
            print(f"   ✅ Archivos verificados: {valid_count}/{min(5, len(files))}")
        else:
            print(f"❌ {name}: Directorio no encontrado")

def main():
    """Función principal"""
    print("🚀 GENERADOR DE DATOS SINTÉTICOS")
    print("🎯 Para Transfer Learning Gradual")
    print("=" * 50)
    
    # Generar dataset masivo
    create_massive_training_pairs(target_count=500)
    
    # Verificar integridad
    verify_dataset()
    
    print(f"\n✨ ¡Dataset listo para Transfer Learning Gradual!")

if __name__ == "__main__":
    main()
