#!/usr/bin/env python3
"""
Script para crear datos sintéticos de entrenamiento
Genera parejas clean/degraded a partir de imágenes limpias
"""

import os
import cv2
import numpy as np
import random
from PIL import Image, ImageFilter, ImageEnhance
import argparse

def add_noise(image, noise_level=0.1):
    """Agregar ruido gaussiano"""
    h, w, c = image.shape
    noise = np.random.normal(0, noise_level * 255, (h, w, c))
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_blur(image, blur_strength=1.5):
    """Agregar desenfoque"""
    return cv2.GaussianBlur(image, (5, 5), blur_strength)

def add_compression_artifacts(image, quality=30):
    """Simular artefactos de compresión JPEG"""
    # Convertir a PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Simular compresión JPEG de baja calidad
    import io
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    
    # Volver a OpenCV
    return cv2.cvtColor(np.array(compressed), cv2.COLOR_RGB2BGR)

def adjust_brightness_contrast(image, brightness=0.8, contrast=0.9):
    """Ajustar brillo y contraste"""
    # Convertir a PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Ajustar brillo
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness)
    
    # Ajustar contraste
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast)
    
    # Volver a OpenCV
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def add_shadow_stains(image):
    """Agregar manchas y sombras simuladas"""
    h, w = image.shape[:2]
    
    # Crear máscara de manchas
    mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # Agregar algunas manchas circulares
    for _ in range(random.randint(2, 5)):
        center = (random.randint(0, w), random.randint(0, h))
        radius = random.randint(20, 60)
        intensity = random.randint(180, 220)
        cv2.circle(mask, center, radius, intensity, -1)
    
    # Suavizar la máscara
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask = mask.astype(np.float32) / 255.0
    
    # Aplicar la máscara
    result = image.copy().astype(np.float32)
    for c in range(3):
        result[:, :, c] *= mask
    
    return np.clip(result, 0, 255).astype(np.uint8)

def degrade_document(image, degradation_level='medium'):
    """
    Aplicar múltiples degradaciones a una imagen de documento
    """
    result = image.copy()
    
    if degradation_level == 'light':
        # Degradación ligera
        result = add_noise(result, 0.05)
        result = add_blur(result, 0.8)
        result = adjust_brightness_contrast(result, 0.95, 0.95)
        
    elif degradation_level == 'medium':
        # Degradación media
        result = add_noise(result, 0.1)
        result = add_blur(result, 1.2)
        result = add_compression_artifacts(result, 40)
        result = adjust_brightness_contrast(result, 0.85, 0.85)
        
        if random.random() > 0.5:
            result = add_shadow_stains(result)
            
    elif degradation_level == 'heavy':
        # Degradación fuerte
        result = add_noise(result, 0.15)
        result = add_blur(result, 2.0)
        result = add_compression_artifacts(result, 25)
        result = adjust_brightness_contrast(result, 0.7, 0.7)
        result = add_shadow_stains(result)
    
    return result

def add_geometric_distortions(image):
    """Agregar distorsiones geométricas sutiles"""
    h, w = image.shape[:2]
    
    try:
        # Crear puntos de control para transformación perspectiva
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # Pequeñas distorsiones aleatorias
        offset = min(w, h) * 0.01  # Reducir a 1% para evitar problemas
        pts2 = pts1.copy()
        
        # Aplicar offsets pequeños y seguros
        for i in range(4):
            pts2[i][0] += random.uniform(-offset, offset)
            pts2[i][1] += random.uniform(-offset, offset)
        
        # Asegurar que los puntos sean válidos
        pts2[0] = np.maximum(pts2[0], [0, 0])  # Top-left
        pts2[1] = np.minimum(pts2[1], [w, 0])  # Top-right
        pts2[2] = np.maximum(pts2[2], [0, h])  # Bottom-left
        pts2[3] = np.minimum(pts2[3], [w, h])  # Bottom-right
        
        # Aplicar transformación perspectiva
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(image, M, (w, h))
        
        return result
    except:
        # Si hay error, retornar imagen original
        return image

def add_color_variations(image):
    """Agregar variaciones de color realistas"""
    # Conversión a HSV para manipular color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Variaciones en hue (tinte)
    hue_shift = random.uniform(-10, 10)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    
    # Variaciones en saturación
    sat_factor = random.uniform(0.8, 1.2)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
    
    # Variaciones en valor (brillo)
    val_factor = random.uniform(0.85, 1.15)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_factor, 0, 255)
    
    # Volver a BGR
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result

def add_texture_noise(image):
    """Agregar ruido de textura tipo papel"""
    h, w, c = image.shape
    
    # Crear textura de papel
    paper_texture = np.random.normal(1.0, 0.05, (h, w))
    paper_texture = cv2.GaussianBlur(paper_texture, (3, 3), 0)
    
    # Aplicar textura
    result = image.astype(np.float32)
    for channel in range(c):
        result[:, :, channel] *= paper_texture
    
    return np.clip(result, 0, 255).astype(np.uint8)

def add_random_patches(image):
    """Agregar parches aleatorios de degradación"""
    result = image.copy()
    h, w = image.shape[:2]
    
    # Número aleatorio de parches
    num_patches = random.randint(0, 3)
    
    for _ in range(num_patches):
        # Tamaño del parche
        patch_size = random.randint(20, 80)
        
        # Posición aleatoria
        x = random.randint(0, max(0, w - patch_size))
        y = random.randint(0, max(0, h - patch_size))
        
        # Tipo de degradación del parche
        patch_type = random.choice(['blur', 'darken', 'noise'])
        
        patch = result[y:y+patch_size, x:x+patch_size].copy()
        
        if patch_type == 'blur':
            patch = cv2.GaussianBlur(patch, (7, 7), 2.0)
        elif patch_type == 'darken':
            patch = (patch * 0.6).astype(np.uint8)
        elif patch_type == 'noise':
            noise = np.random.normal(0, 30, patch.shape)
            patch = np.clip(patch + noise, 0, 255).astype(np.uint8)
        
        result[y:y+patch_size, x:x+patch_size] = patch
    
    return result

def degrade_document_advanced(image, degradation_params=None):
    """
    Aplicar degradaciones avanzadas con parámetros personalizables
    """
    if degradation_params is None:
        # Parámetros aleatorios
        degradation_params = {
            'noise_level': random.uniform(0.02, 0.15),
            'blur_strength': random.uniform(0.5, 2.5),
            'jpeg_quality': random.randint(20, 60),
            'brightness': random.uniform(0.7, 1.1),
            'contrast': random.uniform(0.7, 1.1),
            'add_shadows': random.random() > 0.6,
            'add_geometric': random.random() > 0.7,
            'add_color_shift': random.random() > 0.5,
            'add_texture': random.random() > 0.6,
            'add_patches': random.random() > 0.8
        }
    
    result = image.copy()
    
    # Aplicar degradaciones en orden aleatorio para mayor variedad
    degradations = []
    
    if degradation_params.get('add_geometric', False):
        degradations.append(('geometric', add_geometric_distortions))
    
    if degradation_params.get('add_color_shift', False):
        degradations.append(('color', add_color_variations))
    
    degradations.extend([
        ('noise', lambda img: add_noise(img, degradation_params['noise_level'])),
        ('blur', lambda img: add_blur(img, degradation_params['blur_strength'])),
        ('compression', lambda img: add_compression_artifacts(img, degradation_params['jpeg_quality'])),
        ('brightness', lambda img: adjust_brightness_contrast(img, 
                                                            degradation_params['brightness'], 
                                                            degradation_params['contrast']))
    ])
    
    if degradation_params.get('add_shadows', False):
        degradations.append(('shadows', add_shadow_stains))
    
    if degradation_params.get('add_texture', False):
        degradations.append(('texture', add_texture_noise))
    
    if degradation_params.get('add_patches', False):
        degradations.append(('patches', add_random_patches))
    
    # Mezclar orden de aplicación
    random.shuffle(degradations)
    
    # Aplicar degradaciones
    for name, func in degradations:
        try:
            result = func(result)
        except Exception as e:
            print(f"⚠️ Error en degradación {name}: {e}")
            continue
    
    return result

def create_massive_training_pairs(target_count=500):
    """Crear dataset masivo de 500+ pares de entrenamiento"""
    
    print("🏭 CREANDO DATASET MASIVO DE ENTRENAMIENTO")
    print("=" * 60)
    print(f"🎯 Meta: {target_count} imágenes degradadas")
    
    # Directorios
    clean_dir = "data/train/clean"
    degraded_dir = "data/train/degraded"
    
    # Crear directorios si no existen
    os.makedirs(degraded_dir, exist_ok=True)
    
    # Obtener imágenes limpias
    if not os.path.exists(clean_dir):
        print(f"❌ Error: No existe {clean_dir}")
        return
    
    clean_files = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not clean_files:
        print(f"❌ Error: No hay imágenes en {clean_dir}")
        return
    
    print(f"📁 Encontradas {len(clean_files)} imágenes limpias base")
    
    # Cargar todas las imágenes limpias
    clean_images = {}
    for clean_file in clean_files:
        clean_path = os.path.join(clean_dir, clean_file)
        image = cv2.imread(clean_path)
        if image is not None:
            clean_images[clean_file] = image
            print(f"✅ Cargada: {clean_file} ({image.shape[1]}x{image.shape[0]})")
    
    if not clean_images:
        print("❌ No se pudieron cargar imágenes")
        return
    
    print(f"\n🎲 Generando {target_count} variaciones degradadas...")
    
    # Calcular cuántas variaciones por imagen base
    variations_per_image = target_count // len(clean_images)
    extra_variations = target_count % len(clean_images)
    
    print(f"� {variations_per_image} variaciones por imagen base")
    print(f"📊 {extra_variations} variaciones extra en primeras imágenes")
    
    total_created = 0
    progress_step = max(1, target_count // 20)  # Mostrar progreso cada 5%
    
    for idx, (clean_file, clean_image) in enumerate(clean_images.items()):
        base_name = os.path.splitext(clean_file)[0]
        
        # Calcular variaciones para esta imagen
        num_variations = variations_per_image
        if idx < extra_variations:
            num_variations += 1
        
        print(f"\n📝 Procesando {clean_file}: {num_variations} variaciones")
        
        for var_idx in range(num_variations):
            try:
                # Crear degradación única
                degraded = degrade_document_advanced(clean_image)
                
                # Nombre único
                degraded_filename = f"{base_name}_synthetic_{var_idx+1:03d}.png"
                degraded_path = os.path.join(degraded_dir, degraded_filename)
                
                # Guardar
                cv2.imwrite(degraded_path, degraded)
                total_created += 1
                
                # Mostrar progreso
                if total_created % progress_step == 0:
                    progress = (total_created / target_count) * 100
                    print(f"   📈 Progreso: {total_created}/{target_count} ({progress:.1f}%)")
                
            except Exception as e:
                print(f"❌ Error creando variación {var_idx+1}: {e}")
                continue
    
    print("\n" + "=" * 60)
    print(f"🎯 DATASET MASIVO COMPLETADO!")
    print(f"✅ Creadas {total_created} imágenes degradadas sintéticas")
    print(f"📁 Guardadas en: {degraded_dir}")
    print(f"📊 Ratio de expansión: {total_created/len(clean_images):.1f}x")
    
    # Estadísticas del dataset
    print(f"\n📈 ESTADÍSTICAS DEL DATASET:")
    print(f"   🖼️ Imágenes limpias: {len(clean_images)}")
    print(f"   🔄 Imágenes degradadas: {total_created}")
    print(f"   📊 Total de pares: {total_created}")
    print(f"   💾 Espacio estimado: ~{total_created * 0.5:.1f} MB")
    
    return total_created

def create_training_pairs():
    """Función original mantenida para compatibilidad"""
    return create_massive_training_pairs(target_count=500)
    
    print("\n" + "=" * 60)
    print(f"🎯 RESUMEN:")
    print(f"✅ Creadas {total_created} imágenes degradadas sintéticas")
    print(f"📁 Guardadas en: {degraded_dir}")
    print("\n🚀 ¡Ahora puedes entrenar tu modelo!")
    print("   Ejecuta: python src/train.py --config config/train_config.yaml")

def expand_clean_dataset():
    """Expandir el dataset limpio usando las imágenes de validación"""
    
    print("\n🔄 EXPANDIENDO DATASET LIMPIO")
    print("-" * 40)
    
    val_clean_dir = "data/val/clean" 
    train_clean_dir = "data/train/clean"
    
    if not os.path.exists(val_clean_dir):
        print("❌ No existe directorio val/clean")
        return
    
    val_files = [f for f in os.listdir(val_clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    copied = 0
    for file in val_files:
        src = os.path.join(val_clean_dir, file)
        dst = os.path.join(train_clean_dir, f"val_{file}")
        
        # Verificar que no existe ya
        if not os.path.exists(dst):
            img = cv2.imread(src)
            if img is not None:
                cv2.imwrite(dst, img)
                print(f"   ✅ Copiada: {file} → val_{file}")
                copied += 1
    
    print(f"📊 Copiadas {copied} imágenes adicionales al dataset de entrenamiento")

if __name__ == "__main__":
    print("🎯 GENERADOR DE DATASET MASIVO")
    print("=" * 50)
    
    # Primero expandir dataset si es necesario
    try:
        expand_clean_dataset()
    except Exception as e:
        print(f"⚠️ Error expandiendo dataset: {e}")
    
    # Crear dataset masivo de 500 imágenes
    try:
        total_created = create_massive_training_pairs(target_count=500)
        
        if total_created > 0:
            print(f"\n🎉 ¡ÉXITO! Dataset expandido a {total_created} imágenes")
            print("🚀 ¡Listo para transfer learning gradual!")
        else:
            print("\n❌ No se crearon imágenes")
            
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
