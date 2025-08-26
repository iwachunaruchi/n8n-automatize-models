#!/usr/bin/env python3
"""
Script para probar el modelo con fine-tuning OPTIMIZADO
"""

import sys
import os
sys.path.append('src')

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Importar nuestros modelos
from models.restormer import Restormer

def load_optimized_model(device):
    """Cargar modelo con fine-tuning optimizado"""
    
    # Probar primero el mejor modelo
    best_model_path = "outputs/checkpoints/optimized_best_model.pth"
    final_model_path = "outputs/checkpoints/optimized_restormer_final.pth"
    
    model_path = None
    if os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"🏆 Usando MEJOR modelo optimizado")
    elif os.path.exists(final_model_path):
        model_path = final_model_path
        print(f"🔧 Usando modelo FINAL optimizado")
    else:
        print(f"❌ No se encontraron modelos optimizados")
        print("💡 Ejecuta primero: python optimized_finetuning.py")
        return None
    
    print(f"🔧 Cargando modelo optimizado...")
    
    # Crear modelo con la misma arquitectura
    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias', dual_pixel_task=False
    )
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"✅ Modelo optimizado cargado exitosamente")
        
        # Mostrar información de optimización
        if 'best_loss' in checkpoint:
            print(f"📉 Mejor loss: {checkpoint['best_loss']:.6f}")
        if 'optimization_config' in checkpoint:
            config = checkpoint['optimization_config']
            print(f"⚙️ Configuración optimizada:")
            for key, value in config.items():
                print(f"   • {key}: {value}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None

def test_optimized_model():
    """Probar el modelo con fine-tuning optimizado"""
    
    print("🧪 PROBANDO MODELO CON FINE-TUNING OPTIMIZADO")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    
    # Cargar modelo optimizado
    model = load_optimized_model(device)
    if model is None:
        return
    
    # Directorio de imágenes de prueba
    test_dir = "data/train/degraded"
    output_dir = "outputs/samples/optimized_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Imágenes de prueba
    test_images = ["1.png", "10.png", "11.png"]
    
    print(f"📁 Procesando imágenes de prueba...")
    
    results = {}
    
    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"⚠️ No se encontró: {img_name}")
            continue
        
        print(f"🖼️ Procesando: {img_name}")
        
        # Cargar imagen
        img = cv2.imread(img_path)
        original_shape = img.shape
        
        # Convertir a RGB y normalizar
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Redimensionar para el modelo
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
                output_path = os.path.join(output_dir, f"optimized_{img_name}")
                cv2.imwrite(output_path, restored_bgr)
                
                print(f"   ✅ Guardado: {output_path}")
                
                # Calcular métricas comparativas
                gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_restored = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY)
                
                # Métricas avanzadas
                sharpness_orig = cv2.Laplacian(gray_original, cv2.CV_64F).var()
                sharpness_restored = cv2.Laplacian(gray_restored, cv2.CV_64F).var()
                
                contrast_orig = gray_original.std()
                contrast_restored = gray_restored.std()
                
                brightness_orig = gray_original.mean()
                brightness_restored = gray_restored.mean()
                
                results[img_name] = {
                    'sharpness_improvement': ((sharpness_restored/sharpness_orig)-1)*100,
                    'contrast_improvement': ((contrast_restored/contrast_orig)-1)*100,
                    'brightness_change': brightness_restored - brightness_orig
                }
                
                print(f"   📈 Nitidez: {sharpness_orig:.0f} → {sharpness_restored:.0f} ({results[img_name]['sharpness_improvement']:+.1f}%)")
                print(f"   🔆 Contraste: {contrast_orig:.1f} → {contrast_restored:.1f} ({results[img_name]['contrast_improvement']:+.1f}%)")
                print(f"   💡 Brillo: {brightness_orig:.0f} → {brightness_restored:.0f} ({results[img_name]['brightness_change']:+.1f})")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   ❌ CUDA OOM para {img_name}")
                    torch.cuda.empty_cache()
                else:
                    print(f"   ❌ Error: {e}")
    
    # Resumen de resultados
    print(f"\n📊 RESUMEN DE MEJORAS:")
    print("-" * 40)
    
    avg_sharpness = np.mean([r['sharpness_improvement'] for r in results.values()])
    avg_contrast = np.mean([r['contrast_improvement'] for r in results.values()])
    
    print(f"📈 Nitidez promedio: {avg_sharpness:+.1f}%")
    print(f"🔆 Contraste promedio: {avg_contrast:+.1f}%")
    
    if avg_sharpness > -20:  # Menos del 20% de pérdida es aceptable
        print("✅ OPTIMIZACIÓN EXITOSA - Nitidez preservada")
    else:
        print("⚠️ REQUIERE MÁS AJUSTES - Pérdida de nitidez significativa")
    
    print(f"\n🎯 RESULTADOS GUARDADOS EN: {output_dir}")
    print(f"💡 Compara con modelos anteriores ejecutando:")
    print(f"   python compare_all_models.py")

def compare_optimization_results():
    """Comparar resultados antes y después de la optimización"""
    
    print(f"\n🔍 COMPARACIÓN: FINE-TUNING ORIGINAL vs OPTIMIZADO")
    print("=" * 60)
    
    # Directorios a comparar
    original_ft_dir = "outputs/samples/finetuned_model"
    optimized_dir = "outputs/samples/optimized_model"
    
    if not os.path.exists(original_ft_dir):
        print("⚠️ No se encontraron resultados del fine-tuning original")
        return
    
    if not os.path.exists(optimized_dir):
        print("⚠️ No se encontraron resultados del fine-tuning optimizado")
        return
    
    test_images = ["1.png", "10.png", "11.png"]
    
    for img_name in test_images:
        print(f"\n📸 Comparando {img_name}:")
        
        # Cargar imagen original
        original_path = f"data/train/degraded/{img_name}"
        original_ft_path = f"{original_ft_dir}/finetuned_{img_name}"
        optimized_path = f"{optimized_dir}/optimized_{img_name}"
        
        if all(os.path.exists(p) for p in [original_path, original_ft_path, optimized_path]):
            
            # Calcular métricas para cada versión
            for name, path in [("Original FT", original_ft_path), ("Optimizado", optimized_path)]:
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                contrast = gray.std()
                
                print(f"   {name}: Nitidez={sharpness:.0f}, Contraste={contrast:.1f}")
        else:
            print("   ❌ Archivos faltantes para comparación")

if __name__ == "__main__":
    test_optimized_model()
    compare_optimization_results()
