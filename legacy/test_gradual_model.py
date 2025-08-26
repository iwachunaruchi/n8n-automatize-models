#!/usr/bin/env python3
"""
Test del modelo con Transfer Learning Gradual
Comparación completa con todos los métodos anteriores
"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Imports locales
import sys
sys.path.append('src')
sys.path.append('src/models')

from src.models.restormer import Restormer

def load_model(model_path, device):
    """Cargar modelo entrenado"""
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
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Modelo cargado: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"❌ Error cargando {model_path}: {e}")
        return None
    
    model.to(device)
    model.eval()
    return model

def calculate_metrics(original, restored):
    """Calcular métricas de calidad"""
    # Convertir a float32 y normalizar
    original = original.astype(np.float32) / 255.0
    restored = restored.astype(np.float32) / 255.0
    
    # PSNR
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # SSIM aproximado (correlación normalizada)
    original_flat = original.flatten()
    restored_flat = restored.flatten()
    
    correlation = np.corrcoef(original_flat, restored_flat)[0, 1]
    ssim_approx = (correlation + 1) / 2  # Normalizar a [0, 1]
    
    return psnr, ssim_approx

def test_single_image(model, image_path, device):
    """Testear una imagen con el modelo"""
    try:
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        original_image = image.copy()
        
        # Convertir a tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Procesar
        with torch.no_grad():
            restored_tensor = model(image_tensor)
        
        # Convertir resultado
        restored = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
        
        return original_image, restored
    
    except Exception as e:
        print(f"❌ Error procesando {image_path}: {e}")
        return None, None

def test_all_models():
    """Comparar todos los modelos entrenados"""
    
    print("🧪 EVALUACIÓN COMPLETA DE MODELOS")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    
    # Modelos a evaluar
    models_info = {
        'Pretrained': 'models/pretrained/restormer_denoising.pth',
        'Trained Custom': 'outputs/checkpoints/restormer_epoch_30.pth',
        'Fine-tuned': 'outputs/checkpoints/finetuned_restormer_final.pth',
        'Optimized': 'outputs/checkpoints/optimized_restormer_final.pth',
        'Gradual Transfer': 'outputs/checkpoints/gradual_transfer_final.pth'
    }
    
    # Verificar qué modelos existen
    available_models = {}
    for name, path in models_info.items():
        if os.path.exists(path):
            available_models[name] = path
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: No encontrado")
    
    if not available_models:
        print("❌ No hay modelos disponibles para evaluar")
        return
    
    # Imágenes de prueba
    test_dir = "data/val/degraded"
    if not os.path.exists(test_dir):
        test_dir = "data/train/degraded"
    
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][:5]
    
    if not test_images:
        print("❌ No hay imágenes de prueba")
        return
    
    print(f"\n📁 Usando {len(test_images)} imágenes de prueba")
    
    # Cargar modelos
    models = {}
    for name, path in available_models.items():
        model = load_model(path, device)
        if model is not None:
            models[name] = model
    
    # Evaluar cada imagen
    results = {name: {'psnr': [], 'ssim': []} for name in models.keys()}
    
    print(f"\n🔍 EVALUANDO MODELOS...")
    
    # Para visualización
    comparison_results = []
    
    for i, img_name in enumerate(test_images[:3]):  # Solo 3 para visualización
        img_path = os.path.join(test_dir, img_name)
        
        print(f"\n📸 Imagen {i+1}: {img_name}")
        
        # Buscar imagen limpia correspondiente
        clean_name = img_name.replace('_deg_', '_').replace('_var_', '_').replace('_synthetic_', '_')
        clean_name = clean_name.replace('degraded', 'clean')
        
        # Intentar encontrar imagen limpia
        clean_paths = [
            os.path.join("data/val/clean", clean_name),
            os.path.join("data/train/clean", clean_name.replace('val_', '')),
            os.path.join("data/train/clean", img_name.split('_')[0] + '.png')
        ]
        
        clean_image = None
        for clean_path in clean_paths:
            if os.path.exists(clean_path):
                clean_image = cv2.imread(clean_path)
                break
        
        # Probar cada modelo
        img_results = {}
        
        for model_name, model in models.items():
            degraded, restored = test_single_image(model, img_path, device)
            
            if degraded is not None and restored is not None:
                img_results[model_name] = {
                    'degraded': degraded,
                    'restored': restored
                }
                
                # Calcular métricas si tenemos imagen limpia
                if clean_image is not None:
                    # Redimensionar si es necesario
                    if clean_image.shape != restored.shape:
                        clean_resized = cv2.resize(clean_image, (restored.shape[1], restored.shape[0]))
                    else:
                        clean_resized = clean_image
                    
                    psnr, ssim = calculate_metrics(clean_resized, restored)
                    results[model_name]['psnr'].append(psnr)
                    results[model_name]['ssim'].append(ssim)
                    
                    print(f"   {model_name}: PSNR={psnr:.2f}, SSIM={ssim:.3f}")
        
        if clean_image is not None:
            img_results['clean'] = clean_image
        
        comparison_results.append({
            'name': img_name,
            'results': img_results
        })
    
    # Mostrar estadísticas finales
    print(f"\n📊 ESTADÍSTICAS FINALES")
    print("=" * 60)
    
    model_ranking = []
    
    for model_name in models.keys():
        if results[model_name]['psnr']:
            avg_psnr = np.mean(results[model_name]['psnr'])
            avg_ssim = np.mean(results[model_name]['ssim'])
            model_ranking.append((model_name, avg_psnr, avg_ssim))
            print(f"{model_name:15s}: PSNR={avg_psnr:6.2f}, SSIM={avg_ssim:.3f}")
    
    # Ranking por PSNR
    model_ranking.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 RANKING DE MODELOS (por PSNR):")
    print("-" * 40)
    for i, (name, psnr, ssim) in enumerate(model_ranking, 1):
        print(f"{i}. {name}: {psnr:.2f} dB")
    
    # Crear visualización comparativa
    create_comparison_visualization(comparison_results, model_ranking)
    
    return model_ranking

def create_comparison_visualization(comparison_results, model_ranking):
    """Crear visualización comparativa"""
    
    print(f"\n🎨 Creando visualización comparativa...")
    
    n_images = len(comparison_results)
    n_models = len(model_ranking)
    
    # Crear figura grande
    fig, axes = plt.subplots(n_images, n_models + 2, figsize=(20, 6 * n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, img_data in enumerate(comparison_results):
        img_name = img_data['name']
        results = img_data['results']
        
        # Imagen degradada
        if 'degraded' in results[list(results.keys())[0]]:
            degraded = results[list(results.keys())[0]]['degraded']
            axes[i, 0].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title('Degradada', fontsize=10)
            axes[i, 0].axis('off')
        
        # Imagen limpia (si existe)
        col = 1
        if 'clean' in results:
            clean = results['clean']
            axes[i, col].imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
            axes[i, col].set_title('Original', fontsize=10)
            axes[i, col].axis('off')
            col += 1
        else:
            # Ocultar columna de original si no existe
            axes[i, col].axis('off')
            col += 1
        
        # Resultados de modelos (ordenados por ranking)
        for j, (model_name, _, _) in enumerate(model_ranking):
            if model_name in results and 'restored' in results[model_name]:
                restored = results[model_name]['restored']
                axes[i, col + j].imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
                axes[i, col + j].set_title(f'{model_name}', fontsize=10)
                axes[i, col + j].axis('off')
    
    plt.suptitle('🏆 Comparación Completa de Modelos - Transfer Learning Gradual', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Guardar
    output_path = "outputs/analysis/gradual_transfer_comparison.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Comparación guardada: {output_path}")

if __name__ == "__main__":
    print("🎯 EVALUACIÓN DEL TRANSFER LEARNING GRADUAL")
    print("=" * 50)
    
    try:
        ranking = test_all_models()
        
        if ranking:
            print(f"\n🎉 ¡Evaluación completada!")
            print(f"🥇 Mejor modelo: {ranking[0][0]} ({ranking[0][1]:.2f} dB)")
            print(f"📊 Ver resultados en: outputs/analysis/gradual_transfer_comparison.png")
        
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
