#!/usr/bin/env python3
"""
EVALUACIÓN RÁPIDA - Transfer Learning vs otros métodos
"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Importar directamente
import sys
sys.path.append('src')
from models.restormer import Restormer

def load_model_safe(model_path, device):
    """Cargar modelo de manera segura"""
    if not os.path.exists(model_path):
        return None
    
    try:
        print(f"📥 Cargando: {os.path.basename(model_path)}")
        
        model = Restormer(
            inp_channels=3, out_channels=3, dim=48,
            num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
            heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type='WithBias', dual_pixel_task=False
        )
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print(f"✅ Cargado exitosamente")
        return model
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}...")
        return None

def process_image_safe(model, image, device):
    """Procesar imagen de manera segura"""
    try:
        # Preprocesar
        h, w = image.shape[:2]
        
        # Padding para divisibilidad
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if len(image.shape) == 3:
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        # Convertir a tensor
        input_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        # Inferencia
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Postprocesar
        output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        
        # Recortar al tamaño original
        restored = output[:h, :w]
        return restored
        
    except Exception as e:
        print(f"❌ Error procesando: {e}")
        return image

def calculate_metrics(original, restored):
    """Calcular métricas básicas"""
    original_f = original.astype(np.float32) / 255.0
    restored_f = restored.astype(np.float32) / 255.0
    
    # PSNR
    mse = np.mean((original_f - restored_f) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # SSIM aproximado
    orig_flat = original_f.flatten()
    rest_flat = restored_f.flatten()
    correlation = np.corrcoef(orig_flat, rest_flat)[0, 1]
    ssim_approx = (correlation + 1) / 2 if not np.isnan(correlation) else 0
    
    # Sharpness
    gray_restored = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_restored, cv2.CV_64F).var()
    
    return {
        'psnr': psnr,
        'ssim': ssim_approx,
        'sharpness': laplacian_var
    }

def main():
    """Evaluación rápida"""
    print("🔬 EVALUACIÓN RÁPIDA DE MODELOS")
    print("🎯 Transfer Learning Gradual vs otros métodos")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    
    # Modelos a comparar
    models_to_test = {
        'Transfer Learning Gradual': 'outputs/checkpoints/gradual_transfer_final.pth',
        'Fine-tuning Optimizado': 'outputs/checkpoints/optimized_restormer_final.pth',
        'Fine-tuning Básico': 'outputs/checkpoints/finetuned_restormer_final.pth'
    }
    
    # Cargar modelos
    print(f"\n📥 CARGANDO MODELOS...")
    models = {}
    for name, path in models_to_test.items():
        model = load_model_safe(path, device)
        if model:
            models[name] = model
    
    if not models:
        print("❌ No se pudieron cargar modelos")
        return
    
    print(f"✅ Modelos cargados: {len(models)}")
    
    # Buscar imágenes de prueba
    test_dirs = ["data/train/degraded", "data/val/degraded"]
    test_images = []
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            test_images = [os.path.join(test_dir, f) for f in files[:3]]
            break
    
    if not test_images:
        print("❌ No se encontraron imágenes de prueba")
        return
    
    print(f"\n📸 Imágenes de prueba: {len(test_images)}")
    
    # Evaluar modelos
    print(f"\n🧪 EVALUANDO MODELOS...")
    all_results = {}
    
    for img_path in test_images:
        img_name = os.path.basename(img_path)
        print(f"\n📸 Procesando: {img_name}")
        
        degraded = cv2.imread(img_path)
        if degraded is None:
            continue
        
        all_results[img_name] = {}
        
        for model_name, model in models.items():
            print(f"   🔄 {model_name}...")
            
            restored = process_image_safe(model, degraded, device)
            metrics = calculate_metrics(degraded, restored)
            
            all_results[img_name][model_name] = metrics
            
            print(f"      📊 PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.3f} | Sharpness: {metrics['sharpness']:.0f}")
    
    # Calcular promedios
    print(f"\n📊 RESUMEN FINAL:")
    print("=" * 60)
    
    model_averages = {}
    for model_name in models.keys():
        psnr_vals = []
        ssim_vals = []
        sharpness_vals = []
        
        for img_results in all_results.values():
            if model_name in img_results:
                psnr_vals.append(img_results[model_name]['psnr'])
                ssim_vals.append(img_results[model_name]['ssim'])
                sharpness_vals.append(img_results[model_name]['sharpness'])
        
        if psnr_vals:
            model_averages[model_name] = {
                'psnr': np.mean(psnr_vals),
                'ssim': np.mean(ssim_vals),
                'sharpness': np.mean(sharpness_vals)
            }
    
    # Mostrar ranking
    if model_averages:
        print(f"\n🏆 RANKING POR PSNR:")
        psnr_ranking = sorted(model_averages.items(), key=lambda x: x[1]['psnr'], reverse=True)
        for i, (model, metrics) in enumerate(psnr_ranking):
            print(f"   {i+1}. {model}: {metrics['psnr']:.2f} dB")
        
        print(f"\n🔍 RANKING POR SSIM:")
        ssim_ranking = sorted(model_averages.items(), key=lambda x: x[1]['ssim'], reverse=True)
        for i, (model, metrics) in enumerate(ssim_ranking):
            print(f"   {i+1}. {model}: {metrics['ssim']:.3f}")
        
        print(f"\n⚡ RANKING POR SHARPNESS:")
        sharpness_ranking = sorted(model_averages.items(), key=lambda x: x[1]['sharpness'], reverse=True)
        for i, (model, metrics) in enumerate(sharpness_ranking):
            print(f"   {i+1}. {model}: {metrics['sharpness']:.0f}")
    
    print(f"\n🎉 EVALUACIÓN COMPLETADA")
    print(f"🏆 El Transfer Learning Gradual debería estar en los primeros lugares")

if __name__ == "__main__":
    main()
