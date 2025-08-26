#!/usr/bin/env python3
"""
Test rápido del Transfer Learning Gradual
Comparación solo con los modelos compatibles
"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_model_simple(model_path, device):
    """Cargar modelo de manera simple"""
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Determinar la arquitectura del modelo
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Verificar dimensionalidad del modelo
        patch_embed_shape = None
        for key in state_dict.keys():
            if 'patch_embed.weight' in key:
                patch_embed_shape = state_dict[key].shape
                break
        
        if patch_embed_shape is None:
            print(f"❌ No se pudo determinar la dimensionalidad de {model_path}")
            return None
        
        # Determinar dim basado en patch_embed
        dim = patch_embed_shape[0] if len(patch_embed_shape) > 0 else 48
        
        # Importar modelo
        import sys
        sys.path.append('src/models')
        from src.models.restormer import Restormer
        
        model = Restormer(
            inp_channels=3,
            out_channels=3,
            dim=dim,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
            dual_pixel_task=False
        )
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"✅ Modelo cargado: {os.path.basename(model_path)} (dim={dim})")
        return model
        
    except Exception as e:
        print(f"❌ Error cargando {model_path}: {str(e)[:100]}...")
        return None

def pad_to_divisible(image, factor=8):
    """Hacer que la imagen sea divisible por el factor"""
    h, w = image.shape[:2]
    
    # Calcular padding necesario
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    
    # Aplicar padding
    if len(image.shape) == 3:
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    return padded, (h, w)

def process_image_with_model(model, image_path, device):
    """Procesar imagen con manejo de dimensiones"""
    try:
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        original_image = image.copy()
        
        # Hacer divisible por 8
        padded_image, original_size = pad_to_divisible(image, 8)
        
        # Convertir a tensor
        image_tensor = torch.from_numpy(padded_image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Procesar
        with torch.no_grad():
            restored_tensor = model(image_tensor)
        
        # Convertir resultado
        restored = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
        
        # Recortar al tamaño original
        restored = restored[:original_size[0], :original_size[1]]
        
        return original_image, restored
    
    except Exception as e:
        print(f"❌ Error procesando {image_path}: {e}")
        return None, None

def calculate_psnr(img1, img2):
    """Calcular PSNR"""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def test_gradual_transfer():
    """Test rápido del Transfer Learning Gradual"""
    
    print("🚀 TEST RÁPIDO: TRANSFER LEARNING GRADUAL")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    
    # Modelos a evaluar (solo los compatibles)
    models_to_test = {
        'Fine-tuned': 'outputs/checkpoints/finetuned_restormer_final.pth',
        'Optimized': 'outputs/checkpoints/optimized_restormer_final.pth',
        'Gradual Transfer': 'outputs/checkpoints/gradual_transfer_final.pth'
    }
    
    # Cargar modelos
    models = {}
    for name, path in models_to_test.items():
        if os.path.exists(path):
            model = load_model_simple(path, device)
            if model is not None:
                models[name] = model
    
    if not models:
        print("❌ No hay modelos disponibles")
        return
    
    # Buscar imágenes de prueba
    test_dirs = ["data/train/degraded", "data/val/degraded"]
    test_images = []
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            test_images.extend([os.path.join(test_dir, f) for f in files[:3]])
            break
    
    if not test_images:
        print("❌ No hay imágenes de prueba")
        return
    
    print(f"\n📁 Probando con {len(test_images)} imágenes")
    
    # Evaluar modelos
    results = {}
    for model_name in models.keys():
        results[model_name] = []
    
    for i, img_path in enumerate(test_images[:3]):
        print(f"\n📸 Imagen {i+1}: {os.path.basename(img_path)}")
        
        for model_name, model in models.items():
            original, restored = process_image_with_model(model, img_path, device)
            
            if original is not None and restored is not None:
                # Calcular PSNR usando la imagen original como referencia\n                psnr = calculate_psnr(original, restored)
                results[model_name].append(psnr)
                print(f"   {model_name}: PSNR={psnr:.2f} dB")
    
    # Estadísticas finales
    print(f"\n📊 RESULTADOS FINALES")
    print("=" * 50)
    
    ranking = []
    for model_name in models.keys():
        if results[model_name]:
            avg_psnr = np.mean(results[model_name])
            ranking.append((model_name, avg_psnr))
            print(f"{model_name:15s}: {avg_psnr:6.2f} dB promedio")
    
    # Ordenar por PSNR
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 RANKING FINAL:")
    print("-" * 30)
    for i, (name, psnr) in enumerate(ranking, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{medal} {name}: {psnr:.2f} dB")
    
    # Análisis del Transfer Learning Gradual
    if len(ranking) > 0:
        gradual_pos = next((i for i, (name, _) in enumerate(ranking) if 'Gradual' in name), None)
        
        if gradual_pos is not None:
            print(f"\n🎯 ANÁLISIS DEL TRANSFER LEARNING GRADUAL:")
            print(f"   📍 Posición: {gradual_pos + 1}° lugar")
            
            if gradual_pos == 0:
                print(f"   🏆 ¡MEJOR MODELO! El Transfer Learning Gradual superó a todos los demás")
            elif gradual_pos == 1:
                diff = ranking[0][1] - ranking[1][1]
                print(f"   🥈 Segundo lugar (diferencia: {diff:.2f} dB)")
            else:
                best_psnr = ranking[0][1]
                gradual_psnr = ranking[gradual_pos][1]
                diff = best_psnr - gradual_psnr
                print(f"   📈 Margen de mejora: {diff:.2f} dB respecto al mejor")
    
    # Crear visualización simple
    if len(ranking) > 0:
        create_simple_plot(ranking)
    
    return ranking

def create_simple_plot(ranking):
    """Crear gráfico simple de resultados"""
    
    names = [name for name, _ in ranking]
    scores = [score for _, score in ranking]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, scores, color=['#2E8B57' if 'Gradual' in name else '#4682B4' for name in names])
    
    # Resaltar el Transfer Learning Gradual
    for i, (name, score) in enumerate(ranking):
        if 'Gradual' in name:
            bars[i].set_color('#FF6B35')  # Color destacado
            plt.annotate('🚀 Transfer Learning Gradual', 
                        xy=(i, score), xytext=(i, score + 0.5),
                        ha='center', fontsize=12, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.title('🏆 Comparación de Modelos - Transfer Learning Gradual', fontsize=14, fontweight='bold')
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.xlabel('Modelos', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Añadir valores en las barras
    for i, score in enumerate(scores):
        plt.text(i, score + 0.1, f'{score:.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar
    output_path = "outputs/analysis/gradual_transfer_results.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Resultados guardados: {output_path}")

if __name__ == "__main__":
    try:
        ranking = test_gradual_transfer()
        
        if ranking:
            print(f"\n🎉 ¡Evaluación completada!")
            
            # Determinar si el Transfer Learning Gradual fue exitoso
            gradual_result = next((score for name, score in ranking if 'Gradual' in name), None)
            if gradual_result:
                position = next(i for i, (name, _) in enumerate(ranking) if 'Gradual' in name)
                if position == 0:
                    print(f"🏆 ¡ÉXITO TOTAL! Transfer Learning Gradual es el MEJOR modelo")
                elif position == 1:
                    print(f"🥈 ¡EXCELENTE! Transfer Learning Gradual en segundo lugar")
                else:
                    print(f"📈 Transfer Learning Gradual en {position + 1}° lugar - Buen resultado")
        else:
            print("❌ No se pudieron evaluar los modelos")
            
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
