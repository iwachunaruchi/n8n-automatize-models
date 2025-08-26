#!/usr/bin/env python3
"""
EVALUACIÓN Y COMPARACIÓN DE MODELOS
Compara todos los métodos implementados
"""

import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Paths
import sys
import os

# Agregar el directorio padre al path para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.models.restormer import Restormer

class ModelEvaluator:
    """Evaluador de modelos para comparación"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.results = {}
        
    def load_model(self, name: str, checkpoint_path: str) -> bool:
        """Cargar un modelo específico"""
        if not os.path.exists(checkpoint_path):
            print(f"❌ No encontrado: {checkpoint_path}")
            return False
        
        try:
            print(f"📥 Cargando {name}...")
            
            # Crear modelo
            model = Restormer(
                inp_channels=3, out_channels=3, dim=48,
                num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                bias=False, LayerNorm_type='WithBias', dual_pixel_task=False
            )
            
            # Cargar pesos
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            self.models[name] = model
            print(f"✅ {name} cargado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando {name}: {str(e)[:100]}...")
            return False
    
    def load_all_available_models(self):
        """Cargar todos los modelos disponibles"""
        print("🔧 CARGANDO TODOS LOS MODELOS DISPONIBLES")
        print("=" * 50)
        
        model_paths = {
            "Transfer Learning Gradual": "outputs/checkpoints/gradual_transfer_final.pth",
            "Fine-tuning Optimizado": "outputs/checkpoints/optimized_restormer_final.pth",
            "Fine-tuning Básico": "outputs/checkpoints/finetuned_restormer_final.pth",
            "Modelo Preentrenado": "models/pretrained/restormer_denoising.pth",
            "Entrenado desde Cero": "outputs/checkpoints/best_restormer.pth"
        }
        
        loaded_count = 0
        for name, path in model_paths.items():
            if self.load_model(name, path):
                loaded_count += 1
        
        print(f"\n📊 Modelos cargados: {loaded_count}/{len(model_paths)}")
        return loaded_count > 0
    
    def pad_to_divisible(self, image: np.ndarray, factor: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Hacer imagen divisible por factor"""
        h, w = image.shape[:2]
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        
        if len(image.shape) == 3:
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        return padded, (h, w)
    
    def process_image_with_model(self, model, image: np.ndarray) -> np.ndarray:
        """Procesar imagen con un modelo específico"""
        try:
            # Preprocesar
            padded_image, original_size = self.pad_to_divisible(image, 8)
            input_tensor = torch.from_numpy(padded_image).permute(2, 0, 1).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            # Inferencia
            with torch.no_grad():
                output_tensor = model(input_tensor)
            
            # Postprocesar
            output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
            
            # Recortar al tamaño original
            restored = output[:original_size[0], :original_size[1]]
            return restored
            
        except Exception as e:
            print(f"❌ Error procesando con modelo: {e}")
            return image
    
    def calculate_metrics(self, original: np.ndarray, restored: np.ndarray) -> Dict[str, float]:
        """Calcular métricas de calidad"""
        original_f = original.astype(np.float32) / 255.0
        restored_f = restored.astype(np.float32) / 255.0
        
        # PSNR
        mse = np.mean((original_f - restored_f) ** 2)
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        
        # SSIM aproximado (correlación)
        orig_flat = original_f.flatten()
        rest_flat = restored_f.flatten()
        correlation = np.corrcoef(orig_flat, rest_flat)[0, 1]
        ssim_approx = (correlation + 1) / 2
        
        # Sharpness (varianza del Laplaciano)
        gray_restored = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_restored, cv2.CV_64F).var()
        
        return {
            'psnr': psnr,
            'ssim': ssim_approx,
            'sharpness': laplacian_var
        }
    
    def evaluate_on_test_images(self, test_images: List[str]):
        """Evaluar todos los modelos en imágenes de prueba"""
        print(f"\n🧪 EVALUANDO {len(self.models)} MODELOS EN {len(test_images)} IMÁGENES")
        print("=" * 60)
        
        self.results = {}
        
        for img_path in test_images:
            img_name = os.path.basename(img_path)
            print(f"\n📸 Procesando: {img_name}")
            
            # Cargar imagen
            degraded = cv2.imread(img_path)
            if degraded is None:
                continue
            
            self.results[img_name] = {}
            
            # Evaluar cada modelo
            for model_name, model in self.models.items():
                print(f"   🔄 {model_name}...")
                
                restored = self.process_image_with_model(model, degraded)
                metrics = self.calculate_metrics(degraded, restored)
                
                self.results[img_name][model_name] = {
                    'restored': restored,
                    'metrics': metrics
                }
                
                print(f"      📊 PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.3f} | Sharpness: {metrics['sharpness']:.0f}")
    
    def create_comparison_visualization(self, image_name: str):
        """Crear visualización comparativa para una imagen"""
        if image_name not in self.results:
            return
        
        results = self.results[image_name]
        num_models = len(results)
        
        if num_models == 0:
            return
        
        fig, axes = plt.subplots(2, (num_models + 1) // 2, figsize=(4 * ((num_models + 1) // 2), 8))
        if num_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Mostrar cada modelo
        for i, (model_name, data) in enumerate(results.items()):
            if i >= len(axes):
                break
                
            restored = data['restored']
            metrics = data['metrics']
            
            # Convertir BGR a RGB para matplotlib
            restored_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(restored_rgb)
            axes[i].set_title(f"{model_name}\nPSNR: {metrics['psnr']:.1f} | SSIM: {metrics['ssim']:.3f}", 
                            fontsize=10)
            axes[i].axis('off')
        
        # Ocultar ejes no usados
        for i in range(num_models, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Guardar
        output_path = f"outputs/evaluation/comparison_{image_name.replace('.png', '.png')}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparación guardada: {output_path}")
    
    def create_metrics_summary(self):
        """Crear resumen de métricas de todos los modelos"""
        if not self.results:
            return
        
        # Agregar métricas por modelo
        model_metrics = {}
        
        for img_name, img_results in self.results.items():
            for model_name, data in img_results.items():
                if model_name not in model_metrics:
                    model_metrics[model_name] = {'psnr': [], 'ssim': [], 'sharpness': []}
                
                metrics = data['metrics']
                model_metrics[model_name]['psnr'].append(metrics['psnr'])
                model_metrics[model_name]['ssim'].append(metrics['ssim'])
                model_metrics[model_name]['sharpness'].append(metrics['sharpness'])
        
        # Calcular promedios
        model_averages = {}
        for model_name, metrics in model_metrics.items():
            model_averages[model_name] = {
                'psnr': np.mean(metrics['psnr']),
                'ssim': np.mean(metrics['ssim']),
                'sharpness': np.mean(metrics['sharpness'])
            }
        
        # Crear gráfico
        models = list(model_averages.keys())
        psnr_values = [model_averages[model]['psnr'] for model in models]
        ssim_values = [model_averages[model]['ssim'] for model in models]
        sharpness_values = [model_averages[model]['sharpness'] for model in models]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # PSNR
        bars1 = ax1.bar(models, psnr_values, color='skyblue', alpha=0.7)
        ax1.set_title('PSNR Promedio')
        ax1.set_ylabel('PSNR (dB)')
        ax1.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars1, psnr_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom')
        
        # SSIM
        bars2 = ax2.bar(models, ssim_values, color='lightgreen', alpha=0.7)
        ax2.set_title('SSIM Promedio')
        ax2.set_ylabel('SSIM')
        ax2.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars2, ssim_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Sharpness
        bars3 = ax3.bar(models, sharpness_values, color='lightcoral', alpha=0.7)
        ax3.set_title('Sharpness Promedio')
        ax3.set_ylabel('Varianza Laplaciano')
        ax3.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars3, sharpness_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Guardar
        output_path = "outputs/evaluation/metrics_summary.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Resumen de métricas guardado: {output_path}")
        
        return model_averages
    
    def print_ranking(self, model_averages: Dict):
        """Imprimir ranking de modelos"""
        print(f"\n🏆 RANKING DE MODELOS")
        print("=" * 50)
        
        # Ranking por PSNR
        psnr_ranking = sorted(model_averages.items(), key=lambda x: x[1]['psnr'], reverse=True)
        print(f"\n📊 Por PSNR:")
        for i, (model, metrics) in enumerate(psnr_ranking):
            print(f"   {i+1}. {model}: {metrics['psnr']:.2f} dB")
        
        # Ranking por SSIM
        ssim_ranking = sorted(model_averages.items(), key=lambda x: x[1]['ssim'], reverse=True)
        print(f"\n🔍 Por SSIM:")
        for i, (model, metrics) in enumerate(ssim_ranking):
            print(f"   {i+1}. {model}: {metrics['ssim']:.3f}")
        
        # Ranking por Sharpness
        sharpness_ranking = sorted(model_averages.items(), key=lambda x: x[1]['sharpness'], reverse=True)
        print(f"\n⚡ Por Sharpness:")
        for i, (model, metrics) in enumerate(sharpness_ranking):
            print(f"   {i+1}. {model}: {metrics['sharpness']:.0f}")

def find_test_images() -> List[str]:
    """Encontrar imágenes de prueba"""
    test_dirs = ["data/val/degraded", "data/train/degraded"]
    test_images = []
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            test_images = [os.path.join(test_dir, f) for f in files[:3]]
            break
    
    return test_images

def main():
    """Función principal de evaluación"""
    print("🔬 EVALUACIÓN COMPARATIVA DE MODELOS")
    print("🎯 Transfer Learning vs otros métodos")
    print("=" * 60)
    
    # Inicializar evaluador
    evaluator = ModelEvaluator()
    
    # Cargar modelos
    if not evaluator.load_all_available_models():
        print("❌ No se pudieron cargar modelos")
        return
    
    # Encontrar imágenes de prueba
    test_images = find_test_images()
    if not test_images:
        print("❌ No se encontraron imágenes de prueba")
        return
    
    print(f"\n📁 Imágenes de prueba: {len(test_images)}")
    for img in test_images:
        print(f"   📸 {os.path.basename(img)}")
    
    # Evaluar modelos
    evaluator.evaluate_on_test_images(test_images)
    
    # Crear visualizaciones
    print(f"\n🎨 Creando visualizaciones...")
    for img_path in test_images:
        img_name = os.path.basename(img_path)
        evaluator.create_comparison_visualization(img_name)
    
    # Resumen de métricas
    model_averages = evaluator.create_metrics_summary()
    
    # Mostrar ranking
    if model_averages:
        evaluator.print_ranking(model_averages)
    
    print(f"\n🎉 EVALUACIÓN COMPLETADA")
    print(f"📁 Resultados en: outputs/evaluation/")
    print(f"🏆 El Transfer Learning Gradual debería mostrar los mejores resultados")

if __name__ == "__main__":
    main()
