"""
Entrenamiento para Capa 1: Pipeline de Preprocesamiento
Otsu + CLAHE + Deskew por Hough

Este script no requiere entrenamiento con modelos ML,
pero evalúa la efectividad del preprocesamiento usando imágenes de la API
"""

import os
import sys
import requests
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Agregar path para importar la capa 1
sys.path.append(str(Path(__file__).parent.parent / "layer-1"))
from layer_1 import PreprocessingPipeline

# Configuración
API_BASE_URL = "http://localhost:8000"
MINIO_BUCKETS = {
    'degraded': 'document-degraded',
    'clean': 'document-clean',
    'restored': 'document-restored',
    'training': 'document-training'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Layer1Trainer:
    """Evaluador y optimizador para Capa 1"""
    
    def __init__(self, api_base_url: str = API_BASE_URL):
        self.api_url = api_base_url
        self.pipeline = PreprocessingPipeline()
        self.results = []
        
    def fetch_images_from_api(self, bucket: str, max_images: int = 50) -> List[Tuple[str, np.ndarray]]:
        """Obtener imágenes desde la API/MinIO"""
        print(f"📥 Obteniendo imágenes del bucket '{bucket}'...")
        
        try:
            # Listar archivos en el bucket
            response = requests.get(f"{self.api_url}/files/list/{bucket}")
            if response.status_code != 200:
                print(f"❌ Error obteniendo lista de archivos: {response.status_code}")
                return []
            
            files_data = response.json()
            files = files_data.get('files', [])[:max_images]
            
            images = []
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Descargar imagen
                        img_response = requests.get(f"{self.api_url}/files/view/{bucket}/{filename}")
                        if img_response.status_code == 200:
                            # Convertir bytes a imagen
                            img_array = np.frombuffer(img_response.content, np.uint8)
                            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            
                            if image is not None:
                                images.append((filename, image))
                                print(f"✅ Imagen cargada: {filename}")
                            else:
                                print(f"⚠️ Error decodificando: {filename}")
                        else:
                            print(f"⚠️ Error descargando {filename}: {img_response.status_code}")
                    except Exception as e:
                        print(f"⚠️ Error procesando {filename}: {e}")
            
            print(f"📊 Total de imágenes cargadas: {len(images)}")
            return images
            
        except Exception as e:
            print(f"❌ Error conectando con la API: {e}")
            return []
    
    def evaluate_preprocessing_quality(self, original: np.ndarray, processed: np.ndarray) -> Dict:
        """Evaluar calidad del preprocesamiento"""
        
        def calculate_sharpness(img):
            """Calcular nitidez usando Laplaciano"""
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        
        def calculate_contrast(img):
            """Calcular contraste"""
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            return gray.std()
        
        def calculate_text_clarity(img):
            """Estimar claridad del texto usando gradientes"""
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Detectar bordes
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            return edge_density
        
        # Calcular métricas
        original_sharpness = calculate_sharpness(original)
        processed_sharpness = calculate_sharpness(processed)
        
        original_contrast = calculate_contrast(original)
        processed_contrast = calculate_contrast(processed)
        
        original_clarity = calculate_text_clarity(original)
        processed_clarity = calculate_text_clarity(processed)
        
        return {
            "sharpness_improvement": processed_sharpness / max(original_sharpness, 1e-6),
            "contrast_improvement": processed_contrast / max(original_contrast, 1e-6),
            "text_clarity_improvement": processed_clarity / max(original_clarity, 1e-6),
            "original_metrics": {
                "sharpness": original_sharpness,
                "contrast": original_contrast,
                "text_clarity": original_clarity
            },
            "processed_metrics": {
                "sharpness": processed_sharpness,
                "contrast": processed_contrast,
                "text_clarity": processed_clarity
            }
        }
    
    def run_evaluation(self, max_images_per_bucket: int = 20):
        """Ejecutar evaluación completa de Capa 1"""
        print("🔧 EVALUACIÓN DE CAPA 1: Pipeline de Preprocesamiento")
        print("=" * 60)
        
        # Obtener imágenes degradadas para probar
        degraded_images = self.fetch_images_from_api(MINIO_BUCKETS['degraded'], max_images_per_bucket)
        
        if not degraded_images:
            print("❌ No se encontraron imágenes degradadas")
            return
        
        # Procesar cada imagen
        results = []
        output_dir = Path("outputs/layer1_evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (filename, image) in enumerate(degraded_images):
            print(f"\n📸 Procesando {filename} ({i+1}/{len(degraded_images)})")
            print("-" * 40)
            
            # Aplicar pipeline de preprocesamiento
            processed_image, processing_info = self.pipeline.process_document(image)
            
            # Evaluar calidad
            quality_metrics = self.evaluate_preprocessing_quality(image, processed_image)
            
            # Guardar resultados
            result = {
                "filename": filename,
                "processing_info": processing_info,
                "quality_metrics": quality_metrics,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            # Crear comparación visual
            self.create_comparison_plot(image, processed_image, processing_info, 
                                      output_dir / f"comparison_{i+1:03d}_{Path(filename).stem}.png")
            
            # Mostrar métricas
            print(f"📊 Mejoras:")
            print(f"   🔍 Nitidez: {quality_metrics['sharpness_improvement']:.2f}x")
            print(f"   🎨 Contraste: {quality_metrics['contrast_improvement']:.2f}x")
            print(f"   📝 Claridad texto: {quality_metrics['text_clarity_improvement']:.2f}x")
        
        # Guardar reporte completo
        self.save_evaluation_report(results, output_dir)
        
        # Crear resumen de resultados
        self.create_summary_report(results, output_dir)
        
        print(f"\n🎉 Evaluación completada. Resultados en: {output_dir}")
    
    def create_comparison_plot(self, original: np.ndarray, processed: np.ndarray, 
                              processing_info: Dict, save_path: Path):
        """Crear gráfico de comparación"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Imagen original
        if len(original.shape) == 3:
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        else:
            original_rgb = original
        
        axes[0].imshow(original_rgb, cmap='gray' if len(original.shape) == 2 else None)
        axes[0].set_title('Original', fontsize=14)
        axes[0].axis('off')
        
        # Imagen procesada
        if len(processed.shape) == 3:
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        else:
            processed_rgb = processed
        
        axes[1].imshow(processed_rgb, cmap='gray' if len(processed.shape) == 2 else None)
        
        # Título con procesamiento aplicado
        title_parts = ["Procesado"]
        if processing_info["otsu_applied"]:
            title_parts.append("Otsu")
        if processing_info["clahe_applied"]:
            title_parts.append("CLAHE")
        if processing_info["deskew_applied"]:
            title_parts.append(f"Deskew({processing_info['skew_angle']:.1f}°)")
        
        axes[1].set_title(" + ".join(title_parts), fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_evaluation_report(self, results: List[Dict], output_dir: Path):
        """Guardar reporte detallado en JSON"""
        report_path = output_dir / "evaluation_report.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                "evaluation_date": datetime.now().isoformat(),
                "total_images": len(results),
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Reporte detallado guardado: {report_path}")
    
    def create_summary_report(self, results: List[Dict], output_dir: Path):
        """Crear reporte resumen con estadísticas"""
        
        if not results:
            return
        
        # Calcular estadísticas promedio
        sharpness_improvements = [r["quality_metrics"]["sharpness_improvement"] for r in results]
        contrast_improvements = [r["quality_metrics"]["contrast_improvement"] for r in results]
        clarity_improvements = [r["quality_metrics"]["text_clarity_improvement"] for r in results]
        
        avg_sharpness = np.mean(sharpness_improvements)
        avg_contrast = np.mean(contrast_improvements)
        avg_clarity = np.mean(clarity_improvements)
        
        # Contar procesamientos aplicados
        otsu_count = sum(1 for r in results if r["processing_info"]["otsu_applied"])
        clahe_count = sum(1 for r in results if r["processing_info"]["clahe_applied"])
        deskew_count = sum(1 for r in results if r["processing_info"]["deskew_applied"])
        
        # Crear gráfico resumen
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Gráfico de mejoras promedio
        improvements = [avg_sharpness, avg_contrast, avg_clarity]
        labels = ['Nitidez', 'Contraste', 'Claridad Texto']
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        bars = ax1.bar(labels, improvements, color=colors)
        ax1.set_title('Mejoras Promedio por Métrica')
        ax1.set_ylabel('Factor de Mejora')
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Sin mejora')
        ax1.legend()
        
        # Agregar valores en las barras
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{improvement:.2f}x', ha='center', va='bottom')
        
        # Gráfico de procesamientos aplicados
        processing_counts = [otsu_count, clahe_count, deskew_count]
        processing_labels = ['Otsu', 'CLAHE', 'Deskew']
        
        ax2.bar(processing_labels, processing_counts, color=['orange', 'purple', 'brown'])
        ax2.set_title('Procesamientos Aplicados')
        ax2.set_ylabel('Número de Imágenes')
        
        # Distribución de mejoras de nitidez
        ax3.hist(sharpness_improvements, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Distribución de Mejoras de Nitidez')
        ax3.set_xlabel('Factor de Mejora')
        ax3.set_ylabel('Frecuencia')
        ax3.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
        
        # Distribución de mejoras de contraste
        ax4.hist(contrast_improvements, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.set_title('Distribución de Mejoras de Contraste')
        ax4.set_xlabel('Factor de Mejora')
        ax4.set_ylabel('Frecuencia')
        ax4.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        summary_path = output_dir / "summary_report.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Crear reporte de texto
        text_report = f"""
REPORTE DE EVALUACIÓN - CAPA 1: Pipeline de Preprocesamiento
================================================================

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total de imágenes procesadas: {len(results)}

MEJORAS PROMEDIO:
- Nitidez: {avg_sharpness:.2f}x
- Contraste: {avg_contrast:.2f}x
- Claridad de texto: {avg_clarity:.2f}x

PROCESAMIENTOS APLICADOS:
- Otsu: {otsu_count}/{len(results)} imágenes ({otsu_count/len(results)*100:.1f}%)
- CLAHE: {clahe_count}/{len(results)} imágenes ({clahe_count/len(results)*100:.1f}%)
- Deskew: {deskew_count}/{len(results)} imágenes ({deskew_count/len(results)*100:.1f}%)

ANÁLISIS:
"""
        
        if avg_sharpness > 1.1:
            text_report += "✅ El pipeline mejora significativamente la nitidez\n"
        elif avg_sharpness > 1.0:
            text_report += "⚠️ El pipeline mejora ligeramente la nitidez\n"
        else:
            text_report += "❌ El pipeline no mejora la nitidez\n"
        
        if avg_contrast > 1.1:
            text_report += "✅ El pipeline mejora significativamente el contraste\n"
        elif avg_contrast > 1.0:
            text_report += "⚠️ El pipeline mejora ligeramente el contraste\n"
        else:
            text_report += "❌ El pipeline no mejora el contraste\n"
        
        if avg_clarity > 1.1:
            text_report += "✅ El pipeline mejora significativamente la claridad del texto\n"
        elif avg_clarity > 1.0:
            text_report += "⚠️ El pipeline mejora ligeramente la claridad del texto\n"
        else:
            text_report += "❌ El pipeline no mejora la claridad del texto\n"
        
        # Guardar reporte de texto
        text_path = output_dir / "summary_report.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        print(f"📊 Reporte resumen guardado: {summary_path}")
        print(f"📄 Reporte de texto guardado: {text_path}")
        
        # Mostrar resumen en consola
        print("\n" + "="*60)
        print("📊 RESUMEN DE RESULTADOS")
        print("="*60)
        print(f"📈 Mejora promedio de nitidez: {avg_sharpness:.2f}x")
        print(f"📈 Mejora promedio de contraste: {avg_contrast:.2f}x")
        print(f"📈 Mejora promedio de claridad: {avg_clarity:.2f}x")
        print(f"🔧 Deskew aplicado en: {deskew_count} imágenes")
        print("="*60)

def main():
    """Función principal"""
    print("🔧 ENTRENAMIENTO/EVALUACIÓN CAPA 1")
    print("==================================")
    print("Pipeline: Otsu + CLAHE + Deskew por Hough")
    print()
    
    # Verificar conexión con API
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API conectada exitosamente")
        else:
            print(f"⚠️ API respondió con código: {response.status_code}")
    except Exception as e:
        print(f"❌ Error conectando con API: {e}")
        print("Asegúrate de que la API esté ejecutándose en localhost:8000")
        return
    
    # Crear entrenador/evaluador
    trainer = Layer1Trainer()
    
    # Ejecutar evaluación
    trainer.run_evaluation(max_images_per_bucket=30)

if __name__ == "__main__":
    main()
