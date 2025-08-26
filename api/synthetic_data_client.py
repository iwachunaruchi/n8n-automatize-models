#!/usr/bin/env python3
"""
Cliente para el Workflow de Generación de Datos Sintéticos
Testing del nuevo workflow n8n + API
"""

import requests
import os
import json
import time
from pathlib import Path

class SyntheticDataClient:
    """Cliente para interactuar con el workflow de datos sintéticos"""
    
    def __init__(self, 
                 api_url: str = "http://localhost:8000",
                 n8n_url: str = "http://localhost:5678"):
        self.api_url = api_url.rstrip('/')
        self.n8n_url = n8n_url.rstrip('/')
        self.session = requests.Session()
    
    def classify_image_quality(self, image_path: str):
        """Clasificar calidad de imagen via API directa"""
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/png')}
            response = self.session.post(f"{self.api_url}/classify/image-quality", files=files)
            response.raise_for_status()
            return response.json()
    
    def upload_for_synthetic_generation(self, image_path: str):
        """Subir imagen al workflow de generación sintética via n8n"""
        webhook_url = f"{self.n8n_url}/webhook/data-generation"
        
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/png')}
            response = self.session.post(webhook_url, files=files)
            response.raise_for_status()
            return response.json()
    
    def get_dataset_stats(self):
        """Obtener estadísticas del dataset"""
        response = self.session.get(f"{self.api_url}/dataset/stats?include_new=true")
        response.raise_for_status()
        return response.json()
    
    def start_synthetic_generation(self, 
                                  source_bucket: str,
                                  source_file: str,
                                  target_count: int = 10,
                                  generation_type: str = "degradation"):
        """Iniciar generación sintética directamente via API"""
        data = {
            "source_bucket": source_bucket,
            "source_file": source_file,
            "target_count": target_count,
            "generation_type": generation_type,
            "output_bucket": "document-training"
        }
        
        response = self.session.post(f"{self.api_url}/generate/synthetic-data", json=data)
        response.raise_for_status()
        return response.json()
    
    def monitor_generation_job(self, job_id: str, timeout: int = 300):
        """Monitorear progreso de generación sintética"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.session.get(f"{self.api_url}/jobs/{job_id}")
            response.raise_for_status()
            status = response.json()
            
            print(f"Job {job_id}: {status['status']}")
            
            if status['status'] in ['completed', 'failed']:
                return status
            
            time.sleep(5)
        
        raise TimeoutError(f"Job {job_id} no completó en {timeout} segundos")

def demo_image_classification():
    """Demo: Clasificación automática de imágenes"""
    print("🔍 DEMO: Clasificación Automática de Imágenes")
    print("=" * 50)
    
    client = SyntheticDataClient()
    
    # Buscar imágenes de ejemplo
    test_images = []
    data_dirs = ["data/train/clean", "data/train/degraded"]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir)[:2]:  # Solo 2 por directorio
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    test_images.append(os.path.join(data_dir, file))
    
    if not test_images:
        print("❌ No se encontraron imágenes de prueba")
        return
    
    for image_path in test_images:
        try:
            print(f"\n📸 Analizando: {os.path.basename(image_path)}")
            result = client.classify_image_quality(image_path)
            
            print(f"  🏷️  Clasificación: {result['classification']}")
            print(f"  📊 Confianza: {result['confidence']:.2f}")
            print(f"  📈 Score de calidad: {result['metrics']['quality_score']:.1f}")
            print(f"  🔍 Sharpness: {result['metrics']['sharpness']:.1f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

def demo_workflow_complete():
    """Demo: Workflow completo de generación sintética"""
    print("\n🔄 DEMO: Workflow Completo de Generación Sintética")
    print("=" * 60)
    
    client = SyntheticDataClient()
    
    # Buscar una imagen para probar
    test_image = None
    for data_dir in ["data/train/clean", "data/train/degraded"]:
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    test_image = os.path.join(data_dir, file)
                    break
            if test_image:
                break
    
    if not test_image:
        print("❌ No se encontraron imágenes de prueba")
        return
    
    try:
        print(f"📤 Subiendo imagen al workflow: {os.path.basename(test_image)}")
        
        # Enviar al workflow de n8n
        result = client.upload_for_synthetic_generation(test_image)
        print(f"✅ Workflow iniciado: {result}")
        
    except requests.exceptions.ConnectionError:
        print("❌ n8n no disponible. Probando API directa...")
        
        # Fallback: usar API directa
        try:
            # 1. Clasificar imagen
            classification = client.classify_image_quality(test_image)
            print(f"🏷️  Clasificación: {classification['classification']}")
            
            # 2. Simular carga a bucket apropiado
            bucket = "document-clean" if classification['classification'] == 'clean' else "document-degraded"
            filename = os.path.basename(test_image)
            
            # 3. Iniciar generación sintética
            generation_result = client.start_synthetic_generation(
                source_bucket=bucket,
                source_file=filename,
                target_count=5,  # Reducido para demo
                generation_type="degradation" if classification['classification'] == 'clean' else "variation"
            )
            
            print(f"🔄 Generación iniciada: {generation_result['job_id']}")
            
            # 4. Monitorear progreso
            final_status = client.monitor_generation_job(generation_result['job_id'])
            print(f"✅ Generación completada: {final_status}")
            
        except Exception as e:
            print(f"❌ Error en API directa: {e}")

def demo_dataset_stats():
    """Demo: Estadísticas del dataset"""
    print("\n📊 DEMO: Estadísticas del Dataset")
    print("=" * 40)
    
    client = SyntheticDataClient()
    
    try:
        stats = client.get_dataset_stats()
        
        print(f"📈 Total de muestras: {stats['total_samples']}")
        print(f"⏰ Última actualización: {stats['timestamp']}")
        print("")
        
        for bucket, data in stats['buckets'].items():
            if 'error' not in data:
                print(f"🗂️  {bucket}:")
                print(f"   📁 Archivos: {data['count']}")
                print(f"   💾 Tamaño: {data['total_size_mb']:.1f} MB")
            else:
                print(f"❌ {bucket}: {data['error']}")
        
    except requests.exceptions.ConnectionError:
        print("❌ API no disponible")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Función principal de demostración"""
    print("🎯 CLIENTE WORKFLOW GENERACIÓN DATOS SINTÉTICOS")
    print("=" * 60)
    print("🌐 API: http://localhost:8000")
    print("🔄 n8n: http://localhost:5678")
    print("📁 MinIO: http://localhost:9001")
    print("")
    
    # Verificar servicios
    client = SyntheticDataClient()
    
    try:
        health = requests.get(f"{client.api_url}/health", timeout=5)
        print("✅ API disponible")
    except:
        print("❌ API no disponible")
    
    try:
        n8n_health = requests.get(f"{client.n8n_url}/healthz", timeout=5)
        print("✅ n8n disponible")
    except:
        print("❌ n8n no disponible")
    
    print("")
    
    # Ejecutar demos
    demo_dataset_stats()
    demo_image_classification()
    # demo_workflow_complete()  # Comentado para evitar errores sin servicios
    
    print(f"\n🎊 DEMOS COMPLETADOS")
    print("=" * 30)
    print("📋 Para usar el workflow completo:")
    print("1. 🐳 Iniciar servicios: docker-compose up -d")
    print("2. 🚀 Iniciar API: python api/main.py")
    print("3. 🔄 Importar workflow en n8n: synthetic-data-generation.json")
    print("4. 📤 Enviar imagen a: http://localhost:5678/webhook/data-generation")

if __name__ == "__main__":
    main()
