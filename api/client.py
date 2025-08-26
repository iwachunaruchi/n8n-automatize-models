#!/usr/bin/env python3
"""
Cliente para la API de Restauración de Documentos
Ejemplos de uso con MinIO y n8n
"""

import requests
import os
import json
import time
from typing import List, Dict, Any

class DocumentRestorationClient:
    """Cliente para interactuar with la API"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Verificar estado de la API"""
        response = self.session.get(f"{self.api_url}/health")
        response.raise_for_status()
        return response.json()
    
    def restore_single_image(self, image_path: str, output_path: str = None) -> str:
        """Restaurar una sola imagen"""
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/png')}
            response = self.session.post(f"{self.api_url}/restore/single", files=files)
            response.raise_for_status()
        
        # Guardar resultado
        if output_path is None:
            output_path = f"restored_{os.path.basename(image_path)}"
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return output_path
    
    def restore_from_minio(self, bucket_name: str, object_name: str, 
                          output_bucket: str = "document-restored") -> Dict[str, Any]:
        """Restaurar imagen desde MinIO"""
        data = {
            "bucket_name": bucket_name,
            "object_name": object_name,
            "output_bucket": output_bucket
        }
        
        response = self.session.post(f"{self.api_url}/restore/from-minio", params=data)
        response.raise_for_status()
        return response.json()
    
    def start_batch_processing(self, bucket_name: str, 
                             file_patterns: List[str] = None,
                             output_bucket: str = "document-restored") -> str:
        """Iniciar procesamiento en lote"""
        if file_patterns is None:
            file_patterns = ["*.png", "*.jpg", "*.jpeg"]
        
        data = {
            "bucket_name": bucket_name,
            "file_patterns": file_patterns,
            "output_bucket": output_bucket
        }
        
        response = self.session.post(f"{self.api_url}/restore/batch", json=data)
        response.raise_for_status()
        result = response.json()
        return result['job_id']
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Obtener estado de trabajo"""
        response = self.session.get(f"{self.api_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_job(self, job_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Esperar a que termine un trabajo"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)
            
            if status['status'] in ['completed', 'failed']:
                return status
            
            print(f"Job {job_id}: {status['status']}")
            time.sleep(5)
        
        raise TimeoutError(f"Job {job_id} no completó en {timeout} segundos")
    
    def list_bucket_contents(self, bucket_name: str) -> Dict[str, Any]:
        """Listar contenido de bucket"""
        data = {"bucket_name": bucket_name}
        response = self.session.post(f"{self.api_url}/buckets/list", json=data)
        response.raise_for_status()
        return response.json()
    
    def start_training(self, clean_bucket: str = "document-clean",
                      degraded_bucket: str = "document-degraded",
                      epochs: int = 10, batch_size: int = 2) -> str:
        """Iniciar entrenamiento"""
        data = {
            "clean_bucket": clean_bucket,
            "degraded_bucket": degraded_bucket,
            "epochs": epochs,
            "batch_size": batch_size
        }
        
        response = self.session.post(f"{self.api_url}/training/start", json=data)
        response.raise_for_status()
        result = response.json()
        return result['job_id']

# Ejemplos de uso
def example_single_image():
    """Ejemplo: restaurar una imagen local"""
    client = DocumentRestorationClient()
    
    # Verificar API
    health = client.health_check()
    print(f"API Status: {health['status']}")
    
    # Restaurar imagen
    if os.path.exists("test_image.png"):
        output = client.restore_single_image("test_image.png")
        print(f"Imagen restaurada guardada en: {output}")

def example_minio_workflow():
    """Ejemplo: workflow completo con MinIO"""
    client = DocumentRestorationClient()
    
    # 1. Listar imágenes degradadas
    degraded_contents = client.list_bucket_contents("document-degraded")
    print(f"Imágenes degradadas encontradas: {len(degraded_contents['objects'])}")
    
    # 2. Procesar en lote
    job_id = client.start_batch_processing("document-degraded")
    print(f"Procesamiento iniciado: {job_id}")
    
    # 3. Esperar resultado
    result = client.wait_for_job(job_id)
    print(f"Procesamiento completado: {result}")

def example_n8n_integration():
    """Ejemplo: integración con n8n"""
    
    # Este sería el payload típico que n8n enviaría
    n8n_payload = {
        "bucket_name": "document-degraded",
        "new_files": [
            "documento_001.png",
            "documento_002.jpg"
        ],
        "webhook_url": "http://localhost:5678/webhook/processing-complete"
    }
    
    client = DocumentRestorationClient()
    
    # Procesar cada archivo nuevo
    for file_name in n8n_payload["new_files"]:
        try:
            result = client.restore_from_minio(
                bucket_name=n8n_payload["bucket_name"],
                object_name=file_name
            )
            print(f"Procesado: {file_name} -> {result['output_file']}")
            
            # n8n recibiría este resultado via webhook
            
        except Exception as e:
            print(f"Error procesando {file_name}: {e}")

if __name__ == "__main__":
    print("🚀 Cliente API de Restauración de Documentos")
    print("=" * 50)
    
    # Ejecutar ejemplos
    try:
        print("\n1. Verificando API...")
        client = DocumentRestorationClient()
        health = client.health_check()
        print(f"✅ API activa: {health}")
        
        print("\n2. Ejemplo workflow MinIO...")
        # example_minio_workflow()  # Comentado para evitar errores sin MinIO
        
        print("\n✅ Ejemplos completados")
        
    except requests.exceptions.ConnectionError:
        print("❌ API no disponible. Ejecutar: python api/main.py")
    except Exception as e:
        print(f"❌ Error: {e}")
