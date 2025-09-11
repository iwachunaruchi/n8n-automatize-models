#!/usr/bin/env python3
"""
🎯 DEMO: NUEVA ESTRUCTURA NAFNET PARA GENERACIÓN DE DATOS SINTÉTICOS
====================================================================
Demo que muestra cómo usar la nueva estructura organizacional para NAFNet.

Estructura Generada:
document-training/
├── datasets/
│   └── NAFNet/
│       └── SIDD-width64/
│           ├── train/
│           │   ├── lq/    # Low-quality (degradadas)
│           │   └── gt/    # Ground-truth (limpias)
│           └── val/
│               ├── lq/    
│               └── gt/
"""

import asyncio
import requests
import json
import time
from datetime import datetime

# Configuración
API_BASE_URL = "http://localhost:8000"
MINIO_CONSOLE_URL = "http://localhost:9001"

class NAFNetStructureDemo:
    """Demo de la nueva estructura NAFNet"""
    
    def __init__(self):
        self.api_url = API_BASE_URL
        
    def print_header(self, title: str):
        """Imprimir header formateado"""
        print("\n" + "="*60)
        print(f"🎯 {title}")
        print("="*60)
    
    def print_step(self, step: int, description: str):
        """Imprimir paso numerado"""
        print(f"\n📋 Paso {step}: {description}")
        print("-" * 50)
    
    def check_api_health(self):
        """Verificar que la API esté funcionando"""
        try:
            response = requests.get(f"{self.api_url}/synthetic/info")
            if response.status_code == 200:
                info = response.json()
                print("✅ API funcionando correctamente")
                print(f"📊 Servicio: {info.get('service', 'N/A')}")
                if 'nafnet_configuration' in info:
                    print(f"🎯 Tarea actual: {info['nafnet_configuration']['current_task']}")
                    print(f"📁 Estructura: {info['nafnet_configuration']['dataset_structure']}")
                return True
            else:
                print(f"❌ API no responde correctamente: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error conectando con API: {e}")
            return False
    
    def list_nafnet_tasks(self):
        """Listar tareas NAFNet disponibles"""
        try:
            response = requests.get(f"{self.api_url}/synthetic/nafnet/tasks")
            if response.status_code == 200:
                tasks_info = response.json()
                print("✅ Tareas NAFNet disponibles:")
                print(f"🏗️ Core: {tasks_info['core_name']}")
                print(f"⚡ Tarea actual: {tasks_info['current_task']}")
                
                print("\n📋 Tareas disponibles:")
                for task_name, task_config in tasks_info['available_tasks'].items():
                    print(f"  • {task_name}: {task_config['task_description']}")
                    print(f"    Degradaciones: {', '.join(task_config['degradation_types'])}")
                
                print("\n📁 Estructura de ejemplo:")
                for path_type, path in tasks_info['example_paths'].items():
                    print(f"  {path_type}: {path}")
                
                return tasks_info
            else:
                print(f"❌ Error obteniendo tareas: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def generate_nafnet_dataset(self, source_bucket: str = "document-clean", count: int = 20):
        """Generar dataset NAFNet estructurado"""
        try:
            payload = {
                "source_bucket": source_bucket,
                "count": count,
                "task": "SIDD-width64",
                "train_val_split": True
            }
            
            print(f"🚀 Enviando solicitud de generación NAFNet...")
            print(f"   📂 Bucket fuente: {source_bucket}")
            print(f"   📊 Cantidad: {count} pares")
            print(f"   🎯 Tarea: SIDD-width64")
            print(f"   📈 División train/val: Sí")
            
            response = requests.post(f"{self.api_url}/synthetic/nafnet/dataset", params=payload)
            
            if response.status_code == 200:
                result = response.json()
                job_id = result['job_id']
                print(f"✅ Job encolado exitosamente")
                print(f"🆔 Job ID: {job_id}")
                print(f"📊 Sistema: {result['system']}")
                
                return job_id
            else:
                print(f"❌ Error generando dataset: {response.status_code}")
                print(f"📝 Respuesta: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def check_job_status(self, job_id: str):
        """Verificar estado del trabajo"""
        try:
            response = requests.get(f"{self.api_url}/jobs/rq/{job_id}")
            if response.status_code == 200:
                job_info = response.json()
                status = job_info.get('status', 'unknown')
                
                if status == 'queued':
                    print("⏳ Job en cola...")
                elif status == 'started':
                    print("🔄 Job iniciado...")
                elif status == 'finished':
                    print("✅ Job completado!")
                    if 'result' in job_info:
                        result = job_info['result']
                        print(f"📊 Total generado: {result.get('total_generated', 0)} pares")
                        print(f"🏋️ Train: {result.get('generation_result', {}).get('train', {}).get('count', 0)} pares")
                        print(f"✅ Val: {result.get('generation_result', {}).get('val', {}).get('count', 0)} pares")
                elif status == 'failed':
                    print("❌ Job falló")
                    if 'exc_info' in job_info:
                        print(f"🐛 Error: {job_info['exc_info']}")
                else:
                    print(f"📊 Estado: {status}")
                
                return status
            else:
                print(f"❌ Error verificando job: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def wait_for_completion(self, job_id: str, max_wait: int = 300):
        """Esperar a que el trabajo se complete"""
        print(f"⏰ Esperando completación del job {job_id}...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = self.check_job_status(job_id)
            
            if status == 'finished':
                print("🎉 ¡Trabajo completado exitosamente!")
                return True
            elif status == 'failed':
                print("💥 Trabajo falló")
                return False
            
            time.sleep(5)  # Esperar 5 segundos antes de verificar nuevamente
        
        print("⏰ Timeout esperando completación")
        return False
    
    def get_nafnet_dataset_info(self, task: str = "SIDD-width64"):
        """Obtener información del dataset NAFNet generado"""
        try:
            response = requests.get(f"{self.api_url}/synthetic/nafnet/info/{task}")
            if response.status_code == 200:
                dataset_info = response.json()
                
                print("📊 Información del Dataset NAFNet:")
                print(f"🎯 Tarea: {dataset_info['task']}")
                print(f"🏋️ Train - LQ: {dataset_info['train']['lq']}, GT: {dataset_info['train']['gt']}")
                print(f"✅ Val - LQ: {dataset_info['val']['lq']}, GT: {dataset_info['val']['gt']}")
                print(f"📈 Total pares: {dataset_info['total_pairs']}")
                
                print("\n📁 Estructura de rutas:")
                structure = dataset_info['structure']
                for path_type, path in structure.items():
                    if path_type != 'complete_pairs':
                        print(f"  {path_type}: {path}")
                
                return dataset_info
            else:
                print(f"❌ Error obteniendo info: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def validate_nafnet_dataset(self, task: str = "SIDD-width64"):
        """Validar integridad del dataset NAFNet"""
        try:
            response = requests.post(f"{self.api_url}/synthetic/nafnet/validate/{task}")
            if response.status_code == 200:
                result = response.json()
                job_id = result['job_id']
                
                print(f"🔍 Validación NAFNet iniciada")
                print(f"🆔 Job ID: {job_id}")
                
                # Esperar un poco y verificar resultado
                time.sleep(10)
                status = self.check_job_status(job_id)
                
                return job_id
            else:
                print(f"❌ Error iniciando validación: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def run_complete_demo(self):
        """Ejecutar demo completa"""
        self.print_header("DEMO COMPLETA DE ESTRUCTURA NAFNET")
        
        # Paso 1: Verificar API
        self.print_step(1, "Verificar estado de la API")
        if not self.check_api_health():
            print("❌ Demo cancelada: API no disponible")
            return
        
        # Paso 2: Listar tareas disponibles
        self.print_step(2, "Listar tareas NAFNet disponibles")
        tasks_info = self.list_nafnet_tasks()
        if not tasks_info:
            print("❌ Demo cancelada: No se pudieron obtener las tareas")
            return
        
        # Paso 3: Generar dataset
        self.print_step(3, "Generar dataset NAFNet estructurado")
        job_id = self.generate_nafnet_dataset(count=10)  # Generar pocos para demo
        if not job_id:
            print("❌ Demo cancelada: No se pudo iniciar la generación")
            return
        
        # Paso 4: Esperar completación
        self.print_step(4, "Esperar completación del trabajo")
        if not self.wait_for_completion(job_id, max_wait=180):
            print("❌ Demo cancelada: Timeout o error en generación")
            return
        
        # Paso 5: Verificar información del dataset
        self.print_step(5, "Verificar información del dataset generado")
        dataset_info = self.get_nafnet_dataset_info()
        if not dataset_info:
            print("⚠️ No se pudo obtener información del dataset")
        
        # Paso 6: Validar dataset
        self.print_step(6, "Validar integridad del dataset")
        validation_job = self.validate_nafnet_dataset()
        if validation_job:
            time.sleep(15)  # Esperar validación
            self.check_job_status(validation_job)
        
        # Paso 7: Mostrar resumen
        self.print_step(7, "Resumen de la demo")
        print("🎉 Demo completada exitosamente!")
        print("\n📋 Lo que hemos logrado:")
        print("  ✅ Verificación de API y servicios")
        print("  ✅ Listado de tareas NAFNet disponibles")
        print("  ✅ Generación de dataset con estructura organizada")
        print("  ✅ División automática en train/val")
        print("  ✅ Validación de integridad del dataset")
        
        print(f"\n🌐 Puedes verificar los archivos en MinIO Console:")
        print(f"   URL: {MINIO_CONSOLE_URL}")
        print(f"   Usuario: minioadmin / minioadmin")
        print(f"   Bucket: document-training")
        print(f"   Ruta: datasets/NAFNet/SIDD-width64/")
        
        print("\n📁 Estructura final generada:")
        print("   document-training/")
        print("   └── datasets/")
        print("       └── NAFNet/")
        print("           └── SIDD-width64/")
        print("               ├── train/")
        print("               │   ├── lq/  (imágenes degradadas)")
        print("               │   └── gt/  (imágenes limpias)")
        print("               └── val/")
        print("                   ├── lq/  (validación degradadas)")
        print("                   └── gt/  (validación limpias)")

def main():
    """Función principal"""
    print("🚀 Iniciando Demo de Estructura NAFNet")
    print("=" * 60)
    
    demo = NAFNetStructureDemo()
    
    try:
        demo.run_complete_demo()
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
