"""
Demo completo para entrenar capas via API
Simula el flujo de trabajo que n8n seguiría
"""

import requests
import time
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Verificar que la API esté funcionando"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API está funcionando")
            return True
        else:
            print(f"⚠️ API respondió con código: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error conectando con API: {e}")
        return False

def list_available_images():
    """Mostrar imágenes disponibles para entrenamiento"""
    print("\n📋 IMÁGENES DISPONIBLES PARA ENTRENAMIENTO")
    print("=" * 50)
    
    buckets = ['document-degraded', 'document-clean']
    
    for bucket in buckets:
        try:
            response = requests.get(f"{API_BASE_URL}/files/list/{bucket}")
            if response.status_code == 200:
                data = response.json()
                files = data.get('files', [])
                print(f"\n📁 {bucket}: {len(files)} archivos")
                
                # Mostrar algunos ejemplos
                for i, file in enumerate(files[:5]):
                    print(f"   📄 {file}")
                if len(files) > 5:
                    print(f"   ... y {len(files) - 5} más")
            else:
                print(f"❌ Error listando {bucket}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error accediendo a {bucket}: {e}")

def demo_layer1_evaluation():
    """Demostrar evaluación de Capa 1"""
    print("\n🔧 DEMO: EVALUACIÓN DE CAPA 1")
    print("=" * 50)
    
    # Iniciar evaluación
    try:
        response = requests.post(f"{API_BASE_URL}/training/layer1/evaluate", 
                               params={"max_images": 10})
        
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data["job_id"]
            print(f"✅ Evaluación iniciada. Job ID: {job_id}")
            
            # Monitorear progreso
            monitor_job_progress(job_id, "Evaluación Capa 1")
            
        else:
            print(f"❌ Error iniciando evaluación: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_layer2_training():
    """Demostrar entrenamiento de Capa 2"""
    print("\n🔧 DEMO: ENTRENAMIENTO DE CAPA 2")
    print("=" * 50)
    
    # Iniciar entrenamiento
    try:
        params = {
            "num_epochs": 5,
            "max_pairs": 20,
            "batch_size": 2
        }
        
        response = requests.post(f"{API_BASE_URL}/training/layer2/train", 
                               params=params)
        
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data["job_id"]
            print(f"✅ Entrenamiento iniciado. Job ID: {job_id}")
            print(f"📊 Parámetros: {params}")
            
            # Monitorear progreso
            monitor_job_progress(job_id, "Entrenamiento Capa 2")
            
        else:
            print(f"❌ Error iniciando entrenamiento: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def monitor_job_progress(job_id: str, job_name: str):
    """Monitorear progreso de un trabajo"""
    print(f"\n📊 Monitoreando progreso de: {job_name}")
    print(f"🔍 Job ID: {job_id}")
    
    start_time = time.time()
    
    while True:
        try:
            response = requests.get(f"{API_BASE_URL}/training/status/{job_id}")
            
            if response.status_code == 200:
                status_data = response.json()
                status = status_data["status"]
                progress = status_data["progress"]
                
                elapsed = time.time() - start_time
                
                print(f"\r⏱️ {elapsed:.0f}s | Estado: {status} | Progreso: {progress}%", end="")
                
                if status == "completed":
                    print(f"\n✅ {job_name} completado exitosamente!")
                    
                    # Mostrar resultados
                    show_job_results(job_id)
                    break
                    
                elif status == "failed":
                    print(f"\n❌ {job_name} falló")
                    error = status_data.get("error", "Error desconocido")
                    print(f"Error: {error}")
                    break
                    
                elif status == "cancelled":
                    print(f"\n⚠️ {job_name} fue cancelado")
                    break
                
                # Información adicional para entrenamiento
                if "training_info" in status_data:
                    info = status_data["training_info"]
                    current_epoch = info.get("current_epoch", 0)
                    total_epochs = info.get("num_epochs", 0)
                    if total_epochs > 0:
                        print(f" | Época: {current_epoch}/{total_epochs}", end="")
                
            else:
                print(f"\n❌ Error obteniendo estado: {response.status_code}")
                break
                
        except Exception as e:
            print(f"\n❌ Error monitoreando: {e}")
            break
        
        time.sleep(5)  # Esperar 5 segundos antes de la siguiente consulta

def show_job_results(job_id: str):
    """Mostrar resultados de un trabajo"""
    try:
        response = requests.get(f"{API_BASE_URL}/training/results/{job_id}")
        
        if response.status_code == 200:
            results = response.json()
            
            print(f"\n📊 RESULTADOS DEL TRABAJO")
            print("-" * 30)
            print(f"Tipo: {results['type']}")
            print(f"Estado: {results['status']}")
            
            if "results" in results and results["results"]:
                job_results = results["results"]
                print(f"Éxito: {job_results.get('success', 'N/A')}")
                
                if results['type'] == 'layer1_evaluation':
                    print(f"Imágenes procesadas: {job_results.get('max_images_processed', 'N/A')}")
                elif results['type'] == 'layer2_training':
                    print(f"Épocas completadas: {job_results.get('epochs_completed', 'N/A')}")
                    print(f"Pares usados: {job_results.get('pairs_used', 'N/A')}")
                    print(f"Batch size: {job_results.get('batch_size', 'N/A')}")
            
            # Mostrar archivos de salida
            output_files = results.get("output_files", [])
            if output_files:
                print(f"\n📁 Archivos de salida ({len(output_files)}):")
                for file_info in output_files[:5]:  # Mostrar solo los primeros 5
                    size_mb = file_info["size"] / (1024 * 1024)
                    print(f"   📄 {file_info['filename']} ({size_mb:.2f} MB)")
                if len(output_files) > 5:
                    print(f"   ... y {len(output_files) - 5} archivos más")
        else:
            print(f"❌ Error obteniendo resultados: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error mostrando resultados: {e}")

def list_all_training_jobs():
    """Listar todos los trabajos de entrenamiento"""
    print("\n📋 TRABAJOS DE ENTRENAMIENTO")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_BASE_URL}/training/jobs")
        
        if response.status_code == 200:
            data = response.json()
            jobs = data.get("jobs", [])
            
            if not jobs:
                print("No hay trabajos de entrenamiento")
                return
            
            print(f"Total de trabajos: {data['total_jobs']}")
            print()
            
            for job in jobs:
                print(f"🔍 ID: {job['job_id'][:8]}...")
                print(f"   Tipo: {job['type']}")
                print(f"   Estado: {job['status']}")
                print(f"   Progreso: {job['progress']}%")
                print(f"   Inicio: {job['start_time']}")
                print()
        else:
            print(f"❌ Error listando trabajos: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def show_training_info():
    """Mostrar información sobre las capacidades de entrenamiento"""
    print("\n📋 INFORMACIÓN DE ENTRENAMIENTO")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_BASE_URL}/training/info")
        
        if response.status_code == 200:
            info = response.json()
            
            print("🔧 CAPAS DISPONIBLES:")
            for layer_id, layer_info in info["available_layers"].items():
                print(f"\n📌 {layer_id.upper()}")
                print(f"   Nombre: {layer_info['name']}")
                print(f"   Descripción: {layer_info['description']}")
                print(f"   Tipo: {layer_info['type']}")
                print(f"   Requiere GPU: {layer_info['requires_gpu']}")
                print(f"   Endpoints: {', '.join(layer_info['endpoints'])}")
            
            print(f"\n📊 Fuente de datos: {info['data_source']}")
            print("📁 Buckets utilizados:")
            for bucket in info["buckets_used"]:
                print(f"   - {bucket}")
            
            print(f"\n📁 Ubicación de salida: {info['output_location']}")
            
        else:
            print(f"❌ Error obteniendo información: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Función principal del demo"""
    print("🚀 DEMO COMPLETO: ENTRENAMIENTO DE CAPAS VIA API")
    print("=" * 60)
    print("Este demo simula el flujo que seguiría n8n para entrenar modelos")
    print()
    
    # Verificar API
    if not check_api_health():
        print("❌ No se puede continuar sin conexión a la API")
        return
    
    # Mostrar información del sistema
    show_training_info()
    
    # Mostrar imágenes disponibles
    list_available_images()
    
    # Listar trabajos existentes
    list_all_training_jobs()
    
    # Menú interactivo
    while True:
        print("\n" + "="*60)
        print("🎯 OPCIONES DE DEMO:")
        print("1. Evaluar Capa 1 (Pipeline de preprocesamiento)")
        print("2. Entrenar Capa 2 (NAFNet + DocUNet)")
        print("3. Listar trabajos activos")
        print("4. Mostrar información del sistema")
        print("5. Salir")
        print("="*60)
        
        try:
            choice = input("Selecciona una opción (1-5): ").strip()
            
            if choice == "1":
                demo_layer1_evaluation()
            elif choice == "2":
                demo_layer2_training()
            elif choice == "3":
                list_all_training_jobs()
            elif choice == "4":
                show_training_info()
            elif choice == "5":
                print("👋 ¡Hasta luego!")
                break
            else:
                print("❌ Opción no válida")
                
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
