"""
Workflow completo para n8n: Gestión de datos y entrenamiento de Capa 2
"""

import requests
import time
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

def check_layer2_data_status():
    """Verificar estado de datos para Capa 2"""
    print("\n📊 VERIFICANDO DATOS PARA CAPA 2")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_BASE_URL}/training/layer2/data-status")
        
        if response.status_code == 200:
            data = response.json()
            stats = data["statistics"]
            recs = data["recommendations"]
            
            print(f"📁 Bucket: {data['bucket']}")
            print(f"📊 Archivos totales: {stats['total_files']}")
            print(f"🟢 Archivos limpios: {stats['clean_files']}")
            print(f"🔴 Archivos degradados: {stats['degraded_files']}")
            print(f"✅ Pares válidos: {stats['valid_pairs']}")
            print(f"📄 Otros archivos: {stats['other_files']}")
            
            print(f"\n🎯 RECOMENDACIONES:")
            print(f"   Mínimo requerido: {recs['minimum_pairs']} pares")
            print(f"   Recomendado: {recs['recommended_pairs']} pares")
            print(f"   Estado actual: {recs['current_status']}")
            print(f"   Listo para entrenar: {data['ready_for_training']}")
            
            return data
        else:
            print(f"❌ Error verificando datos: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def prepare_training_data(target_pairs: int = 100):
    """Preparar datos para entrenamiento"""
    print(f"\n🔧 PREPARANDO DATOS PARA ENTRENAMIENTO ({target_pairs} pares)")
    print("=" * 60)
    
    try:
        response = requests.post(f"{API_BASE_URL}/training/layer2/prepare-data",
                               params={
                                   "target_pairs": target_pairs,
                                   "source_bucket": "document-clean"
                               })
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("action") == "none_needed":
                print(f"✅ {data['message']}")
                return True
            else:
                job_id = data["job_id"]
                print(f"🚀 Generando {data['needed_pairs']} pares adicionales...")
                print(f"🔍 Job ID: {job_id}")
                
                # Monitorear progreso
                return monitor_job_progress(job_id, "Preparación de datos")
        else:
            print(f"❌ Error preparando datos: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def train_layer2_with_options():
    """Entrenar Capa 2 con opciones configurables"""
    print("\n🔧 CONFIGURANDO ENTRENAMIENTO DE CAPA 2")
    print("=" * 60)
    
    # Mostrar opciones
    print("📋 OPCIONES DE ENTRENAMIENTO:")
    print("1. Usar bucket 'document-training' (RECOMENDADO - pares sintéticos)")
    print("2. Usar buckets separados 'document-degraded' y 'document-clean'")
    
    choice = input("\nSelecciona opción (1-2): ").strip()
    use_training_bucket = choice == "1"
    
    # Configurar parámetros
    print(f"\n🔧 CONFIGURACIÓN:")
    print(f"   Fuente de datos: {'bucket de entrenamiento' if use_training_bucket else 'buckets separados'}")
    
    epochs = int(input("Número de épocas (10): ") or "10")
    max_pairs = int(input("Máximo de pares (50): ") or "50")
    batch_size = int(input("Batch size (2): ") or "2")
    
    print(f"\n🚀 INICIANDO ENTRENAMIENTO:")
    print(f"   Épocas: {epochs}")
    print(f"   Máx. pares: {max_pairs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Usar bucket entrenamiento: {use_training_bucket}")
    
    try:
        response = requests.post(f"{API_BASE_URL}/training/layer2/train",
                               params={
                                   "num_epochs": epochs,
                                   "max_pairs": max_pairs,
                                   "batch_size": batch_size,
                                   "use_training_bucket": use_training_bucket
                               })
        
        if response.status_code == 200:
            data = response.json()
            job_id = data["job_id"]
            
            print(f"✅ Entrenamiento iniciado!")
            print(f"🔍 Job ID: {job_id}")
            print(f"📊 Parámetros: {data['parameters']}")
            
            # Monitorear progreso
            return monitor_job_progress(job_id, "Entrenamiento Capa 2")
        else:
            print(f"❌ Error iniciando entrenamiento: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def monitor_job_progress(job_id: str, job_name: str):
    """Monitorear progreso de un trabajo"""
    print(f"\n📊 Monitoreando: {job_name}")
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
                
                # Información específica por tipo
                if "training_info" in status_data:
                    info = status_data["training_info"]
                    current_epoch = info.get("current_epoch", 0)
                    total_epochs = info.get("num_epochs", 0)
                    if total_epochs > 0:
                        print(f" | Época: {current_epoch}/{total_epochs}", end="")
                
                if status == "completed":
                    print(f"\n✅ {job_name} completado exitosamente!")
                    show_job_results(job_id)
                    return True
                    
                elif status == "failed":
                    print(f"\n❌ {job_name} falló")
                    error = status_data.get("error", "Error desconocido")
                    print(f"Error: {error}")
                    return False
                    
                elif status == "cancelled":
                    print(f"\n⚠️ {job_name} fue cancelado")
                    return False
                
            else:
                print(f"\n❌ Error obteniendo estado: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"\n❌ Error monitoreando: {e}")
            return False
        
        time.sleep(5)

def show_job_results(job_id: str):
    """Mostrar resultados detallados"""
    try:
        response = requests.get(f"{API_BASE_URL}/training/results/{job_id}")
        
        if response.status_code == 200:
            results = response.json()
            
            print(f"\n📊 RESULTADOS DETALLADOS")
            print("-" * 30)
            
            if "results" in results and results["results"]:
                job_results = results["results"]
                
                if results['type'] == 'data_preparation':
                    print(f"Pares generados: {job_results.get('generated_count', 'N/A')}")
                    pairs = job_results.get('pairs', [])
                    if pairs:
                        print(f"Ejemplos de pares:")
                        for i, pair in enumerate(pairs[:3]):
                            print(f"  Par {i+1}: {pair['clean_file']} -> {pair['degraded_file']}")
                
                elif results['type'] == 'layer2_training':
                    print(f"Épocas completadas: {job_results.get('epochs_completed', 'N/A')}")
                    print(f"Pares utilizados: {job_results.get('pairs_used', 'N/A')}")
                    print(f"Batch size: {job_results.get('batch_size', 'N/A')}")
            
            # Archivos de salida
            output_files = results.get("output_files", [])
            if output_files:
                print(f"\n📁 Archivos generados ({len(output_files)}):")
                for file_info in output_files[:5]:
                    size_mb = file_info["size"] / (1024 * 1024)
                    print(f"   📄 {file_info['filename']} ({size_mb:.2f} MB)")
                    
        else:
            print(f"❌ Error obteniendo resultados: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error mostrando resultados: {e}")

def workflow_completo():
    """Workflow completo recomendado para n8n"""
    print("🚀 WORKFLOW COMPLETO PARA CAPA 2")
    print("=" * 60)
    print("Este es el flujo recomendado para usar en n8n")
    print()
    
    # Paso 1: Verificar datos actuales
    print("🔍 PASO 1: Verificar datos existentes")
    data_status = check_layer2_data_status()
    
    if not data_status:
        print("❌ No se pudo verificar el estado de los datos")
        return
    
    # Paso 2: Preparar datos si es necesario
    valid_pairs = data_status["statistics"]["valid_pairs"]
    recommended_pairs = data_status["recommendations"]["recommended_pairs"]
    
    if valid_pairs < recommended_pairs:
        print(f"\n🔧 PASO 2: Generar pares adicionales")
        print(f"Actual: {valid_pairs} | Recomendado: {recommended_pairs}")
        
        if not prepare_training_data(recommended_pairs):
            print("❌ No se pudieron preparar los datos")
            return
    else:
        print(f"\n✅ PASO 2: Datos suficientes ({valid_pairs} pares)")
    
    # Paso 3: Entrenar modelo
    print(f"\n🚀 PASO 3: Entrenar Capa 2")
    
    # Configuración recomendada para n8n
    params = {
        "num_epochs": 15,
        "max_pairs": min(valid_pairs, 100),
        "batch_size": 2,
        "use_training_bucket": True
    }
    
    print(f"Configuración automática: {params}")
    
    try:
        response = requests.post(f"{API_BASE_URL}/training/layer2/train", params=params)
        
        if response.status_code == 200:
            data = response.json()
            job_id = data["job_id"]
            
            print(f"✅ Entrenamiento iniciado automáticamente!")
            print(f"🔍 Job ID: {job_id}")
            
            # Para n8n, aquí retornarías el job_id para monitoreo posterior
            print(f"\n📋 PARA N8N:")
            print(f"   - Guardar job_id: {job_id}")
            print(f"   - Endpoint de monitoreo: /training/status/{job_id}")
            print(f"   - Endpoint de resultados: /training/results/{job_id}")
            
            # Monitorear progreso (en n8n esto sería un nodo separado)
            monitor_job_progress(job_id, "Entrenamiento Automático")
            
        else:
            print(f"❌ Error en entrenamiento automático: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Menú principal"""
    print("🎯 GESTIÓN COMPLETA DE CAPA 2")
    print("=" * 60)
    print("Opciones para tu flujo con n8n")
    print()
    
    while True:
        print("\n" + "="*60)
        print("🎯 OPCIONES:")
        print("1. Verificar estado de datos")
        print("2. Preparar datos (generar pares)")
        print("3. Entrenar Capa 2 (configuración manual)")
        print("4. Workflow completo automatizado (RECOMENDADO)")
        print("5. Ver trabajos activos")
        print("6. Salir")
        print("="*60)
        
        try:
            choice = input("Selecciona una opción (1-6): ").strip()
            
            if choice == "1":
                check_layer2_data_status()
            elif choice == "2":
                target = int(input("Número de pares objetivo (100): ") or "100")
                prepare_training_data(target)
            elif choice == "3":
                train_layer2_with_options()
            elif choice == "4":
                workflow_completo()
            elif choice == "5":
                response = requests.get(f"{API_BASE_URL}/training/jobs")
                if response.status_code == 200:
                    jobs = response.json()
                    print(f"\n📋 Trabajos activos: {jobs['total_jobs']}")
                    for job in jobs['jobs'][:5]:
                        print(f"   🔍 {job['job_id'][:8]}... | {job['type']} | {job['status']}")
                else:
                    print("❌ Error obteniendo trabajos")
            elif choice == "6":
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
