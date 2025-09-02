"""
Helper para n8n: Entrenamiento de Capa 2
Funciones auxiliares para el workflow de entrenamiento de Layer 2
"""

import requests
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional

API_BASE_URL = "http://localhost:8000"

class Layer2TrainingHelper:
    """Helper para gestionar entrenamiento de Capa 2 desde n8n"""
    
    def __init__(self, api_base_url: str = API_BASE_URL):
        self.api_base_url = api_base_url
    
    def validate_training_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validar parámetros de entrenamiento"""
        defaults = {
            "num_epochs": 10,
            "max_pairs": 100,
            "batch_size": 2,
            "use_training_bucket": True
        }
        
        # Aplicar defaults
        validated = {}
        for key, default_value in defaults.items():
            validated[key] = params.get(key, default_value)
        
        # Validaciones
        errors = []
        
        if not (1 <= validated["num_epochs"] <= 100):
            errors.append("num_epochs debe estar entre 1 y 100")
        
        if not (10 <= validated["max_pairs"] <= 1000):
            errors.append("max_pairs debe estar entre 10 y 1000")
        
        if not (1 <= validated["batch_size"] <= 8):
            errors.append("batch_size debe estar entre 1 y 8")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "parameters": validated
        }
    
    def check_data_status(self) -> Dict[str, Any]:
        """Verificar estado de datos para entrenamiento"""
        try:
            response = requests.get(f"{self.api_base_url}/training/layer2/data-status")
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": f"Error {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def prepare_training_data(self, target_pairs: int = 100, source_bucket: str = "document-clean") -> Dict[str, Any]:
        """Preparar datos adicionales si es necesario"""
        try:
            response = requests.post(
                f"{self.api_base_url}/training/layer2/prepare-data",
                params={
                    "target_pairs": target_pairs,
                    "source_bucket": source_bucket
                }
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": f"Error {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def start_training(self, **params) -> Dict[str, Any]:
        """Iniciar entrenamiento de Capa 2"""
        try:
            response = requests.post(
                f"{self.api_base_url}/training/layer2/train",
                params=params
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": f"Error {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Obtener estado del entrenamiento"""
        try:
            response = requests.get(f"{self.api_base_url}/training/status/{job_id}")
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": f"Error {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_training_results(self, job_id: str) -> Dict[str, Any]:
        """Obtener resultados del entrenamiento"""
        try:
            response = requests.get(f"{self.api_base_url}/training/results/{job_id}")
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": f"Error {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def monitor_training(self, job_id: str, max_wait_time: int = 3600, check_interval: int = 30) -> Dict[str, Any]:
        """Monitorear entrenamiento hasta completarse o fallar"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_result = self.get_training_status(job_id)
            
            if not status_result["success"]:
                return {
                    "success": False,
                    "error": "Error obteniendo estado",
                    "details": status_result["error"]
                }
            
            status_data = status_result["data"]
            current_status = status_data["status"]
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Estado: {current_status}")
            
            if current_status == "completed":
                results = self.get_training_results(job_id)
                return {
                    "success": True,
                    "status": "completed",
                    "results": results["data"] if results["success"] else None,
                    "final_status": status_data
                }
            
            elif current_status == "failed":
                return {
                    "success": False,
                    "status": "failed",
                    "error": status_data.get("error", "Entrenamiento falló"),
                    "final_status": status_data
                }
            
            elif current_status == "cancelled":
                return {
                    "success": False,
                    "status": "cancelled",
                    "error": "Entrenamiento cancelado",
                    "final_status": status_data
                }
            
            # Mostrar progreso si está disponible
            if "training_info" in status_data:
                training_info = status_data["training_info"]
                current_epoch = training_info.get("current_epoch", 0)
                total_epochs = training_info.get("num_epochs", 0)
                progress = status_data.get("progress", 0)
                
                print(f"    Época: {current_epoch}/{total_epochs} | Progreso: {progress}%")
            
            time.sleep(check_interval)
        
        return {
            "success": False,
            "status": "timeout",
            "error": f"Tiempo de espera agotado ({max_wait_time}s)",
            "final_status": status_data
        }

def demo_complete_training_workflow():
    """Demo completo del workflow de entrenamiento"""
    print("🚀 DEMO: Workflow Completo de Entrenamiento Capa 2")
    print("=" * 60)
    
    helper = Layer2TrainingHelper()
    
    # 1. Verificar estado de datos
    print("\n📊 1. Verificando estado de datos...")
    data_status = helper.check_data_status()
    
    if not data_status["success"]:
        print(f"❌ Error verificando datos: {data_status['error']}")
        return
    
    status_info = data_status["data"]
    print(f"✅ Pares disponibles: {status_info['statistics']['valid_pairs']}")
    print(f"✅ Listo para entrenar: {status_info['ready_for_training']}")
    
    # 2. Preparar datos si es necesario
    if status_info["statistics"]["valid_pairs"] < 50:
        print("\n🔧 2. Preparando datos adicionales...")
        prep_result = helper.prepare_training_data(target_pairs=100)
        
        if prep_result["success"]:
            print(f"✅ Preparación iniciada: {prep_result['data']['message']}")
        else:
            print(f"❌ Error preparando datos: {prep_result['error']}")
            return
    
    # 3. Validar parámetros
    print("\n⚙️ 3. Validando parámetros...")
    params = {
        "num_epochs": 5,  # Reducido para demo
        "max_pairs": 50,
        "batch_size": 2,
        "use_training_bucket": True
    }
    
    validation = helper.validate_training_parameters(params)
    
    if not validation["valid"]:
        print(f"❌ Parámetros inválidos: {validation['errors']}")
        return
    
    print(f"✅ Parámetros válidos: {validation['parameters']}")
    
    # 4. Iniciar entrenamiento
    print("\n🎯 4. Iniciando entrenamiento...")
    training_result = helper.start_training(**validation["parameters"])
    
    if not training_result["success"]:
        print(f"❌ Error iniciando entrenamiento: {training_result['error']}")
        return
    
    job_id = training_result["data"]["job_id"]
    print(f"✅ Entrenamiento iniciado: {job_id}")
    
    # 5. Monitorear entrenamiento
    print("\n📈 5. Monitoreando entrenamiento...")
    monitoring_result = helper.monitor_training(job_id, max_wait_time=1800, check_interval=30)
    
    if monitoring_result["success"]:
        print(f"🎉 ¡Entrenamiento completado exitosamente!")
        print(f"📊 Resultados disponibles en: /training/results/{job_id}")
    else:
        print(f"❌ Entrenamiento falló o fue interrumpido: {monitoring_result['error']}")
    
    print("\n" + "=" * 60)
    print("Demo completado")

if __name__ == "__main__":
    # Ejecutar demo
    demo_complete_training_workflow()
