#!/usr/bin/env python3
"""
🔄 COLA COMPARTIDA ENTRE API Y WORKER
====================================
Sistema de cola compartida usando archivos JSON para comunicación
entre el API server y el worker.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import tempfile

# Solo importar fcntl en sistemas Unix
try:
    import fcntl
    FCNTL_AVAILABLE = True
except ImportError:
    FCNTL_AVAILABLE = False

class SharedJobQueue:
    """Cola de jobs compartida entre API server y worker usando archivos"""
    
    def __init__(self, queue_file: str = None):
        if queue_file is None:
            # Usar directorio temporal del sistema
            temp_dir = tempfile.gettempdir()
            self.queue_file = os.path.join(temp_dir, "n8n_job_queue.json")
        else:
            self.queue_file = queue_file
        
        # Inicializar archivo si no existe
        if not os.path.exists(self.queue_file):
            self._write_queue_data({"queue": [], "status": {}})
    
    def _read_queue_data(self) -> Dict[str, Any]:
        """Leer datos de la cola con lock"""
        try:
            with open(self.queue_file, 'r', encoding='utf-8') as f:
                if FCNTL_AVAILABLE:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock para lectura
                data = json.load(f)
                if FCNTL_AVAILABLE:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Unlock
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            return {"queue": [], "status": {}}
    
    def _write_queue_data(self, data: Dict[str, Any]):
        """Escribir datos de la cola con lock"""
        with open(self.queue_file, 'w', encoding='utf-8') as f:
            if FCNTL_AVAILABLE:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock para escritura
            json.dump(data, f, indent=2, ensure_ascii=False)
            if FCNTL_AVAILABLE:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Unlock
    
    def enqueue_job(self, job_data: Dict[str, Any]) -> str:
        """Encolar job"""
        data = self._read_queue_data()
        
        job_id = job_data["job_id"]
        
        # Agregar a la cola
        data["queue"].append(job_data)
        
        # Actualizar estado
        data["status"][job_id] = {
            "status": "queued",
            "created_at": job_data["created_at"],
            "job_type": job_data["job_type"],
            "parameters": job_data["parameters"]
        }
        
        self._write_queue_data(data)
        return job_id
    
    def dequeue_job(self) -> Optional[Dict[str, Any]]:
        """Obtener siguiente job de la cola"""
        data = self._read_queue_data()
        
        if data["queue"]:
            job = data["queue"].pop(0)  # FIFO
            self._write_queue_data(data)
            return job
        
        return None
    
    def update_job_status(self, job_id: str, status: str, **extra_data):
        """Actualizar status del job"""
        data = self._read_queue_data()
        
        if job_id in data["status"]:
            data["status"][job_id]["status"] = status
            data["status"][job_id]["updated_at"] = datetime.now().isoformat()
            data["status"][job_id].update(extra_data)
            
            self._write_queue_data(data)
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener status de un job específico"""
        data = self._read_queue_data()
        return data["status"].get(job_id)
    
    def get_all_jobs(self) -> Dict[str, Any]:
        """Obtener todos los jobs y estadísticas"""
        data = self._read_queue_data()
        
        # Calcular estadísticas
        stats = {}
        for job_info in data["status"].values():
            status = job_info.get("status", "unknown")
            stats[status] = stats.get(status, 0) + 1
        
        return {
            "total_jobs": len(data["status"]),
            "jobs": data["status"],
            "statistics": stats,
            "queued_in_worker": len(data["queue"])
        }
    
    def clear_completed_jobs(self, max_age_hours: int = 24):
        """Limpiar jobs completados antiguos"""
        data = self._read_queue_data()
        current_time = datetime.now()
        
        jobs_to_remove = []
        for job_id, job_info in data["status"].items():
            if job_info.get("status") in ["completed", "failed"]:
                created_at = datetime.fromisoformat(job_info.get("created_at", ""))
                age_hours = (current_time - created_at).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    jobs_to_remove.append(job_id)
        
        # Remover jobs antiguos
        for job_id in jobs_to_remove:
            del data["status"][job_id]
        
        self._write_queue_data(data)
        return len(jobs_to_remove)

# Instancia global para Windows (sin fcntl)
class WindowsSharedJobQueue:
    """Versión para Windows sin fcntl"""
    
    def __init__(self, queue_file: str = None):
        if queue_file is None:
            temp_dir = tempfile.gettempdir()
            self.queue_file = os.path.join(temp_dir, "n8n_job_queue.json")
        else:
            self.queue_file = queue_file
        
        # Inicializar archivo si no existe
        if not os.path.exists(self.queue_file):
            self._write_queue_data({"queue": [], "status": {}})
    
    def _read_queue_data(self) -> Dict[str, Any]:
        """Leer datos de la cola"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with open(self.queue_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, PermissionError):
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Pequeña espera antes de reintentar
                    continue
                return {"queue": [], "status": {}}
    
    def _write_queue_data(self, data: Dict[str, Any]):
        """Escribir datos de la cola"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with open(self.queue_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                break
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.1)
                    continue
                raise
    
    def enqueue_job(self, job_data: Dict[str, Any]) -> str:
        """Encolar job"""
        data = self._read_queue_data()
        
        job_id = job_data["job_id"]
        
        # Agregar a la cola
        data["queue"].append(job_data)
        
        # Actualizar estado
        data["status"][job_id] = {
            "status": "queued",
            "created_at": job_data["created_at"],
            "job_type": job_data["job_type"],
            "parameters": job_data["parameters"]
        }
        
        self._write_queue_data(data)
        return job_id
    
    def dequeue_job(self) -> Optional[Dict[str, Any]]:
        """Obtener siguiente job de la cola"""
        data = self._read_queue_data()
        
        if data["queue"]:
            job = data["queue"].pop(0)  # FIFO
            self._write_queue_data(data)
            return job
        
        return None
    
    def update_job_status(self, job_id: str, status: str, **extra_data):
        """Actualizar status del job"""
        data = self._read_queue_data()
        
        if job_id in data["status"]:
            data["status"][job_id]["status"] = status
            data["status"][job_id]["updated_at"] = datetime.now().isoformat()
            data["status"][job_id].update(extra_data)
            
            self._write_queue_data(data)
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener status de un job específico"""
        data = self._read_queue_data()
        return data["status"].get(job_id)
    
    def get_all_jobs(self) -> Dict[str, Any]:
        """Obtener todos los jobs y estadísticas"""
        data = self._read_queue_data()
        
        # Calcular estadísticas
        stats = {}
        for job_info in data["status"].values():
            status = job_info.get("status", "unknown")
            stats[status] = stats.get(status, 0) + 1
        
        return {
            "total_jobs": len(data["status"]),
            "jobs": data["status"],
            "statistics": stats,
            "queued_in_worker": len(data["queue"])
        }
    
    def clear_completed_jobs(self, max_age_hours: int = 24):
        """Limpiar jobs completados antiguos"""
        data = self._read_queue_data()
        current_time = datetime.now()
        
        jobs_to_remove = []
        for job_id, job_info in data["status"].items():
            if job_info.get("status") in ["completed", "failed"]:
                created_at = datetime.fromisoformat(job_info.get("created_at", ""))
                age_hours = (current_time - created_at).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    jobs_to_remove.append(job_id)
        
        # Remover jobs antiguos
        for job_id in jobs_to_remove:
            del data["status"][job_id]
        
        self._write_queue_data(data)
        return len(jobs_to_remove)

# Factory function para crear la cola apropiada según el OS
def create_shared_queue() -> 'WindowsSharedJobQueue':
    """Crear instancia de cola compartida"""
    # En Windows usamos siempre WindowsSharedJobQueue
    # En producción con Linux se podría usar SharedJobQueue con fcntl
    return WindowsSharedJobQueue()

if __name__ == "__main__":
    # Test de la cola compartida
    print("🧪 PROBANDO COLA COMPARTIDA")
    print("=" * 40)
    
    queue = create_shared_queue()
    
    # Crear job de prueba
    test_job = {
        "job_id": "test_shared_123",
        "job_type": "test",
        "parameters": {"test": True},
        "created_at": datetime.now().isoformat()
    }
    
    # Encolar
    print(f"📝 Encolando job: {test_job['job_id']}")
    queue.enqueue_job(test_job)
    
    # Verificar estado
    status = queue.get_job_status(test_job["job_id"])
    print(f"📊 Estado: {status}")
    
    # Desencolar
    dequeued = queue.dequeue_job()
    print(f"📤 Desencolado: {dequeued['job_id'] if dequeued else 'Ninguno'}")
    
    print("✅ Cola compartida funcionando")
