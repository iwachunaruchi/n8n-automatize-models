#!/usr/bin/env python3
"""
🔍 VALIDADOR DE CONFIGURACIÓN DOCKER
===================================
Script para validar que toda la configuración Docker esté correcta antes del deployment.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any

def validate_paths() -> Dict[str, Any]:
    """Validar que todos los paths referenciados existan"""
    print("🔍 Validando paths...")
    
    docker_dir = Path(__file__).parent
    project_root = docker_dir.parent
    
    issues = []
    warnings = []
    
    # Validar paths en docker-compose-rq.yml
    compose_file = docker_dir / "docker-compose-rq.yml"
    
    paths_to_check = [
        # API volumes
        ("../outputs", "outputs directory"),
        ("../models", "models directory"),
        ("../src", "src directory"),
        ("../api", "api directory"),
        ("../config", "config directory"),
        ("../scripts", "scripts directory"),
        ("../workers", "workers directory"),
        ("../n8n/workflows", "n8n workflows directory"),
        
        # Files
        ("../pyproject.toml", "Poetry configuration"),
        ("Dockerfile", "Main Dockerfile"),
        ("Dockerfile.worker", "Worker Dockerfile"),
    ]
    
    for rel_path, description in paths_to_check:
        abs_path = docker_dir / rel_path
        if not abs_path.exists():
            issues.append(f"❌ Missing {description}: {abs_path}")
        else:
            print(f"✅ Found {description}")
    
    return {"issues": issues, "warnings": warnings}

def validate_dockerfiles() -> Dict[str, Any]:
    """Validar Dockerfiles"""
    print("\n🐳 Validando Dockerfiles...")
    
    docker_dir = Path(__file__).parent
    issues = []
    warnings = []
    
    # Validar Dockerfile principal
    dockerfile = docker_dir / "Dockerfile"
    if dockerfile.exists():
        content = dockerfile.read_text()
        if "poetry install" in content:
            print("✅ Dockerfile usa Poetry")
        else:
            warnings.append("⚠️ Dockerfile no parece usar Poetry")
            
        if "uvicorn" in content:
            print("✅ Dockerfile configura uvicorn")
        else:
            issues.append("❌ Dockerfile no tiene comando uvicorn")
    else:
        issues.append("❌ Dockerfile principal no encontrado")
    
    # Validar Dockerfile.worker
    dockerfile_worker = docker_dir / "Dockerfile.worker"
    if dockerfile_worker.exists():
        content = dockerfile_worker.read_text()
        if "rq_worker.py" in content:
            print("✅ Dockerfile.worker usa rq_worker.py")
        else:
            issues.append("❌ Dockerfile.worker no ejecuta rq_worker.py")
            
        if "optimized_watcher.py" in content:
            issues.append("❌ Dockerfile.worker referencia optimized_watcher.py (archivo eliminado)")
    else:
        issues.append("❌ Dockerfile.worker no encontrado")
    
    return {"issues": issues, "warnings": warnings}

def validate_docker_compose() -> Dict[str, Any]:
    """Validar docker-compose-rq.yml"""
    print("\n📦 Validando docker-compose-rq.yml...")
    
    docker_dir = Path(__file__).parent
    issues = []
    warnings = []
    
    compose_file = docker_dir / "docker-compose-rq.yml"
    if not compose_file.exists():
        issues.append("❌ docker-compose-rq.yml no encontrado")
        return {"issues": issues, "warnings": warnings}
    
    try:
        with open(compose_file) as f:
            compose_data = yaml.safe_load(f)
        
        # Validar servicios requeridos
        required_services = ["redis", "minio", "postgres", "doc-restoration-api", "rq-worker"]
        services = compose_data.get("services", {})
        
        for service in required_services:
            if service in services:
                print(f"✅ Servicio {service} configurado")
            else:
                issues.append(f"❌ Servicio {service} faltante")
        
        # Validar configuración de Redis
        if "redis" in services:
            redis_config = services["redis"]
            if redis_config.get("image") == "redis:7-alpine":
                print("✅ Redis usa imagen correcta")
            else:
                warnings.append("⚠️ Redis no usa imagen recomendada")
        
        # Validar environment variables
        api_service = services.get("doc-restoration-api", {})
        api_env = api_service.get("environment", {})
        
        required_env = ["REDIS_URL", "MINIO_ENDPOINT", "PYTHONPATH"]
        for env_var in required_env:
            if env_var in api_env:
                print(f"✅ Variable {env_var} configurada en API")
            else:
                issues.append(f"❌ Variable {env_var} faltante en API")
        
        # Validar volúmenes
        api_volumes = api_service.get("volumes", [])
        worker_service = services.get("rq-worker", {})
        worker_volumes = worker_service.get("volumes", [])
        
        if any("../api:/app/api" in v for v in api_volumes):
            print("✅ API tiene hot reload configurado")
        else:
            warnings.append("⚠️ API no tiene hot reload configurado")
            
        if any("../workers:/app/workers" in v for v in worker_volumes):
            print("✅ Worker tiene hot reload configurado")
        else:
            warnings.append("⚠️ Worker no tiene hot reload configurado")
            
    except yaml.YAMLError as e:
        issues.append(f"❌ Error parsing docker-compose-rq.yml: {e}")
    except Exception as e:
        issues.append(f"❌ Error validando docker-compose-rq.yml: {e}")
    
    return {"issues": issues, "warnings": warnings}

def validate_python_files() -> Dict[str, Any]:
    """Validar archivos Python críticos"""
    print("\n🐍 Validando archivos Python críticos...")
    
    docker_dir = Path(__file__).parent
    project_root = docker_dir.parent
    issues = []
    warnings = []
    
    # Validar archivos críticos
    critical_files = [
        ("workers/rq_worker.py", "RQ Worker principal"),
        ("workers/rq_tasks.py", "RQ Tasks registry"),
        ("api/main.py", "API principal"),
        ("rq_job_system.py", "RQ Job Manager"),
    ]
    
    for file_path, description in critical_files:
        abs_path = project_root / file_path
        if abs_path.exists():
            print(f"✅ {description} encontrado")
            
            # Validaciones específicas
            try:
                content = abs_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                content = abs_path.read_text(encoding='latin-1')
            
            if "rq_worker.py" in file_path:
                if "from redis import Redis" in content:
                    print("  ✅ Importa Redis correctamente")
                else:
                    issues.append(f"❌ {description} no importa Redis")
                    
                if "optimized_watcher" in content:
                    issues.append(f"❌ {description} referencia optimized_watcher (obsoleto)")
                    
            elif "main.py" in file_path:
                if "@app.get(\"/health\")" in content:
                    print("  ✅ Endpoint /health configurado")
                else:
                    issues.append(f"❌ {description} sin endpoint /health")
        else:
            issues.append(f"❌ {description} no encontrado: {abs_path}")
    
    return {"issues": issues, "warnings": warnings}

def validate_obsolete_files() -> Dict[str, Any]:
    """Validar que archivos obsoletos hayan sido eliminados"""
    print("\n🗑️ Validando eliminación de archivos obsoletos...")
    
    docker_dir = Path(__file__).parent
    issues = []
    warnings = []
    
    obsolete_files = [
        "docker-compose.yml",
        "docker-compose.dev.yml", 
        "Dockerfile.dev",
        "Dockerfile.minio-setup",
        "minio-setup-complete.sh",
        "minio-setup-python.py",
        "start_dev.ps1",
        "start_dev.sh",
        "start_dev_simple.ps1", 
        "start_system.ps1",
        "outputs/"
    ]
    
    for obsolete_file in obsolete_files:
        path = docker_dir / obsolete_file
        if path.exists():
            issues.append(f"❌ Archivo obsoleto aún presente: {obsolete_file}")
        else:
            print(f"✅ Archivo obsoleto eliminado: {obsolete_file}")
    
    return {"issues": issues, "warnings": warnings}

def main():
    """Función principal de validación"""
    print("🔍 VALIDADOR DE CONFIGURACIÓN DOCKER")
    print("=" * 60)
    
    all_issues = []
    all_warnings = []
    
    # Ejecutar validaciones
    validations = [
        validate_paths,
        validate_dockerfiles,
        validate_docker_compose,
        validate_python_files,
        validate_obsolete_files
    ]
    
    for validation in validations:
        result = validation()
        all_issues.extend(result.get("issues", []))
        all_warnings.extend(result.get("warnings", []))
    
    # Mostrar resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE VALIDACIÓN")
    print("=" * 60)
    
    if not all_issues and not all_warnings:
        print("🎉 ¡PERFECTO! La configuración Docker está lista")
        print("✅ No se encontraron problemas")
        print("\n🚀 Puedes iniciar el sistema con:")
        print("   ./start_rq_system.ps1")
        return 0
    
    if all_warnings:
        print("\n⚠️ ADVERTENCIAS:")
        for warning in all_warnings:
            print(f"  {warning}")
    
    if all_issues:
        print("\n❌ PROBLEMAS CRÍTICOS:")
        for issue in all_issues:
            print(f"  {issue}")
        print(f"\n❌ Total de problemas críticos: {len(all_issues)}")
        print("🔧 Corrige estos problemas antes de continuar")
        return 1
    
    if all_warnings and not all_issues:
        print(f"\n⚠️ Total de advertencias: {len(all_warnings)}")
        print("✅ Sistema funcional pero con mejoras posibles")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
