#!/usr/bin/env python3
"""
Script para validar la configuración del proyecto con Poetry
"""
import subprocess
import sys
import importlib.util
from pathlib import Path

def check_import(module_name, description):
    """Verificar que un módulo se pueda importar"""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            print(f"✅ {description}: OK")
            return True
        else:
            print(f"❌ {description}: Módulo no encontrado")
            return False
    except ImportError as e:
        print(f"❌ {description}: Error - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description}: Advertencia - {e}")
        return True  # Consideramos como éxito parcial

def check_poetry_env():
    """Verificar el entorno de Poetry"""
    try:
        result = subprocess.run(["poetry", "env", "info"], capture_output=True, text=True, check=True)
        print("✅ Entorno de Poetry: OK")
        print(f"   {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error verificando entorno de Poetry: {e}")
        return False

def check_project_structure():
    """Verificar estructura del proyecto"""
    project_root = Path(__file__).parent.parent
    required_dirs = [
        "api",
        "src", 
        "layers",
        "config",
        "data",
        "models",
        "outputs",
        "n8n",
        "scripts"
    ]
    
    required_files = [
        "pyproject.toml",
        "README.md",
        ".env.example",
        "docker-compose.yml",
        "Dockerfile"
    ]
    
    print("🏗️ Verificando estructura del proyecto...")
    
    all_good = True
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ Directorio {dir_name}: OK")
        else:
            print(f"❌ Directorio {dir_name}: No encontrado")
            all_good = False
    
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"✅ Archivo {file_name}: OK")
        else:
            print(f"❌ Archivo {file_name}: No encontrado")
            all_good = False
    
    return all_good

def main():
    """Validar todo el entorno"""
    print("🔍 Validando configuración del proyecto n8n-automatize-models\n")
    
    # Verificar Poetry
    check_poetry_env()
    print()
    
    # Verificar estructura del proyecto
    check_project_structure()
    print()
    
    # Verificar imports críticos
    print("📦 Verificando imports críticos...")
    critical_imports = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("boto3", "AWS SDK (MinIO)"),
        ("pydantic", "Pydantic"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("albumentations", "Albumentations"),
        ("yaml", "PyYAML"),
        ("requests", "Requests"),
    ]
    
    failed_imports = []
    for module, description in critical_imports:
        if not check_import(module, description):
            failed_imports.append(module)
    
    print()
    
    if failed_imports:
        print(f"❌ {len(failed_imports)} imports fallaron. Ejecuta:")
        print("   poetry install --no-root")
        print("   para reinstalar las dependencias.")
    else:
        print("🎉 Todas las dependencias están correctamente instaladas!")
    
    print("\n📋 Comandos útiles:")
    print("  poetry shell                    # Activar entorno virtual")
    print("  poetry run python scripts/start_api.py  # Iniciar API")
    print("  poetry add <paquete>            # Agregar dependencia")
    print("  poetry show                     # Ver dependencias")
    print("  docker-compose up -d            # Iniciar servicios")

if __name__ == "__main__":
    main()
