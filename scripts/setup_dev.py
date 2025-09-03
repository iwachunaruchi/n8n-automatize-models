#!/usr/bin/env python3
"""
Script para inicializar el entorno de desarrollo
"""
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Ejecutar comando con descripción"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return None

def main():
    """Configurar el entorno de desarrollo"""
    project_root = Path(__file__).parent.parent
    print(f"🏗️  Configurando entorno de desarrollo en: {project_root}")
    
    # Verificar que Poetry está instalado
    poetry_check = run_command("poetry --version", "Verificando Poetry")
    if not poetry_check:
        print("❌ Poetry no está instalado. Instálalo con: pip install poetry")
        sys.exit(1)
    
    # Cambiar al directorio del proyecto
    import os
    os.chdir(project_root)
    
    # Configurar Poetry
    run_command("poetry config virtualenvs.in-project true", "Configurando Poetry para usar .venv local")
    
    # Instalar dependencias
    run_command("poetry install --no-root", "Instalando dependencias")
    
    # Configurar pre-commit hooks (opcional)
    activate_venv = run_command("poetry run pre-commit install", "Configurando pre-commit hooks")
    
    print("\n🎉 Entorno configurado exitosamente!")
    print("\n📋 Comandos útiles:")
    print("  poetry shell                    # Activar entorno virtual")
    print("  poetry run python scripts/start_api.py  # Iniciar API")
    print("  poetry add <paquete>            # Agregar dependencia")
    print("  poetry remove <paquete>         # Remover dependencia")
    print("  poetry show                     # Ver dependencias instaladas")
    print("  poetry update                   # Actualizar dependencias")
    
    print("\n🐳 Docker:")
    print("  docker-compose up -d            # Iniciar servicios")
    print("  docker-compose down             # Detener servicios")
    
    print("\n🌐 URLs de desarrollo:")
    print("  API: http://localhost:8000")
    print("  API Docs: http://localhost:8000/docs")
    print("  n8n: http://localhost:5678")
    print("  MinIO: http://localhost:9001")

if __name__ == "__main__":
    main()
