#!/usr/bin/env python3
"""
Script de configuración para el entorno API + MinIO + n8n
"""

import os
import requests
import time
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Instalar dependencias de la API"""
    print("📦 Instalando dependencias de la API...")
    
    api_requirements = Path("api/requirements.txt")
    if api_requirements.exists():
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(api_requirements)
        ], check=True)
        print("✅ Dependencias instaladas")
    else:
        print("❌ Archivo api/requirements.txt no encontrado")

def setup_environment():
    """Configurar variables de entorno"""
    print("🔧 Configurando entorno...")
    
    env_example = Path("api/.env.example")
    env_file = Path("api/.env")
    
    if env_example.exists() and not env_file.exists():
        # Copiar archivo de ejemplo
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Archivo .env creado desde .env.example")
        print("🔧 Edita api/.env con tu configuración específica")
    else:
        print("✅ Archivo .env ya existe")

def start_infrastructure():
    """Iniciar MinIO y n8n con Docker"""
    print("🐳 Iniciando infraestructura con Docker...")
    
    if not Path("docker-compose.yml").exists():
        print("❌ docker-compose.yml no encontrado")
        return False
    
    try:
        # Iniciar servicios
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        print("✅ Servicios Docker iniciados")
        
        # Esperar a que los servicios estén listos
        print("⏳ Esperando a que los servicios estén listos...")
        time.sleep(30)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error iniciando Docker: {e}")
        return False

def verify_minio():
    """Verificar que MinIO esté funcionando"""
    print("🔍 Verificando MinIO...")
    
    try:
        response = requests.get("http://localhost:9000/minio/health/live", timeout=10)
        if response.status_code == 200:
            print("✅ MinIO está activo")
            return True
    except requests.exceptions.RequestException:
        pass
    
    print("❌ MinIO no responde")
    return False

def verify_n8n():
    """Verificar que n8n esté funcionando"""
    print("🔍 Verificando n8n...")
    
    try:
        response = requests.get("http://localhost:5678/healthz", timeout=10)
        if response.status_code == 200:
            print("✅ n8n está activo")
            print("🌐 n8n disponible en: http://localhost:5678")
            print("👤 Usuario: admin | Contraseña: admin123")
            return True
    except requests.exceptions.RequestException:
        pass
    
    print("❌ n8n no responde")
    return False

def create_sample_data():
    """Crear datos de ejemplo para testing"""
    print("📸 Creando datos de ejemplo...")
    
    # Verificar si hay imágenes en data/train/degraded
    degraded_dir = Path("data/train/degraded")
    if degraded_dir.exists():
        degraded_files = list(degraded_dir.glob("*.png"))
        if degraded_files:
            print(f"✅ Encontradas {len(degraded_files)} imágenes degradadas para testing")
            return True
    
    print("⚠️  No se encontraron imágenes degradadas para testing")
    print("💡 Puedes agregar imágenes a data/train/degraded/ para testing")
    return False

def test_api():
    """Probar la API básica"""
    print("🧪 Probando la API...")
    
    try:
        # Verificar que el modelo se pueda cargar
        from api.main import load_model
        model = load_model()
        print("✅ Modelo cargado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False

def show_next_steps():
    """Mostrar próximos pasos"""
    print("\n🎉 CONFIGURACIÓN COMPLETADA")
    print("=" * 50)
    print("📋 Próximos pasos:")
    print()
    print("1. 🚀 Iniciar la API:")
    print("   cd api && python main.py")
    print()
    print("2. 🌐 Acceder a n8n:")
    print("   http://localhost:5678")
    print("   Usuario: admin | Contraseña: admin123")
    print()
    print("3. 📁 Acceder a MinIO Console:")
    print("   http://localhost:9001")
    print("   Usuario: minio | Contraseña: minio123")
    print()
    print("4. 🧪 Probar el cliente:")
    print("   python api/client.py")
    print()
    print("5. 📊 Importar workflow en n8n:")
    print("   Importar: n8n/workflows/document-processing.json")
    print()

def main():
    """Función principal"""
    print("🔧 CONFIGURACIÓN DEL ENTORNO API + MinIO + n8n")
    print("=" * 60)
    
    success_steps = 0
    total_steps = 6
    
    # Paso 1: Instalar dependencias
    try:
        install_dependencies()
        success_steps += 1
    except Exception as e:
        print(f"❌ Error instalando dependencias: {e}")
    
    # Paso 2: Configurar entorno
    setup_environment()
    success_steps += 1
    
    # Paso 3: Iniciar infraestructura
    if start_infrastructure():
        success_steps += 1
    
    # Paso 4: Verificar MinIO
    if verify_minio():
        success_steps += 1
    
    # Paso 5: Verificar n8n
    if verify_n8n():
        success_steps += 1
    
    # Paso 6: Verificar datos y modelo
    create_sample_data()
    if test_api():
        success_steps += 1
    
    print(f"\n📊 Progreso: {success_steps}/{total_steps} pasos completados")
    
    if success_steps >= 4:  # Al menos infraestructura básica
        show_next_steps()
    else:
        print("\n❌ Configuración incompleta. Revisar errores arriba.")

if __name__ == "__main__":
    main()
