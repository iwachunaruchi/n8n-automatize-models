#!/usr/bin/env python3
"""
SCRIPT DE CONFIGURACIÓN Y VERIFICACIÓN DEL PROYECTO
Transfer Learning Gradual - Setup Inicial
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def create_directory_structure():
    """Crear estructura de directorios del proyecto"""
    print("📁 CREANDO ESTRUCTURA DE DIRECTORIOS")
    print("=" * 50)
    
    directories = [
        "training",
        "evaluation", 
        "data_generation",
        "outputs/checkpoints",
        "outputs/analysis",
        "outputs/evaluation",
        "outputs/pipeline_results",
        "outputs/samples",
        "models/pretrained",
        "data/train/clean",
        "data/train/degraded",
        "data/val/clean",
        "data/val/degraded"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")
    
    print("📁 Estructura de directorios creada")

def check_python_environment():
    """Verificar entorno de Python"""
    print(f"\n🐍 VERIFICANDO ENTORNO DE PYTHON")
    print("=" * 50)
    
    # Versión de Python
    python_version = sys.version
    print(f"🔧 Versión de Python: {python_version}")
    
    # Verificar si estamos en virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detectado")
    else:
        print("⚠️  No se detectó virtual environment")
        print("💡 Recomendación: Activar venv_3.11\\Scripts\\activate")
    
    return True

def check_dependencies():
    """Verificar dependencias instaladas"""
    print(f"\n📦 VERIFICANDO DEPENDENCIAS")
    print("=" * 50)
    
    required_packages = [
        'torch',
        'torchvision', 
        'opencv-python',
        'numpy',
        'pillow',
        'matplotlib',
        'tqdm',
        'pyyaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Faltan {len(missing_packages)} dependencias")
        print("💡 Instalar con: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ Todas las dependencias están instaladas")
        return True

def check_cuda_availability():
    """Verificar disponibilidad de CUDA"""
    print(f"\n🔧 VERIFICANDO CUDA")
    print("=" * 50)
    
    try:
        import torch
        
        print(f"🔧 PyTorch versión: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✅ CUDA disponible")
            print(f"🎯 Dispositivo: {torch.cuda.get_device_name(0)}")
            print(f"💾 Memoria VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
            
            # Test simple
            test_tensor = torch.randn(100, 100).cuda()
            print("✅ Test CUDA exitoso")
            
            return True
        else:
            print("⚠️  CUDA no disponible - usando CPU")
            return False
            
    except ImportError:
        print("❌ PyTorch no instalado")
        return False
    except Exception as e:
        print(f"❌ Error verificando CUDA: {e}")
        return False

def download_pretrained_model():
    """Descargar modelo preentrenado si no existe"""
    print(f"\n📥 VERIFICANDO MODELO PREENTRENADO")
    print("=" * 50)
    
    model_path = "models/pretrained/restormer_denoising.pth"
    
    if os.path.exists(model_path):
        print(f"✅ Modelo ya existe: {model_path}")
        return True
    
    print("📥 Descargando modelo Restormer preentrenado...")
    
    url = "https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_blind.pth"
    
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Descargar
        print(f"🔽 Descargando desde: {url}")
        urllib.request.urlretrieve(url, model_path)
        
        # Verificar descarga
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) // (1024*1024)
            print(f"✅ Modelo descargado: {file_size} MB")
            return True
        else:
            print("❌ Error en la descarga")
            return False
            
    except Exception as e:
        print(f"❌ Error descargando modelo: {e}")
        print("💡 Descarga manual desde: https://github.com/swz30/Restormer/releases")
        return False

def check_data_availability():
    """Verificar disponibilidad de datos de entrenamiento"""
    print(f"\n📊 VERIFICANDO DATOS DE ENTRENAMIENTO")
    print("=" * 50)
    
    data_dirs = {
        "Clean Training": "data/train/clean",
        "Degraded Training": "data/train/degraded"
    }
    
    total_files = 0
    
    for name, path in data_dirs.items():
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            print(f"📁 {name}: {len(files)} archivos")
            total_files += len(files)
        else:
            print(f"❌ {name}: No encontrado")
    
    if total_files > 0:
        print(f"✅ Total archivos de datos: {total_files}")
        
        if total_files < 20:
            print("💡 Pocos datos - ejecutar: python data_generation\\create_synthetic_data.py")
        
        return True
    else:
        print("⚠️  No se encontraron datos de entrenamiento")
        print("💡 Ejecutar: python data_generation\\create_synthetic_data.py")
        return False

def check_existing_models():
    """Verificar modelos ya entrenados"""
    print(f"\n🏆 VERIFICANDO MODELOS ENTRENADOS")
    print("=" * 50)
    
    model_files = {
        "Transfer Learning Gradual": "outputs/checkpoints/gradual_transfer_final.pth",
        "Fine-tuning Optimizado": "outputs/checkpoints/optimized_restormer_final.pth", 
        "Fine-tuning Básico": "outputs/checkpoints/finetuned_restormer_final.pth",
        "Entrenado desde Cero": "outputs/checkpoints/best_restormer.pth"
    }
    
    available_models = 0
    
    for name, path in model_files.items():
        if os.path.exists(path):
            file_size = os.path.getsize(path) // (1024*1024)
            print(f"✅ {name}: {file_size} MB")
            available_models += 1
        else:
            print(f"❌ {name}: No encontrado")
    
    if available_models > 0:
        print(f"✅ {available_models} modelos disponibles")
        return True
    else:
        print("⚠️  No hay modelos entrenados")
        print("💡 Ejecutar: python training\\gradual_transfer_learning.py")
        return False

def create_quick_start_guide():
    """Crear guía de inicio rápido"""
    guide_content = """# GUÍA DE INICIO RÁPIDO
## Transfer Learning Gradual - Restauración de Documentos

### 🚀 PASOS PARA EMPEZAR:

#### 1. Activar entorno virtual:
```bash
venv_3.11\\Scripts\\activate
```

#### 2. Generar datos sintéticos (si es necesario):
```bash
python data_generation\\create_synthetic_data.py
```

#### 3. Entrenar modelo con Transfer Learning Gradual:
```bash
python training\\gradual_transfer_learning.py
```

#### 4. Usar el modelo entrenado:
```bash
python main_pipeline.py
```

#### 5. Evaluar y comparar modelos:
```bash
python evaluation\\compare_models.py
```

### 📁 ARCHIVOS PRINCIPALES:

- **main_pipeline.py**: Pipeline principal de restauración
- **training/gradual_transfer_learning.py**: Entrenamiento avanzado
- **evaluation/compare_models.py**: Comparación de modelos
- **data_generation/create_synthetic_data.py**: Generación de datos

### 🏆 RESULTADOS ESPERADOS:

El Transfer Learning Gradual debería superar a otros métodos en:
- ✅ PSNR (> 25 dB)
- ✅ SSIM (> 0.85)  
- ✅ Preservación de texto
- ✅ Eliminación de ruido

### 🔧 CONFIGURACIÓN:

Editar `config/pipeline_config.yaml` para personalizar:
- Parámetros del modelo
- Configuración de entrenamiento
- Etapas del Transfer Learning
"""
    
    with open("QUICK_START.md", "w", encoding="utf-8") as f:
        f.write(guide_content)
    
    print("📝 Guía de inicio rápido creada: QUICK_START.md")

def print_project_status():
    """Mostrar estado completo del proyecto"""
    print(f"\n🎯 ESTADO ACTUAL DEL PROYECTO")
    print("=" * 60)
    
    # Verificar componentes principales
    components = [
        ("Pipeline Principal", "main_pipeline.py"),
        ("Transfer Learning", "training/gradual_transfer_learning.py"),
        ("Evaluación", "evaluation/compare_models.py"),
        ("Generación de Datos", "data_generation/create_synthetic_data.py"),
        ("Configuración", "config/pipeline_config.yaml")
    ]
    
    for name, path in components:
        status = "✅" if os.path.exists(path) else "❌"
        print(f"{status} {name}: {path}")
    
    # Estado del dataset
    degraded_files = []
    if os.path.exists("data/train/degraded"):
        degraded_files = [f for f in os.listdir("data/train/degraded") 
                         if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n📊 Dataset: {len(degraded_files)} imágenes degradadas")
    
    # Modelos disponibles
    models_dir = "outputs/checkpoints"
    available_models = 0
    if os.path.exists(models_dir):
        available_models = len([f for f in os.listdir(models_dir) 
                              if f.endswith('.pth')])
    
    print(f"🏆 Modelos entrenados: {available_models}")
    
    # Recomendaciones
    print(f"\n💡 PRÓXIMOS PASOS RECOMENDADOS:")
    
    if len(degraded_files) < 100:
        print("   1. 📊 python data_generation\\create_synthetic_data.py")
    
    if available_models == 0:
        print("   2. 🏋️ python training\\gradual_transfer_learning.py")
    
    print("   3. 🔬 python evaluation\\compare_models.py")
    print("   4. 🚀 python main_pipeline.py")

def main():
    """Función principal de configuración"""
    print("🛠️  CONFIGURACIÓN DEL PROYECTO")
    print("🎯 Transfer Learning Gradual - Document Restoration")
    print("=" * 60)
    
    # Crear estructura
    create_directory_structure()
    
    # Verificaciones
    check_python_environment()
    deps_ok = check_dependencies()
    cuda_ok = check_cuda_availability()
    model_ok = download_pretrained_model()
    data_ok = check_data_availability()
    trained_ok = check_existing_models()
    
    # Crear guía
    create_quick_start_guide()
    
    # Estado final
    print_project_status()
    
    # Resumen
    print(f"\n🎉 CONFIGURACIÓN COMPLETADA")
    print("=" * 60)
    
    if deps_ok and model_ok:
        print("✅ Proyecto listo para usar")
        print("📖 Ver QUICK_START.md para instrucciones")
    else:
        print("⚠️  Revisar problemas anteriores antes de continuar")
    
    print(f"\n🚀 COMANDO RECOMENDADO:")
    if not data_ok:
        print("   python data_generation\\create_synthetic_data.py")
    elif not trained_ok:
        print("   python training\\gradual_transfer_learning.py")
    else:
        print("   python main_pipeline.py")

if __name__ == "__main__":
    main()
