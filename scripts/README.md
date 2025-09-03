# 🛠️ Scripts de Utilidad y Gestión

Este directorio contiene scripts organizados para desarrollo, gestión de modelos y configuración del sistema.

## 📁 Estructura Organizada

```
scripts/
├── 🔧 Core Development           # Scripts básicos de desarrollo
│   ├── start_api.py
│   ├── validate_setup.py
│   ├── manage_deps.py
│   └── setup_dev.py
│
├── 📦 model-management/          # Gestión de modelos por layers
│   ├── layer-1/                 # Scripts para Layer 1 (Restormer)
│   │   ├── download_restormer_models.py
│   │   └── upload_local_models.py
│   ├── layer-2/                 # Scripts para Layer 2 (NAFNet/DocUNet)
│   │   ├── download_nafnet_models.py
│   │   └── upload_local_models.py
│   └── README.md
│
├── 🚀 minio-setup/              # Configuración de MinIO
│   ├── setup_pretrained_folders.py
│   └── README.md
│
└── 🔬 investigations/           # Scripts de investigación
    ├── investigate_nafnet.py
    └── README.md
```

## 🎯 Scripts por Categoría

### Core Development
Scripts básicos para desarrollo y mantenimiento:

```bash
# Iniciar API
poetry run python scripts/start_api.py

# Validar configuración
poetry run python scripts/validate_setup.py

# Gestionar dependencias
poetry run python scripts/manage_deps.py

# Setup desarrollo
poetry run python scripts/setup_dev.py
```

### Model Management
Scripts organizados por layers para gestión completa de modelos:

```bash
# Layer 1 - Restormer models
cd scripts/model-management/layer-1
python download_restormer_models.py
python upload_local_models.py

# Layer 2 - NAFNet/DocUNet models
cd scripts/model-management/layer-2
python download_nafnet_models.py
python upload_local_models.py
```

### MinIO Setup
Configuración inicial de la infraestructura de almacenamiento:

```bash
# Setup completo de MinIO
cd scripts/minio-setup
python setup_pretrained_folders.py
```

### Investigations
Scripts para investigar y entender modelos:

```bash
# Investigar estructura de modelos
cd scripts/investigations
python investigate_nafnet.py
```

## 🚀 Flujo de Trabajo Recomendado

### Setup Inicial (Nueva Máquina)
1. **Configurar desarrollo**:
   ```bash
   poetry run python scripts/setup_dev.py
   poetry run python scripts/validate_setup.py
   ```

2. **Preparar MinIO**:
   ```bash
   cd scripts/minio-setup
   python setup_pretrained_folders.py
   ```

3. **Descargar modelos**:
   ```bash
   # Layer 1
   cd scripts/model-management/layer-1
   python download_restormer_models.py
   
   # Layer 2
   cd scripts/model-management/layer-2
   python download_nafnet_models.py
   ```

### Desarrollo Diario
```bash
# Iniciar API para desarrollo
poetry run python scripts/start_api.py

# Validar cambios
poetry run python scripts/validate_setup.py
```

### Gestión de Modelos
```bash
# Investigar modelo nuevo
cd scripts/investigations
python investigate_[modelo].py

# Subir modelo local
cd scripts/model-management/layer-X
python upload_local_models.py
```

## 🌟 Características Destacadas

### Escalabilidad
- ✅ **Estructura por layers**: Fácil agregar nuevas capas
- ✅ **Scripts modulares**: Cada script tiene propósito específico
- ✅ **Configuración consistente**: MinIO config unificada

### Reutilización
- ✅ **Plantillas**: Scripts base para nuevos modelos
- ✅ **Documentación**: README detallado en cada carpeta
- ✅ **Ejemplos**: Casos de uso claros

### Mantenimiento
- ✅ **Organización clara**: Fácil encontrar el script correcto
- ✅ **Funcionalidad específica**: Un script, una tarea
- ✅ **Error handling**: Manejo robusto de errores

## 📚 Documentación Detallada

Cada subdirectorio tiene su README específico:
- [`model-management/README.md`](./model-management/README.md) - Gestión de modelos
- [`minio-setup/README.md`](./minio-setup/README.md) - Configuración MinIO
- [`investigations/README.md`](./investigations/README.md) - Scripts de investigación

## 🔧 Uso con Poetry

Todos los scripts están diseñados para Poetry:

```bash
# Activar entorno virtual
poetry shell

# Ejecutar desde raíz del proyecto
python scripts/[categoria]/script.py

# O navegar a directorio específico
cd scripts/categoria
python script.py
```

## 🎯 Próximas Expansiones

### Layer 3+
```bash
mkdir scripts/model-management/layer-3
# Copiar y adaptar scripts de layer-2
```

### Nuevos Tipos de Modelos
```bash
mkdir scripts/model-management/layer-2/new_model_type
# Crear scripts específicos
```

### Automatización CI/CD
```bash
mkdir scripts/automation
# Scripts para deployment y testing
```
