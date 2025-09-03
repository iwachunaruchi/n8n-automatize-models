# 📁 Model Management Scripts

Este directorio contiene scripts organizados para la gestión de modelos preentrenados, divididos por capas (layers) del pipeline de restauración.

## 🏗️ Estructura

```
model-management/
├── layer-1/           # Scripts para modelos de Layer 1
│   ├── download_restormer_models.py
│   └── upload_local_models.py
└── layer-2/           # Scripts para modelos de Layer 2
    ├── download_nafnet_models.py
    └── upload_local_models.py
```

## 🎯 Propósito por Layer

### Layer 1 - Primera Etapa de Restauración
- **Objetivo**: Denoising básico, mejora inicial de calidad
- **Modelos**: Restormer (Transformer-based)
- **Tareas**: Denoising, deraining, motion deblurring
- **Preparación**: Base para procesamiento de Layer 2

### Layer 2 - Segunda Etapa de Restauración
- **Objetivo**: Restauración avanzada, fine-tuning específico
- **Modelos**: NAFNet (Noise Aware Filtering), DocUNet (Document Unwarping)
- **Tareas**: Denoising avanzado, corrección geométrica
- **Especialización**: Optimización para documentos específicos

## 🔧 Uso de Scripts

### Scripts de Descarga (`download_*_models.py`)
Descargan modelos preentrenados desde repositorios oficiales y los organizan en MinIO:

```bash
# Para Layer 1
cd scripts/model-management/layer-1
python download_restormer_models.py

# Para Layer 2
cd scripts/model-management/layer-2
python download_nafnet_models.py
```

### Scripts de Upload (`upload_local_models.py`)
Suben modelos que tienes localmente a MinIO con metadatos completos:

```bash
# Para Layer 1
cd scripts/model-management/layer-1
python upload_local_models.py

# Para Layer 2
cd scripts/model-management/layer-2
python upload_local_models.py
```

## 📋 Funcionalidades Comunes

✅ **Verificación de duplicados**: Evita subir modelos existentes
✅ **Hash MD5**: Verificación de integridad
✅ **Metadatos completos**: Información detallada de cada modelo
✅ **Manejo de errores**: Recuperación robusta de fallos
✅ **Progreso visual**: Barras de progreso para descargas
✅ **Documentación auto**: Genera README para cada modelo

## 🌟 Escalabilidad

Esta estructura permite:
- **Agregar nuevos layers** fácilmente (layer-3, layer-4, etc.)
- **Diferentes tipos de modelos** por layer
- **Configuración específica** por arquitectura
- **Reutilización de código** común

## 🔗 Integración

Los modelos gestionados aquí se integran automáticamente con:
- Pipeline de entrenamiento automatizado
- Sistema de fine-tuning diferencial
- API de restauración de documentos
- Workflows de n8n

## 📚 Próximos Pasos

Para agregar un nuevo layer:
1. Crear carpeta `layer-X/`
2. Copiar y adaptar scripts de referencia
3. Actualizar configuración de modelos
4. Documentar en este README

## 🛠️ Configuración MinIO

Antes de usar estos scripts, asegúrate de que MinIO esté configurado:
```bash
cd scripts/minio-setup
python setup_pretrained_folders.py
```
