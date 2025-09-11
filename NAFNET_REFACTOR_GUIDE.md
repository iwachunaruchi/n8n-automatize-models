# 🎯 Refactor NAFNet: Nueva Estructura de Datos Sintéticos

## 📋 Resumen del Cambio

Se ha refactorizado completamente el sistema de generación de datos sintéticos para seguir la estructura recomendada de entrenamiento de modelos **NAFNet**, organizando los datasets en una jerarquía clara y escalable.

## 🏗️ Nueva Estructura Organizacional

### Estructura de Directorios en MinIO

```
document-training/
└── datasets/
    └── NAFNet/                    # Core del modelo
        ├── SIDD-width64/          # Tarea específica (actual)
        │   ├── train/             # 80% de los datos
        │   │   ├── lq/           # Low-quality (degradadas)
        │   │   └── gt/           # Ground-truth (limpias)
        │   └── val/              # 20% de los datos
        │       ├── lq/           # Validación degradadas
        │       └── gt/           # Validación limpias
        ├── GoPro/                # Futura tarea (motion deblurring)
        │   ├── train/
        │   └── val/
        └── FLICKR1024/           # Futura tarea (super resolution)
            ├── train/
            └── val/
```

### Nomenclatura de Archivos

```
train_uuid-123_lq.png    # Imagen de entrenamiento degradada
train_uuid-123_gt.png    # Imagen de entrenamiento limpia
val_uuid-456_lq.png      # Imagen de validación degradada
val_uuid-456_gt.png      # Imagen de validación limpia
```

## 🔧 Nuevas Funcionalidades

### 1. Configuración de Tareas NAFNet

```python
NAFNET_CONFIG = {
    "CORE_NAME": "NAFNet",
    "CURRENT_TASK": "SIDD-width64",
    "AVAILABLE_TASKS": {
        "SIDD-width64": {
            "model_name": "NAFNet-SIDD-width64.pth",
            "task_description": "Image denoising trained on SIDD dataset",
            "degradation_types": ["gaussian_noise", "real_noise", "mixed_noise"],
            "recommended_intensity": {"min": 0.01, "max": 0.3},
            "image_size": [256, 256]
        },
        "GoPro": {
            "degradation_types": ["motion_blur", "gaussian_blur"],
            "recommended_intensity": {"min": 0.1, "max": 0.8}
        },
        "FLICKR1024": {
            "degradation_types": ["downsampling", "compression", "blur"],
            "recommended_intensity": {"min": 0.1, "max": 0.5}
        }
    }
}
```

### 2. Nuevos Métodos del Servicio

#### `generate_nafnet_training_dataset()`

```python
result = synthetic_data_service.generate_nafnet_training_dataset(
    source_bucket="document-clean",
    count=100,
    task="SIDD-width64",
    train_val_split=True
)
```

#### `get_nafnet_dataset_info()`

```python
info = synthetic_data_service.get_nafnet_dataset_info("SIDD-width64")
# Retorna estadísticas de train/val y rutas organizadas
```

#### `list_available_nafnet_tasks()`

```python
tasks = synthetic_data_service.list_available_nafnet_tasks()
# Lista todas las tareas disponibles y sus configuraciones
```

### 3. Nuevos Endpoints API

#### Generar Dataset Estructurado

```http
POST /synthetic/nafnet/dataset
{
    "source_bucket": "document-clean",
    "count": 100,
    "task": "SIDD-width64",
    "train_val_split": true
}
```

#### Obtener Información del Dataset

```http
GET /synthetic/nafnet/info/SIDD-width64
```

#### Listar Tareas Disponibles

```http
GET /synthetic/nafnet/tasks
```

#### Validar Dataset

```http
POST /synthetic/nafnet/validate/SIDD-width64
```

### 4. Nuevas Tareas RQ

#### `generate_nafnet_dataset_job`

- Genera dataset completo con estructura NAFNet
- División automática train/val
- Aplicación de degradaciones específicas por tarea

#### `validate_nafnet_dataset_job`

- Valida integridad de la estructura
- Verifica balance train/val
- Reporta estadísticas de salud del dataset

## 🎯 Degradaciones Específicas por Tarea

### SIDD-width64 (Image Denoising)

```python
degradation_types = ["gaussian_noise", "real_noise", "mixed_noise"]
# Optimizado para tareas de denoising
```

### GoPro (Motion Deblurring)

```python
degradation_types = ["motion_blur", "gaussian_blur"]
# Simula motion blur realista
```

### FLICKR1024 (Super Resolution)

```python
degradation_types = ["downsampling", "compression", "blur"]
# Simula pérdida de resolución
```

## 🚀 Guía de Uso

### 1. Verificar Tareas Disponibles

```bash
curl http://localhost:8000/synthetic/nafnet/tasks
```

### 2. Generar Dataset NAFNet

```bash
curl -X POST "http://localhost:8000/synthetic/nafnet/dataset" \
     -d "source_bucket=document-clean&count=50&task=SIDD-width64"
```

### 3. Verificar Progreso

```bash
curl http://localhost:8000/jobs/rq/{job_id}
```

### 4. Obtener Estadísticas

```bash
curl http://localhost:8000/synthetic/nafnet/info/SIDD-width64
```

### 5. Validar Dataset

```bash
curl -X POST http://localhost:8000/synthetic/nafnet/validate/SIDD-width64
```

## 📊 Ventajas del Nuevo Sistema

### ✅ **Organización Clara**

- Estructura jerárquica por modelo y tarea
- Separación clara de train/val
- Nomenclatura consistente

### ✅ **Escalabilidad**

- Fácil agregar nuevas tareas NAFNet
- Configuración centralizada
- Degradaciones específicas por tarea

### ✅ **Compatibilidad**

- Mantiene retrocompatibilidad con método legacy
- Nuevos endpoints especializados
- Migración gradual posible

### ✅ **Integración con Entrenamiento**

- Estructura directamente compatible con frameworks ML
- División automática train/val
- Metadatos de trazabilidad

### ✅ **Monitoreo y Validación**

- Estadísticas detalladas por split
- Validación de integridad automática
- Reportes de salud del dataset

## 🔄 Migración desde Sistema Anterior

### Método Legacy (Mantiene Compatibilidad)

```python
# Sigue funcionando como antes
result = synthetic_data_service.generate_training_pairs(
    clean_bucket="document-clean",
    count=10,
    use_nafnet_structure=False  # Usa estructura antigua
)
```

### Método NAFNet (Recomendado)

```python
# Nueva estructura organizacional
result = synthetic_data_service.generate_training_pairs(
    clean_bucket="document-clean",
    count=10,
    task="SIDD-width64",
    use_nafnet_structure=True  # Usa estructura NAFNet
)
```

## 🧪 Script de Demostración

Ejecutar el script de demo incluido:

```bash
python demo_nafnet_structure.py
```

Este script demuestra:

1. ✅ Verificación de servicios
2. 📋 Listado de tareas disponibles
3. 🎯 Generación de dataset estructurado
4. ⏰ Monitoreo de progreso
5. 📊 Obtención de estadísticas
6. 🔍 Validación de integridad

## 📁 Verificación en MinIO

Después de generar datos, verificar en MinIO Console:

- **URL**: http://localhost:9001
- **Credenciales**: minioadmin / minioadmin
- **Bucket**: document-training
- **Ruta**: `datasets/NAFNet/SIDD-width64/`

## 🎯 Próximos Pasos

1. **Entrenar con Nueva Estructura**: Actualizar scripts de entrenamiento para usar las nuevas rutas
2. **Agregar Más Tareas**: Implementar GoPro y FLICKR1024 cuando sea necesario
3. **Optimizar Degradaciones**: Refinar algoritmos de degradación por tarea
4. **Métricas Avanzadas**: Implementar métricas específicas por tipo de tarea

## 🐛 Troubleshooting

### Dataset No Generado

```bash
# Verificar logs del worker
docker logs doc-restoration-worker

# Verificar estado del job
curl http://localhost:8000/jobs/rq/{job_id}
```

### Estructura Incorrecta

```bash
# Validar dataset
curl -X POST http://localhost:8000/synthetic/nafnet/validate/SIDD-width64
```

### Problemas de Performance

```bash
# Verificar estadísticas de cola RQ
curl http://localhost:8000/jobs/rq/stats
```

---

🎉 **¡El sistema NAFNet está listo para usar!** La nueva estructura organizacional facilitará el entrenamiento de modelos y la gestión de datasets de manera escalable y profesional.
