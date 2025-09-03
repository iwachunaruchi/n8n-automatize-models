# Sistema de Entrenamiento Automatizado de Modelos de Restauración de Imágenes

## 🎯 Descripción General

Este proyecto implementa un sistema completo para el entrenamiento automatizado de modelos de restauración de imágenes con las siguientes características principales:

- **Generación automática de datos sintéticos** para entrenar modelos
- **Pipeline de entrenamiento Layer 2** con seguimiento de métricas
- **Almacenamiento organizado en MinIO** con buckets especializados
- **Generación automática de reportes** detallados de entrenamiento
- **API REST completa** para gestión y monitoreo
- **Integración con n8n** para workflows automatizados
- **Arquitectura Docker** completamente containerizada

## 🚀 Inicio Rápido

### 1. Prerrequisitos

- Docker y Docker Compose
- Python 3.11+ (opcional, para desarrollo)
- 8GB RAM mínimo recomendado

### 2. Levantar el Sistema

```bash
# Clonar y navegar al directorio
cd c:\tesis\n8n-automatize-models

# Construir y levantar todos los servicios
docker-compose up --build -d

# Verificar que todo esté funcionando
python verify_system.py
```

### 3. Acceso a Servicios

- **API REST**: http://localhost:8000
- **Documentación API**: http://localhost:8000/docs
- **n8n Interface**: http://localhost:5678 (admin/admin)
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)

## 📁 Estructura del Proyecto

```
n8n-automatize-models/
├── 🐳 docker-compose.yml          # Orquestación de servicios
├── 🐳 Dockerfile                  # Imagen principal Python
├── 📦 pyproject.toml              # Dependencias Poetry
├── 🔧 verify_system.py            # Verificación rápida
├── 🎯 demo_complete_system.py     # Demo completa
│
├── api/                           # 🌐 API REST Principal
│   ├── main.py                    # FastAPI app
│   ├── routers/                   # Endpoints organizados
│   │   ├── training.py           # Entrenamiento + Reportes
│   │   ├── models.py             # Gestión de modelos
│   │   ├── synthetic_data.py     # Datos sintéticos
│   │   ├── files.py              # Gestión archivos
│   │   └── ...
│   ├── services/                  # 🔧 Lógica de negocio
│   │   ├── training_service.py           # Entrenamiento Layer 2
│   │   ├── training_report_service.py   # 📊 Reportes automáticos
│   │   ├── minio_service.py             # Almacenamiento
│   │   ├── synthetic_data_service.py    # Generación de datos
│   │   └── ...
│   └── models/                    # 📋 Schemas y modelos
│
├── layers/                        # 🧠 Modelos de ML
│   ├── layer-1/                  # Modelo Layer 1
│   └── train-layers/             # Scripts entrenamiento
│
├── data/                         # 📊 Datos de entrenamiento
│   ├── train/                    # Entrenamiento
│   └── val/                      # Validación
│
├── outputs/                      # 📤 Resultados
│   ├── checkpoints/              # Modelos entrenados
│   ├── analysis/                 # Análisis y gráficos
│   └── samples/                  # Muestras generadas
│
└── n8n/                         # 🔄 Workflows automatizados
    ├── workflows/                # Definiciones JSON
    └── helper scripts/           # Scripts auxiliares
```

## 🔧 Servicios y Componentes

### 🌐 API REST (Puerto 8000)

FastAPI con documentación automática en `/docs`

**Endpoints principales:**

- `GET /training/status` - Estado del sistema de entrenamiento
- `POST /training/layer2/start` - Iniciar entrenamiento Layer 2
- `GET /training/reports` - Listar reportes generados
- `GET /models/layer/{layer}` - Listar modelos por capa
- `POST /synthetic/generate` - Generar datos sintéticos

### 🗄️ MinIO (Puertos 9000/9001)

Almacenamiento de objetos organizado en buckets:

```
training-data/          # Datos de entrenamiento
  ├── clean/           # Imágenes limpias
  └── degraded/        # Imágenes degradadas

models/                 # Modelos entrenados organizados
  ├── layer_1/         # Modelos Layer 1
  └── layer_2/         # Modelos Layer 2

training-outputs/       # Resultados de entrenamiento
  ├── checkpoints/     # Puntos de control
  ├── analysis/        # Análisis y gráficos
  └── reports/         # 📊 Reportes automáticos

synthetic-data/         # Datos generados sintéticamente
evaluation/            # Resultados de evaluación
```

### 🔄 n8n (Puerto 5678)

Workflows automatizados:

- **Layer2 Training Workflow**: Entrenamiento automatizado
- **Synthetic Data Generation**: Generación de datos
- **Model Evaluation Pipeline**: Evaluación automática

### 🗃️ PostgreSQL

Base de datos para n8n y metadatos del sistema

## 🎯 Funcionalidades Principales

### 1. 🤖 Entrenamiento Automatizado

```python
# Ejemplo de uso del API
import requests

# Iniciar entrenamiento Layer 2
response = requests.post("http://localhost:8000/training/layer2/start", json={
    "num_epochs": 10,
    "batch_size": 4,
    "max_pairs": 30
})

job_id = response.json()["job_id"]
print(f"Entrenamiento iniciado: {job_id}")
```

### 2. 📊 Reportes Automáticos

Cada entrenamiento genera automáticamente un reporte detallado con:

- **Información del trabajo**: ID, timestamps, duración
- **Parámetros de entrenamiento**: épocas, batch size, learning rate
- **Estadísticas de datos**: archivos utilizados, pares válidos
- **Métricas por época**: Loss, PSNR, SSIM, accuracy
- **Información del modelo**: arquitectura, parámetros, tamaño
- **Comparación con modelos anteriores**
- **Configuración del entorno**: Hardware, versiones
- **Recomendaciones**: Sugerencias para mejoras

### 3. 🎨 Generación de Datos Sintéticos

```python
# Generar datos adicionales
response = requests.post("http://localhost:8000/synthetic/generate", json={
    "num_images": 20,
    "degradation_type": "document-clean"
})
```

### 4. 💾 Gestión de Modelos

- Almacenamiento automático en MinIO
- Organización por capas (layer_1, layer_2)
- Versionado automático con timestamps
- API para descarga y gestión

## 🔍 Monitoreo y Verificación

### Verificación Rápida

```bash
python verify_system.py
```

### Demo Completa

```bash
python demo_complete_system.py
```

### Logs en Tiempo Real

```bash
# Ver logs de todos los servicios
docker-compose logs -f

# Ver logs específicos
docker-compose logs -f api
docker-compose logs -f n8n
```

## 📊 Ejemplo de Reporte de Entrenamiento

```
================================================================================
                           REPORTE DE ENTRENAMIENTO LAYER 2
================================================================================

🔍 INFORMACIÓN DEL TRABAJO
Job ID: training_20241213_143022
Iniciado: 2024-12-13 14:30:22
Completado: 2024-12-13 14:35:18
Duración: 4 minutos 56 segundos
Estado: COMPLETADO

⚙️ PARÁMETROS DE ENTRENAMIENTO
Épocas: 8
Batch Size: 2
Learning Rate: 0.0001
Pares máximos: 25
Optimizador: AdamW

📊 ESTADÍSTICAS DE DATOS
Total archivos procesados: 62
Pares válidos utilizados: 25
Archivos clean: 31
Archivos degraded: 31
Efectividad de datos: 80.6%

📈 MÉTRICAS DE ENTRENAMIENTO
Época 1: Loss=0.0932, PSNR=28.45, SSIM=0.815, Accuracy=85.2%
Época 2: Loss=0.0654, PSNR=29.12, SSIM=0.834, Accuracy=87.8%
...
Época 8: Loss=0.0234, PSNR=32.18, SSIM=0.951, Accuracy=95.3%

Mejora Total:
- Loss: 0.0932 → 0.0234 (-75.1%)
- PSNR: 28.45 → 32.18 (+13.1%)
- SSIM: 0.815 → 0.951 (+16.7%)
- Accuracy: 85.2% → 95.3% (+10.1%)

🤖 INFORMACIÓN DEL MODELO
Modelo guardado: models/layer_2/restormer_layer2_20241213_143518.pth
Tamaño: 100.21 KB
Parámetros: ~25,042
Arquitectura: Restormer

🔄 COMPARACIÓN CON MODELO ANTERIOR
Modelo anterior: restormer_layer2_20241213_141205.pth
Mejora en PSNR: +1.23 dB
Mejora en SSIM: +0.043
Reducción en Loss: -0.0156

💻 CONFIGURACIÓN DEL ENTORNO
Python: 3.11.6
PyTorch: 2.1.0
CUDA disponible: No
Memoria utilizada: ~2.1 GB

💡 RECOMENDACIONES
✅ Entrenamiento exitoso con buenas métricas
✅ Mejora consistente respecto al modelo anterior
⚠️ Considerar usar GPU para entrenamientos más largos
💡 Los datos actuales son suficientes para este nivel de entrenamiento

================================================================================
Reporte generado automáticamente el 2024-12-13 14:35:18
================================================================================
```

## 🛠️ Desarrollo y Personalización

### Agregar Nuevos Endpoints

1. Crear nuevo router en `api/routers/`
2. Implementar servicio en `api/services/`
3. Registrar en `api/main.py`

### Modificar Pipeline de Entrenamiento

- Editar `api/services/training_service.py`
- Personalizar métricas en `train-layers/train_layer_2.py`
- Ajustar reportes en `api/services/training_report_service.py`

### Crear Nuevos Workflows n8n

1. Acceder a http://localhost:5678
2. Crear workflow visualmente
3. Exportar JSON a `n8n/workflows/`

## 🔧 Solución de Problemas

### Problema: Servicios no inician

```bash
# Verificar puertos ocupados
netstat -an | findstr "8000\|5678\|9000"

# Reiniciar servicios
docker-compose down
docker-compose up --build -d
```

### Problema: Falta de datos para entrenamiento

```bash
# Generar datos sintéticos
curl -X POST "http://localhost:8000/synthetic/generate" \
     -H "Content-Type: application/json" \
     -d '{"num_images": 30, "degradation_type": "document-clean"}'
```

### Problema: Modelos no se guardan

- Verificar conexión MinIO en logs
- Comprobar buckets existen: http://localhost:9001
- Revisar permisos en `docker-compose.yml`

## 📈 Métricas y KPIs

El sistema rastrea automáticamente:

- **Tiempo de entrenamiento** por época
- **Mejora de métricas** (Loss, PSNR, SSIM)
- **Uso de recursos** (memoria, CPU)
- **Eficiencia de datos** (pares válidos/total)
- **Progreso histórico** comparando modelos

## 🎉 Características Destacadas

✅ **Completamente dockerizado** - Sin problemas de dependencias  
✅ **Reportes automáticos** - Documentación detallada de cada entrenamiento  
✅ **Almacenamiento organizado** - Buckets especializados en MinIO  
✅ **API REST completa** - Endpoints para todas las operaciones  
✅ **Workflows n8n** - Automatización visual sin código  
✅ **Monitoreo en tiempo real** - Estado y progreso del entrenamiento  
✅ **Generación de datos sintéticos** - Ampliación automática del dataset  
✅ **Gestión de modelos** - Versionado y comparación automática

## 🏆 Estado del Sistema

**✅ SISTEMA COMPLETO Y OPERATIVO**

- Migración exitosa a Poetry para gestión de dependencias
- Dockerización completa con Docker Compose
- Sistema de almacenamiento MinIO con organización automática
- Generación automática de reportes detallados
- Pipeline de entrenamiento Layer 2 completamente funcional
- API REST con más de 50 endpoints
- Integración completa con n8n workflows

¡El sistema está listo para uso en producción! 🚀
