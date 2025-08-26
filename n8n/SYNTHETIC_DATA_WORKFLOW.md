# 🔄 WORKFLOW: Generación Automática de Datos Sintéticos

## 📋 Descripción General

Este workflow automatiza la **generación de datos sintéticos** para expandir tu dataset de entrenamiento. Clasifica automáticamente las imágenes como **clean** o **degraded** y genera **10 variaciones sintéticas** para cada imagen subida.

## 🎯 Objetivo Principal

**Automatizar la primera fase del proceso de generación de datos sintéticos:**

1. ✅ Recepción automática de imágenes
2. ✅ Clasificación inteligente (clean vs degraded)
3. ✅ Almacenamiento en buckets apropiados
4. ✅ Generación de 10 variaciones sintéticas
5. ✅ Actualización automática de estadísticas

## 🔄 Flujo del Workflow

### 📥 **PASO 1: Webhook - Data Upload**

- **Endpoint**: `/webhook/data-generation`
- **Método**: POST multipart/form-data
- **Input**: Imagen original (cualquier formato)

### 🔍 **PASO 2: Classify Image Quality**

- **API**: `POST /classify/image-quality`
- **Función**: Análisis automático de calidad de imagen
- **Métricas**:
  - 🔍 **Sharpness** (Varianza del Laplaciano)
  - 📊 **Gradient** (Magnitud promedio)
  - 🌈 **Contrast** (Desviación estándar)
  - 🔧 **Noise Level** (Análisis gaussiano)
- **Output**: `"clean"` o `"degraded"` + métricas

### 🔀 **PASO 3: Check Classification**

- **Decisión**: Rutear según clasificación
- **Branch A**: Si `classification == "clean"` → Upload to Clean Bucket
- **Branch B**: Si `classification == "degraded"` → Upload to Degraded Bucket

### ☁️ **PASO 4A: Upload to Clean Bucket**

- **Destino**: `document-clean` bucket en MinIO
- **Siguiente**: Generar versiones degradadas

### ☁️ **PASO 4B: Upload to Degraded Bucket**

- **Destino**: `document-degraded` bucket en MinIO
- **Siguiente**: Generar variaciones adicionales

### 🤖 **PASO 5: Generate Synthetic Data**

- **API**: `POST /generate/synthetic-data`
- **Parámetros**:
  ```json
  {
    "source_bucket": "document-clean|document-degraded",
    "source_file": "filename.png",
    "target_count": 10,
    "generation_type": "degradation|variation",
    "output_bucket": "document-training"
  }
  ```

### ⏱️ **PASO 6: Check Generation Status + Wait Loop**

- **API**: `GET /jobs/{job_id}`
- **Monitoreo**: Verifica cada 5 segundos
- **Estados**: `pending` → `processing` → `completed|failed`

### 📊 **PASO 7: Update Dataset Stats**

- **API**: `GET /dataset/stats?include_new=true`
- **Función**: Recalcula estadísticas del dataset

### 🔔 **PASO 8: Notify Completion**

- **API**: `POST /notify/dataset-updated`
- **Función**: Notifica expansión del dataset
- **Output**: Log de eventos y métricas

### ✅ **PASO 9: Response**

- **Success**: Retorna estadísticas de generación
- **Error**: Retorna detalles del error

## 🎯 Tipos de Generación Sintética

### 🔧 **Degradation (desde imágenes clean)**

1. **Ruido Gaussiano**: σ = 5-15, probabilidad 80%
2. **Blur Gaussiano**: kernel 3x3, 5x5, 7x7, probabilidad 60%
3. **Compresión JPEG**: quality 30-70, probabilidad 70%
4. **Ajuste Brillo/Contraste**: α = 0.7-1.3, β = ±30, probabilidad 50%

### 🔄 **Variation (desde imágenes degraded)**

1. **Rotación Sutil**: ±2 grados, probabilidad 40%
2. **Gamma Correction**: γ = 0.8-1.2, probabilidad 60%
3. **Salt-and-Pepper**: 2% pixels, probabilidad 30%
4. **Transformaciones Geométricas**: Preservando degradación

## 📊 Métricas de Clasificación

### 🔍 **Quality Score Calculation**

```python
quality_score = (
    laplacian_var * 0.4 +      # Sharpness weight
    avg_gradient * 0.3 +       # Edge definition
    contrast * 0.2 +           # Dynamic range
    (100 - noise_level) * 0.1  # Noise penalty
)

classification = "clean" if quality_score > 150 else "degraded"
```

### 📈 **Thresholds Empíricos**

- **Clean**: Quality Score > 150
- **Degraded**: Quality Score ≤ 150
- **Confidence**: Normalizado a [0,1]

## 🌐 Endpoints API Nuevos

### `POST /classify/image-quality`

```python
# Input: Multipart file
# Output:
{
  "classification": "clean|degraded",
  "confidence": 0.85,
  "metrics": {
    "sharpness": 245.7,
    "gradient": 12.3,
    "contrast": 45.2,
    "noise": 8.1,
    "quality_score": 187.5
  },
  "filename": "document.png"
}
```

### `POST /generate/synthetic-data`

```python
# Input: JSON parameters
# Output:
{
  "job_id": "uuid-string",
  "status": "generation_started"
}
```

### `GET /dataset/stats`

```python
# Output:
{
  "buckets": {
    "document-clean": {"count": 15, "total_size_mb": 45.2},
    "document-degraded": {"count": 180, "total_size_mb": 234.7},
    "document-training": {"count": 1500, "total_size_mb": 890.3}
  },
  "total_samples": 1695,
  "timestamp": "2025-08-23T16:30:00"
}
```

## 🧪 Testing del Workflow

### **Método 1: n8n Webhook**

```bash
curl -X POST \
  -F "file=@document.png" \
  http://localhost:5678/webhook/data-generation
```

### **Método 2: Cliente Python**

```python
from api.synthetic_data_client import SyntheticDataClient

client = SyntheticDataClient()
result = client.upload_for_synthetic_generation("test_image.png")
print(result)
```

### **Método 3: API Directa**

```python
# 1. Clasificar
classification = client.classify_image_quality("image.png")

# 2. Generar sintéticos
job = client.start_synthetic_generation(
    source_bucket="document-clean",
    source_file="image.png",
    target_count=10
)

# 3. Monitorear
status = client.monitor_generation_job(job['job_id'])
```

## 📈 Beneficios del Workflow

✅ **Automatización Completa**: Sin intervención manual  
✅ **Clasificación Inteligente**: Análisis automático de calidad  
✅ **Expansión Masiva**: 10x multiplicación del dataset  
✅ **Almacenamiento Organizado**: Buckets separados por tipo  
✅ **Monitoreo en Tiempo Real**: Estado de generación  
✅ **Escalabilidad**: Procesamiento en background  
✅ **Integración Total**: Compatible con pipeline existente

## 🔄 Integración con Pipeline Principal

Este workflow **alimenta automáticamente** tu pipeline de entrenamiento:

```
Imagen Original → Clasificación → Generación Sintética → Training Dataset → Transfer Learning Gradual
```

¡Tu sistema ahora puede **expandir automáticamente** el dataset con solo subir imágenes! 🚀
