# 🔧 Train Layers - Entrenamiento de Capas

Esta carpeta contiene únicamente los scripts de entrenamiento para las capas del pipeline.

## 📁 Estructura

```
train-layers/
├── train_layer_1.py    # Entrenamiento/evaluación de Capa 1 (Otsu + CLAHE + Deskew)
└── train_layer_2.py    # Entrenamiento de Capa 2 (NAFNet + DocUNet)
```

## 🚀 Uso

### Capa 1 - Pipeline de Preprocesamiento

```bash
cd layers/train-layers
python train_layer_1.py
```

- **Función**: Evalúa la efectividad del pipeline de preprocesamiento
- **Entrada**: Imágenes del bucket `document-degraded`
- **Salida**: Métricas de mejora y reportes visuales

### Capa 2 - NAFNet + DocUNet

```bash
cd layers/train-layers
python train_layer_2.py
```

- **Función**: Entrena modelos de denoising y dewarping
- **Entrada**: Pares sintéticos del bucket `document-training` (OPCIÓN 1)
- **Salida**: Modelos entrenados y métricas de pérdida

## 📊 Configuración actual

**Capa 2 está configurada para usar OPCIÓN 1:**

- ✅ Fuente de datos: bucket `document-training`
- ✅ Pares sintéticos generados por n8n (10 limpios + 10 degradados)
- ✅ Emparejamiento por UUID automático
- ✅ Optimizado para tu flujo actual

## 🔗 Archivos relacionados

- **Demos**: `demos/demo_api_training.py` - Demo interactivo completo
- **Workflows n8n**: `n8n/workflow_n8n_layer2.py` - Scripts específicos para n8n
- **API**: `api/routers/training.py` - Endpoints para entrenar via API

## 🎯 Para n8n

Usa los endpoints de la API en lugar de ejecutar directamente:

```http
# Evaluar Capa 1
POST /training/layer1/evaluate?max_images=30

# Entrenar Capa 2
POST /training/layer2/train?use_training_bucket=true&num_epochs=15&max_pairs=100

# Verificar estado
GET /training/status/{job_id}
```
