# 🔬 Model Investigation Scripts

Scripts para investigar y analizar la estructura de modelos preentrenados, útiles para entender cómo integrar nuevos modelos en el pipeline.

## 📋 Scripts Incluidos

### `investigate_nafnet.py`
Script de referencia para investigar modelos NAFNet y entender su estructura interna.

**Funcionalidades:**
- 🔍 Descarga modelo desde MinIO
- 📊 Analiza estructura del checkpoint
- 🏗️ Inspecciona arquitectura y parámetros
- 📝 Genera reporte detallado
- 🧹 Limpieza automática de archivos temporales

## 🎯 Propósito

### Análisis de Modelos Nuevos
Cuando quieras integrar un modelo nuevo:
1. **Estructura**: ¿Qué claves tiene el checkpoint?
2. **Parámetros**: ¿Cómo están organizados los pesos?
3. **Compatibilidad**: ¿Es compatible con nuestro pipeline?
4. **Adaptación**: ¿Qué cambios necesita el código?

### Debugging de Integración
Si hay problemas al cargar un modelo:
- Verificar formato del checkpoint
- Identificar claves faltantes o incorrectas
- Entender diferencias entre arquitecturas

## 🔧 Uso

### Investigación Básica
```bash
cd scripts/investigations
python investigate_nafnet.py
```

### Para Otros Modelos
Copia y adapta el script:
```bash
cp investigate_nafnet.py investigate_restormer.py
# Edita las rutas y configuración específica
```

## 📊 Output Típico

El script genera información como:
```
🔍 Investigando estructura del modelo NAFNet...
📥 Descargando modelo...
🔍 Cargando checkpoint...

Tipo de checkpoint: <class 'dict'>
Claves principales: ['params', 'optimizer', 'epoch', 'best_psnr']

Clave 'params':
  Tipo: <class 'dict'>
  Sub-claves: ['intro.weight', 'intro.bias', 'downs.0.weight', ...]

Parámetros en 'params':
  intro.weight: torch.Size([64, 3, 3, 3])
  intro.bias: torch.Size([64])
  downs.0.weight: torch.Size([128, 64, 2, 2])
  ...
```

## 🛠️ Plantilla para Nuevos Modelos

### Script Base
```python
#!/usr/bin/env python3
"""
Script para investigar modelo [NOMBRE_MODELO]
"""
import sys
sys.path.append('.')

from api.services.minio_service import minio_service
import torch
import os

def investigate_model():
    """Investigar estructura del modelo"""
    print(f"🔍 Investigando modelo [NOMBRE_MODELO]...")
    
    try:
        # Configuración específica
        bucket = 'models'
        model_path = 'pretrained_models/layer_X/[tipo]/[modelo].pth'
        
        # Descargar y analizar
        model_data = minio_service.download_file(bucket, model_path)
        
        # Análisis detallado aquí...
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    investigate_model()
```

## 📚 Casos de Uso Comunes

### 1. Integrar Modelo Nuevo
```bash
# Investigar estructura
python investigate_new_model.py

# Identificar parámetros compatibles
# Adaptar código de carga en el pipeline
```

### 2. Debug de Transferencia de Pesos
```bash
# Verificar qué capas son compatibles
# Entender diferencias de arquitectura
# Optimizar transferencia de parámetros
```

### 3. Análisis de Checkpoints
```bash
# Entender formato específico del modelo
# Identificar metadatos útiles
# Verificar integridad del archivo
```

## 🔄 Integración con Pipeline

Los resultados de estos scripts ayudan a:

### Código de Carga
```python
# En train_layer_X.py
def load_pretrained_model():
    # Basado en investigación previa
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    else:
        model.load_state_dict(checkpoint)
```

### Transferencia de Capas
```python
# Transferir solo capas compatibles
compatible_layers = [
    'intro.weight', 'intro.bias',
    'downs.0.weight', 'downs.0.bias'
]
```

### Adaptación de Arquitectura
```python
# Modificaciones basadas en estructura analizada
if model_type == 'nafnet':
    # Configuración específica para NAFNet
elif model_type == 'restormer':
    # Configuración específica para Restormer
```

## 📋 Checklist de Investigación

Para cada modelo nuevo:
- [ ] ¿Qué tipo de checkpoint es? (dict, state_dict, modelo completo)
- [ ] ¿Qué claves principales tiene?
- [ ] ¿Están los pesos en 'params', 'state_dict', o directamente?
- [ ] ¿Qué metadatos adicionales incluye?
- [ ] ¿Es compatible con nuestra arquitectura?
- [ ] ¿Qué capas podemos transferir?
- [ ] ¿Necesita adaptaciones específicas?

## 🌟 Scripts Futuros

Plantillas para crear:
- `investigate_restormer.py` - Para modelos Restormer
- `investigate_docunet.py` - Para modelos DocUNet
- `investigate_custom.py` - Para modelos personalizados
- `compare_models.py` - Comparar estructuras entre modelos

## 🔗 Referencias

Estos scripts complementan:
- **Model management**: Entender antes de descargar/subir
- **Training pipeline**: Integración correcta en el entrenamiento
- **API service**: Carga eficiente en producción

## 🎓 Consejos de Uso

1. **Siempre investiga antes de integrar** un modelo nuevo
2. **Documenta los hallazgos** para futura referencia
3. **Compara con modelos existentes** para identificar patrones
4. **Prueba la compatibilidad** antes de usar en producción
