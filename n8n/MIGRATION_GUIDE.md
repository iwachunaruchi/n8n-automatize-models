# Migración del Workflow "Generar imagenes sinteticas"

## Resumen de Cambios Realizados

Este documento describe las modificaciones necesarias para migrar el workflow de n8n desde la API original (`Proyectos docker`) a la nueva API refactorizada (`refactor api v1`).

## Cambios en las URLs de los Nodos

### 1. **Nodo: "HTTP Request" → "Classify Document"**

```
ANTES: /classify/image-quality-flexible
AHORA: /classify/document
```

**Cambios en la respuesta:**

- `classification` → `document_type`
- La respuesta ahora incluye `analysis` con métricas detalladas

### 2. **Nodo: "Generate Synthetic Data" → "Generate Training Pairs"**

```
ANTES: /generate/synthetic-data
AHORA: /synthetic/training-pairs
```

**Cambios en parámetros:**

- `source_bucket` + `source_file` + `target_count` → `clean_bucket` + `count`
- La nueva API genera pares automáticamente desde el bucket clean

### 3. **Nodos: "Upload to Clean/Degraded Bucket"**

```
ANTES: /upload/classified
AHORA: /files/upload
```

**Cambios en parámetros:**

- `classification` + `filename` (query params) → `bucket` (query param)
- La nueva API requiere especificar el bucket directamente

### 4. **Nodo: "Update Dataset Stats"**

```
ANTES: /dataset/stats
AHORA: /synthetic/stats/document-training
```

**Cambios:**

- Ahora es específico por bucket
- Estructura de respuesta simplificada

### 5. **Nodo: "Notify Completion" - REMOVIDO**

```
ANTES: /notify/dataset-updated
AHORA: No disponible en la nueva API
```

**Acción:** Este nodo fue removido del workflow actualizado ya que la funcionalidad no existe en la nueva API.

## Cambios en las Respuestas de la API

### Clasificación de Documentos

**ANTES:**

```json
{
  "classification": "clean|degraded",
  "confidence": 0.85,
  "metrics": {...}
}
```

**AHORA:**

```json
{
  "document_type": "documento_limpio|documento_borroso|documento_bajo_contraste|documento_oscuro",
  "analysis": {
    "sharpness": 150.5,
    "gradient": 45.2,
    "contrast": 67.8,
    "noise": 12.3,
    "quality_score": 180.4
  }
}
```

### Generación de Datos Sintéticos

**ANTES:**

```json
{
  "job_id": "uuid",
  "status": "generation_started"
}
```

**AHORA:**

```json
{
  "job_id": "uuid",
  "status": "pending",
  "requested_count": 10,
  "source_bucket": "document-clean"
}
```

## Cambios en la Lógica del Workflow

### 1. **Clasificación Mejorada**

- La nueva API clasifica con más detalle (documento_limpio, documento_borroso, etc.)
- Se actualizaron las condiciones del nodo "If" para usar `document_type` en lugar de `classification`

### 2. **Subida de Archivos Simplificada**

- Ya no se necesita especificar `classification` como parámetro
- El bucket se especifica directamente en el parámetro `bucket`

### 3. **Generación de Datos Sintéticos**

- La nueva API genera automáticamente pares de entrenamiento
- Se especifica el bucket fuente y la cantidad deseada
- La API maneja internamente la creación de versiones degradadas

## Nuevas Funcionalidades Disponibles

La nueva API incluye funcionalidades adicionales que podrías integrar:

### 1. **Datos Sintéticos Avanzados**

- `/synthetic/noise` - Agregar ruido específico
- `/synthetic/degrade` - Degradar imágenes limpias
- `/synthetic/augment` - Aumentar dataset

### 2. **Gestión de Archivos Mejorada**

- `/files/download/{bucket}/{filename}` - Descargar archivos
- `/files/delete/{bucket}/{filename}` - Eliminar archivos
- `/files/list/{bucket}` - Listar contenido

### 3. **Restauración Asíncrona**

- `/restore/document/async` - Restauración en background
- `/restore/batch` - Restauración por lotes

## Instalación del Workflow Actualizado

1. **Exporta** el workflow actual desde n8n
2. **Importa** el nuevo archivo: `Generar imagenes sinteticas_UPDATED.json`
3. **Verifica** que la URL de la API sea correcta en cada nodo
4. **Testa** el workflow con una imagen de prueba

## Verificación de Funcionamiento

Para verificar que el workflow migrado funciona correctamente:

1. **Envía una imagen** al webhook
2. **Verifica** que la clasificación funcione
3. **Confirma** que la subida al bucket correcto funcione
4. **Comprueba** que la generación de datos sintéticos se complete
5. **Revisa** las estadísticas del dataset

## Notas Importantes

- La nueva API es completamente modular y más robusta
- Los nombres de los tipos de documentos son más descriptivos
- La gestión de trabajos (jobs) es más completa
- Se removió la funcionalidad de notificaciones (puede reimplementarse si es necesaria)

## Próximos Pasos

1. Importar el workflow actualizado en n8n
2. Configurar las URLs correctas de la nueva API
3. Probar con datos reales
4. Considerar integrar las nuevas funcionalidades disponibles
