# Guía de Optimización de Memoria para API de Procesamiento de Imágenes

## Resumen de Mejoras Implementadas

La API ahora está optimizada para manejar imágenes de alta resolución (como 1200 DPI) sin causar errores de memoria que puedan derribar la aplicación.

## Nuevas Características

### 1. Manejo Inteligente de Memoria

- **Verificación de memoria disponible**: La API verifica la memoria disponible antes de procesar imágenes
- **Redimensionamiento automático**: Imágenes grandes se redimensionan automáticamente para análisis
- **Limpieza de memoria**: Liberación proactiva de memoria durante el procesamiento

### 2. Manejo Robusto de Errores

- **Error 507 (Insufficient Storage)**: Para errores de memoria específicos
- **Respuestas graceful**: La aplicación continúa funcionando incluso cuando una imagen es demasiado grande
- **Mensajes informativos**: Sugerencias claras sobre cómo resolver problemas de memoria

### 3. Procesamiento Optimizado

#### Configuración de Tamaños:

- **analysis_size**: 1024px (tamaño máximo para análisis)
- **max_image_size**: 2048px (tamaño máximo general)
- **sample_size**: 512px (para análisis de frecuencias)

#### Algoritmos Mejorados:

- **Interpolación LANCZOS4**: Para redimensionamiento de alta calidad
- **Análisis por muestras**: Para imágenes muy grandes, se analizan secciones representativas
- **Limpieza automática**: Liberación de memoria con `gc.collect()`

## Nuevos Formatos de Respuesta

### Clasificación de Documentos

**Antes:**

```json
{
  "document_type": "documento_limpio",
  "analysis": {...}
}
```

**Ahora:**

```json
{
  "status": "success",
  "classification": {
    "type": "clean",
    "confidence": 0.85,
    "details": {
      "reason": "Imagen de alta calidad detectada",
      "quality_indicators": {
        "low_noise": true,
        "high_sharpness": true,
        "good_contrast": true,
        "clear_edges": true
      },
      "quality_score": 0.85,
      "metrics": {
        "noise_level": 12.3,
        "blur_score": 650.2,
        "contrast": 85.7,
        "edge_density": 0.15,
        "high_freq_energy": 850000.0
      },
      "resolution": "2400x3200",
      "memory_optimized": true
    }
  },
  "analysis": {...}
}
```

### Respuestas de Error de Memoria

```json
{
  "status": "error",
  "error": "Memoria insuficiente",
  "message": "La imagen es demasiado grande para procesar. Intente con una imagen de menor resolución.",
  "details": "Cannot allocate memory (os error 12)"
}
```

## Endpoints Actualizados

### 1. `/files/analyze`

- Ahora maneja errores de memoria gracefully
- Retorna código 507 para problemas de memoria
- Incluye métricas detalladas de calidad

### 2. `/classify/document`

- Clasificación mejorada con confianza
- Manejo de errores de memoria
- Upload condicional (solo si hay memoria suficiente)

### 3. `/classify/batch`

- Procesamiento por lotes con resiliencia
- Estadísticas de errores de memoria
- Continuación del procesamiento aunque fallen algunas imágenes

## Configuración Recomendada

### Para n8n Workflows

```javascript
// Verificar respuesta de análisis
if (response.status === 'error' && response.error === 'Memoria insuficiente') {
  // Manejar error de memoria
  console.log('Imagen demasiado grande:', response.message);
  // Posible acción: redimensionar imagen o usar endpoint diferente
} else if (response.status === 'success') {
  // Procesar clasificación normal
  const documentType = response.classification.type;
  const confidence = response.classification.confidence;
}
```

### Valores de Confianza

- **0.8 - 1.0**: Alta confianza
- **0.6 - 0.8**: Confianza media
- **0.0 - 0.6**: Baja confianza (revisar manualmente)

## Monitoreo y Diagnóstico

### Logs Informativos

La API ahora incluye logs detallados:

```
INFO: Imagen redimensionada para análisis: 2400x3200 -> 1024x1365
INFO: Imagen: 15.2MB, Memoria disponible: 1024.5MB
ERROR: Error de memoria analizando imagen: Cannot allocate memory
```

### Métricas Incluidas en Respuestas

- `memory_optimized`: Indica si la imagen fue redimensionada
- `resolution`: Resolución original de la imagen
- `total_pixels`: Número total de píxeles
- `quality_score`: Puntuación de calidad general (0-1)

## Recomendaciones de Uso

### Para Imágenes Grandes (>10MB)

1. **Usar `/files/analyze` primero**: Para verificar si la imagen se puede procesar
2. **Preparar handling de errores**: Implementar manejo para códigos 507
3. **Considerar pre-procesamiento**: Redimensionar imágenes antes de enviar a la API

### Para Procesamiento en Lote

1. **Usar `/classify/batch`**: Optimizado para múltiples archivos
2. **Revisar estadísticas**: Verificar cuántas imágenes tuvieron errores de memoria
3. **Procesar por partes**: Para buckets muy grandes, procesar en chunks

## Resolución de Problemas

### Error "Cannot allocate memory (os error 12)"

**Soluciones:**

1. Reducir resolución de imagen antes de enviar
2. Verificar memoria disponible del servidor
3. Procesar imágenes de una en una en lugar de lotes grandes

### Clasificación con Baja Confianza

**Causas posibles:**

1. Imagen en zona límite entre clean/degraded
2. Tipo de degradación no común
3. Imagen redimensionada perdió detalles importantes

**Soluciones:**

1. Revisar métricas detalladas en `classification.details.metrics`
2. Considerar análisis manual para casos de baja confianza
3. Ajustar umbrales si es necesario

## Próximos Pasos Sugeridos

1. **Implementar pre-procesamiento**: Redimensionar imágenes automáticamente en n8n
2. **Monitoreo de memoria**: Alertas cuando la memoria del servidor esté baja
3. **Cache de resultados**: Para evitar re-procesar imágenes grandes
4. **Análisis progresivo**: Análisis en múltiples resoluciones si es necesario

La API ahora es mucho más robusta y puede manejar imágenes de cualquier tamaño sin comprometer la estabilidad del sistema.
