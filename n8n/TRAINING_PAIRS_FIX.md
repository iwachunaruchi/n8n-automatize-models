# Corrección del Endpoint generate_training_pairs

## Problema Identificado

El endpoint `generate_training_pairs` solo generaba **2 archivos** (1 par) en lugar de **20 archivos** (10 pares) cuando había una sola imagen en el bucket.

## Cambios Realizados

### 1. **Servicio Corregido (`synthetic_data_service.py`)**

#### **Lógica Mejorada:**

```python
def generate_training_pairs(self, clean_bucket: str, count: int = 10) -> dict:
    # Si solo hay un archivo, generar múltiples versiones degradadas de ese archivo
    if len(clean_files) == 1:
        logger.info(f"Generando {count} versiones degradadas del archivo único")
        clean_file = clean_files[0]
        clean_data = minio_service.download_file(clean_bucket, clean_file)

        # Generar múltiples versiones degradadas
        for i in range(count):
            degraded_data = self.generate_degraded_version(clean_data, "mixed")

            # Crear archivos únicos
            pair_id = str(uuid.uuid4())
            clean_filename = f"clean_{pair_id}.png"
            degraded_filename = f"degraded_{pair_id}.png"

            # Subir AMBOS archivos al bucket de entrenamiento
            minio_service.upload_file(clean_data, BUCKETS['training'], clean_filename)
            minio_service.upload_file(degraded_data, BUCKETS['training'], degraded_filename)
```

#### **Resultado Esperado:**

- **Input:** 1 imagen limpia + count=10
- **Output:** 20 archivos (10 clean + 10 degraded) en `document-training`

### 2. **Workflow Corregido (`Generar imagenes sinteticas_FINAL_CORRECTED.json`)**

#### **Flujo Optimizado:**

```
Webhook → Analyze Only → Check Classification → Route → Upload → Generate Training Pairs → Monitor → Success
```

#### **Cambios Clave:**

1. **`/files/analyze`** - Solo analiza, no sube automáticamente
2. **Routing correcto** - Sube al bucket apropiado según clasificación
3. **`clean_bucket: "document-clean"`** - Fijo, no dinámico desde clasificación
4. **Lógica clara** - Solo genera sintéticos si es documento limpio

## Flujo Detallado

### **Para Imagen LIMPIA:**

1. **Analyze** → `document_type: "documento_limpio"`
2. **Upload** → Sube a `document-clean`
3. **Generate** → Toma de `document-clean`, genera 10 pares en `document-training`
4. **Result** → 20 archivos creados (10 clean + 10 degraded)

### **Para Imagen DEGRADADA:**

1. **Analyze** → `document_type: "documento_borroso"`
2. **Upload** → Sube a `document-degraded`
3. **Error Response** → "Only clean documents can generate synthetic training data"

## Ventajas de la Corrección

### ✅ **Funcionalidad Restaurada:**

- Ahora genera **10 pares** (20 archivos) como esperabas
- Cada par tiene variaciones aleatorias en la degradación
- Compatible con tu flujo actual de n8n

### ✅ **Logging Mejorado:**

```python
logger.info(f"Generando {count} versiones degradadas del archivo único: {clean_files[0]}")
logger.info(f"Generado par {i + 1}/{count}: {clean_filename} -> {degraded_filename}")
```

### ✅ **Respuesta Detallada:**

```json
{
  "status": "success",
  "generated_count": 10,
  "pairs": [...],
  "source_bucket": "document-clean",
  "total_files_created": 20
}
```

## Testing

### **Probar con Imagen Limpia:**

```bash
curl -X POST "http://localhost:5678/webhook/data-generation" \
  -F "File=@imagen_limpia.jpg"
```

**Resultado esperado:**

- 1 archivo en `document-clean`
- 20 archivos en `document-training` (10 clean_xxx.png + 10 degraded_xxx.png)

### **Verificar en MinIO:**

```bash
# Listar bucket de entrenamiento
curl "http://doc-restoration-api:8000/files/list/document-training"

# Debería mostrar 20 archivos con prefijos clean_ y degraded_
```

## Archivos Actualizados

1. **`synthetic_data_service.py`** - Lógica corregida para múltiples pares
2. **`Generar imagenes sinteticas_FINAL_CORRECTED.json`** - Workflow optimizado
3. **`TRAINING_PAIRS_FIX.md`** - Esta documentación

## Próximos Pasos

1. **Importar** el workflow corregido en n8n
2. **Probar** con una imagen limpia
3. **Verificar** que se crean 20 archivos en `document-training`
4. **Confirmar** que las degradaciones tienen variaciones aleatorias

¿Quieres que probemos el endpoint corregido ahora?
