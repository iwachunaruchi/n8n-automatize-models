#!/bin/sh
# Script mejorado para configuración completa de MinIO
set -e

echo "🚀 Iniciando configuración completa de MinIO..."

# Configurar alias de MinIO
mc alias set myminio http://minio:9000 minio minio123

# Crear buckets básicos
echo "📦 Creando buckets básicos..."
mc mb myminio/document-degraded || true
mc mb myminio/document-clean || true
mc mb myminio/document-restored || true
mc mb myminio/document-training || true
mc mb myminio/models || true

# Configurar permisos públicos
echo "🔓 Configurando permisos públicos..."
mc anonymous set public myminio/document-degraded
mc anonymous set public myminio/document-clean
mc anonymous set public myminio/document-restored
mc anonymous set public myminio/document-training
mc anonymous set public myminio/models

# Crear estructura de carpetas para modelos preentrenados
echo "📁 Creando estructura de modelos preentrenados..."
echo '' | mc pipe myminio/models/pretrained_models/.keep
echo '' | mc pipe myminio/models/pretrained_models/layer_1/.keep
echo '' | mc pipe myminio/models/pretrained_models/layer_2/.keep
echo '' | mc pipe myminio/models/pretrained_models/layer_2/nafnet/.keep
echo '' | mc pipe myminio/models/pretrained_models/layer_2/docunet/.keep
echo '' | mc pipe myminio/models/pretrained_models/general/.keep

# Crear carpetas adicionales para organización
echo "📁 Creando carpetas adicionales..."
echo '' | mc pipe myminio/models/training_outputs/.keep
echo '' | mc pipe myminio/models/evaluation_results/.keep
echo '' | mc pipe myminio/models/checkpoints/.keep

# Crear archivos README detallados
echo "📄 Creando documentación..."

# README principal
cat > /tmp/main_readme.md << 'EOF'
# Modelos Preentrenados

Esta carpeta contiene todos los modelos preentrenados organizados por capas y tipos.

## Estructura:
- `layer_1/`: Modelos para la primera capa de restauración
- `layer_2/`: Modelos para la segunda capa de restauración
  - `nafnet/`: Modelos NAFNet (Noise Aware Filtering Network)
  - `docunet/`: Modelos DocUNet (Document Unwarping Network)
- `general/`: Modelos de propósito general

## Uso:
Los modelos en esta carpeta se utilizan como punto de partida para fine-tuning
específico del dominio de restauración de documentos.

## Configuración automática:
Esta estructura fue creada automáticamente por Docker Compose.
EOF

mc cp /tmp/main_readme.md myminio/models/pretrained_models/README.md

# README Layer 1
cat > /tmp/layer1_readme.md << 'EOF'
# Modelos Preentrenados - Layer 1

Modelos preentrenados para la primera capa de restauración de documentos.

## Propósito:
Primera etapa de procesamiento, enfocada en denoising básico y mejora inicial de calidad.

## Modelos recomendados:
- Restormer para denoising general
- DnCNN para ruido gaussiano
- FFDNet para ruido real

## Descarga automática:
Los modelos se descargan automáticamente durante el entrenamiento.
EOF

mc cp /tmp/layer1_readme.md myminio/models/pretrained_models/layer_1/README.md

# README NAFNet
cat > /tmp/nafnet_readme.md << 'EOF'
# NAFNet Pretrained Models

Modelos NAFNet (Noise Aware Filtering Network) preentrenados.

## Modelo recomendado:
- **NAFNet-SIDD-width64**: Óptimo para denoising de documentos
  - Entrenado en dataset SIDD (real-world noise)
  - Arquitectura equilibrada (width=64)
  - Excelente para fine-tuning

## Descarga:
Los modelos se descargan automáticamente durante el entrenamiento si no están presentes.

## Uso:
Utilizados como backbone para fine-tuning específico en documentos degradados.
EOF

mc cp /tmp/nafnet_readme.md myminio/models/pretrained_models/layer_2/nafnet/README.md

echo "✅ Configuración completa de MinIO finalizada exitosamente!"
echo "📋 Buckets creados:"
echo "   - document-degraded"
echo "   - document-clean" 
echo "   - document-restored"
echo "   - document-training"
echo "   - models (con estructura de carpetas)"
echo ""
echo "📁 Estructura de modelos configurada:"
echo "   - pretrained_models/layer_1/"
echo "   - pretrained_models/layer_2/nafnet/"
echo "   - pretrained_models/layer_2/docunet/"
echo "   - pretrained_models/general/"
echo ""
echo "🌐 Acceso MinIO Console: http://localhost:9001"
echo "   Usuario: minio"
echo "   Contraseña: minio123"
