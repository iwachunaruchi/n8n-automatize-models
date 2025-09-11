#!/bin/sh
# MinIO Setup Script - Configuración completa de buckets y estructura NAFNet
# Este script configura toda la infraestructura de almacenamiento para el sistema

set -e  # Detener en cualquier error

echo "🚀 INICIANDO CONFIGURACIÓN DE MINIO"
echo "=================================="

# Configurar colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo "${RED}[ERROR]${NC} $1"
}

# Esperar a que MinIO esté listo
log_info "Esperando a que MinIO esté disponible..."
sleep 10

# Configurar alias MinIO
log_info "Configurando alias MinIO..."
mc alias set myminio http://minio:9000 minio minio123
if [ $? -eq 0 ]; then
    log_success "Alias MinIO configurado correctamente"
else
    log_error "Error configurando alias MinIO"
    exit 1
fi

# ===== CREAR BUCKETS BÁSICOS =====
log_info "Creando buckets básicos..."

create_bucket() {
    local bucket_name=$1
    log_info "Creando bucket: $bucket_name"
    mc mb myminio/$bucket_name 2>/dev/null || log_warning "Bucket $bucket_name ya existe"
    mc anonymous set public myminio/$bucket_name
    log_success "Bucket $bucket_name configurado"
}

create_bucket "document-degraded"
create_bucket "document-clean"
create_bucket "document-restored"
create_bucket "document-training"
create_bucket "models"

# ===== CREAR ESTRUCTURA DE MODELOS PREENTRENADOS =====
log_info "Creando estructura de modelos preentrenados..."

create_folder() {
    local path=$1
    local description=$2
    log_info "Creando: $description"
    printf '' | mc pipe myminio/$path
    if [ $? -eq 0 ]; then
        log_success "✓ $description"
    else
        log_error "✗ Error creando $description"
        return 1
    fi
}

# Estructura de modelos
create_folder "models/pretrained_models/.keep" "models/pretrained_models/"
create_folder "models/pretrained_models/layer_1/.keep" "models/pretrained_models/layer_1/"
create_folder "models/pretrained_models/layer_2/.keep" "models/pretrained_models/layer_2/"
create_folder "models/pretrained_models/layer_2/nafnet/.keep" "models/pretrained_models/layer_2/nafnet/"
create_folder "models/pretrained_models/layer_2/docunet/.keep" "models/pretrained_models/layer_2/docunet/"
create_folder "models/pretrained_models/general/.keep" "models/pretrained_models/general/"

# ===== CREAR ESTRUCTURA NAFNET COMPLETA =====
log_info "🎯 Creando estructura NAFNet para datasets..."

# Crear estructura base
create_folder "document-training/datasets/.keep" "datasets base folder"
create_folder "document-training/datasets/NAFNet/.keep" "NAFNet base folder"

# Función para crear estructura completa de una tarea NAFNet
create_nafnet_task() {
    local task_name=$1
    local description=$2
    
    log_info "📁 Creando estructura para tarea: $task_name ($description)"
    
    # Carpeta principal de la tarea
    create_folder "document-training/datasets/NAFNet/$task_name/.keep" "$task_name base"
    
    # Carpetas de entrenamiento
    create_folder "document-training/datasets/NAFNet/$task_name/train/.keep" "$task_name/train"
    create_folder "document-training/datasets/NAFNet/$task_name/train/lq/.keep" "$task_name/train/lq (Low Quality)"
    create_folder "document-training/datasets/NAFNet/$task_name/train/gt/.keep" "$task_name/train/gt (Ground Truth)"
    
    # Carpetas de validación
    create_folder "document-training/datasets/NAFNet/$task_name/val/.keep" "$task_name/val"
    create_folder "document-training/datasets/NAFNet/$task_name/val/lq/.keep" "$task_name/val/lq (Low Quality)"
    create_folder "document-training/datasets/NAFNet/$task_name/val/gt/.keep" "$task_name/val/gt (Ground Truth)"
    
    log_success "✅ Estructura $task_name completada"
    sleep 1  # Pausa para evitar sobrecarga
}

# Crear las tres tareas NAFNet
create_nafnet_task "SIDD-width64" "Image Denoising"
create_nafnet_task "GoPro" "Motion Deblurring"
create_nafnet_task "FLICKR1024" "Super Resolution"

# ===== CREAR ARCHIVOS README Y DOCUMENTACIÓN =====
log_info "📝 Creando documentación README..."

create_readme() {
    local path=$1
    local content=$2
    local description=$3
    
    log_info "Creando README: $description"
    printf '%s\n' "$content" | mc pipe myminio/$path
    if [ $? -eq 0 ]; then
        log_success "✓ README: $description"
    else
        log_error "✗ Error creando README: $description"
    fi
}

# READMEs de modelos
create_readme "models/pretrained_models/README.md" \
    "# Modelos Preentrenados - Estructura configurada automáticamente" \
    "Modelos preentrenados"

create_readme "models/pretrained_models/layer_1/README.md" \
    "# Layer 1 Models - Para primera capa de restauración" \
    "Layer 1"

create_readme "models/pretrained_models/layer_2/nafnet/README.md" \
    "# NAFNet Models - Modelos especializados en denoising" \
    "NAFNet"

create_readme "models/pretrained_models/layer_2/docunet/README.md" \
    "# DocUNet Models - Modelos para unwrapping de documentos" \
    "DocUNet"

# READMEs de datasets NAFNet
create_readme "document-training/datasets/NAFNet/README.md" \
    "# NAFNet Datasets Structure - Configurado automáticamente" \
    "NAFNet datasets"

create_readme "document-training/datasets/NAFNet/SIDD-width64/README.md" \
    "# SIDD-width64 Dataset - Image Denoising" \
    "SIDD-width64"

create_readme "document-training/datasets/NAFNet/GoPro/README.md" \
    "# GoPro Dataset - Motion Deblurring" \
    "GoPro"

create_readme "document-training/datasets/NAFNet/FLICKR1024/README.md" \
    "# FLICKR1024 Dataset - Super Resolution" \
    "FLICKR1024"

# ===== VERIFICACIÓN FINAL =====
log_info "🔍 Verificando estructura creada..."

echo ""
log_info "Estructura de buckets:"
mc ls myminio/

echo ""
log_info "Estructura de document-training:"
mc ls -r myminio/document-training/ | head -20

echo ""
log_info "Estructura de modelos:"
mc ls -r myminio/models/ | head -10

# ===== RESUMEN FINAL =====
echo ""
echo "🎉 ==============================================="
echo "🎉 CONFIGURACIÓN DE MINIO COMPLETADA EXITOSAMENTE"
echo "🎉 ==============================================="
echo ""
log_success "Buckets creados: document-degraded, document-clean, document-restored, document-training, models"
log_success "Estructura de modelos preentrenados configurada"
log_success "Estructura NAFNet completa configurada:"
echo "   📁 document-training/datasets/NAFNet/SIDD-width64/train|val/lq|gt"
echo "   📁 document-training/datasets/NAFNet/GoPro/train|val/lq|gt"
echo "   📁 document-training/datasets/NAFNet/FLICKR1024/train|val/lq|gt"
log_success "Documentación README creada para todas las estructuras"
echo ""
echo "🌐 Acceso MinIO Console: http://localhost:9001 (minio/minio123)"
echo "📊 RQ Dashboard: http://localhost:9181"
echo "🎯 Sistema listo para generación de datasets NAFNet!"
echo ""
