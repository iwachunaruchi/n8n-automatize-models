#!/bin/bash

# Script de inicio para desarrollo con Docker
# Levanta API + Worker con hot reload habilitado

echo "🚀 Iniciando entorno de desarrollo..."
echo "🔧 API + Worker Modular con Hot Reload"
echo "=" * 50

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Función para mostrar logs de manera bonita
show_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

show_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

show_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar que Docker está corriendo
if ! docker info > /dev/null 2>&1; then
    show_error "Docker no está corriendo. Por favor inicia Docker Desktop."
    exit 1
fi

# Cambiar al directorio docker
cd "$(dirname "$0")"

# Limpiar contenedores previos si existen
show_status "Limpiando contenedores previos..."
docker-compose down --remove-orphans

# Construir imágenes con cache
show_status "Construyendo imágenes..."
docker-compose build --parallel

# Iniciar servicios base (MinIO, PostgreSQL)
show_status "Iniciando servicios base (MinIO, PostgreSQL)..."
docker-compose up -d minio postgres

# Esperar a que los servicios base estén listos
show_status "Esperando a que los servicios base estén listos..."
sleep 10

# Configurar MinIO
show_status "Configurando MinIO..."
docker-compose up --no-deps minio-setup

# Iniciar API y Worker con hot reload
show_status "Iniciando API y Worker con hot reload..."
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d doc-restoration-api job-worker

# Mostrar información de conexión
echo ""
echo "=" * 60
show_status "🎉 Entorno de desarrollo listo!"
echo ""
echo "📊 Servicios disponibles:"
echo "   • API de Restauración: http://localhost:8000"
echo "   • API Docs (Swagger):   http://localhost:8000/docs"
echo "   • MinIO Console:        http://localhost:9001 (minio/minio123)"
echo "   • n8n (opcional):       http://localhost:5678 (admin/admin123)"
echo ""
echo "🔄 Hot Reload habilitado en:"
echo "   • API: /api, /src, /config"
echo "   • Worker: /workers, /api/services, shared_job_queue.py"
echo ""
echo "📋 Comandos útiles:"
echo "   • Ver logs API:    docker-compose logs -f doc-restoration-api"
echo "   • Ver logs Worker: docker-compose logs -f job-worker"
echo "   • Detener todo:    docker-compose down"
echo ""
show_warning "Presiona Ctrl+C para ver logs en tiempo real..."

# Mostrar logs en tiempo real
docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f doc-restoration-api job-worker
