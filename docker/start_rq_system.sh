#!/bin/bash
# Script alternativo para sistemas Unix/Linux/macOS

echo "🚀 INICIANDO SISTEMA DE RESTAURACIÓN CON RQ"
echo "============================================================"
echo ""

# Colores
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

show_status() {
    echo -e "${CYAN}📋 $1${NC}"
}

show_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

show_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

show_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Verificar Docker
if ! command -v docker &> /dev/null; then
    show_error "Docker no está disponible"
    exit 1
fi

docker_version=$(docker --version)
show_success "Docker detectado: $docker_version"

# Cambiar al directorio docker
cd "$(dirname "$0")"
show_status "Directorio de trabajo: $(pwd)"

# Limpiar contenedores anteriores
show_status "Limpiando contenedores anteriores..."
docker-compose -f docker-compose-rq.yml down --remove-orphans

# Construir imágenes
show_status "Construyendo imágenes con caché..."
docker-compose -f docker-compose-rq.yml build --parallel

# Iniciar servicios base
show_status "Iniciando servicios base (Redis, MinIO, PostgreSQL)..."
docker-compose -f docker-compose-rq.yml up -d redis minio postgres

# Esperar servicios base
show_status "Esperando que los servicios base estén listos..."
sleep 15

# Configurar MinIO
show_status "Configurando MinIO..."
docker-compose -f docker-compose-rq.yml up --no-deps minio-setup

# Iniciar API y Worker RQ
show_status "Iniciando API y Worker RQ..."
docker-compose -f docker-compose-rq.yml up -d doc-restoration-api rq-worker

# Opcional: Iniciar Dashboard RQ
show_status "Iniciando RQ Dashboard..."
docker-compose -f docker-compose-rq.yml up -d rq-dashboard

# Mostrar servicios
echo ""
show_success "SISTEMA INICIADO EXITOSAMENTE!"
echo ""
echo -e "\033[0;35m🌐 SERVICIOS DISPONIBLES:\033[0m"
echo "  • API REST:        http://localhost:8000"
echo "  • Documentación:   http://localhost:8000/docs"
echo "  • MinIO Console:   http://localhost:9001 (minio/minio123)"
echo "  • RQ Dashboard:    http://localhost:9181"
echo "  • n8n (opcional):  http://localhost:5678 (admin/admin123)"
echo ""

echo -e "${YELLOW}📊 COMANDOS ÚTILES:${NC}"
echo "  • Ver logs API:    docker-compose -f docker-compose-rq.yml logs -f doc-restoration-api"
echo "  • Ver logs Worker: docker-compose -f docker-compose-rq.yml logs -f rq-worker"
echo "  • Detener todo:    docker-compose -f docker-compose-rq.yml down"
echo ""

echo -e "${CYAN}🧪 TESTS DISPONIBLES:${NC}"
echo "  • Test job simple: POST http://localhost:8000/jobs/rq/test"
echo "  • Ver estadísticas: GET http://localhost:8000/jobs/rq/stats"
echo "  • Health check:    GET http://localhost:8000/health"
echo ""

# Verificar que los servicios están corriendo
show_status "Verificando servicios..."
sleep 10

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    show_success "API funcionando correctamente"
else
    show_warning "API aún iniciándose... (normal en primer arranque)"
fi

show_success "Setup completado! 🎉"
