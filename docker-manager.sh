#!/bin/bash
# Script para gestionar Docker con Poetry

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para mostrar ayuda
show_help() {
    echo -e "${BLUE}🐳 Docker Manager para n8n-automatize-models${NC}"
    echo ""
    echo "Uso: ./docker-manager.sh [COMANDO] [OPCIONES]"
    echo ""
    echo "COMANDOS PRINCIPALES:"
    echo -e "  ${GREEN}build${NC}           Construir todas las imágenes"
    echo -e "  ${GREEN}up${NC}              Iniciar servicios de producción"
    echo -e "  ${GREEN}dev${NC}             Iniciar servicios de desarrollo"
    echo -e "  ${GREEN}down${NC}            Detener todos los servicios"
    echo -e "  ${GREEN}logs${NC}            Ver logs de servicios"
    echo -e "  ${GREEN}clean${NC}           Limpiar imágenes y volúmenes"
    echo -e "  ${GREEN}status${NC}          Estado de servicios"
    echo ""
    echo "COMANDOS ESPECÍFICOS:"
    echo -e "  ${YELLOW}build-api${NC}       Construir solo la API"
    echo -e "  ${YELLOW}restart-api${NC}     Reiniciar solo la API"
    echo -e "  ${YELLOW}shell-api${NC}       Abrir shell en contenedor API"
    echo -e "  ${YELLOW}training${NC}        Iniciar servicio de entrenamiento"
    echo ""
    echo "OPCIONES:"
    echo -e "  ${YELLOW}--force${NC}         Forzar recreación de contenedores"
    echo -e "  ${YELLOW}--no-cache${NC}      Construir sin usar cache"
    echo ""
    echo "EJEMPLOS:"
    echo "  ./docker-manager.sh build           # Construir todo"
    echo "  ./docker-manager.sh dev             # Desarrollo"
    echo "  ./docker-manager.sh up --force      # Producción forzada"
    echo "  ./docker-manager.sh logs api        # Ver logs de API"
}

# Función para verificar si Docker está corriendo
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}❌ Error: Docker no está corriendo${NC}"
        exit 1
    fi
}

# Función para construir imágenes
build_images() {
    local no_cache=""
    if [[ "$2" == "--no-cache" ]]; then
        no_cache="--no-cache"
    fi
    
    echo -e "${BLUE}🔨 Construyendo imágenes Docker...${NC}"
    
    # Construir imagen de producción
    echo -e "${YELLOW}📦 Construyendo imagen de producción...${NC}"
    docker build $no_cache -t n8n-automatize-models:latest -f Dockerfile .
    
    # Construir imagen de desarrollo
    echo -e "${YELLOW}🛠️ Construyendo imagen de desarrollo...${NC}"
    docker build $no_cache -t n8n-automatize-models:dev -f Dockerfile.dev .
    
    echo -e "${GREEN}✅ Imágenes construidas exitosamente${NC}"
}

# Función para iniciar servicios de producción
start_production() {
    local force=""
    if [[ "$2" == "--force" ]]; then
        force="--force-recreate"
    fi
    
    echo -e "${BLUE}🚀 Iniciando servicios de producción...${NC}"
    docker-compose up -d $force
    
    echo -e "${GREEN}✅ Servicios iniciados${NC}"
    echo -e "${YELLOW}🌐 URLs disponibles:${NC}"
    echo "  API: http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
    echo "  n8n: http://localhost:5678"
    echo "  MinIO: http://localhost:9001"
}

# Función para iniciar servicios de desarrollo
start_development() {
    echo -e "${BLUE}🛠️ Iniciando servicios de desarrollo...${NC}"
    docker-compose -f docker-compose.dev.yml up -d
    
    echo -e "${GREEN}✅ Servicios de desarrollo iniciados${NC}"
    echo -e "${YELLOW}🌐 URLs disponibles:${NC}"
    echo "  API (Dev): http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
    echo "  MinIO: http://localhost:9001"
}

# Función para detener servicios
stop_services() {
    echo -e "${BLUE}⏹️ Deteniendo servicios...${NC}"
    docker-compose down
    docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
    echo -e "${GREEN}✅ Servicios detenidos${NC}"
}

# Función para ver logs
show_logs() {
    local service="$2"
    if [[ -z "$service" ]]; then
        echo -e "${BLUE}📋 Mostrando logs de todos los servicios...${NC}"
        docker-compose logs -f --tail=100
    else
        echo -e "${BLUE}📋 Mostrando logs de $service...${NC}"
        docker-compose logs -f --tail=100 "$service"
    fi
}

# Función para limpiar Docker
clean_docker() {
    echo -e "${YELLOW}🧹 Limpiando imágenes y volúmenes...${NC}"
    
    # Detener servicios
    stop_services
    
    # Limpiar contenedores detenidos
    docker container prune -f
    
    # Limpiar imágenes sin usar
    docker image prune -f
    
    # Opcionalmente limpiar volúmenes (comentado por seguridad)
    # docker volume prune -f
    
    echo -e "${GREEN}✅ Limpieza completada${NC}"
}

# Función para mostrar estado
show_status() {
    echo -e "${BLUE}📊 Estado de servicios Docker...${NC}"
    echo ""
    echo -e "${YELLOW}Contenedores:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo -e "${YELLOW}Volúmenes:${NC}"
    docker volume ls --filter name=doc_restoration
    echo ""
    echo -e "${YELLOW}Imágenes del proyecto:${NC}"
    docker images --filter reference=n8n-automatize-models
}

# Función para construir solo la API
build_api() {
    echo -e "${BLUE}🔨 Construyendo imagen de API...${NC}"
    docker build -t n8n-automatize-models:latest -f Dockerfile .
    echo -e "${GREEN}✅ Imagen de API construida${NC}"
}

# Función para reiniciar API
restart_api() {
    echo -e "${BLUE}🔄 Reiniciando API...${NC}"
    docker-compose restart doc-restoration-api
    echo -e "${GREEN}✅ API reiniciada${NC}"
}

# Función para abrir shell en API
shell_api() {
    echo -e "${BLUE}🐚 Abriendo shell en contenedor API...${NC}"
    docker-compose exec doc-restoration-api /bin/bash
}

# Función para iniciar entrenamiento
start_training() {
    echo -e "${BLUE}🎯 Iniciando servicio de entrenamiento...${NC}"
    docker-compose --profile training up -d training-service
    echo -e "${GREEN}✅ Servicio de entrenamiento iniciado${NC}"
}

# Script principal
main() {
    check_docker
    
    case "$1" in
        "build")
            build_images "$@"
            ;;
        "up"|"start")
            start_production "$@"
            ;;
        "dev"|"development")
            start_development
            ;;
        "down"|"stop")
            stop_services
            ;;
        "logs")
            show_logs "$@"
            ;;
        "clean")
            clean_docker
            ;;
        "status")
            show_status
            ;;
        "build-api")
            build_api
            ;;
        "restart-api")
            restart_api
            ;;
        "shell-api")
            shell_api
            ;;
        "training")
            start_training
            ;;
        "help"|"-h"|"--help"|"")
            show_help
            ;;
        *)
            echo -e "${RED}❌ Comando desconocido: $1${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Ejecutar script principal
main "$@"
