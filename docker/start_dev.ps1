# Script de inicio para desarrollo con Docker en Windows
# Levanta API + Worker con hot reload habilitado

Write-Host "Iniciando entorno de desarrollo..." -ForegroundColor Green
Write-Host "API + Worker Modular con Hot Reload" -ForegroundColor Green
Write-Host "=================================================="

# Función para mostrar mensajes con colores
function Show-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Show-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Show-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Verificar que Docker está corriendo
try {
    docker info | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker no responde"
    }
} catch {
    Show-Error "Docker no esta corriendo. Por favor inicia Docker Desktop."
    exit 1
}

# Cambiar al directorio docker
Set-Location $PSScriptRoot

# Limpiar contenedores previos si existen
Show-Status "Limpiando contenedores previos..."
docker-compose down --remove-orphans

# Construir imágenes con cache
Show-Status "Construyendo imagenes..."
docker-compose build --parallel

# Iniciar servicios base (MinIO, PostgreSQL)
Show-Status "Iniciando servicios base (MinIO, PostgreSQL)..."
docker-compose up -d minio postgres

# Esperar a que los servicios base estén listos
Show-Status "Esperando a que los servicios base esten listos..."
Start-Sleep -Seconds 10

# Configurar MinIO
Show-Status "Configurando MinIO..."
docker-compose up --no-deps minio-setup

# Iniciar API y Worker con hot reload
Show-Status "Iniciando API y Worker con hot reload..."
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d doc-restoration-api job-worker

# Mostrar información de conexión
Write-Host ""
Write-Host "============================================================"
Show-Status "Entorno de desarrollo listo!"
Write-Host ""
Write-Host "Servicios disponibles:" -ForegroundColor Cyan
Write-Host "   • API de Restauracion: http://localhost:8000"
Write-Host "   • API Docs (Swagger):   http://localhost:8000/docs"
Write-Host "   • MinIO Console:        http://localhost:9001 (minio/minio123)"
Write-Host "   • n8n (opcional):       http://localhost:5678 (admin/admin123)"
Write-Host ""
Write-Host "Hot Reload habilitado en:" -ForegroundColor Cyan
Write-Host "   • API: /api, /src, /config"
Write-Host "   • Worker: /workers, /api/services, shared_job_queue.py"
Write-Host ""
Write-Host "Comandos utiles:" -ForegroundColor Cyan
Write-Host "   • Ver logs API:    docker-compose logs -f doc-restoration-api"
Write-Host "   • Ver logs Worker: docker-compose logs -f job-worker"
Write-Host "   • Detener todo:    docker-compose down"
Write-Host ""
Show-Warning "Presiona Ctrl+C para detener los logs..."

# Mostrar logs en tiempo real
docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f doc-restoration-api job-worker
