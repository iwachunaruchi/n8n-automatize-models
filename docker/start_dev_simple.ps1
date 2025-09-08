# Script simple de inicio para desarrollo con Docker
# Versión sin caracteres especiales para compatibilidad

Write-Host "=== Iniciando entorno de desarrollo ===" -ForegroundColor Green
Write-Host "API + Worker Modular con Hot Reload" -ForegroundColor Cyan

# Verificar Docker
Write-Host "Verificando Docker..." -ForegroundColor Yellow
try {
    docker info | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker no disponible"
    }
    Write-Host "Docker OK" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker no esta corriendo. Inicia Docker Desktop." -ForegroundColor Red
    exit 1
}

# Ir al directorio correcto
Set-Location $PSScriptRoot
Write-Host "Directorio: $PWD" -ForegroundColor Gray

# Limpiar contenedores anteriores
Write-Host "Limpiando contenedores previos..." -ForegroundColor Yellow
docker-compose down --remove-orphans

# Construir imagenes
Write-Host "Construyendo imagenes..." -ForegroundColor Yellow
docker-compose build --parallel

# Iniciar servicios base
Write-Host "Iniciando MinIO y PostgreSQL..." -ForegroundColor Yellow
docker-compose up -d minio postgres

# Esperar
Write-Host "Esperando servicios base..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Configurar MinIO
Write-Host "Configurando MinIO..." -ForegroundColor Yellow
docker-compose up --no-deps minio-setup

# Iniciar API y Worker
Write-Host "Iniciando API y Worker..." -ForegroundColor Yellow
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d doc-restoration-api job-worker

# Mostrar resultados
Write-Host ""
Write-Host "=== ENTORNO LISTO ===" -ForegroundColor Green
Write-Host ""
Write-Host "Servicios:" -ForegroundColor Cyan
Write-Host "  API:       http://localhost:8000" -ForegroundColor White
Write-Host "  Docs:      http://localhost:8000/docs" -ForegroundColor White  
Write-Host "  MinIO:     http://localhost:9001" -ForegroundColor White
Write-Host "  Usuario:   minio / minio123" -ForegroundColor Gray
Write-Host ""
Write-Host "Hot Reload activo en API y Worker" -ForegroundColor Green
Write-Host ""
Write-Host "Ver logs:" -ForegroundColor Cyan
Write-Host "  docker-compose logs -f doc-restoration-api" -ForegroundColor Gray
Write-Host "  docker-compose logs -f job-worker" -ForegroundColor Gray
Write-Host ""
Write-Host "Presiona Ctrl+C para ver logs en tiempo real..." -ForegroundColor Yellow

# Logs en tiempo real
docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f doc-restoration-api job-worker
