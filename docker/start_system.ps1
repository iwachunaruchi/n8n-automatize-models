# Script para iniciar el sistema completo con RQ
Write-Host "Iniciando Sistema de Restauracion con RQ..." -ForegroundColor Green

# Verificar Docker
try {
    $dockerVersion = docker --version
    Write-Host "Docker detectado: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "Docker no esta disponible" -ForegroundColor Red
    exit 1
}

# Cambiar al directorio docker
Set-Location $PSScriptRoot
Write-Host "Directorio de trabajo: $PWD" -ForegroundColor Cyan

# Limpiar contenedores anteriores
Write-Host "Limpiando contenedores anteriores..." -ForegroundColor Yellow
docker-compose -f docker-compose-rq.yml down --remove-orphans

# Construir imagenes
Write-Host "Construyendo imagenes..." -ForegroundColor Yellow
docker-compose -f docker-compose-rq.yml build --parallel

# Iniciar servicios base
Write-Host "Iniciando servicios base..." -ForegroundColor Yellow
docker-compose -f docker-compose-rq.yml up -d redis minio postgres

# Esperar servicios base
Write-Host "Esperando servicios base..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Configurar MinIO
Write-Host "Configurando MinIO..." -ForegroundColor Yellow
docker-compose -f docker-compose-rq.yml up --no-deps minio-setup

# Iniciar API y Worker RQ
Write-Host "Iniciando API y Worker RQ..." -ForegroundColor Yellow
docker-compose -f docker-compose-rq.yml up -d doc-restoration-api rq-worker

# Iniciar Dashboard RQ
Write-Host "Iniciando RQ Dashboard..." -ForegroundColor Yellow
docker-compose -f docker-compose-rq.yml up -d rq-dashboard

# Mostrar servicios
Write-Host ""
Write-Host "SISTEMA INICIADO EXITOSAMENTE!" -ForegroundColor Green
Write-Host ""
Write-Host "SERVICIOS DISPONIBLES:" -ForegroundColor Magenta
Write-Host "  - API REST:        http://localhost:8000" -ForegroundColor White
Write-Host "  - Documentacion:   http://localhost:8000/docs" -ForegroundColor White
Write-Host "  - MinIO Console:   http://localhost:9001 (minio/minio123)" -ForegroundColor White
Write-Host "  - RQ Dashboard:    http://localhost:9181" -ForegroundColor White
Write-Host ""

Write-Host "COMANDOS UTILES:" -ForegroundColor Yellow
Write-Host "  - Ver logs API:    docker-compose -f docker-compose-rq.yml logs -f doc-restoration-api" -ForegroundColor Gray
Write-Host "  - Ver logs Worker: docker-compose -f docker-compose-rq.yml logs -f rq-worker" -ForegroundColor Gray
Write-Host "  - Detener todo:    docker-compose -f docker-compose-rq.yml down" -ForegroundColor Gray
Write-Host ""

# Verificar servicios
Write-Host "Verificando servicios..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

try {
    $apiStatus = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 5
    if ($apiStatus.status -eq "healthy") {
        Write-Host "API funcionando correctamente" -ForegroundColor Green
    }
} catch {
    Write-Host "API aun iniciandose... (normal en primer arranque)" -ForegroundColor Yellow
}

Write-Host "Setup completado!" -ForegroundColor Green
