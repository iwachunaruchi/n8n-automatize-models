#!/usr/bin/env pwsh
# Script para iniciar el sistema completo con RQ

Write-Host "🚀 INICIANDO SISTEMA DE RESTAURACIÓN CON RQ" -ForegroundColor Green
Write-Host "=" * 60
Write-Host ""

# Función para mostrar status
function Show-Status {
    param($Message)
    Write-Host "📋 $Message" -ForegroundColor Cyan
}

function Show-Success {
    param($Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Show-Warning {
    param($Message)
    Write-Host "⚠️ $Message" -ForegroundColor Yellow
}

function Show-Error {
    param($Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

# Verificar Docker
try {
    $dockerVersion = docker --version
    Show-Success "Docker detectado: $dockerVersion"
} catch {
    Show-Error "Docker no está disponible"
    exit 1
}

# Cambiar al directorio docker
Set-Location $PSScriptRoot
Show-Status "Directorio de trabajo: $PWD"

# Limpiar contenedores anteriores
Show-Status "Limpiando contenedores anteriores..."
docker-compose -f docker-compose-rq.yml down --remove-orphans

# Construir imágenes
Show-Status "Construyendo imágenes con caché..."
docker-compose -f docker-compose-rq.yml build --parallel

# Iniciar servicios base
Show-Status "Iniciando servicios base (Redis, MinIO, PostgreSQL)..."
docker-compose -f docker-compose-rq.yml up -d redis minio postgres

# Esperar servicios base
Show-Status "Esperando que los servicios base estén listos..."
Start-Sleep -Seconds 15

# Configurar MinIO
Show-Status "Configurando MinIO..."
docker-compose -f docker-compose-rq.yml up --no-deps minio-setup

# Iniciar API y Worker RQ
Show-Status "Iniciando API y Worker RQ..."
docker-compose -f docker-compose-rq.yml up -d doc-restoration-api rq-worker

# Opcional: Iniciar Dashboard RQ
Show-Status "Iniciando RQ Dashboard..."
docker-compose -f docker-compose-rq.yml up -d rq-dashboard

# Mostrar servicios
Write-Host ""
Show-Success "SISTEMA INICIADO EXITOSAMENTE!"
Write-Host ""
Write-Host "🌐 SERVICIOS DISPONIBLES:" -ForegroundColor Magenta
Write-Host "  • API REST:        http://localhost:8000" -ForegroundColor White
Write-Host "  • Documentación:   http://localhost:8000/docs" -ForegroundColor White
Write-Host "  • MinIO Console:   http://localhost:9001 (minio/minio123)" -ForegroundColor White
Write-Host "  • RQ Dashboard:    http://localhost:9181" -ForegroundColor White
Write-Host "  • n8n (opcional):  http://localhost:5678 (admin/admin123)" -ForegroundColor Gray
Write-Host ""

Write-Host "📊 COMANDOS ÚTILES:" -ForegroundColor Yellow
Write-Host "  • Ver logs API:    docker-compose -f docker-compose-rq.yml logs -f doc-restoration-api" -ForegroundColor Gray
Write-Host "  • Ver logs Worker: docker-compose -f docker-compose-rq.yml logs -f rq-worker" -ForegroundColor Gray
Write-Host "  • Detener todo:    docker-compose -f docker-compose-rq.yml down" -ForegroundColor Gray
Write-Host ""

Write-Host "🧪 TESTS DISPONIBLES:" -ForegroundColor Cyan
Write-Host "  • Test job simple: POST http://localhost:8000/jobs/rq/test" -ForegroundColor White
Write-Host "  • Ver estadísticas: GET http://localhost:8000/jobs/rq/stats" -ForegroundColor White
Write-Host "  • Health check:    GET http://localhost:8000/health" -ForegroundColor White
Write-Host ""

# Verificar que los servicios están corriendo
Show-Status "Verificando servicios..."
Start-Sleep -Seconds 10

try {
    $apiStatus = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 5
    if ($apiStatus.status -eq "healthy") {
        Show-Success "API funcionando correctamente"
    }
} catch {
    Show-Warning "API aún iniciándose... (normal en primer arranque)"
}

Show-Success "Setup completado!"
