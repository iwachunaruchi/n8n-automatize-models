# Docker Manager para n8n-automatize-models (PowerShell)
param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Service = "",
    
    [switch]$Force,
    [switch]$NoCache,
    [switch]$Help
)

# Función para mostrar ayuda
function Show-Help {
    Write-Host "🐳 Docker Manager para n8n-automatize-models" -ForegroundColor Blue
    Write-Host ""
    Write-Host "Uso: .\docker-manager.ps1 [COMANDO] [OPCIONES]"
    Write-Host ""
    Write-Host "COMANDOS PRINCIPALES:" -ForegroundColor Green
    Write-Host "  build           Construir todas las imágenes"
    Write-Host "  up              Iniciar servicios de producción"
    Write-Host "  dev             Iniciar servicios de desarrollo"
    Write-Host "  down            Detener todos los servicios"
    Write-Host "  logs            Ver logs de servicios"
    Write-Host "  clean           Limpiar imágenes y volúmenes"
    Write-Host "  status          Estado de servicios"
    Write-Host ""
    Write-Host "COMANDOS ESPECÍFICOS:" -ForegroundColor Yellow
    Write-Host "  build-api       Construir solo la API"
    Write-Host "  restart-api     Reiniciar solo la API"
    Write-Host "  shell-api       Abrir shell en contenedor API"
    Write-Host "  training        Iniciar servicio de entrenamiento"
    Write-Host ""
    Write-Host "OPCIONES:" -ForegroundColor Cyan
    Write-Host "  -Force          Forzar recreación de contenedores"
    Write-Host "  -NoCache        Construir sin usar cache"
    Write-Host ""
    Write-Host "EJEMPLOS:"
    Write-Host "  .\docker-manager.ps1 build           # Construir todo"
    Write-Host "  .\docker-manager.ps1 dev             # Desarrollo"
    Write-Host "  .\docker-manager.ps1 up -Force       # Producción forzada"
    Write-Host "  .\docker-manager.ps1 logs api        # Ver logs de API"
}

# Función para verificar Docker
function Test-Docker {
    try {
        docker info | Out-Null
        return $true
    }
    catch {
        Write-Host "❌ Error: Docker no está corriendo" -ForegroundColor Red
        return $false
    }
}

# Función para construir imágenes
function Build-Images {
    $noCacheFlag = if ($NoCache) { "--no-cache" } else { "" }
    
    Write-Host "🔨 Construyendo imágenes Docker..." -ForegroundColor Blue
    
    # Construir imagen de producción
    Write-Host "📦 Construyendo imagen de producción..." -ForegroundColor Yellow
    docker build $noCacheFlag -t n8n-automatize-models:latest -f Dockerfile .
    
    # Construir imagen de desarrollo
    Write-Host "🛠️ Construyendo imagen de desarrollo..." -ForegroundColor Yellow
    docker build $noCacheFlag -t n8n-automatize-models:dev -f Dockerfile.dev .
    
    Write-Host "✅ Imágenes construidas exitosamente" -ForegroundColor Green
}

# Función para iniciar producción
function Start-Production {
    $forceFlag = if ($Force) { "--force-recreate" } else { "" }
    
    Write-Host "🚀 Iniciando servicios de producción..." -ForegroundColor Blue
    docker-compose up -d $forceFlag
    
    Write-Host "✅ Servicios iniciados" -ForegroundColor Green
    Write-Host "🌐 URLs disponibles:" -ForegroundColor Yellow
    Write-Host "  API: http://localhost:8000"
    Write-Host "  API Docs: http://localhost:8000/docs"
    Write-Host "  n8n: http://localhost:5678"
    Write-Host "  MinIO: http://localhost:9001"
}

# Función para iniciar desarrollo
function Start-Development {
    Write-Host "🛠️ Iniciando servicios de desarrollo..." -ForegroundColor Blue
    docker-compose -f docker-compose.dev.yml up -d
    
    Write-Host "✅ Servicios de desarrollo iniciados" -ForegroundColor Green
    Write-Host "🌐 URLs disponibles:" -ForegroundColor Yellow
    Write-Host "  API (Dev): http://localhost:8000"
    Write-Host "  API Docs: http://localhost:8000/docs"
    Write-Host "  MinIO: http://localhost:9001"
}

# Función para detener servicios
function Stop-Services {
    Write-Host "⏹️ Deteniendo servicios..." -ForegroundColor Blue
    docker-compose down
    docker-compose -f docker-compose.dev.yml down 2>$null
    Write-Host "✅ Servicios detenidos" -ForegroundColor Green
}

# Función para mostrar logs
function Show-Logs {
    if ([string]::IsNullOrEmpty($Service)) {
        Write-Host "📋 Mostrando logs de todos los servicios..." -ForegroundColor Blue
        docker-compose logs -f --tail=100
    } else {
        Write-Host "📋 Mostrando logs de $Service..." -ForegroundColor Blue
        docker-compose logs -f --tail=100 $Service
    }
}

# Función para limpiar Docker
function Clean-Docker {
    Write-Host "🧹 Limpiando imágenes y volúmenes..." -ForegroundColor Yellow
    
    # Detener servicios
    Stop-Services
    
    # Limpiar contenedores detenidos
    docker container prune -f
    
    # Limpiar imágenes sin usar
    docker image prune -f
    
    Write-Host "✅ Limpieza completada" -ForegroundColor Green
}

# Función para mostrar estado
function Show-Status {
    Write-Host "📊 Estado de servicios Docker..." -ForegroundColor Blue
    Write-Host ""
    Write-Host "Contenedores:" -ForegroundColor Yellow
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    Write-Host ""
    Write-Host "Volúmenes:" -ForegroundColor Yellow
    docker volume ls --filter name=doc_restoration
    Write-Host ""
    Write-Host "Imágenes del proyecto:" -ForegroundColor Yellow
    docker images --filter reference=n8n-automatize-models
}

# Función para construir solo API
function Build-Api {
    Write-Host "🔨 Construyendo imagen de API..." -ForegroundColor Blue
    docker build -t n8n-automatize-models:latest -f Dockerfile .
    Write-Host "✅ Imagen de API construida" -ForegroundColor Green
}

# Función para reiniciar API
function Restart-Api {
    Write-Host "🔄 Reiniciando API..." -ForegroundColor Blue
    docker-compose restart doc-restoration-api
    Write-Host "✅ API reiniciada" -ForegroundColor Green
}

# Función para shell de API
function Shell-Api {
    Write-Host "🐚 Abriendo shell en contenedor API..." -ForegroundColor Blue
    docker-compose exec doc-restoration-api /bin/bash
}

# Función para entrenamiento
function Start-Training {
    Write-Host "🎯 Iniciando servicio de entrenamiento..." -ForegroundColor Blue
    docker-compose --profile training up -d training-service
    Write-Host "✅ Servicio de entrenamiento iniciado" -ForegroundColor Green
}

# Script principal
if ($Help) {
    Show-Help
    exit 0
}

if (-not (Test-Docker)) {
    exit 1
}

switch ($Command.ToLower()) {
    "build" { Build-Images }
    "up" { Start-Production }
    "start" { Start-Production }
    "dev" { Start-Development }
    "development" { Start-Development }
    "down" { Stop-Services }
    "stop" { Stop-Services }
    "logs" { Show-Logs }
    "clean" { Clean-Docker }
    "status" { Show-Status }
    "build-api" { Build-Api }
    "restart-api" { Restart-Api }
    "shell-api" { Shell-Api }
    "training" { Start-Training }
    "help" { Show-Help }
    default {
        Write-Host "❌ Comando desconocido: $Command" -ForegroundColor Red
        Write-Host ""
        Show-Help
        exit 1
    }
}
