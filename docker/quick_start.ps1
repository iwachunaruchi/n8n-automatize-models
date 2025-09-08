# Comando único para iniciar todo el entorno de desarrollo
# Ejecutar en PowerShell desde el directorio docker/

# Limpiar y construir
docker-compose down --remove-orphans
docker-compose build --parallel

# Iniciar servicios base
docker-compose up -d minio postgres

# Esperar 10 segundos
Start-Sleep -Seconds 10

# Configurar MinIO
docker-compose up --no-deps minio-setup

# Iniciar API y Worker con hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d doc-restoration-api job-worker

# Mostrar status
Write-Host ""
Write-Host "=== SERVICIOS INICIADOS ===" -ForegroundColor Green
Write-Host "API:   http://localhost:8000" -ForegroundColor Cyan
Write-Host "Docs:  http://localhost:8000/docs" -ForegroundColor Cyan  
Write-Host "MinIO: http://localhost:9001 (minio/minio123)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Ver logs:" -ForegroundColor Yellow
Write-Host "docker-compose logs -f doc-restoration-api job-worker" -ForegroundColor Gray
