# Script de verificacion del sistema completo
Write-Host "Sistema de Restauracion de Documentos - Verificacion Completa" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan

Write-Host "`nURLs de acceso:" -ForegroundColor Yellow
Write-Host "   API Principal: http://localhost:8000/health" -ForegroundColor White
Write-Host "   API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "   RQ Dashboard: http://localhost:9181" -ForegroundColor White
Write-Host "   MinIO Console: http://localhost:9001" -ForegroundColor White
Write-Host "   n8n Interface: http://localhost:5678" -ForegroundColor White

Write-Host "`nVerificando servicios..." -ForegroundColor Yellow

# Verificar API Principal
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 5 -UseBasicParsing
    Write-Host "   API Principal: OK" -ForegroundColor Green
} catch {
    Write-Host "   API Principal: Error" -ForegroundColor Red
}

# Verificar RQ Dashboard
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9181" -Method GET -TimeoutSec 5 -UseBasicParsing
    Write-Host "   RQ Dashboard: OK" -ForegroundColor Green
} catch {
    Write-Host "   RQ Dashboard: Error" -ForegroundColor Red
}

# Verificar MinIO
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9001" -Method GET -TimeoutSec 5 -UseBasicParsing
    Write-Host "   MinIO Console: OK" -ForegroundColor Green
} catch {
    Write-Host "   MinIO Console: Error" -ForegroundColor Red
}

# Verificar n8n
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5678" -Method GET -TimeoutSec 5 -UseBasicParsing
    Write-Host "   n8n Interface: OK" -ForegroundColor Green
} catch {
    Write-Host "   n8n Interface: Error" -ForegroundColor Red
}

Write-Host "`nEstado de contenedores:" -ForegroundColor Yellow
docker ps --format "table {{.Names}}\t{{.Status}}"

Write-Host "`nCredenciales:" -ForegroundColor Cyan
Write-Host "   MinIO Console: minio / minio123" -ForegroundColor White
Write-Host "   n8n Interface: admin / admin123" -ForegroundColor White

Write-Host "`nSistema verificado!" -ForegroundColor Green
