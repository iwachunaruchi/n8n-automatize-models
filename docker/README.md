# 🐳 Docker Development Environment

Este directorio contiene la configuración completa para ejecutar el sistema de restauración de documentos con **API + Worker Modular** en Docker con **Hot Reload** habilitado.

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Environment                       │
├─────────────────────────────────────────────────────────────┤
│  📦 API Container (Port 8000)                              │
│  ├── FastAPI with Hot Reload                               │
│  ├── Cola Compartida Integration                           │
│  └── Swagger Docs: /docs                                   │
├─────────────────────────────────────────────────────────────┤
│  ⚙️  Worker Container                                       │
│  ├── Worker Modular con Hot Reload                         │
│  ├── Training Handler                                      │
│  ├── Synthetic Data Handler                                │
│  └── Restoration Handler                                   │
├─────────────────────────────────────────────────────────────┤
│  💾 MinIO Container (Ports 9000, 9001)                     │
│  ├── Object Storage                                        │
│  ├── Buckets: document-*, models                           │
│  └── Console: http://localhost:9001                        │
├─────────────────────────────────────────────────────────────┤
│  🗄️  PostgreSQL Container                                   │
│  └── Database for n8n                                      │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Inicio Rápido

### Opción 1: Script Automático (Recomendado)

**Windows:**
```powershell
.\start_dev.ps1
```

**Linux/macOS:**
```bash
./start_dev.sh
```

### Opción 2: Manual

```bash
# 1. Iniciar servicios base
docker-compose up -d minio postgres minio-setup

# 2. Iniciar API + Worker con hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d doc-restoration-api job-worker

# 3. Ver logs
docker-compose logs -f doc-restoration-api job-worker
```

## 📊 Servicios Disponibles

| Servicio | URL | Credenciales |
|----------|-----|--------------|
| **API Principal** | http://localhost:8000 | - |
| **Swagger Docs** | http://localhost:8000/docs | - |
| **MinIO Console** | http://localhost:9001 | minio/minio123 |
| **n8n (opcional)** | http://localhost:5678 | admin/admin123 |

## 🔄 Hot Reload

### API (Port 8000):
- **Archivos vigilados:** `/api`, `/src`, `/config`
- **Reinicio automático:** Sí, en cambios de código
- **Logs:** `docker-compose logs -f doc-restoration-api`

### Worker Modular:
- **Archivos vigilados:** `/workers`, `/api/services`, `shared_job_queue.py`
- **Reinicio automático:** Sí, en cambios de código
- **Logs:** `docker-compose logs -f job-worker`

## 🛠️ Comandos Útiles

### Desarrollo Diario:
```bash
# Ver logs en tiempo real
docker-compose logs -f doc-restoration-api job-worker

# Reiniciar un servicio específico
docker-compose restart doc-restoration-api
docker-compose restart job-worker

# Acceder al shell del contenedor
docker-compose exec doc-restoration-api bash
docker-compose exec job-worker bash

# Ver estado de servicios
docker-compose ps
```

### Debugging:
```bash
# Ver logs completos
docker-compose logs doc-restoration-api
docker-compose logs job-worker

# Inspeccionar salud de contenedores
docker-compose exec doc-restoration-api curl http://localhost:8000/health
docker inspect job-worker

# Ver métricas de sistema
docker stats
```

### Gestión de Datos:
```bash
# Backup de volúmenes
docker run --rm -v doc_restoration_minio_data:/data -v $(pwd):/backup alpine tar czf /backup/minio_backup.tar.gz /data

# Limpiar volúmenes (¡CUIDADO!)
docker-compose down -v
docker volume prune
```

## 📁 Estructura de Archivos

```
docker/
├── docker-compose.yml          # Configuración base
├── docker-compose.dev.yml      # Override para desarrollo
├── Dockerfile                  # API container
├── Dockerfile.worker           # Worker container  
├── start_dev.ps1              # Script Windows
├── start_dev.sh               # Script Linux/macOS
└── README.md                  # Esta documentación
```

## 🐛 Solución de Problemas

### El worker no inicia:
```bash
# Verificar logs
docker-compose logs job-worker

# Verificar health check
docker inspect job-worker | grep Health -A 10

# Reiniciar worker
docker-compose restart job-worker
```

### API no responde:
```bash
# Verificar puerto
netstat -an | grep 8000

# Verificar logs
docker-compose logs doc-restoration-api

# Verificar health
curl http://localhost:8000/health
```

### MinIO no accesible:
```bash
# Reiniciar y reconfigurar
docker-compose restart minio
docker-compose up --no-deps minio-setup
```

### Hot reload no funciona:
```bash
# Verificar montajes de volúmenes
docker-compose config

# Verificar variables de entorno
docker-compose exec doc-restoration-api env | grep RELOAD
docker-compose exec job-worker env | grep RELOAD
```

## 🔧 Configuración Avanzada

### Variables de Entorno:
- `DEBUG=true` - Habilitar modo debug
- `LOG_LEVEL=debug` - Nivel de logging
- `WORKER_POLL_INTERVAL=2` - Intervalo de polling del worker
- `MAX_CONCURRENT_JOBS=3` - Jobs simultáneos máximos

### Perfiles de Docker Compose:
```bash
# Solo servicios base
docker-compose up minio postgres

# Con n8n
docker-compose --profile=training up

# Completo con hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

---

## 🎯 Flujo de Desarrollo

1. **Ejecutar:** `.\start_dev.ps1` (Windows) o `./start_dev.sh` (Linux/macOS)
2. **Desarrollar:** Editar archivos en `/api`, `/workers`, `/src`
3. **Probar:** Los cambios se aplican automáticamente
4. **Logs:** Visibles en tiempo real en la consola
5. **Detener:** `Ctrl+C` + `docker-compose down`

**¡Hot reload funcionando en ambas consolas! 🔥**
