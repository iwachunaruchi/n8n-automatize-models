# 🐳 Guía de Docker para n8n-automatize-models

## 📋 Resumen de la Dockerización

Tu proyecto ahora está completamente dockerizado con **Poetry** como gestor de dependencias. Incluye múltiples servicios y configuraciones para diferentes entornos.

## 🏗️ Arquitectura Docker

### Servicios Principales:
- **API de Restauración** - FastAPI con Poetry
- **MinIO** - Object Storage (S3 compatible)
- **PostgreSQL** - Base de datos para n8n
- **n8n** - Workflow automation
- **Servicio de Entrenamiento** - Modelos ML (opcional)

### Imágenes:
- `n8n-automatize-models:latest` - Producción
- `n8n-automatize-models:dev` - Desarrollo (con hot reload)

## 🚀 Comandos Principales

### Usando PowerShell (Windows):
```powershell
# Construir todas las imágenes
.\docker-manager.ps1 build

# Iniciar servicios de producción
.\docker-manager.ps1 up

# Iniciar servicios de desarrollo (con hot reload)
.\docker-manager.ps1 dev

# Ver logs
.\docker-manager.ps1 logs

# Detener servicios
.\docker-manager.ps1 down

# Limpiar contenedores e imágenes
.\docker-manager.ps1 clean

# Ver estado
.\docker-manager.ps1 status
```

### Usando Docker Compose directamente:
```bash
# Producción
docker-compose up -d

# Desarrollo
docker-compose -f docker-compose.dev.yml up -d

# Solo servicios específicos
docker-compose up -d minio postgres

# Ver logs
docker-compose logs -f doc-restoration-api

# Detener
docker-compose down
```

## 🌐 URLs de Servicios

### Producción:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **n8n**: http://localhost:5678 (admin/admin123)
- **MinIO Console**: http://localhost:9001 (minio/minio123)

### Desarrollo:
- **API Dev**: http://localhost:8001 (con hot reload)
- **MinIO Console**: http://localhost:9001

## 📁 Estructura de Volúmenes

```
doc_restoration_minio_data/     # Datos de MinIO
doc_restoration_postgres_data/  # Base de datos PostgreSQL
doc_restoration_n8n_data/       # Configuración de n8n
doc_restoration_api_logs/       # Logs de la API
doc_restoration_training_logs/  # Logs de entrenamiento
```

## 🔧 Configuraciones por Entorno

### Producción (`docker-compose.yml`):
- Imagen optimizada sin dependencias de desarrollo
- Usuario no-root para seguridad
- Healthchecks configurados
- Logs persistentes
- Reinicio automático

### Desarrollo (`docker-compose.dev.yml`):
- Hot reload activado
- Dependencias de desarrollo incluidas
- Volúmenes montados para código fuente
- Configuración de debug

## 🛠️ Desarrollo con Hot Reload

Para desarrollar con recarga automática:

```powershell
# Iniciar servicios de desarrollo
.\docker-manager.ps1 dev

# Los cambios en estos directorios se recargan automáticamente:
# - api/
# - src/
# - config/
# - scripts/
```

## 🎯 Entrenamiento de Modelos

Para ejecutar entrenamiento en contenedor:

```powershell
# Iniciar servicio de entrenamiento
.\docker-manager.ps1 training

# O directamente:
docker-compose --profile training up -d training-service
```

## 🔍 Diagnóstico y Debugging

### Ver logs en tiempo real:
```bash
# Todos los servicios
docker-compose logs -f

# Solo API
docker-compose logs -f doc-restoration-api

# Solo MinIO
docker-compose logs -f minio
```

### Acceder a shell del contenedor:
```bash
# API
docker-compose exec doc-restoration-api /bin/bash

# MinIO
docker-compose exec minio /bin/bash
```

### Verificar estado de servicios:
```powershell
.\docker-manager.ps1 status
```

## 🚨 Solución de Problemas

### 1. **Error de permisos en volúmenes**:
```bash
# Cambiar permisos en Windows
icacls .\outputs /grant Everyone:(OI)(CI)F
icacls .\models /grant Everyone:(OI)(CI)F
```

### 2. **Puerto ocupado**:
```bash
# Verificar puertos en uso
netstat -an | findstr :8000
netstat -an | findstr :9000
```

### 3. **MinIO no conecta**:
```bash
# Verificar salud de MinIO
docker-compose exec minio mc admin info local
```

### 4. **API no responde**:
```bash
# Verificar logs de API
docker-compose logs doc-restoration-api

# Verificar healthcheck
docker inspect doc-restoration-api | findstr Health
```

### 5. **Limpiar todo y empezar de nuevo**:
```powershell
# Detener y limpiar todo
.\docker-manager.ps1 down
.\docker-manager.ps1 clean

# Reconstruir
.\docker-manager.ps1 build

# Iniciar de nuevo
.\docker-manager.ps1 up
```

## 📊 Monitoreo

### Recursos del sistema:
```bash
# Ver uso de recursos
docker stats

# Ver espacio de volúmenes
docker system df
```

### Logs importantes:
- **API**: `/app/logs/app.log`
- **MinIO**: Logs internos del contenedor
- **PostgreSQL**: Logs de base de datos
- **n8n**: Logs de workflows

## 🔒 Seguridad

### Configuraciones aplicadas:
- ✅ Usuario no-root en contenedores
- ✅ Variables sensibles en environment
- ✅ Red aislada entre servicios
- ✅ Healthchecks configurados
- ✅ Volúmenes con permisos limitados

### Para producción real:
1. Cambiar credenciales por defecto
2. Usar secrets de Docker
3. Configurar SSL/TLS
4. Implementar backup de volúmenes
5. Configurar logging centralizado

## 📈 Escalabilidad

Para escalar servicios:

```bash
# Escalar API a 3 instancias
docker-compose up -d --scale doc-restoration-api=3

# Usar load balancer (nginx, traefik, etc.)
```

## 🔄 CI/CD

Para automatización:

```dockerfile
# Build en CI
docker build -t $REGISTRY/n8n-automatize-models:$TAG .

# Deploy
docker-compose -f docker-compose.prod.yml up -d
```

---

**🎉 Tu proyecto está completamente dockerizado y listo para producción!**

Para cualquier problema, consulta los logs con: `.\docker-manager.ps1 logs`
