# 🎯 Guía Completa de Poetry para n8n-automatize-models

## ✅ ¿Qué hemos logrado?

Hemos migrado exitosamente tu proyecto a **Poetry** como gestor de dependencias, resolviendo los problemas de dependencias y creando un entorno de desarrollo más robusto.

## 🛠️ Configuración Actual

### Archivos importantes creados/modificados:

- ✅ `pyproject.toml` - Configuración de Poetry con todas las dependencias
- ✅ `README.md` - Documentación del proyecto
- ✅ `.env.example` - Variables de entorno actualizadas
- ✅ `scripts/` - Scripts de utilidad para desarrollo
- ✅ `.gitignore` - Actualizado para Poetry

### Entorno Virtual:

- 📁 Ubicación: `.venv/` (en el directorio del proyecto)
- 🐍 Python: 3.12.2
- 📦 Dependencias: 100+ paquetes instalados

## 🚀 Comandos Principales

### Gestión del entorno:

```bash
# Activar entorno virtual
poetry shell

# Instalar dependencias
poetry install --no-root

# Actualizar dependencias
poetry update

# Ver dependencias instaladas
poetry show
```

### Gestión de paquetes:

```bash
# Agregar nueva dependencia
poetry add nombre-paquete

# Agregar dependencia de desarrollo
poetry add --group dev nombre-paquete

# Remover dependencia
poetry remove nombre-paquete
```

### Scripts personalizados:

```bash
# Validar configuración
poetry run python scripts/validate_setup.py

# Iniciar API (opción 1 - recomendada)
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Iniciar API (opción 2 - script personalizado)
poetry run python scripts/start_api.py

# Gestionar dependencias
poetry run python scripts/manage_deps.py list
```

## 🐳 Docker con Poetry

El proyecto mantiene compatibilidad con Docker. El `Dockerfile` y `docker-compose.yml` siguen funcionando:

```bash
# Iniciar todos los servicios
docker-compose up -d

# Solo la API
docker-compose up doc-restoration-api
```

## 📦 Estructura de Dependencias

### Principales grupos:

- **Core ML/DL**: torch, torchvision, opencv-python, albumentations
- **API**: fastapi, uvicorn, pydantic
- **Cloud Storage**: boto3, botocore (MinIO)
- **Data Science**: numpy, matplotlib, scikit-learn
- **Development**: black, isort, flake8, mypy, pytest

### Dependencias de desarrollo:

```bash
# Solo instalar dependencias de producción
poetry install --no-root --only main

# Instalar todo (desarrollo + producción)
poetry install --no-root
```

## 🔧 Solución a Problemas Comunes

### 1. **Error "Restormer no disponible"**

```bash
# Agregar dependencias faltantes específicas
poetry add BasicSR
poetry add facexlib
poetry add gfpgan
```

### 2. **Problemas de importación**

```bash
# Validar instalación
poetry run python scripts/validate_setup.py

# Reinstalar dependencias
poetry install --no-root --force
```

### 3. **Conflictos de versiones**

```bash
# Ver dependencias desactualizadas
poetry show --outdated

# Actualizar específico
poetry update nombre-paquete
```

## 📊 Ventajas de Poetry vs pip

| Aspecto                        | Poetry            | pip + requirements.txt |
| ------------------------------ | ----------------- | ---------------------- |
| **Resolución de dependencias** | ✅ Automática     | ❌ Manual              |
| **Lock file**                  | ✅ poetry.lock    | ❌ No                  |
| **Gestión de versiones**       | ✅ Inteligente    | ❌ Manual              |
| **Entornos virtuales**         | ✅ Automático     | ❌ Manual              |
| **Scripts personalizados**     | ✅ pyproject.toml | ❌ setup.py            |
| **Compatibilidad**             | ✅ pip compatible | ✅ Estándar            |

## 🎯 Siguientes Pasos Recomendados

1. **Crear archivo .env**:

   ```bash
   cp .env.example .env
   # Editar .env con tus configuraciones
   ```

2. **Configurar pre-commit hooks**:

   ```bash
   poetry run pre-commit install
   ```

3. **Probar la API**:

   ```bash
   poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   # Visitar: http://localhost:8000/docs
   ```

4. **Ejecutar tests** (cuando los tengas):
   ```bash
   poetry run pytest
   ```

## 🆘 Comandos de Emergencia

Si algo no funciona:

```bash
# Recrear entorno virtual
poetry env remove python
poetry install --no-root

# Limpiar cache
poetry cache clear pypi --all

# Verificar configuración
poetry config --list
```

## 📱 URLs de Desarrollo

- **API**: http://localhost:8000
- **Documentación**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **n8n**: http://localhost:5678
- **MinIO**: http://localhost:9001

---

**🎉 ¡Tu proyecto ahora usa Poetry y está libre de problemas de dependencias!**

Para cualquier duda, ejecuta: `poetry run python scripts/validate_setup.py`
