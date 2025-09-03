# n8n-automatize-models

Sistema de automatización de restauración de documentos con n8n

## Descripción

Este proyecto combina modelos de machine learning para restauración de documentos con automatización de workflows usando n8n.

## Estructura del Proyecto

- `api/` - API REST para los servicios de restauración
- `src/` - Código fuente de los modelos
- `layers/` - Componentes de entrenamiento por capas
- `config/` - Archivos de configuración
- `data/` - Datos de entrenamiento y validación
- `models/` - Modelos preentrenados y checkpoints
- `outputs/` - Resultados y análisis generados
- `n8n/` - Workflows y helpers de n8n

## Instalación

1. Instalar Poetry:

```bash
pip install poetry
```

2. Instalar dependencias:

```bash
poetry install
```

3. Activar el entorno virtual:

```bash
poetry shell
```

## Uso

### Iniciar la API

```bash
poetry run start-api
```

### Entrenar modelos

```bash
poetry run train-layer1
poetry run train-layer2
```

## Docker

Ejecutar con docker-compose:

```bash
docker-compose up -d
```

## Servicios

- API: http://localhost:8000
- n8n: http://localhost:5678
- MinIO: http://localhost:9001
