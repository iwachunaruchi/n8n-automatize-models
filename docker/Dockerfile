# Dockerfile para la API de Restauración con Poetry
FROM python:3.11-slim AS base

# Configurar variables de entorno para Poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=1.8.3 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-dev \
    curl \
    wget \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar Poetry
RUN pip install poetry==$POETRY_VERSION

# Configurar directorio de trabajo
WORKDIR /app

# Copiar archivos de Poetry primero (para cache de Docker)
COPY pyproject.toml poetry.lock* ./

# Configurar Poetry y instalar dependencias
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only=main \
    && rm -rf $POETRY_CACHE_DIR

# Copiar código fuente
COPY . .

# Crear directorios necesarios
RUN mkdir -p outputs/checkpoints models/pretrained logs

# Crear usuario no-root para seguridad
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Exponer puerto
EXPOSE 8000

# Variables de entorno por defecto (sin credenciales sensibles)
ENV PYTHONPATH=/app \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
