# Dockerfile para la API de Restauración
FROM python:3.11-slim

# Instalar dependencias del sistema (actualizadas para Debian trixie)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Configurar directorio de trabajo
WORKDIR /app

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt .
COPY api/requirements.txt ./api/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Instalar watchfiles para hot reload
RUN pip install --no-cache-dir watchfiles

# Copiar código fuente
COPY . .

# Crear directorios necesarios
RUN mkdir -p outputs/checkpoints models/pretrained

# Exponer puerto
EXPOSE 8000

# Variables de entorno por defecto
ENV PYTHONPATH=/app
ENV MINIO_ENDPOINT=minio:9000
ENV MINIO_ACCESS_KEY=minio
ENV MINIO_SECRET_KEY=minio123

# Comando por defecto
CMD ["python", "api/main.py"]
