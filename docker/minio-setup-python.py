#!/usr/bin/env python3
"""
Script integrado para configuración completa de MinIO en Docker
Basado en setup_pretrained_folders.py pero adaptado para Docker
"""
import time
import boto3
from botocore.exceptions import ClientError

# Configuración MinIO para Docker
MINIO_CONFIG = {
    'endpoint': 'minio:9000',  # Nombre del servicio en Docker
    'access_key': 'minio',
    'secret_key': 'minio123'
}

# Buckets principales
MAIN_BUCKETS = [
    'document-degraded',
    'document-clean', 
    'document-restored',
    'document-training',
    'models'
]

# Estructura de carpetas para modelos preentrenados
PRETRAINED_FOLDERS = [
    'pretrained_models/',
    'pretrained_models/layer_1/',
    'pretrained_models/layer_2/',
    'pretrained_models/layer_2/nafnet/',
    'pretrained_models/layer_2/docunet/',
    'pretrained_models/general/',
    'training_outputs/',
    'evaluation_results/',
    'checkpoints/'
]

def wait_for_minio():
    """Esperar a que MinIO esté disponible"""
    print("⏳ Esperando a que MinIO esté disponible...")
    max_attempts = 30
    
    for attempt in range(max_attempts):
        try:
            client = boto3.client(
                's3',
                endpoint_url=f"http://{MINIO_CONFIG['endpoint']}",
                aws_access_key_id=MINIO_CONFIG['access_key'],
                aws_secret_access_key=MINIO_CONFIG['secret_key'],
                region_name='us-east-1'
            )
            client.list_buckets()
            print("✅ MinIO está disponible!")
            return client
        except Exception as e:
            print(f"⏳ Intento {attempt + 1}/{max_attempts} - MinIO no disponible aún...")
            time.sleep(2)
    
    raise Exception("❌ MinIO no disponible después de esperar")

def create_buckets(client):
    """Crear buckets principales"""
    print("📦 Creando buckets principales...")
    
    for bucket in MAIN_BUCKETS:
        try:
            client.create_bucket(Bucket=bucket)
            print(f"✅ Bucket creado: {bucket}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                print(f"ℹ️  Bucket ya existe: {bucket}")
            else:
                print(f"❌ Error creando bucket {bucket}: {e}")

def create_folder_structure(client):
    """Crear estructura de carpetas para modelos"""
    print("📁 Creando estructura de carpetas...")
    
    bucket = 'models'
    for folder in PRETRAINED_FOLDERS:
        try:
            # Crear marcador de carpeta
            if not folder.endswith('/'):
                folder += '/'
            
            client.put_object(
                Bucket=bucket,
                Key=folder,
                Body='',
                ContentType='application/x-directory'
            )
            print(f"✅ Carpeta creada: {folder}")
            
        except Exception as e:
            print(f"❌ Error creando carpeta {folder}: {e}")

def create_readme_files(client):
    """Crear archivos README explicativos"""
    print("📄 Creando archivos README...")
    
    bucket = 'models'
    
    readme_contents = {
        'pretrained_models/README.md': """# Modelos Preentrenados

Esta carpeta contiene todos los modelos preentrenados organizados por capas y tipos.

## Estructura:
- `layer_1/`: Modelos para la primera capa de restauración
- `layer_2/`: Modelos para la segunda capa de restauración
  - `nafnet/`: Modelos NAFNet (Noise Aware Filtering Network)
  - `docunet/`: Modelos DocUNet (Document Unwarping Network)
- `general/`: Modelos de propósito general

## Uso:
Los modelos en esta carpeta se utilizan como punto de partida para fine-tuning
específico del dominio de restauración de documentos.

## Configurado automáticamente:
Esta estructura fue creada por Docker Compose al inicializar el sistema.
""",
        
        'pretrained_models/layer_1/README.md': """# Modelos Preentrenados - Layer 1

Modelos preentrenados para la primera capa de restauración de documentos.

## Propósito:
Primera etapa de procesamiento, enfocada en denoising básico y mejora inicial de calidad.

## Modelos recomendados:
- Restormer para denoising general
- DnCNN para ruido gaussiano
- FFDNet para ruido real

## Descarga automática:
Los modelos se descargan automáticamente durante el entrenamiento si no están presentes.
""",
        
        'pretrained_models/layer_2/nafnet/README.md': """# NAFNet Pretrained Models

Modelos NAFNet (Noise Aware Filtering Network) preentrenados.

## Modelo recomendado:
- **NAFNet-SIDD-width64**: Óptimo para denoising de documentos
  - Entrenado en dataset SIDD (real-world noise)
  - Arquitectura equilibrada (width=64)
  - Excelente para fine-tuning

## Descarga:
Los modelos se descargan automáticamente durante el entrenamiento si no están presentes.

## Uso:
Utilizados como backbone para fine-tuning específico en documentos degradados.
""",
        
        'pretrained_models/layer_2/docunet/README.md': """# DocUNet Pretrained Models

Modelos DocUNet especializados en unwrapping de documentos.

## Propósito:
Corrección de distorsiones geométricas en documentos fotografiados.

## Modelos disponibles:
- DocUNet base para documentos generales
- Variantes especializadas para diferentes tipos de distorsión

## Aplicación:
Segunda etapa del pipeline de restauración, enfocada en corrección geométrica.
""",
        
        'training_outputs/README.md': """# Training Outputs

Carpeta para almacenar resultados y artefactos de entrenamiento.

## Contenido:
- Checkpoints de modelos durante entrenamiento
- Logs y métricas de entrenamiento
- Gráficos de progreso y análisis
- Reportes automáticos generados
""",
        
        'checkpoints/README.md': """# Model Checkpoints

Puntos de control de modelos durante el entrenamiento.

## Organización:
- Checkpoints por época
- Mejores modelos según métricas
- Modelos finales entrenados

## Uso:
Para reanudar entrenamientos y seleccionar mejores versiones.
"""
    }
    
    for path, content in readme_contents.items():
        try:
            client.put_object(
                Bucket=bucket,
                Key=path,
                Body=content.encode('utf-8'),
                ContentType='text/markdown'
            )
            print(f"✅ README creado: {path}")
        except Exception as e:
            print(f"❌ Error creando README {path}: {e}")

def main():
    """Función principal"""
    print("🚀 Iniciando configuración completa de MinIO...")
    
    try:
        # Esperar y conectar a MinIO
        client = wait_for_minio()
        
        # Crear buckets principales
        create_buckets(client)
        
        # Crear estructura de carpetas
        create_folder_structure(client)
        
        # Crear archivos README
        create_readme_files(client)
        
        print("\n✅ ¡Configuración completa de MinIO finalizada exitosamente!")
        print("\n📋 Resumen:")
        print(f"   - Buckets creados: {', '.join(MAIN_BUCKETS)}")
        print(f"   - Carpetas configuradas: {len(PRETRAINED_FOLDERS)}")
        print(f"   - Documentación: 6 archivos README")
        print("\n🌐 Acceso MinIO Console: http://localhost:9001")
        print("   Usuario: minio")
        print("   Contraseña: minio123")
        
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        exit(1)

if __name__ == "__main__":
    main()
