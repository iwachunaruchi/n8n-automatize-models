#!/usr/bin/env python3
"""
Script para crear la estructura de carpetas para modelos preentrenados en MinIO
"""
import sys
import os
import boto3
from botocore.exceptions import ClientError

# Configuración MinIO
MINIO_CONFIG = {
    'endpoint': 'localhost:9000',
    'access_key': 'minio',
    'secret_key': 'minio123'
}

# Estructura de carpetas a crear
PRETRAINED_FOLDERS = [
    'pretrained_models/',
    'pretrained_models/layer_1/',
    'pretrained_models/layer_2/',
    'pretrained_models/layer_2/nafnet/',
    'pretrained_models/layer_2/docunet/',
    'pretrained_models/general/'
]

def create_s3_client():
    """Crear cliente S3 para MinIO"""
    return boto3.client(
        's3',
        endpoint_url=f"http://{MINIO_CONFIG['endpoint']}",
        aws_access_key_id=MINIO_CONFIG['access_key'],
        aws_secret_access_key=MINIO_CONFIG['secret_key'],
        region_name='us-east-1'
    )

def create_folder_marker(client, bucket, folder_path):
    """Crear marcador de carpeta en S3/MinIO"""
    try:
        # En S3, las carpetas se representan con objetos que terminan en '/'
        if not folder_path.endswith('/'):
            folder_path += '/'
        
        # Crear un objeto vacío que actúa como marcador de carpeta
        client.put_object(
            Bucket=bucket,
            Key=folder_path,
            Body='',
            ContentType='application/x-directory'
        )
        print(f"✅ Carpeta creada: {folder_path}")
        
    except Exception as e:
        print(f"❌ Error creando carpeta {folder_path}: {e}")

def create_readme_file(client, bucket, folder_path, content):
    """Crear archivo README en una carpeta"""
    try:
        readme_path = folder_path + 'README.md'
        client.put_object(
            Bucket=bucket,
            Key=readme_path,
            Body=content.encode('utf-8'),
            ContentType='text/markdown'
        )
        print(f"📄 README creado: {readme_path}")
        
    except Exception as e:
        print(f"❌ Error creando README en {folder_path}: {e}")

def main():
    """Función principal"""
    print("🚀 Configurando estructura de modelos preentrenados en MinIO...")
    
    # Crear cliente S3
    client = create_s3_client()
    bucket = 'models'
    
    # Verificar que el bucket existe
    try:
        client.head_bucket(Bucket=bucket)
        print(f"✅ Bucket '{bucket}' encontrado")
    except ClientError:
        print(f"❌ Bucket '{bucket}' no encontrado")
        return
    
    # Crear estructura de carpetas
    print("\n📁 Creando estructura de carpetas...")
    for folder in PRETRAINED_FOLDERS:
        create_folder_marker(client, bucket, folder)
    
    # Crear archivos README explicativos
    print("\n📄 Creando archivos README...")
    
    readme_content = {
        'pretrained_models/': """# Modelos Preentrenados

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
""",
        
        'pretrained_models/layer_1/': """# Modelos Preentrenados - Layer 1

Modelos preentrenados para la primera capa de restauración de documentos.

## Propósito:
Primera etapa de procesamiento, enfocada en denoising básico y mejora inicial de calidad.

## Modelos recomendados:
- Restormer para denoising general
- DnCNN para ruido gaussiano
- FFDNet para ruido real
""",
        
        'pretrained_models/layer_2/': """# Modelos Preentrenados - Layer 2

Modelos preentrenados para la segunda capa de restauración de documentos.

## Subdirectorios:
- `nafnet/`: Modelos NAFNet especializados
- `docunet/`: Modelos DocUNet para unwrapping

## Estrategia:
Fine-tuning de modelos preentrenados en datasets específicos de documentos.
""",
        
        'pretrained_models/layer_2/nafnet/': """# NAFNet Pretrained Models

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
        
        'pretrained_models/layer_2/docunet/': """# DocUNet Pretrained Models

Modelos DocUNet especializados en unwrapping de documentos.

## Propósito:
Corrección de distorsiones geométricas en documentos fotografiados.

## Modelos disponibles:
- DocUNet base para documentos generales
- Variantes especializadas para diferentes tipos de distorsión

## Aplicación:
Segunda etapa del pipeline de restauración, enfocada en corrección geométrica.
""",
        
        'pretrained_models/general/': """# Modelos Generales

Modelos preentrenados de propósito general para restauración de imágenes.

## Contenido:
- Modelos base para transferencia de aprendizaje
- Backbones preentrenados en ImageNet
- Modelos auxiliares para tareas específicas

## Uso:
Punto de partida para entrenamiento cuando no hay modelos específicos disponibles.
"""
    }
    
    for folder, content in readme_content.items():
        create_readme_file(client, bucket, folder, content)
    
    print("\n✅ ¡Estructura de modelos preentrenados creada exitosamente!")
    print("\n📋 Carpetas creadas:")
    for folder in PRETRAINED_FOLDERS:
        print(f"   - {folder}")
    
    print(f"\n🌐 Accede a MinIO en: http://localhost:9000")
    print(f"   Usuario: {MINIO_CONFIG['access_key']}")
    print(f"   Contraseña: {MINIO_CONFIG['secret_key']}")

if __name__ == "__main__":
    main()
