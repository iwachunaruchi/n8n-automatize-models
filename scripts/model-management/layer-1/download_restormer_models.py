#!/usr/bin/env python3
"""
Script para descargar modelos preentrenados Restormer y organizarlos en MinIO
Plantilla base para Layer-1 - Adaptable para diferentes modelos
"""
import os
import sys
import boto3
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib

# Configuración MinIO
MINIO_CONFIG = {
    'endpoint': 'localhost:9000',
    'access_key': 'minio',
    'secret_key': 'minio123'
}

# URLs de modelos Restormer para Layer-1
RESTORMER_MODELS = {
    'restormer-denoising': {
        'url': 'https://github.com/swz30/Restormer/releases/download/v1.0/restormer_denoising.pth',
        'filename': 'restormer_denoising.pth',
        'description': 'Restormer modelo preentrenado para denoising general',
        'size_mb': 'aprox. 50MB',
        'path': 'pretrained_models/layer_1/restormer/restormer_denoising.pth'
    },
    'restormer-deraining': {
        'url': 'https://github.com/swz30/Restormer/releases/download/v1.0/restormer_deraining.pth',
        'filename': 'restormer_deraining.pth',
        'description': 'Restormer modelo preentrenado para eliminación de lluvia',
        'size_mb': 'aprox. 50MB',
        'path': 'pretrained_models/layer_1/restormer/restormer_deraining.pth'
    },
    'restormer-motion-deblurring': {
        'url': 'https://github.com/swz30/Restormer/releases/download/v1.0/restormer_motion_deblurring.pth',
        'filename': 'restormer_motion_deblurring.pth',
        'description': 'Restormer modelo preentrenado para corrección de blur por movimiento',
        'size_mb': 'aprox. 50MB',
        'path': 'pretrained_models/layer_1/restormer/restormer_motion_deblurring.pth'
    }
    # Agregar más modelos según necesidades de Layer-1
}

def create_s3_client():
    """Crear cliente S3 para MinIO"""
    return boto3.client(
        's3',
        endpoint_url=f"http://{MINIO_CONFIG['endpoint']}",
        aws_access_key_id=MINIO_CONFIG['access_key'],
        aws_secret_access_key=MINIO_CONFIG['secret_key'],
        region_name='us-east-1'
    )

def download_file_with_progress(url, local_path):
    """Descargar archivo con barra de progreso"""
    print(f"📥 Descargando desde: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_path, 'wb') as file, tqdm(
            desc=local_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                pbar.update(size)
        
        print(f"✅ Descarga completada: {local_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error en descarga: {e}")
        raise

def calculate_file_hash(file_path):
    """Calcular hash MD5 del archivo"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def upload_to_minio(client, local_path, bucket, remote_path):
    """Subir archivo a MinIO"""
    try:
        with open(local_path, 'rb') as file:
            client.put_object(
                Bucket=bucket,
                Key=remote_path,
                Body=file,
                ContentType='application/octet-stream'
            )
        print(f"✅ Subido a MinIO: {remote_path}")
        
    except Exception as e:
        print(f"❌ Error subiendo a MinIO: {e}")
        raise

def create_model_info_file(client, bucket, model_name, model_info):
    """Crear archivo de información del modelo"""
    info_content = f"""# {model_name.upper()}

## Información del Modelo - Layer 1
- **Nombre**: {model_info['filename']}
- **Descripción**: {model_info['description']}
- **Tamaño**: {model_info['size_mb']}
- **URL Original**: {model_info['url']}
- **Capa**: Layer 1 (Primera etapa de procesamiento)

## Propósito en Layer 1
Primera etapa del pipeline de restauración de documentos. Se enfoca en:
- Denoising básico y mejora inicial de calidad
- Corrección de artefactos comunes
- Preparación para procesamiento de Layer 2

## Arquitectura
Restormer (Transformer-based) optimizado para tareas de restauración de imágenes:
- Eficiente en memoria y computación
- Excelente para capturar dependencias de largo alcance
- Ideal como backbone para fine-tuning en documentos

## Dataset de Entrenamiento
Entrenado en datasets específicos según la tarea:
- Denoising: Datasets sintéticos y reales de ruido
- Deraining: Datasets de eliminación de lluvia
- Motion Deblurring: Datasets de blur por movimiento

## Integración
Se carga automáticamente en el entrenamiento de Layer 1 cuando está disponible.
Compatible con arquitectura de transferencia de aprendizaje.

## Ubicación en MinIO
- **Bucket**: models
- **Ruta**: {model_info['path']}
"""
    
    info_path = model_info['path'].replace('.pth', '_info.md')
    
    try:
        client.put_object(
            Bucket=bucket,
            Key=info_path,
            Body=info_content.encode('utf-8'),
            ContentType='text/markdown'
        )
        print(f"📄 Archivo de información creado: {info_path}")
        
    except Exception as e:
        print(f"❌ Error creando archivo de información: {e}")

def main():
    """Función principal"""
    print("🚀 Descargando modelos preentrenados Restormer para Layer-1...")
    
    # Crear cliente S3
    client = create_s3_client()
    bucket = 'models'
    
    # Verificar bucket
    try:
        client.head_bucket(Bucket=bucket)
        print(f"✅ Bucket '{bucket}' encontrado")
    except:
        print(f"❌ Bucket '{bucket}' no encontrado. Ejecuta primero el setup de MinIO.")
        return
    
    # Crear directorio temporal
    temp_dir = Path('./temp_downloads_layer1')
    temp_dir.mkdir(exist_ok=True)
    
    for model_name, model_info in RESTORMER_MODELS.items():
        print(f"\n📦 Procesando modelo Layer-1: {model_name}")
        
        local_path = temp_dir / model_info['filename']
        
        # Verificar si ya existe en MinIO
        try:
            client.head_object(Bucket=bucket, Key=model_info['path'])
            print(f"ℹ️ Modelo ya existe en MinIO: {model_info['path']}")
            
            # Preguntar si reemplazar
            response = input(f"¿Deseas reemplazar el modelo existente? (y/N): ")
            if response.lower() != 'y':
                print("⏭️ Saltando descarga...")
                continue
                
        except:
            # El modelo no existe, proceder con descarga
            pass
        
        try:
            # Descargar modelo
            print(f"📥 Descargando {model_info['description']}...")
            download_file_with_progress(model_info['url'], local_path)
            
            # Calcular hash para verificación
            file_hash = calculate_file_hash(local_path)
            print(f"🔍 Hash MD5: {file_hash}")
            
            # Subir a MinIO
            print(f"☁️ Subiendo a MinIO...")
            upload_to_minio(client, local_path, bucket, model_info['path'])
            
            # Crear archivo de información
            create_model_info_file(client, bucket, model_name, model_info)
            
            # Limpiar archivo temporal
            local_path.unlink()
            print(f"🧹 Archivo temporal eliminado")
            
            print(f"✅ Modelo {model_name} procesado exitosamente!")
            
        except Exception as e:
            print(f"❌ Error procesando modelo {model_name}: {e}")
    
    # Limpiar directorio temporal
    try:
        temp_dir.rmdir()
    except:
        pass
    
    print(f"\n✅ ¡Proceso completado para Layer-1!")
    print(f"\n🌐 Revisa los modelos en MinIO: http://localhost:9000")
    print(f"   Bucket: models")
    print(f"   Carpeta: pretrained_models/layer_1/restormer/")

if __name__ == "__main__":
    main()
