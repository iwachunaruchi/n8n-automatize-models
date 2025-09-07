#!/usr/bin/env python3
"""
Script para subir modelos preentrenados locales a MinIO
"""
import os
import sys
import boto3
from pathlib import Path
import hashlib

# Configuración MinIO
MINIO_CONFIG = {
    'endpoint': 'localhost:9000',
    'access_key': 'minio',
    'secret_key': 'minio123'
}

# Mapeo de modelos locales a rutas en MinIO
# Obtener la ruta base del proyecto
project_root = Path(__file__).parent.parent.parent.parent
LOCAL_MODELS = {
    str(project_root / 'models/NAFnet/NAFNet-SIDD-width64.pth'): {
        'minio_path': 'pretrained_models/layer_2/nafnet/NAFNet-SIDD-width64.pth',
        'description': 'NAFNet modelo preentrenado en SIDD dataset para denoising',
        'type': 'nafnet',
        'layer': 'layer_2'
    }
    # Aquí puedes agregar más modelos según los tengas localmente
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

def calculate_file_hash(file_path):
    """Calcular hash MD5 del archivo"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_file_size_mb(file_path):
    """Obtener tamaño del archivo en MB"""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)

def upload_to_minio(client, local_path, bucket, remote_path):
    """Subir archivo a MinIO con progreso"""
    try:
        file_size = os.path.getsize(local_path)
        print(f"📤 Subiendo {local_path} ({get_file_size_mb(local_path)} MB)...")
        
        with open(local_path, 'rb') as file:
            client.put_object(
                Bucket=bucket,
                Key=remote_path,
                Body=file,
                ContentType='application/octet-stream'
            )
        print(f"✅ Subido exitosamente: {remote_path}")
        
    except Exception as e:
        print(f"❌ Error subiendo a MinIO: {e}")
        raise

def create_model_info_file(client, bucket, model_info, local_path):
    """Crear archivo de información del modelo"""
    file_hash = calculate_file_hash(local_path)
    file_size = get_file_size_mb(local_path)
    
    info_content = f"""# {Path(local_path).stem.upper()}

## Información del Modelo
- **Archivo**: {Path(local_path).name}
- **Descripción**: {model_info['description']}
- **Tamaño**: {file_size} MB
- **Tipo**: {model_info['type']}
- **Capa**: {model_info['layer']}
- **Hash MD5**: {file_hash}

## Uso
Este modelo preentrenado se utiliza como punto de partida para fine-tuning
en tareas de restauración de documentos, específicamente para denoising.

## Arquitectura
NAFNet (Noise Aware Filtering Network) con width=64, optimizado para
balance entre calidad y eficiencia computacional.

## Dataset de Entrenamiento
SIDD (Smartphone Image Denoising Dataset) - contiene ruido real de smartphones,
ideal para aplicaciones de denoising en documentos fotografiados.

## Integración
Se carga automáticamente en el entrenamiento de Layer 2 cuando está disponible.

## Ubicación en MinIO
- **Bucket**: models
- **Ruta**: {model_info['minio_path']}

## Fecha de Subida
{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    info_path = model_info['minio_path'].replace('.pth', '_info.md')
    
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
    print("🚀 Subiendo modelos preentrenados locales a MinIO...")
    
    # Crear cliente S3
    client = create_s3_client()
    bucket = 'models'
    
    # Verificar bucket
    try:
        client.head_bucket(Bucket=bucket)
        print(f"✅ Bucket '{bucket}' encontrado")
    except:
        print(f"❌ Bucket '{bucket}' no encontrado")
        return
    
    for local_path, model_info in LOCAL_MODELS.items():
        print(f"\n📦 Procesando modelo: {local_path}")
        
        # Verificar que el archivo local existe
        if not os.path.exists(local_path):
            print(f"❌ Archivo no encontrado: {local_path}")
            continue
        
        # Verificar si ya existe en MinIO
        try:
            client.head_object(Bucket=bucket, Key=model_info['minio_path'])
            print(f"ℹ️ Modelo ya existe en MinIO: {model_info['minio_path']}")
            
            # Preguntar si reemplazar
            response = input(f"¿Deseas reemplazar el modelo existente? (y/N): ")
            if response.lower() != 'y':
                print("⏭️ Saltando subida...")
                continue
                
        except:
            # El modelo no existe, proceder con subida
            pass
        
        try:
            # Calcular hash para verificación
            print(f"🔍 Calculando hash del archivo...")
            file_hash = calculate_file_hash(local_path)
            print(f"🔍 Hash MD5: {file_hash}")
            
            # Subir a MinIO
            upload_to_minio(client, local_path, bucket, model_info['minio_path'])
            
            # Crear archivo de información
            create_model_info_file(client, bucket, model_info, local_path)
            
            print(f"✅ Modelo procesado exitosamente!")
            
        except Exception as e:
            print(f"❌ Error procesando modelo: {e}")
    
    print(f"\n✅ ¡Proceso de subida completado!")
    print(f"\n🌐 Revisa los modelos en MinIO: http://localhost:9000")
    print(f"   Bucket: models")
    print(f"   Carpeta: pretrained_models/layer_2/nafnet/")

if __name__ == "__main__":
    import datetime
    main()
