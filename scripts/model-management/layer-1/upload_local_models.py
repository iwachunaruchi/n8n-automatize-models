#!/usr/bin/env python3
"""
Script para subir modelos preentrenados locales de Layer-1 a MinIO
Específico para modelos Restormer y otros modelos de la primera capa
"""
import os
import sys
import boto3
from pathlib import Path
import hashlib
import datetime

# Configuración MinIO
MINIO_CONFIG = {
    'endpoint': 'localhost:9000',
    'access_key': 'minio',
    'secret_key': 'minio123'
}

# Mapeo de modelos locales Layer-1 a rutas en MinIO
LOCAL_MODELS_LAYER1 = {
    'models/restormer/restormer_denoising.pth': {
        'minio_path': 'pretrained_models/layer_1/restormer/restormer_denoising.pth',
        'description': 'Restormer modelo preentrenado para denoising general',
        'type': 'restormer',
        'layer': 'layer_1',
        'task': 'denoising'
    },
    'models/restormer/restormer_deraining.pth': {
        'minio_path': 'pretrained_models/layer_1/restormer/restormer_deraining.pth',
        'description': 'Restormer modelo preentrenado para eliminación de lluvia',
        'type': 'restormer',
        'layer': 'layer_1',
        'task': 'deraining'
    },
    'models/restormer/restormer_motion_deblurring.pth': {
        'minio_path': 'pretrained_models/layer_1/restormer/restormer_motion_deblurring.pth',
        'description': 'Restormer modelo preentrenado para corrección de blur por movimiento',
        'type': 'restormer',
        'layer': 'layer_1',
        'task': 'motion_deblurring'
    }
    # Agregar más modelos según tengas localmente para Layer-1
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
    """Crear archivo de información del modelo Layer-1"""
    file_hash = calculate_file_hash(local_path)
    file_size = get_file_size_mb(local_path)
    
    info_content = f"""# {Path(local_path).stem.upper()} - LAYER 1

## Información del Modelo
- **Archivo**: {Path(local_path).name}
- **Descripción**: {model_info['description']}
- **Tamaño**: {file_size} MB
- **Tipo**: {model_info['type']}
- **Capa**: {model_info['layer']}
- **Tarea específica**: {model_info['task']}
- **Hash MD5**: {file_hash}

## Propósito en Layer 1
Primera etapa del pipeline de restauración de documentos:
- **Denoising**: Eliminación de ruido básico y artefactos
- **Mejora inicial**: Preparación para procesamiento más avanzado
- **Compatibilidad**: Base sólida para transferencia a Layer 2

## Arquitectura Restormer
Transformer-based architecture optimizada para restauración:
- **Multi-Dconv head**: Captura características locales y globales
- **Gated-Dconv FeedForward**: Procesamiento eficiente
- **Progressive learning**: Ideal para fine-tuning progresivo

## Dataset de Entrenamiento
Según la tarea específica:
- **Denoising**: Datasets de ruido sintético y real
- **Deraining**: Datasets específicos de eliminación de lluvia
- **Motion Deblurring**: Datasets de corrección de blur por movimiento

## Integración en Pipeline
- Se carga automáticamente en Layer 1 cuando está disponible
- Compatible con fine-tuning diferencial
- Prepara características para Layer 2 (NAFNet/DocUNet)

## Uso Recomendado
```python
# Carga en Layer 1
model = load_pretrained_restormer(task='{model_info['task']}')
# Fine-tune en documentos específicos
fine_tuned_model = fine_tune_layer1(model, document_dataset)
```

## Ubicación en MinIO
- **Bucket**: models
- **Ruta**: {model_info['minio_path']}

## Fecha de Subida
{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Compatibilidad
- ✅ Layer 1 training pipeline
- ✅ Progressive transfer learning
- ✅ Multi-task fine-tuning
- ✅ Document domain adaptation
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
    print("🚀 Subiendo modelos preentrenados Layer-1 locales a MinIO...")
    
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
    
    processed_count = 0
    
    for local_path, model_info in LOCAL_MODELS_LAYER1.items():
        print(f"\n📦 Procesando modelo Layer-1: {local_path}")
        
        # Verificar que el archivo local existe
        if not os.path.exists(local_path):
            print(f"❌ Archivo no encontrado: {local_path}")
            print(f"   ℹ️ Coloca el modelo en la ruta especificada o actualiza el mapeo")
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
            
            print(f"✅ Modelo Layer-1 procesado exitosamente!")
            processed_count += 1
            
        except Exception as e:
            print(f"❌ Error procesando modelo: {e}")
    
    print(f"\n✅ ¡Proceso de subida completado!")
    print(f"📊 Modelos procesados: {processed_count}")
    print(f"\n🌐 Revisa los modelos en MinIO: http://localhost:9000")
    print(f"   Bucket: models")
    print(f"   Carpeta: pretrained_models/layer_1/restormer/")

if __name__ == "__main__":
    main()
