"""
Servicio para manejo de MinIO
"""
import boto3
from botocore.exceptions import ClientError
import logging
from fastapi import HTTPException

# Importar configuración con path relativo
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.settings import MINIO_CONFIG, BUCKETS
except ImportError:
    # Fallback configuration
    MINIO_CONFIG = {
        'endpoint': os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
        'access_key': os.getenv('MINIO_ACCESS_KEY', 'minio'),
        'secret_key': os.getenv('MINIO_SECRET_KEY', 'minio123'),
        'secure': os.getenv('MINIO_SECURE', 'false').lower() == 'true'
    }
    BUCKETS = {
        'degraded': 'document-degraded',
        'clean': 'document-clean',
        'restored': 'document-restored',
        'training': 'document-training',
        'models': 'models'
    }

logger = logging.getLogger(__name__)

class MinIOService:
    """Servicio para operaciones con MinIO"""
    
    def __init__(self):
        self.client = None
        self._connect()
    
    def _connect(self):
        """Crear cliente MinIO"""
        try:
            self.client = boto3.client(
                's3',
                endpoint_url=f"http://{MINIO_CONFIG['endpoint']}",
                aws_access_key_id=MINIO_CONFIG['access_key'],
                aws_secret_access_key=MINIO_CONFIG['secret_key'],
                region_name='us-east-1'
            )
        except Exception as e:
            logger.error(f"Error conectando a MinIO: {e}")
            raise HTTPException(status_code=500, detail="Error de conexión a MinIO")
    
    def ensure_buckets(self):
        """Crear buckets si no existen"""
        for bucket_name in BUCKETS.values():
            try:
                self.client.head_bucket(Bucket=bucket_name)
                logger.info(f"Bucket '{bucket_name}' existe")
            except ClientError:
                try:
                    self.client.create_bucket(Bucket=bucket_name)
                    logger.info(f"Bucket '{bucket_name}' creado")
                except Exception as e:
                    logger.error(f"Error creando bucket '{bucket_name}': {e}")
    
    def upload_file(self, file_data: bytes, bucket: str, filename: str) -> str:
        """Subir archivo a MinIO"""
        try:
            self.client.put_object(
                Bucket=bucket,
                Key=filename,
                Body=file_data
            )
            return filename
        except Exception as e:
            logger.error(f"Error subiendo archivo: {e}")
            raise HTTPException(status_code=500, detail="Error subiendo archivo")
    
    def download_file(self, bucket: str, filename: str) -> bytes:
        """Descargar archivo de MinIO"""
        try:
            response = self.client.get_object(Bucket=bucket, Key=filename)
            return response['Body'].read()
        except Exception as e:
            logger.error(f"Error descargando archivo: {e}")
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
    
    def list_files(self, bucket: str, prefix: str = "") -> list:
        """Listar archivos en bucket"""
        try:
            response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            logger.error(f"Error listando archivos: {e}")
            return []
    
    def delete_file(self, bucket: str, filename: str) -> bool:
        """Eliminar archivo de MinIO"""
        try:
            self.client.delete_object(Bucket=bucket, Key=filename)
            return True
        except Exception as e:
            logger.error(f"Error eliminando archivo: {e}")
            return False
    
    def upload_model(self, model_data: bytes, layer: str, model_name: str) -> str:
        """Subir modelo entrenado a MinIO organizando por capas"""
        try:
            # Crear path organizado por capa
            model_path = f"layer_{layer}/{model_name}"
            
            # Subir al bucket models
            self.client.put_object(
                Bucket=BUCKETS.get('models', 'models'),
                Key=model_path,
                Body=model_data,
                ContentType='application/octet-stream'
            )
            
            logger.info(f"Modelo guardado exitosamente: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error subiendo modelo: {e}")
            raise HTTPException(status_code=500, detail=f"Error guardando modelo: {str(e)}")
    
    def download_model(self, layer: str, model_name: str) -> bytes:
        """Descargar modelo específico de una capa"""
        try:
            model_path = f"layer_{layer}/{model_name}"
            response = self.client.get_object(
                Bucket=BUCKETS.get('models', 'models'),
                Key=model_path
            )
            return response['Body'].read()
            
        except Exception as e:
            logger.error(f"Error descargando modelo {layer}/{model_name}: {e}")
            raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {layer}/{model_name}")
    
    def list_models(self, layer: str = None) -> list:
        """Listar modelos disponibles, opcionalmente filtrados por capa"""
        try:
            prefix = f"layer_{layer}/" if layer else ""
            
            response = self.client.list_objects_v2(
                Bucket=BUCKETS.get('models', 'models'),
                Prefix=prefix
            )
            
            models = []
            for obj in response.get('Contents', []):
                model_info = {
                    'path': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'layer': obj['Key'].split('/')[0] if '/' in obj['Key'] else 'unknown',
                    'filename': obj['Key'].split('/')[-1] if '/' in obj['Key'] else obj['Key']
                }
                models.append(model_info)
            
            return models
            
        except Exception as e:
            logger.error(f"Error listando modelos: {e}")
            return []

# Instancia global del servicio
minio_service = MinIOService()
