#!/usr/bin/env python3
"""
API REST para Restauración de Documentos
Integración con n8n y MinIO para automatización
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import io
import os
import uuid
import asyncio
import logging
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import cv2
import numpy as np
import torch
import random
import base64
import uvicorn
from PIL import Image

# Importar nuestros modelos
import sys
sys.path.append('src')
from src.models.restormer import Restormer

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la API
app = FastAPI(
    title="Document Restoration API",
    description="API para restauración de documentos con Transfer Learning Gradual",
    version="1.0.0"
)

# CORS para n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de MinIO usando variables de entorno
MINIO_CONFIG = {
    'endpoint': os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
    'access_key': os.getenv('MINIO_ACCESS_KEY', 'minio'),
    'secret_key': os.getenv('MINIO_SECRET_KEY', 'minio123'),
    'secure': os.getenv('MINIO_SECURE', 'false').lower() == 'true'
}

# Buckets de MinIO
BUCKETS = {
    'degraded': 'document-degraded',
    'clean': 'document-clean',
    'restored': 'document-restored',
    'training': 'document-training'
}

# Estado global del modelo
model_state = {
    'model': None,
    'device': None,
    'loaded': False
}

# Modelos de datos
class ProcessingJob(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    created_at: datetime
    completed_at: Optional[datetime] = None
    input_file: str
    output_file: Optional[str] = None
    error: Optional[str] = None
    # Campos para generación de datos sintéticos
    generated_count: Optional[int] = None
    source_file: Optional[str] = None
    generation_type: Optional[str] = None

class BatchProcessRequest(BaseModel):
    bucket_name: str
    file_patterns: List[str] = ["*.png", "*.jpg", "*.jpeg"]
    output_bucket: str = "document-restored"

class TrainingRequest(BaseModel):
    clean_bucket: str = "document-clean"
    degraded_bucket: str = "document-degraded"
    epochs: int = 10
    batch_size: int = 2

# Estado de trabajos
jobs_state = {}

# Cliente MinIO
def get_minio_client():
    """Crear cliente MinIO"""
    try:
        client = boto3.client(
            's3',
            endpoint_url=f"http://{MINIO_CONFIG['endpoint']}",
            aws_access_key_id=MINIO_CONFIG['access_key'],
            aws_secret_access_key=MINIO_CONFIG['secret_key'],
            region_name='us-east-1'
        )
        return client
    except Exception as e:
        logger.error(f"Error conectando a MinIO: {e}")
        raise HTTPException(status_code=500, detail="Error de conexión a MinIO")

def ensure_buckets():
    """Crear buckets si no existen"""
    client = get_minio_client()
    
    for bucket_name in BUCKETS.values():
        try:
            client.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket '{bucket_name}' existe")
        except ClientError:
            try:
                client.create_bucket(Bucket=bucket_name)
                logger.info(f"Bucket '{bucket_name}' creado")
            except Exception as e:
                logger.error(f"Error creando bucket '{bucket_name}': {e}")

def load_model():
    """Cargar modelo de restauración"""
    if model_state['loaded']:
        return model_state['model']
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Cargando modelo en dispositivo: {device}")
        
        # Cargar el mejor modelo disponible
        model_paths = [
            "outputs/checkpoints/gradual_transfer_final.pth",
            "outputs/checkpoints/optimized_restormer_final.pth",
            "outputs/checkpoints/finetuned_restormer_final.pth"
        ]
        
        model = None
        for model_path in model_paths:
            if os.path.exists(model_path):
                logger.info(f"Cargando modelo: {model_path}")
                
                model = Restormer(
                    inp_channels=3, out_channels=3, dim=48,
                    num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                    heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                    bias=False, LayerNorm_type='WithBias', dual_pixel_task=False
                )
                
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.to(device)
                model.eval()
                break
        
        if model is None:
            raise Exception("No se encontraron modelos entrenados")
        
        model_state['model'] = model
        model_state['device'] = device
        model_state['loaded'] = True
        
        logger.info("Modelo cargado exitosamente")
        return model
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {str(e)}")

def restore_image(image_bytes: bytes) -> bytes:
    """Restaurar imagen usando el modelo"""
    try:
        model = load_model()
        device = model_state['device']
        
        # Convertir bytes a imagen
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("No se pudo decodificar la imagen")
        
        # Preprocesar
        h, w = image.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # Convertir a tensor
        input_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        # Inferencia
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Postprocesar
        output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        
        # Recortar al tamaño original
        restored = output[:h, :w]
        
        # Convertir a bytes
        _, encoded = cv2.imencode('.png', restored)
        return encoded.tobytes()
        
    except Exception as e:
        logger.error(f"Error restaurando imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error en restauración: {str(e)}")

# Inicialización
@app.on_event("startup")
async def startup_event():
    """Inicialización de la API"""
    logger.info("Iniciando Document Restoration API")
    ensure_buckets()
    load_model()
    logger.info("API lista")

# Endpoints

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Document Restoration API - HOT RELOAD ACTIVO! 🔥",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model_state['loaded'],
        "device": str(model_state['device']) if model_state['device'] else None,
        "hot_reload": "✅ Funcionando"
    }

@app.get("/health")
async def health_check():
    """Health check para n8n"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if model_state['loaded'] else "not_loaded",
        "minio_status": "connected"  # TODO: verificar conexión real
    }

@app.post("/restore/single")
async def restore_single_image(file: UploadFile = File(...)):
    """Restaurar una sola imagen"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Leer imagen
        image_bytes = await file.read()
        
        # Restaurar
        restored_bytes = restore_image(image_bytes)
        
        # Retornar imagen restaurada
        return StreamingResponse(
            io.BytesIO(restored_bytes),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=restored_{file.filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error en restore_single_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/restore/from-minio")
async def restore_from_minio(
    bucket_name: str,
    object_name: str,
    output_bucket: str = "document-restored"
):
    """Restaurar imagen desde MinIO"""
    try:
        client = get_minio_client()
        
        # Descargar imagen de MinIO
        response = client.get_object(Bucket=bucket_name, Key=object_name)
        image_bytes = response['Body'].read()
        
        # Restaurar
        restored_bytes = restore_image(image_bytes)
        
        # Subir imagen restaurada a MinIO
        output_name = f"restored_{object_name}"
        client.put_object(
            Bucket=output_bucket,
            Key=output_name,
            Body=restored_bytes,
            ContentType='image/png'
        )
        
        return {
            "status": "success",
            "input_file": f"{bucket_name}/{object_name}",
            "output_file": f"{output_bucket}/{output_name}",
            "processing_time": "calculated_time"
        }
        
    except Exception as e:
        logger.error(f"Error en restore_from_minio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/restore/batch")
async def restore_batch(
    background_tasks: BackgroundTasks,
    request: BatchProcessRequest
):
    """Procesar lote de imágenes en background"""
    job_id = str(uuid.uuid4())
    
    # Crear trabajo
    job = ProcessingJob(
        job_id=job_id,
        status="pending",
        created_at=datetime.now(),
        input_file=f"{request.bucket_name}/*",
        output_file=f"{request.output_bucket}/"
    )
    
    jobs_state[job_id] = job
    
    # Procesar en background
    background_tasks.add_task(
        process_batch_background,
        job_id,
        request.bucket_name,
        request.output_bucket,
        request.file_patterns
    )
    
    return {"job_id": job_id, "status": "pending"}

async def process_batch_background(
    job_id: str,
    input_bucket: str,
    output_bucket: str,
    file_patterns: List[str]
):
    """Procesar lote en background"""
    try:
        jobs_state[job_id].status = "processing"
        
        client = get_minio_client()
        
        # Listar objetos
        response = client.list_objects_v2(Bucket=input_bucket)
        objects = response.get('Contents', [])
        
        processed_count = 0
        
        for obj in objects:
            object_name = obj['Key']
            
            # Verificar extensión
            if any(object_name.lower().endswith(pattern.replace('*', '')) 
                   for pattern in file_patterns):
                
                try:
                    # Procesar imagen
                    img_response = client.get_object(Bucket=input_bucket, Key=object_name)
                    image_bytes = img_response['Body'].read()
                    
                    restored_bytes = restore_image(image_bytes)
                    
                    # Subir resultado
                    output_name = f"restored_{object_name}"
                    client.put_object(
                        Bucket=output_bucket,
                        Key=output_name,
                        Body=restored_bytes,
                        ContentType='image/png'
                    )
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error procesando {object_name}: {e}")
        
        # Completar trabajo
        jobs_state[job_id].status = "completed"
        jobs_state[job_id].completed_at = datetime.now()
        jobs_state[job_id].output_file = f"{output_bucket}/ ({processed_count} files)"
        
    except Exception as e:
        jobs_state[job_id].status = "failed"
        jobs_state[job_id].error = str(e)
        logger.error(f"Error en batch processing: {e}")

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Obtener estado de trabajo"""
    if job_id not in jobs_state:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    
    return jobs_state[job_id]

@app.get("/jobs")
async def list_jobs():
    """Listar todos los trabajos"""
    return {"jobs": list(jobs_state.values())}

@app.post("/buckets/list")
async def list_bucket_contents(bucket_name: str):
    """Listar contenido de bucket (para n8n)"""
    try:
        client = get_minio_client()
        response = client.list_objects_v2(Bucket=bucket_name)
        
        objects = []
        for obj in response.get('Contents', []):
            objects.append({
                "key": obj['Key'],
                "size": obj['Size'],
                "last_modified": obj['LastModified'].isoformat(),
                "url": f"http://{MINIO_CONFIG['endpoint']}/{bucket_name}/{obj['Key']}"
            })
        
        return {"bucket": bucket_name, "objects": objects}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/image-quality")
async def classify_image_quality(file: UploadFile = File()):
    """Clasificar imagen como clean o degraded usando análisis de calidad"""
    print("=== INICIO classify_image_quality ===")
    print(f"File object: {file}")
    print(f"Filename: {getattr(file, 'filename', 'N/A')}")
    print(f"Content-type: {getattr(file, 'content_type', 'N/A')}")
    print(f"Size: {getattr(file, 'size', 'N/A')}")
    
    try:
        # Logging más detallado para debug
        content_type = getattr(file, 'content_type', None)
        print(f"Content-type detectado: '{content_type}'")
        
        # Validar tipo de contenido de manera más permisiva
        if content_type and content_type not in ['application/octet-stream', None]:
            print(f"Validando content-type: {content_type}")
            if not content_type.startswith('image/'):
                print(f"WARNING: Content-type '{content_type}' no es imagen, pero continuando...")
                # Comentar temporalmente esta validación para debug
                # raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        print("Leyendo archivo...")
        # Leer imagen
        image_bytes = await file.read()
        print(f"Bytes leídos: {len(image_bytes)}")
        
        # Verificar que tenemos datos
        if not image_bytes:
            raise HTTPException(status_code=400, detail="No se recibieron datos de imagen")
        
        # Log de primeros bytes para debug
        if len(image_bytes) >= 10:
            first_bytes = image_bytes[:10]
            print(f"Primeros 10 bytes: {[hex(b) for b in first_bytes]}")
        
        print("Creando numpy array...")
        nparr = np.frombuffer(image_bytes, np.uint8)
        print(f"Numpy array shape: {nparr.shape}")
        
        print("Decodificando con OpenCV...")
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            print("ERROR: cv2.imdecode retornó None")
            
            # Intentar detectar el formato
            if len(image_bytes) >= 4:
                header = image_bytes[:4]
                print(f"Header de imagen: {[hex(b) for b in header]}")
                
                if header.startswith(b'\xff\xd8\xff'):
                    print("Detectado: JPEG")
                elif header.startswith(b'\x89PNG'):
                    print("Detectado: PNG")
                elif header.startswith(b'GIF8'):
                    print("Detectado: GIF")
                else:
                    print("Formato no reconocido")
            
            # Intentar con PIL
            try:
                print("Intentando con PIL...")
                pil_image = Image.open(io.BytesIO(image_bytes))
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                print(f"PIL exitoso! Shape: {image.shape}")
            except Exception as pil_error:
                print(f"PIL falló: {pil_error}")
                raise ValueError("No se pudo decodificar la imagen")
        else:
            print(f"OpenCV exitoso! Shape: {image.shape}")
        
        print("Iniciando análisis de calidad...")
        # Análisis de calidad de imagen
        # Convertir a escala de grises para análisis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"Escala de grises: {gray.shape}")
        
        # Calcular métricas de calidad
        print("Calculando métricas...")
        # 1. Varianza del Laplaciano (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Laplacian variance: {laplacian_var}")
        
        # 2. Gradiente promedio (sharpness)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        avg_gradient = np.mean(gradient_magnitude)
        print(f"Average gradient: {avg_gradient}")
        
        # 3. Contraste (desviación estándar)
        contrast = np.std(gray)
        print(f"Contrast: {contrast}")
        
        # 4. Análisis de ruido (usando filtro gaussiano)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_level = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
        print(f"Noise level: {noise_level}")
        
        # Clasificación basada en thresholds empíricos
        quality_score = (laplacian_var * 0.4 + avg_gradient * 0.3 + 
                        contrast * 0.2 + (100 - noise_level) * 0.1)
        print(f"Quality score: {quality_score}")
        
        # Threshold para clasificación (ajustable)
        classification = "clean" if quality_score > 150 else "degraded"
        print(f"Clasificación final: {classification}")
        
        result = {
            "classification": classification,
            "confidence": min(quality_score / 300, 1.0),  # Normalizar a [0,1]
            "metrics": {
                "sharpness": float(laplacian_var),
                "gradient": float(avg_gradient),
                "contrast": float(contrast),
                "noise": float(noise_level),
                "quality_score": float(quality_score)
            },
            "filename": getattr(file, 'filename', 'unknown')
        }
        
        print("=== FIN classify_image_quality EXITOSO ===")
        return result
        
    except Exception as e:
        print(f"ERROR EN classify_image_quality: {e}")
        logger.error(f"=== ERROR EN classify_image_quality ===")
        logger.error(f"Tipo de error: {type(e).__name__}")
        logger.error(f"Mensaje: {str(e)}")
        logger.error(f"Traceback completo:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/image-quality-json")
async def classify_image_quality_json(request: Request):
    """Clasificar imagen recibida como JSON con base64"""
    logger.info("=== INICIO classify_image_quality_json ===")
    try:
        # Obtener el JSON body
        json_data = await request.json()
        logger.info(f"JSON recibido con keys: {list(json_data.keys())}")
        
        # Verificar que existe image_data
        if "image_data" not in json_data:
            raise HTTPException(status_code=400, detail="Campo 'image_data' requerido en JSON")
        
        image_data_b64 = json_data["image_data"]
        logger.info(f"image_data recibido: {len(image_data_b64)} caracteres")
        logger.info(f"Primeros 50 caracteres: {image_data_b64[:50]}")
        
        # Decodificar base64
        try:
            image_bytes = base64.b64decode(image_data_b64)
            logger.info(f"Base64 decodificado: {len(image_bytes)} bytes")
            if len(image_bytes) > 0:
                logger.info(f"Primeros 10 bytes decodificados: {[hex(b) for b in image_bytes[:10]]}")
        except Exception as e:
            logger.error(f"Error decodificando base64: {e}")
            raise HTTPException(status_code=400, detail=f"Error decodificando base64: {e}")
        
        # Decodificar imagen con OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("cv2.imdecode retornó None")
            raise HTTPException(status_code=400, detail="Los datos no corresponden a una imagen válida")
        
        logger.info(f"Imagen decodificada: {image.shape}")
        
        # Análisis de calidad (misma lógica que el otro endpoint)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness (varianza del Laplaciano)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        avg_gradient = np.mean(gradient_magnitude)
        
        # 3. Contraste
        contrast = np.std(gray)
        
        # 4. Análisis de ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_level = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
        
        # Clasificación
        quality_score = (laplacian_var * 0.4 + avg_gradient * 0.3 + 
                        contrast * 0.2 + (100 - noise_level) * 0.1)
        
        classification = "clean" if quality_score > 150 else "degraded"
        
        logger.info(f"Clasificación: {classification}, score: {quality_score}")
        
        return {
            "classification": classification,
            "confidence": min(quality_score / 300, 1.0),
            "metrics": {
                "sharpness": float(laplacian_var),
                "gradient": float(avg_gradient),
                "contrast": float(contrast),
                "noise": float(noise_level),
                "quality_score": float(quality_score)
            },
            "filename": "json_image"
        }
        
    except Exception as e:
        logger.error(f"Error en classify_image_quality_json: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/classify/image-quality-flexible")
async def classify_image_quality_flexible(
    file: Optional[UploadFile] = File(None),
    image_data: Optional[str] = None,
    bucket_name: Optional[str] = None,
    object_name: Optional[str] = None
):
    """Clasificar imagen con múltiples métodos de entrada"""
    logger.info("=== INICIO classify_image_quality_flexible ===")
    try:
        image_bytes = None
        filename = "unknown"
        
        # Log de parámetros recibidos
        logger.info(f"Parámetros: file={file is not None}, image_data={image_data is not None}, bucket_name={bucket_name}, object_name={object_name}")
        
        # Método 1: Archivo directo
        if file:
            logger.info(f"Procesando archivo: {file.filename}, content_type: {getattr(file, 'content_type', 'N/A')}")
            logger.info(f"Tamaño del archivo: {getattr(file, 'size', 'N/A')} bytes")
            
            # Leer los bytes del archivo directamente
            image_bytes = await file.read()
            filename = getattr(file, 'filename', 'unknown')
            logger.info(f"Archivo leído: {len(image_bytes)} bytes")
            
            # Log de los primeros bytes para debug
            if len(image_bytes) > 10:
                first_bytes = image_bytes[:10]
                logger.info(f"Primeros 10 bytes: {[hex(b) for b in first_bytes]}")
        
        # Método 2: Datos base64
        elif image_data:
            logger.info("Procesando imagen desde base64")
            try:
                image_bytes = base64.b64decode(image_data)
                filename = "base64_image"
                logger.info(f"Base64 decodificado: {len(image_bytes)} bytes")
            except Exception as e:
                logger.error(f"Error decodificando base64: {e}")
                raise HTTPException(status_code=400, detail=f"Error decodificando base64: {e}")
        
        # Método 3: Desde MinIO
        elif bucket_name and object_name:
            logger.info(f"Procesando desde MinIO: {bucket_name}/{object_name}")
            client = get_minio_client()
            response = client.get_object(Bucket=bucket_name, Key=object_name)
            image_bytes = response['Body'].read()
            filename = object_name
            logger.info(f"MinIO descargado: {len(image_bytes)} bytes")
        
        else:
            logger.error("No se proporcionó ningún método de entrada válido")
            raise HTTPException(status_code=400, detail="Debe proporcionar file, image_data o bucket_name+object_name")
        
        # Verificar que tenemos datos
        if not image_bytes:
            logger.error("image_bytes está vacío después del procesamiento")
            raise HTTPException(status_code=400, detail="No se recibieron datos de imagen")
        
        logger.info("Iniciando decodificación con OpenCV...")
        
        # Decodificar imagen
        nparr = np.frombuffer(image_bytes, np.uint8)
        logger.info(f"Numpy array creado: shape={nparr.shape}, dtype={nparr.dtype}")
        
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("cv2.imdecode retornó None - archivo no es una imagen válida")
            
            # Debug adicional: verificar si son bytes de imagen válidos
            logger.info("Intentando detectar formato de imagen...")
            
            # Verificar headers comunes de imágenes
            if len(image_bytes) >= 4:
                header = image_bytes[:4]
                if header.startswith(b'\xff\xd8\xff'):
                    logger.info("Detectado header JPEG")
                elif header.startswith(b'\x89PNG'):
                    logger.info("Detectado header PNG")
                elif header.startswith(b'GIF8'):
                    logger.info("Detectado header GIF")
                else:
                    logger.warning(f"Header no reconocido: {[hex(b) for b in header]}")
            
            # Intentar con PIL como alternativa
            try:
                logger.info("Intentando decodificar con PIL...")
                pil_image = Image.open(io.BytesIO(image_bytes))
                # Convertir PIL a OpenCV
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                logger.info(f"Imagen decodificada con PIL exitosamente: shape={image.shape}")
            except Exception as pil_error:
                logger.error(f"PIL también falló: {pil_error}")
                raise HTTPException(status_code=400, detail="El archivo no es una imagen válida o está corrupto")
        
        if image is not None:
            logger.info(f"Imagen decodificada exitosamente: shape={image.shape}")
        
        # Análisis de calidad de imagen
        logger.info("Iniciando análisis de calidad...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.info("Conversión a escala de grises completada")
        
        # Calcular métricas de calidad
        logger.info("Calculando métricas de calidad...")
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        logger.info(f"Laplacian variance: {laplacian_var}")
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        avg_gradient = np.mean(gradient_magnitude)
        logger.info(f"Average gradient: {avg_gradient}")
        
        contrast = np.std(gray)
        logger.info(f"Contrast: {contrast}")
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_level = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
        logger.info(f"Noise level: {noise_level}")
        
        quality_score = (laplacian_var * 0.4 + avg_gradient * 0.3 + 
                        contrast * 0.2 + (100 - noise_level) * 0.1)
        
        classification = "clean" if quality_score > 150 else "degraded"
        
        logger.info(f"Clasificación completada: {classification}, score: {quality_score}")
        
        result = {
            "classification": classification,
            "confidence": min(quality_score / 300, 1.0),
            "metrics": {
                "sharpness": float(laplacian_var),
                "gradient": float(avg_gradient),
                "contrast": float(contrast),
                "noise": float(noise_level),
                "quality_score": float(quality_score)
            },
            "filename": filename,
            "input_method": "file" if file else "base64" if image_data else "minio"
        }
        
        logger.info("=== FIN classify_image_quality_flexible EXITOSO ===")
        return result
        
    except HTTPException as e:
        # Re-lanzar HTTPExceptions tal como están
        logger.error(f"HTTPException en classify_image_quality_flexible: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"=== ERROR EN classify_image_quality_flexible ===")
        logger.error(f"Tipo de error: {type(e).__name__}")
        logger.error(f"Mensaje: {str(e)}")
        logger.error(f"Traceback completo:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/synthetic-data")
async def generate_synthetic_data(
    background_tasks: BackgroundTasks,
    source_bucket: str,
    source_file: str,
    target_count: int = 10,
    generation_type: str = "degradation",  # degradation, variation
    output_bucket: str = "document-training"
):
    """Generar datos sintéticos a partir de una imagen base"""
    job_id = str(uuid.uuid4())
    
    job = ProcessingJob(
        job_id=job_id,
        status="pending",
        created_at=datetime.now(),
        input_file=f"{source_bucket}/{source_file}",
        output_file=f"{output_bucket}/"
    )
    
    jobs_state[job_id] = job
    
    background_tasks.add_task(
        generate_synthetic_background,
        job_id,
        source_bucket,
        source_file,
        target_count,
        generation_type,
        output_bucket
    )
    
    return {"job_id": job_id, "status": "generation_started"}

async def generate_synthetic_background(
    job_id: str,
    source_bucket: str,
    source_file: str,
    target_count: int,
    generation_type: str,
    output_bucket: str
):
    """Generar datos sintéticos en background"""
    try:
        jobs_state[job_id].status = "processing"
        
        client = get_minio_client()
        
        # Descargar imagen base
        response = client.get_object(Bucket=source_bucket, Key=source_file)
        image_bytes = response['Body'].read()
        
        # Decodificar imagen
        nparr = np.frombuffer(image_bytes, np.uint8)
        base_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if base_image is None:
            raise ValueError("No se pudo decodificar la imagen base")
        
        generated_count = 0
        base_name = source_file.split('.')[0]
        
        for i in range(target_count):
            try:
                if generation_type == "degradation":
                    # Generar versión degradada desde imagen limpia
                    synthetic_image = apply_degradation_effects(base_image.copy())
                else:
                    # Generar variación desde imagen degradada
                    synthetic_image = apply_variation_effects(base_image.copy())
                
                # Codificar imagen
                _, encoded = cv2.imencode('.png', synthetic_image)
                synthetic_bytes = encoded.tobytes()
                
                # Nombre del archivo sintético
                synthetic_name = f"{base_name}_synthetic_{generation_type}_{i+1:03d}.png"
                
                # Subir a MinIO
                client.put_object(
                    Bucket=output_bucket,
                    Key=synthetic_name,
                    Body=synthetic_bytes,
                    ContentType='image/png'
                )
                
                generated_count += 1
                
            except Exception as e:
                logger.error(f"Error generando imagen sintética {i+1}: {e}")
        
        # Completar trabajo
        jobs_state[job_id].status = "completed"
        jobs_state[job_id].completed_at = datetime.now()
        jobs_state[job_id].output_file = f"{output_bucket}/ ({generated_count} files)"
        
        # Agregar metadata adicional
        jobs_state[job_id].generated_count = generated_count
        jobs_state[job_id].source_file = source_file
        jobs_state[job_id].generation_type = generation_type
        
    except Exception as e:
        jobs_state[job_id].status = "failed"
        jobs_state[job_id].error = str(e)
        logger.error(f"Error en generación sintética: {e}")

def apply_degradation_effects(image):
    """Aplicar efectos de degradación a imagen limpia"""
    import random
    
    # Efectos aleatorios de degradación
    effects = []
    
    # 1. Ruido gaussiano
    if random.random() < 0.8:
        noise = np.random.normal(0, random.uniform(5, 15), image.shape)
        image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # 2. Blur
    if random.random() < 0.6:
        blur_size = random.choice([3, 5, 7])
        image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    
    # 3. Compresión JPEG
    if random.random() < 0.7:
        quality = random.randint(30, 70)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    # 4. Ajuste de brillo/contraste
    if random.random() < 0.5:
        alpha = random.uniform(0.7, 1.3)  # Contraste
        beta = random.randint(-30, 30)    # Brillo
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return image

def apply_variation_effects(image):
    """Aplicar variaciones a imagen degradada"""
    import random
    
    # Variaciones sutiles para imágenes ya degradadas
    
    # 1. Rotación pequeña
    if random.random() < 0.4:
        angle = random.uniform(-2, 2)
        h, w = image.shape[:2]
        center = (w//2, h//2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    
    # 2. Cambio de gamma
    if random.random() < 0.6:
        gamma = random.uniform(0.8, 1.2)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, table)
    
    # 3. Ruido salt-and-pepper
    if random.random() < 0.3:
        noise_mask = np.random.random(image.shape[:2]) < 0.02
        image[noise_mask] = np.random.choice([0, 255], size=np.sum(noise_mask))
    
    return image

@app.get("/dataset/stats")
async def get_dataset_stats(include_new: bool = False):
    """Obtener estadísticas del dataset"""
    try:
        client = get_minio_client()
        
        stats = {}
        buckets = ["document-clean", "document-degraded", "document-training"]
        
        for bucket in buckets:
            try:
                response = client.list_objects_v2(Bucket=bucket)
                objects = response.get('Contents', [])
                
                # Filtrar solo imágenes
                image_objects = [obj for obj in objects 
                               if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                stats[bucket] = {
                    "count": len(image_objects),
                    "total_size_mb": sum(obj['Size'] for obj in image_objects) / (1024*1024),
                    "files": [obj['Key'] for obj in image_objects] if include_new else []
                }
                
            except Exception as e:
                stats[bucket] = {"count": 0, "error": str(e)}
        
        # Calcular totales
        total_samples = sum(bucket.get("count", 0) for bucket in stats.values())
        
        return {
            "buckets": stats,
            "total_samples": total_samples,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/notify/dataset-updated")
async def notify_dataset_updated(
    event: str,
    source_file: str,
    classification: str,
    new_samples: int,
    total_samples: int
):
    """Notificar actualización del dataset"""
    try:
        # Log del evento
        logger.info(f"Dataset updated: {event}, source: {source_file}, "
                   f"classification: {classification}, new: {new_samples}, total: {total_samples}")
        
        # Aquí podrías agregar notificaciones adicionales:
        # - Webhook a sistemas externos
        # - Actualización de métricas
        # - Trigger de reentrenamiento automático
        
        return {
            "status": "notified",
            "event": event,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training/start")
async def start_training(
    background_tasks: BackgroundTasks,
    request: TrainingRequest
):
    """Iniciar entrenamiento con datos de MinIO"""
    job_id = str(uuid.uuid4())
    
    job = ProcessingJob(
        job_id=job_id,
        status="pending",
        created_at=datetime.now(),
        input_file=f"{request.clean_bucket} + {request.degraded_bucket}",
        output_file="outputs/checkpoints/"
    )
    
    jobs_state[job_id] = job
    
    background_tasks.add_task(
        start_training_background,
        job_id,
        request
    )
    
    return {"job_id": job_id, "status": "training_started"}

@app.post("/test/simple-upload")
async def test_simple_upload(file: UploadFile = File(...)):
    """Test endpoint más simple para verificar que FastAPI puede procesar archivos"""
    logger.info("=== TEST SIMPLE UPLOAD ===")
    try:
        # Log básico
        logger.info(f"Archivo recibido: {file.filename}")
        logger.info(f"Content-type: {file.content_type}")
        
        # Leer bytes
        content = await file.read()
        logger.info(f"Bytes leídos: {len(content)}")
        
        # Verificar si son bytes de imagen válidos
        if len(content) >= 4:
            header = content[:4]
            logger.info(f"Header: {[hex(b) for b in header]}")
            
            # Intentar decodificar directamente
            nparr = np.frombuffer(content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                logger.info(f"✅ Imagen válida! Shape: {image.shape}")
                return {
                    "status": "success",
                    "filename": file.filename,
                    "size": len(content),
                    "image_shape": list(image.shape),
                    "message": "Imagen procesada correctamente"
                }
            else:
                logger.error("❌ cv2.imdecode falló")
                return {
                    "status": "error",
                    "filename": file.filename,
                    "size": len(content),
                    "header": [hex(b) for b in header],
                    "message": "No se pudo decodificar como imagen"
                }
        else:
            logger.error("❌ Archivo demasiado pequeño")
            return {
                "status": "error",
                "size": len(content),
                "message": "Archivo demasiado pequeño"
            }
            
    except Exception as e:
        logger.error(f"Error en test: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/upload/to-bucket")
async def upload_to_bucket(
    file: UploadFile = File(...),
    bucket_name: str = None,
    filename: Optional[str] = None
):
    """Subir archivo a bucket específico de MinIO"""
    logger.info(f"=== UPLOAD TO BUCKET: {bucket_name} ===")
    try:
        # Usar filename original o el proporcionado
        object_name = filename or file.filename or f"uploaded_{uuid.uuid4().hex}.png"
        
        logger.info(f"Subiendo archivo: {object_name} a bucket: {bucket_name}")
        
        # Leer contenido del archivo
        file_content = await file.read()
        logger.info(f"Archivo leído: {len(file_content)} bytes")
        
        # Obtener cliente MinIO
        client = get_minio_client()
        
        # Detectar content type
        content_type = file.content_type
        if not content_type or content_type == 'application/octet-stream':
            if object_name.lower().endswith('.png'):
                content_type = 'image/png'
            elif object_name.lower().endswith(('.jpg', '.jpeg')):
                content_type = 'image/jpeg'
            else:
                content_type = 'image/png'  # default
        
        # Subir a MinIO
        client.put_object(
            Bucket=bucket_name,
            Key=object_name,
            Body=file_content,
            ContentType=content_type
        )
        
        logger.info(f"✅ Archivo subido exitosamente: {bucket_name}/{object_name}")
        
        return {
            "status": "success",
            "bucket": bucket_name,
            "filename": object_name,
            "size": len(file_content),
            "content_type": content_type,
            "url": f"http://{MINIO_CONFIG['endpoint']}/{bucket_name}/{object_name}"
        }
        
    except Exception as e:
        logger.error(f"Error subiendo archivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/classified")
async def upload_classified_image(
    file: UploadFile = File(...),
    classification: str = "",  # "clean" o "degraded"
    filename: Optional[str] = None
):
    """Subir imagen al bucket correspondiente según clasificación"""
    logger.info(f"=== UPLOAD CLASSIFIED: {classification} ===")
    
    # Mapear clasificación a bucket
    bucket_mapping = {
        "clean": "document-clean",
        "degraded": "document-degraded"
    }
    
    if classification not in bucket_mapping:
        raise HTTPException(status_code=400, detail="Classification debe ser 'clean' o 'degraded'")
    
    bucket_name = bucket_mapping[classification]
    
    # Usar el endpoint genérico
    return await upload_to_bucket(file, bucket_name, filename)










async def start_training_background(job_id: str, request: TrainingRequest):
    """Entrenar modelo en background"""
    try:
        jobs_state[job_id].status = "processing"
        
        # TODO: Implementar descarga de datos de MinIO y entrenamiento
        # Esto requiere integrar con el script de entrenamiento
        
        await asyncio.sleep(10)  # Simulación
        
        jobs_state[job_id].status = "completed"
        jobs_state[job_id].completed_at = datetime.now()
        
    except Exception as e:
        jobs_state[job_id].status = "failed"
        jobs_state[job_id].error = str(e)

if __name__ == "__main__":
    # Configuración desde variables de entorno
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    
    uvicorn.run(app, host=host, port=port)
