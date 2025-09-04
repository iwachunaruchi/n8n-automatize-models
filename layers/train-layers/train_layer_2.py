"""
Entrenamiento para Capa 2: NAFNet + DocUNet
NAFNet para denoising/deblurring + DocUNet para dewarping

Este módulo contiene clases y funciones para ser importadas por la API.
No ejecuta entrenamiento directamente - debe ser llamado desde la API.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

# Importar servicios directamente (no HTTP)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from api.services.minio_service import minio_service
    from api.services.image_analysis_service import image_analysis_service
    SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"Servicios no disponibles: {e}")
    SERVICES_AVAILABLE = False

# ============================================================================
# CONFIGURACIÓN DE MODELOS PREENTRENADOS
# ============================================================================

PRETRAINED_MODELS = {
    'nafnet_sidd_width64': {
        'minio_path': 'pretrained_models/layer_2/nafnet/NAFNet-SIDD-width64.pth',
        'bucket': 'models',
        'description': 'NAFNet preentrenado en SIDD dataset',
        'width': 64,
        'compatible_arch': 'NAFNet'
    }
}

# Configuración de buckets
MINIO_BUCKETS = {
    'degraded': 'document-degraded',
    'clean': 'document-clean',
    'restored': 'document-restored',
    'training': 'document-training'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MODELOS SIMPLIFICADOS PARA ENTRENAMIENTO
# ============================================================================

class SimpleChannelAttention(nn.Module):
    """Atención de canal simplificada"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class NAFBlock(nn.Module):
    """Bloque NAF simplificado"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 2, 1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.ca = SimpleChannelAttention(channels)
        self.norm = nn.LayerNorm([channels, 1, 1])
        
    def forward(self, x):
        residual = x
        
        # Feature processing
        x = self.conv1(x)
        x = self.conv2(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2  # Simple gate
        x = self.conv3(x)
        
        # Attention
        x = self.ca(x)
        
        return residual + x

class SimpleNAFNet(nn.Module):
    """NAFNet simplificado para entrenamiento rápido"""
    def __init__(self, in_channels=3, width=32, num_blocks=4):
        super().__init__()
        
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList([
            NAFBlock(width),
            NAFBlock(width * 2),
        ])
        self.downs = nn.ModuleList([
            nn.Conv2d(width, width * 2, 2, stride=2),
        ])
        
        # Middle
        self.middle = nn.Sequential(*[NAFBlock(width * 2) for _ in range(num_blocks)])
        
        # Decoder
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(width * 2, width, 2, stride=2),
        ])
        self.decoder = nn.ModuleList([
            NAFBlock(width),
        ])
        
        self.ending = nn.Conv2d(width, in_channels, 3, padding=1)
        
    def forward(self, x):
        # Intro
        x = self.intro(x)
        
        # Encoder
        skips = []
        for i, (enc, down) in enumerate(zip(self.encoder, self.downs)):
            x = enc(x)
            skips.append(x)
            x = down(x)
        
        # Middle
        x = self.middle(x)
        
        # Decoder
        for i, (up, dec) in enumerate(zip(self.ups, self.decoder)):
            x = up(x)
            x = x + skips[-(i+1)]
            x = dec(x)
        
        # Output
        x = self.ending(x)
        return x

# ============================================================================
# FUNCIONES PARA MANEJO DE MODELOS PREENTRENADOS
# ============================================================================

def download_pretrained_model(model_key: str, cache_dir: str = "./temp_models") -> Optional[str]:
    """
    Descargar modelo preentrenado desde MinIO y guardarlo localmente
    
    Args:
        model_key: Clave del modelo en PRETRAINED_MODELS
        cache_dir: Directorio para guardar el modelo temporalmente
        
    Returns:
        Ruta local del modelo descargado o None si hay error
    """
    if not SERVICES_AVAILABLE:
        logger.warning("Servicios MinIO no disponibles")
        return None
        
    if model_key not in PRETRAINED_MODELS:
        logger.error(f"Modelo preentrenado no encontrado: {model_key}")
        return None
    
    model_info = PRETRAINED_MODELS[model_key]
    
    try:
        # Crear directorio de cache
        os.makedirs(cache_dir, exist_ok=True)
        
        # Ruta local
        local_path = os.path.join(cache_dir, f"{model_key}.pth")
        
        # Verificar si ya existe localmente
        if os.path.exists(local_path):
            logger.info(f"Modelo ya existe localmente: {local_path}")
            return local_path
        
        # Descargar desde MinIO
        logger.info(f"Descargando modelo preentrenado: {model_key}")
        model_data = minio_service.download_file(
            bucket=model_info['bucket'],
            filename=model_info['minio_path']
        )
        
        # Guardar localmente
        with open(local_path, 'wb') as f:
            f.write(model_data)
        
        logger.info(f"Modelo descargado exitosamente: {local_path}")
        return local_path
        
    except Exception as e:
        logger.error(f"Error descargando modelo preentrenado {model_key}: {e}")
        return None

def load_pretrained_nafnet(model_path: str, target_model: SimpleNAFNet, strict: bool = False) -> bool:
    """
    Cargar pesos preentrenados en un modelo NAFNet
    
    Args:
        model_path: Ruta al archivo de modelo preentrenado
        target_model: Modelo NAFNet donde cargar los pesos
        strict: Si True, requiere coincidencia exacta de capas
        
    Returns:
        True si se cargó exitosamente, False si hay error
    """
    try:
        logger.info(f"Cargando modelo preentrenado desde: {model_path}")
        
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extraer state_dict (diferentes formatos posibles)
        if isinstance(checkpoint, dict):
            if 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Obtener estado actual del modelo
        model_dict = target_model.state_dict()
        compatible_dict = {}
        
        # Mapeo de nombres de capas (de NAFNet original a SimpleNAFNet)
        layer_mapping = {
            'encoders.': 'encoder.',
            'decoders.': 'decoder.',
            'ups.': 'ups.',
            'downs.': 'downs.',
            'middle.': 'middle.'
        }
        
        for name, param in state_dict.items():
            mapped_name = name
            
            # Aplicar mapeo de nombres
            for original, target in layer_mapping.items():
                if original in mapped_name:
                    mapped_name = mapped_name.replace(original, target)
            
            # Verificar compatibilidad
            if mapped_name in model_dict:
                if param.shape == model_dict[mapped_name].shape:
                    compatible_dict[mapped_name] = param
                    logger.debug(f"Capa compatible: {name} -> {mapped_name} {param.shape}")
                else:
                    logger.debug(f"Forma incompatible: {name} -> {mapped_name} {param.shape} vs {model_dict[mapped_name].shape}")
            else:
                # Intentar cargar capas básicas (intro, ending) directamente
                if name in model_dict and param.shape == model_dict[name].shape:
                    compatible_dict[name] = param
                    logger.debug(f"Capa directa compatible: {name} {param.shape}")
                else:
                    logger.debug(f"Capa no encontrada: {name}")
        
        if not compatible_dict:
            logger.warning("No se encontraron capas compatibles")
            return False
        
        # Cargar capas compatibles
        model_dict.update(compatible_dict)
        target_model.load_state_dict(model_dict, strict=strict)
        
        logger.info(f"Modelo preentrenado cargado: {len(compatible_dict)}/{len(state_dict)} capas transferidas")
        logger.info(f"Capas del modelo objetivo: {len(model_dict)}")
        
        # Listar capas transferidas exitosamente
        transferred_layers = list(compatible_dict.keys())
        if transferred_layers:
            logger.info(f"Capas transferidas: {transferred_layers[:5]}..." if len(transferred_layers) > 5 else f"Capas transferidas: {transferred_layers}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error cargando modelo preentrenado: {e}")
        return False

def setup_pretrained_nafnet(width: int = 64, use_pretrained: bool = True) -> SimpleNAFNet:
    """
    Crear y configurar un modelo NAFNet con pesos preentrenados
    
    Args:
        width: Ancho del modelo (debe coincidir con el preentrenado)
        use_pretrained: Si cargar pesos preentrenados
        
    Returns:
        Modelo NAFNet configurado
    """
    # Crear modelo
    model = SimpleNAFNet(width=width)
    
    if use_pretrained and SERVICES_AVAILABLE:
        # Intentar cargar modelo preentrenado
        pretrained_path = download_pretrained_model('nafnet_sidd_width64')
        
        if pretrained_path:
            success = load_pretrained_nafnet(pretrained_path, model, strict=False)
            if success:
                logger.info("Modelo NAFNet inicializado con pesos preentrenados")
            else:
                logger.warning("Falló la carga de pesos preentrenados, usando inicialización aleatoria")
        else:
            logger.warning("No se pudo descargar modelo preentrenado, usando inicialización aleatoria")
    else:
        logger.info("Usando inicialización aleatoria para NAFNet")
    
    return model

def setup_finetuning_params(model: nn.Module, freeze_backbone: bool = False, learning_rate_factor: float = 0.1) -> Dict:
    """
    Configurar parámetros para fine-tuning
    
    Args:
        model: Modelo a configurar
        freeze_backbone: Si congelar las capas iniciales
        learning_rate_factor: Factor para reducir learning rate en capas preentrenadas
        
    Returns:
        Diccionario con grupos de parámetros para optimizador
    """
    if freeze_backbone:
        # Congelar primeras capas
        for name, param in model.named_parameters():
            if 'intro' in name or 'encoder.0' in name:
                param.requires_grad = False
                logger.info(f"Capa congelada: {name}")
    
    # Crear grupos de parámetros con diferentes learning rates
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'intro' in name or 'encoder' in name or 'middle' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr_factor': learning_rate_factor},
        {'params': head_params, 'lr_factor': 1.0}
    ]
    
    return param_groups

class SimpleDocUNet(nn.Module):
    """DocUNet simplificado para dewarping"""
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        
        # Decoder
        self.dec2 = self._upconv_block(128, 64)
        self.dec1 = self._upconv_block(64, 32)
        
        # Output - campo de deformación
        self.final = nn.Conv2d(32, 2, 1)
        
        self.pool = nn.MaxPool2d(2)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Decoder
        d2 = self.dec2(e3) + e2
        d1 = self.dec1(d2) + e1
        
        # Output
        flow = self.final(d1)
        return flow

# ============================================================================
# DATASET PARA CARGAR IMÁGENES DESDE API
# ============================================================================

class DocumentDataset(Dataset):
    """Dataset que carga imágenes usando servicios directos (sin HTTP)"""
    
    def __init__(self, max_pairs: int = 100, patch_size: int = 128, 
                 use_training_bucket: bool = True):
        self.patch_size = patch_size
        self.pairs = []
        self.use_training_bucket = use_training_bucket
        
        if not SERVICES_AVAILABLE:
            raise ImportError("Servicios MinIO no disponibles")
        
        # Cargar pares de imágenes
        if use_training_bucket:
            self._load_training_bucket_pairs(max_pairs)
        else:
            self._load_image_pairs(max_pairs)
        
        # Transformaciones
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _load_training_bucket_pairs(self, max_pairs: int):
        """Cargar pares desde bucket de entrenamiento usando servicios directos"""
        print(f"📥 Cargando pares desde bucket 'document-training'...")
        
        try:
            # Obtener archivos del bucket usando servicio directo
            files = minio_service.list_files(MINIO_BUCKETS['training'])
            
            # Separar archivos clean y degraded
            clean_files = [f for f in files if f.startswith('clean_')]
            degraded_files = [f for f in files if f.startswith('degraded_')]
            
            print(f"📊 Archivos limpios encontrados: {len(clean_files)}")
            print(f"📊 Archivos degradados encontrados: {len(degraded_files)}")
            
            # Crear pares basados en el UUID del par
            pairs_found = 0
            for clean_file in clean_files:
                if pairs_found >= max_pairs:
                    break
                
                # Extraer UUID del nombre: clean_{uuid}.png -> {uuid}
                if '_' in clean_file and '.' in clean_file:
                    uuid_part = clean_file.split('_', 1)[1].rsplit('.', 1)[0]
                    
                    # Buscar archivo degradado correspondiente
                    degraded_match = f"degraded_{uuid_part}.png"
                    
                    if degraded_match in degraded_files:
                        self.pairs.append((degraded_match, clean_file))  # (input, target)
                        pairs_found += 1
                        
                        if pairs_found <= 5:  # Mostrar solo los primeros 5
                            print(f"✅ Par encontrado: {degraded_match} -> {clean_file}")
            
            print(f"📊 Pares válidos encontrados: {len(self.pairs)}")
            
        except Exception as e:
            print(f"❌ Error cargando pares del bucket de entrenamiento: {e}")
            # Fallback al método original
            self._load_image_pairs(max_pairs)
    
    def _load_image_pairs(self, max_pairs: int):
        """Cargar pares de imágenes degradadas y limpias usando servicios directos"""
        print(f"📥 Cargando pares de imágenes desde servicios...")
        
        try:
            # Obtener listas de archivos usando servicios directos
            degraded_files = minio_service.list_files(MINIO_BUCKETS['degraded'])
            clean_files = minio_service.list_files(MINIO_BUCKETS['clean'])
            
            # Crear pares válidos
            pairs_found = 0
            for degraded_file in degraded_files:
                if pairs_found >= max_pairs:
                    break
                    
                if not degraded_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                # Buscar archivo limpio correspondiente
                base_name = degraded_file.replace('_deg', '').replace('_degraded', '')
                base_name = base_name.replace('_synthetic', '').replace('_heavy', '')
                base_name = base_name.replace('_medium', '').replace('_light', '')
                
                for clean_file in clean_files:
                    if (base_name in clean_file or 
                        any(part in clean_file for part in base_name.split('_')[:2])):
                        self.pairs.append((degraded_file, clean_file))
                        pairs_found += 1
                        break
            
            print(f"📊 Pares de imágenes encontrados: {len(self.pairs)}")
            
        except Exception as e:
            print(f"❌ Error cargando pares: {e}")
    
    def _download_image(self, bucket: str, filename: str) -> Optional[np.ndarray]:
        """Descargar imagen usando servicios directos"""
        try:
            # Usar servicio MinIO directamente
            image_data = minio_service.download_file(bucket, filename)
            if image_data:
                img_array = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
            return None
        except Exception as e:
            print(f"Error descargando {bucket}/{filename}: {e}")
            return None
    
    def _extract_patch(self, degraded: np.ndarray, clean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extraer patch aleatorio"""
        h, w = min(degraded.shape[0], clean.shape[0]), min(degraded.shape[1], clean.shape[1])
        
        if h < self.patch_size or w < self.patch_size:
            # Redimensionar si es muy pequeña
            scale = max(self.patch_size / h, self.patch_size / w) * 1.1
            new_h, new_w = int(h * scale), int(w * scale)
            degraded = cv2.resize(degraded, (new_w, new_h))
            clean = cv2.resize(clean, (new_w, new_h))
            h, w = new_h, new_w
        
        # Extraer patch aleatorio
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        
        degraded_patch = degraded[top:top+self.patch_size, left:left+self.patch_size]
        clean_patch = clean[top:top+self.patch_size, left:left+self.patch_size]
        
        return degraded_patch, clean_patch
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        degraded_file, clean_file = self.pairs[idx]
        
        # Determinar bucket según el método usado y descargar imágenes
        if self.use_training_bucket:
            # Ambos archivos están en el bucket de entrenamiento
            degraded = self._download_image(MINIO_BUCKETS['training'], degraded_file)
            clean = self._download_image(MINIO_BUCKETS['training'], clean_file)
        else:
            # Método original: buckets separados
            degraded = self._download_image(MINIO_BUCKETS['degraded'], degraded_file)
            clean = self._download_image(MINIO_BUCKETS['clean'], clean_file)
        
        if degraded is None or clean is None:
            # Fallback más robusto: intentar con otros índices sin recursión infinita
            print(f"❌ Error descargando par {idx}, intentando con otros...")
            for fallback_idx in range(len(self.pairs)):
                if fallback_idx != idx:  # Evitar el mismo índice
                    try:
                        fallback_degraded_file, fallback_clean_file = self.pairs[fallback_idx]
                        
                        if self.use_training_bucket:
                            fallback_degraded = self._download_image(MINIO_BUCKETS['training'], fallback_degraded_file)
                            fallback_clean = self._download_image(MINIO_BUCKETS['training'], fallback_clean_file)
                        else:
                            fallback_degraded = self._download_image(MINIO_BUCKETS['degraded'], fallback_degraded_file)
                            fallback_clean = self._download_image(MINIO_BUCKETS['clean'], fallback_clean_file)
                        
                        if fallback_degraded is not None and fallback_clean is not None:
                            degraded, clean = fallback_degraded, fallback_clean
                            print(f"✅ Usando par fallback {fallback_idx}")
                            break
                    except:
                        continue
            
            # Si aún no tenemos datos válidos, crear tensores dummy
            if degraded is None or clean is None:
                print(f"⚠️ Creando tensores dummy para índice {idx}")
                dummy_shape = (self.patch_size, self.patch_size, 3)
                degraded = np.random.rand(*dummy_shape).astype(np.uint8)
                clean = np.random.rand(*dummy_shape).astype(np.uint8)
        
        # Extraer patches
        degraded_patch, clean_patch = self._extract_patch(degraded, clean)
        
        # Aplicar transformaciones
        if self.transform:
            seed = random.randint(0, 2**32)
            
            random.seed(seed)
            np.random.seed(seed)
            degraded_transformed = self.transform(image=degraded_patch)["image"]
            
            random.seed(seed)
            np.random.seed(seed) 
            clean_transformed = self.transform(image=clean_patch)["image"]
        else:
            # Convertir a tensor manualmente si no hay transformaciones
            degraded_transformed = torch.from_numpy(degraded_patch.transpose(2, 0, 1)).float() / 255.0
            clean_transformed = torch.from_numpy(clean_patch.transpose(2, 0, 1)).float() / 255.0
        
        # RETORNAR TUPLA, NO DICCIONARIO (para compatibilidad con DataLoader)
        return degraded_transformed, clean_transformed

# ============================================================================
# ENTRENADOR PARA CAPA 2
# ============================================================================

class Layer2Trainer:
    """Entrenador para modelos de Capa 2 usando servicios directos"""
    
    def __init__(self):
        if not SERVICES_AVAILABLE:
            raise ImportError("Servicios MinIO no disponibles para entrenamiento")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Modelos - NAFNet con posible carga de modelo preentrenado
        logger.info("Inicializando modelo NAFNet...")
        self.nafnet = setup_pretrained_nafnet(width=64, use_pretrained=True).to(self.device)
        self.docunet = SimpleDocUNet(in_channels=3).to(self.device)
        
        # Optimizadores
        self.nafnet_optimizer = optim.Adam(self.nafnet.parameters(), lr=1e-4)
        self.docunet_optimizer = optim.Adam(self.docunet.parameters(), lr=1e-4)
        
        # Criterios de pérdida
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Historial de entrenamiento
        self.training_history = {
            'nafnet_losses': [],
            'docunet_losses': [],
            'combined_losses': []
        }
    
    def apply_dewarping(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Aplicar dewarping con campo de flujo"""
        B, C, H, W = image.shape
        
        # Crear grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Aplicar flujo
        deformed_grid = grid + flow * 0.1  # Factor de escala para estabilidad
        deformed_grid = deformed_grid.permute(0, 2, 3, 1)
        
        # Grid sampling
        dewarped = torch.nn.functional.grid_sample(
            image, deformed_grid, mode='bilinear', 
            padding_mode='border', align_corners=True
        )
        
        return dewarped
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Entrenar una época"""
        self.nafnet.train()
        self.docunet.train()
        
        total_nafnet_loss = 0
        total_docunet_loss = 0
        total_combined_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Época {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Manejar tanto tuplas como listas del DataLoader
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                degraded, clean = batch[0], batch[1]
            else:
                print(f"❌ DEBUG: Batch inesperado - tipo: {type(batch)}")
                continue
                
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            
            # ===== Entrenamiento NAFNet =====
            self.nafnet_optimizer.zero_grad()
            
            # Forward NAFNet
            denoised = self.nafnet(degraded)
            nafnet_loss = self.l1_loss(denoised, clean) + 0.1 * self.mse_loss(denoised, clean)
            
            # Backward NAFNet
            nafnet_loss.backward()
            self.nafnet_optimizer.step()
            
            # ===== Entrenamiento DocUNet =====
            self.docunet_optimizer.zero_grad()
            
            with torch.no_grad():
                denoised_detached = self.nafnet(degraded).detach()
            
            # Forward DocUNet
            flow = self.docunet(denoised_detached)
            dewarped = self.apply_dewarping(denoised_detached, flow)
            
            # Pérdida de DocUNet
            docunet_loss = self.l1_loss(dewarped, clean) + 0.05 * torch.mean(torch.abs(flow))
            
            # Backward DocUNet
            docunet_loss.backward()
            self.docunet_optimizer.step()
            
            # ===== Pérdida combinada (para registro) =====
            with torch.no_grad():
                combined_output = self.apply_dewarping(self.nafnet(degraded), self.docunet(self.nafnet(degraded)))
                combined_loss = self.l1_loss(combined_output, clean)
            
            # Acumular pérdidas
            total_nafnet_loss += nafnet_loss.item()
            total_docunet_loss += docunet_loss.item()
            total_combined_loss += combined_loss.item()
            
            # Actualizar barra de progreso
            pbar.set_postfix({
                'NAF': f'{nafnet_loss.item():.4f}',
                'Doc': f'{docunet_loss.item():.4f}',
                'Comb': f'{combined_loss.item():.4f}'
            })
        
        return {
            'nafnet_loss': total_nafnet_loss / len(dataloader),
            'docunet_loss': total_docunet_loss / len(dataloader),
            'combined_loss': total_combined_loss / len(dataloader)
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validación"""
        self.nafnet.eval()
        self.docunet.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Manejar tanto tuplas como listas del DataLoader
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    degraded, clean = batch[0], batch[1]
                else:
                    print(f"❌ DEBUG VAL: Batch inesperado - tipo: {type(batch)}")
                    continue
                    
                degraded = degraded.to(self.device)
                clean = clean.to(self.device)
                
                # Pipeline completo
                denoised = self.nafnet(degraded)
                flow = self.docunet(denoised)
                dewarped = self.apply_dewarping(denoised, flow)
                
                loss = self.l1_loss(dewarped, clean)
                total_loss += loss.item()
        
        return {'val_loss': total_loss / len(dataloader)}
    
    def train(self, num_epochs: int = 10, max_pairs: int = 100, batch_size: int = 4, 
              use_training_bucket: bool = True, use_finetuning: bool = True, 
              freeze_backbone: bool = False, finetuning_lr_factor: float = 0.1):
        """Entrenamiento completo"""
        print("🔧 ENTRENAMIENTO CAPA 2: NAFNet + DocUNet")
        print("=" * 60)
        print(f"🔧 Dispositivo: {self.device}")
        print(f"📊 Épocas: {num_epochs}")
        print(f"📦 Batch size: {batch_size}")
        print(f"📁 Usando bucket de entrenamiento: {use_training_bucket}")
        
        # Crear dataset usando servicios directos
        dataset = DocumentDataset(max_pairs=max_pairs, 
                                 patch_size=128, use_training_bucket=use_training_bucket)
        
        if len(dataset) == 0:
            print("❌ No se encontraron pares de imágenes para entrenar")
            return
        
        # Dividir en train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"📊 Imágenes de entrenamiento: {len(train_dataset)}")
        print(f"📊 Imágenes de validación: {len(val_dataset)}")
        
        # Configurar fine-tuning si está habilitado
        if use_finetuning:
            print(f"\n🎯 Configurando fine-tuning...")
            print(f"   - Congelar backbone: {freeze_backbone}")
            print(f"   - Factor LR backbone: {finetuning_lr_factor}")
            
            # Configurar parámetros del NAFNet para fine-tuning
            nafnet_param_groups = setup_finetuning_params(
                self.nafnet, 
                freeze_backbone=freeze_backbone, 
                learning_rate_factor=finetuning_lr_factor
            )
            
            # Debug: verificar qué devuelve setup_finetuning_params
            print(f"🔍 DEBUG nafnet_param_groups tipo: {type(nafnet_param_groups)}")
            print(f"🔍 DEBUG nafnet_param_groups contenido: {nafnet_param_groups}")
            
            # Recrear optimizador con grupos de parámetros diferenciados
            base_lr = 1e-4
            self.nafnet_optimizer = optim.Adam([
                {'params': nafnet_param_groups[0]['params'], 'lr': base_lr * nafnet_param_groups[0]['lr_factor']},
                {'params': nafnet_param_groups[1]['params'], 'lr': base_lr * nafnet_param_groups[1]['lr_factor']}
            ])
            
            print(f"   - LR backbone: {base_lr * finetuning_lr_factor}")
            print(f"   - LR head: {base_lr}")
        else:
            print(f"\n🔧 Usando entrenamiento estándar (sin fine-tuning)")
        
        # Entrenamiento
        output_dir = Path("outputs/layer2_training")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n🚀 Época {epoch}/{num_epochs}")
            
            # Entrenar
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validar
            val_metrics = self.validate(val_loader)
            
            # Guardar métricas
            self.training_history['nafnet_losses'].append(train_metrics['nafnet_loss'])
            self.training_history['docunet_losses'].append(train_metrics['docunet_loss'])
            self.training_history['combined_losses'].append(train_metrics['combined_loss'])
            
            # Mostrar resultados
            print(f"📊 Train - NAFNet: {train_metrics['nafnet_loss']:.4f}, "
                  f"DocUNet: {train_metrics['docunet_loss']:.4f}, "
                  f"Combined: {train_metrics['combined_loss']:.4f}")
            print(f"📊 Val - Loss: {val_metrics['val_loss']:.4f}")
            
            # Guardar checkpoints cada 5 épocas
            if epoch % 5 == 0:
                checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
                self.save_models(output_dir / checkpoint_name)
        
        # Guardar modelos finales
        self.save_models(output_dir / "final_models.pth")
        
        # Crear gráficos de entrenamiento
        self.plot_training_history(output_dir)
        
        print(f"\n🎉 Entrenamiento completado. Modelos guardados en MinIO (bucket: models/layer_2/)")
        print(f"📁 Archivos generados:")
        print(f"   - final_models.pth (modelo final)")
        print(f"   - training_history.png (gráficas de entrenamiento)")
        if SERVICES_AVAILABLE:
            print(f"   - checkpoints cada 5 épocas")
        else:
            print(f"⚠️ Algunos archivos pueden haberse guardado localmente en: {output_dir}")
    
    def save_models(self, path: Path):
        """Guardar modelos en MinIO"""
        import tempfile
        import io
        
        # Crear datos del modelo
        model_data = {
            'nafnet_state_dict': self.nafnet.state_dict(),
            'docunet_state_dict': self.docunet.state_dict(),
            'nafnet_optimizer': self.nafnet_optimizer.state_dict(),
            'docunet_optimizer': self.docunet_optimizer.state_dict(),
            'training_history': self.training_history
        }
        
        # Guardar en memoria como bytes
        buffer = io.BytesIO()
        torch.save(model_data, buffer)
        buffer.seek(0)
        model_bytes = buffer.getvalue()
        
        # Extraer nombre del archivo del path
        model_filename = path.name
        
        # Subir a MinIO si el servicio está disponible
        if SERVICES_AVAILABLE and minio_service:
            try:
                # Subir usando el servicio de MinIO
                minio_path = minio_service.upload_model(
                    model_data=model_bytes,
                    layer="2",
                    model_name=model_filename
                )
                print(f"✅ Modelo guardado en MinIO: {minio_path}")
            except Exception as e:
                print(f"⚠️ Error guardando en MinIO: {e}")
                # Fallback: guardar localmente
                torch.save(model_data, path)
                print(f"💾 Modelo guardado localmente: {path}")
        else:
            # Fallback: guardar localmente si MinIO no está disponible
            torch.save(model_data, path)
            print(f"💾 Modelo guardado localmente (MinIO no disponible): {path}")
        
        buffer.close()
    
    def plot_training_history(self, output_dir: Path):
        """Crear gráficos del entrenamiento y guardarlos en MinIO"""
        import tempfile
        import io
        
        plt.figure(figsize=(15, 5))
        
        epochs = range(1, len(self.training_history['nafnet_losses']) + 1)
        
        # Pérdidas por modelo
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.training_history['nafnet_losses'], 'b-', label='NAFNet')
        plt.plot(epochs, self.training_history['docunet_losses'], 'r-', label='DocUNet')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.title('Pérdidas por Modelo')
        plt.legend()
        plt.grid(True)
        
        # Pérdida combinada
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.training_history['combined_losses'], 'g-', label='Combinada')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.title('Pérdida Combinada')
        plt.legend()
        plt.grid(True)
        
        # Comparación todas las pérdidas
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.training_history['nafnet_losses'], 'b-', label='NAFNet', alpha=0.7)
        plt.plot(epochs, self.training_history['docunet_losses'], 'r-', label='DocUNet', alpha=0.7)
        plt.plot(epochs, self.training_history['combined_losses'], 'g-', label='Combinada', alpha=0.7)
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.title('Todas las Pérdidas')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Guardar en memoria como bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        
        # Subir a MinIO si está disponible
        if SERVICES_AVAILABLE and minio_service:
            try:
                # Subir gráfica a MinIO
                graph_filename = "training_history.png"
                minio_service.upload_file(
                    file_data=image_bytes,
                    bucket='models',
                    filename=f"layer_2/{graph_filename}"
                )
                print(f"📊 Gráfica guardada en MinIO: layer_2/{graph_filename}")
            except Exception as e:
                print(f"⚠️ Error guardando gráfica en MinIO: {e}")
                # Fallback: guardar localmente
                plt.savefig(output_dir / "training_history.png", dpi=150, bbox_inches='tight')
                print(f"📊 Gráfica guardada localmente: {output_dir / 'training_history.png'}")
        else:
            # Fallback: guardar localmente
            plt.savefig(output_dir / "training_history.png", dpi=150, bbox_inches='tight')
            print(f"📊 Gráfica guardada localmente (MinIO no disponible): {output_dir / 'training_history.png'}")
        
        plt.close()
        buffer.close()
        plt.ylabel('Pérdida')
        plt.title('Todas las Pérdidas')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "training_history.png", dpi=150, bbox_inches='tight')
        plt.close()

# ============================================================================
# FUNCIONES DE UTILIDAD PARA LA API
# ============================================================================

def create_layer2_trainer() -> 'Layer2Trainer':
    """
    Función factory para crear un entrenador de Capa 2
    Para ser usada por la API (sin dependencias HTTP)
    """
    return Layer2Trainer()

def validate_training_parameters(num_epochs: int, max_pairs: int, batch_size: int) -> Dict[str, str]:
    """Validar parámetros de entrenamiento"""
    errors = {}
    
    if num_epochs < 1 or num_epochs > 1000:
        errors['num_epochs'] = "Debe estar entre 1 y 1000"
    
    if max_pairs < 1 or max_pairs > 10000:
        errors['max_pairs'] = "Debe estar entre 1 y 10000"
    
    if batch_size < 1 or batch_size > 32:
        errors['batch_size'] = "Debe estar entre 1 y 32"
    
    return errors
