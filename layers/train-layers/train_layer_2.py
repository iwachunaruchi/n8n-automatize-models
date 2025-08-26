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

# Importaciones que serán inyectadas por la API
try:
    import requests
except ImportError:
    requests = None

# Configuración de buckets (será inyectada por la API)
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

class APIDocumentDataset(Dataset):
    """Dataset que carga imágenes desde la API"""
    
    def __init__(self, api_base_url: str, max_pairs: int = 100, patch_size: int = 128, 
                 use_training_bucket: bool = True):
        self.api_url = api_base_url
        self.patch_size = patch_size
        self.pairs = []
        self.use_training_bucket = use_training_bucket
        
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
        """Cargar pares desde bucket de entrenamiento (método recomendado)"""
        print(f"📥 Cargando pares desde bucket 'document-training'...")
        
        try:
            # Obtener archivos del bucket de entrenamiento
            response = requests.get(f"{self.api_url}/files/list/{MINIO_BUCKETS['training']}")
            
            if response.status_code != 200:
                print("❌ Error obteniendo archivos del bucket de entrenamiento")
                return
            
            files = response.json().get('files', [])
            
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
        """Cargar pares de imágenes degradadas y limpias"""
        print(f"📥 Cargando pares de imágenes desde API...")
        
        try:
            # Obtener listas de archivos
            degraded_response = requests.get(f"{self.api_url}/files/list/{MINIO_BUCKETS['degraded']}")
            clean_response = requests.get(f"{self.api_url}/files/list/{MINIO_BUCKETS['clean']}")
            
            if degraded_response.status_code != 200 or clean_response.status_code != 200:
                print("❌ Error obteniendo listas de archivos")
                return
            
            degraded_files = degraded_response.json().get('files', [])
            clean_files = clean_response.json().get('files', [])
            
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
        """Descargar imagen desde API"""
        try:
            response = requests.get(f"{self.api_url}/files/view/{bucket}/{filename}")
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
            return None
        except:
            return None
    
    def __getitem__(self, idx):
        degraded_file, clean_file = self.pairs[idx]
        
        # Determinar bucket según el método usado
        if self.use_training_bucket:
            # Ambos archivos están en el bucket de entrenamiento
            degraded = self._download_image(MINIO_BUCKETS['training'], degraded_file)
            clean = self._download_image(MINIO_BUCKETS['training'], clean_file)
        else:
            # Método original: buckets separados
            degraded = self._download_image(MINIO_BUCKETS['degraded'], degraded_file)
            clean = self._download_image(MINIO_BUCKETS['clean'], clean_file)
    
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
        
        # Descargar imágenes
        degraded = self._download_image(MINIO_BUCKETS['degraded'], degraded_file)
        clean = self._download_image(MINIO_BUCKETS['clean'], clean_file)
        
        if degraded is None or clean is None:
            # Fallback a un índice válido
            return self.__getitem__(0 if idx != 0 else 1)
        
        # Extraer patches
        degraded_patch, clean_patch = self._extract_patch(degraded, clean)
        
        # Aplicar transformaciones
        seed = random.randint(0, 2**32)
        
        random.seed(seed)
        np.random.seed(seed)
        degraded_transformed = self.transform(image=degraded_patch)
        
        random.seed(seed)
        np.random.seed(seed) 
        clean_transformed = self.transform(image=clean_patch)
        
        return {
            'degraded': degraded_transformed['image'],
            'clean': clean_transformed['image']
        }

# ============================================================================
# ENTRENADOR PARA CAPA 2
# ============================================================================

class Layer2Trainer:
    """Entrenador para modelos de Capa 2"""
    
    def __init__(self, api_base_url: str = API_BASE_URL):
        self.api_url = api_base_url
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Modelos
        self.nafnet = SimpleNAFNet(in_channels=3, width=32, num_blocks=2).to(self.device)
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
            degraded = batch['degraded'].to(self.device)
            clean = batch['clean'].to(self.device)
            
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
                degraded = batch['degraded'].to(self.device)
                clean = batch['clean'].to(self.device)
                
                # Pipeline completo
                denoised = self.nafnet(degraded)
                flow = self.docunet(denoised)
                dewarped = self.apply_dewarping(denoised, flow)
                
                loss = self.l1_loss(dewarped, clean)
                total_loss += loss.item()
        
        return {'val_loss': total_loss / len(dataloader)}
    
    def train(self, num_epochs: int = 10, max_pairs: int = 100, batch_size: int = 4, 
              use_training_bucket: bool = True):
        """Entrenamiento completo"""
        print("🔧 ENTRENAMIENTO CAPA 2: NAFNet + DocUNet")
        print("=" * 60)
        print(f"🔧 Dispositivo: {self.device}")
        print(f"📊 Épocas: {num_epochs}")
        print(f"📦 Batch size: {batch_size}")
        print(f"📁 Usando bucket de entrenamiento: {use_training_bucket}")
        
        # Crear dataset
        dataset = APIDocumentDataset(self.api_url, max_pairs=max_pairs, 
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
                self.save_models(output_dir / f"checkpoint_epoch_{epoch}.pth")
        
        # Guardar modelos finales
        self.save_models(output_dir / "final_models.pth")
        
        # Crear gráficos de entrenamiento
        self.plot_training_history(output_dir)
        
        print(f"\n🎉 Entrenamiento completado. Modelos guardados en: {output_dir}")
    
    def save_models(self, path: Path):
        """Guardar modelos"""
        torch.save({
            'nafnet_state_dict': self.nafnet.state_dict(),
            'docunet_state_dict': self.docunet.state_dict(),
            'nafnet_optimizer': self.nafnet_optimizer.state_dict(),
            'docunet_optimizer': self.docunet_optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
    
    def plot_training_history(self, output_dir: Path):
        """Crear gráficos del entrenamiento"""
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
        plt.savefig(output_dir / "training_history.png", dpi=150, bbox_inches='tight')
        plt.close()

# ============================================================================
# FUNCIONES DE UTILIDAD PARA LA API
# ============================================================================

def create_layer2_trainer(api_base_url: str = "http://localhost:8000") -> 'Layer2Trainer':
    """
    Función factory para crear un entrenador de Capa 2
    Para ser usada por la API
    """
    return Layer2Trainer(api_base_url)

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
