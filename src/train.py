"""
Script de entrenamiento para el modelo Restormer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import cv2
import numpy as np
from PIL import Image
import yaml
import argparse
from tqdm import tqdm
import logging
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.restormer import Restormer

class DocumentDataset(Dataset):
    """Dataset para pares de imágenes degradadas y limpias"""
    
    def __init__(self, degraded_dir, clean_dir, transform=None, size=(256, 256)):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.size = size
        
        # Obtener lista de archivos
        self.files = []
        for file in os.listdir(degraded_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
                degraded_path = os.path.join(degraded_dir, file)
                clean_path = os.path.join(clean_dir, file)
                if os.path.exists(clean_path):
                    self.files.append((degraded_path, clean_path))
        
        print(f"Dataset cargado: {len(self.files)} pares de imágenes")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        degraded_path, clean_path = self.files[idx]
        
        # Cargar imágenes
        degraded = cv2.imread(degraded_path)
        clean = cv2.imread(clean_path)
        
        degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        degraded = cv2.resize(degraded, self.size)
        clean = cv2.resize(clean, self.size)
        
        # Aplicar transformaciones
        if self.transform:
            augmented = self.transform(image=degraded, mask=clean)
            degraded = augmented['image']
            clean = augmented['mask']
        
        # Normalizar a [0, 1]
        degraded = degraded.astype(np.float32) / 255.0
        clean = clean.astype(np.float32) / 255.0
        
        # Convertir a tensor
        degraded = torch.from_numpy(degraded).permute(2, 0, 1)
        clean = torch.from_numpy(clean).permute(2, 0, 1)
        
        return degraded, clean

def get_transforms(training=True):
    """Obtener transformaciones de datos"""
    if training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.3),
        ])
    else:
        return None

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 suave)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

class EdgeLoss(nn.Module):
    """Pérdida de bordes usando filtros Sobel"""
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
        # Filtros Sobel
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
    def forward(self, pred, target):
        if pred.is_cuda:
            self.sobel_x = self.sobel_x.cuda()
            self.sobel_y = self.sobel_y.cuda()
            
        # Calcular gradientes
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        return F.l1_loss(pred_grad, target_grad)

def train_epoch(model, dataloader, optimizer, criterion, edge_loss, device, epoch):
    """Entrenar una época"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (degraded, clean) in enumerate(pbar):
        degraded = degraded.to(device)
        clean = clean.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        restored = model(degraded)
        
        # Calcular pérdidas
        l1_loss = criterion(restored, clean)
        edge_loss_val = edge_loss(restored, clean)
        
        # Pérdida total
        loss = l1_loss + 0.1 * edge_loss_val
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Actualizar barra de progreso
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'L1': f'{l1_loss.item():.4f}',
            'Edge': f'{edge_loss_val.item():.4f}'
        })
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, edge_loss, device):
    """Validar modelo"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for degraded, clean in tqdm(dataloader, desc='Validating'):
            degraded = degraded.to(device)
            clean = clean.to(device)
            
            restored = model(degraded)
            
            l1_loss = criterion(restored, clean)
            edge_loss_val = edge_loss(restored, clean)
            loss = l1_loss + 0.1 * edge_loss_val
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Guardar checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    """Cargar checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def main():
    parser = argparse.ArgumentParser(description='Entrenar Restormer')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='Ruta al archivo de configuración')
    parser.add_argument('--resume', type=str, default=None,
                       help='Ruta al checkpoint para reanudar entrenamiento')
    
    args = parser.parse_args()
    
    # Cargar configuración
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Usando device: {device}')
    
    # Crear directorios de salida
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['sample_dir'], exist_ok=True)
    
    # Datasets y DataLoaders
    train_transform = get_transforms(training=True)
    val_transform = get_transforms(training=False)
    
    train_dataset = DocumentDataset(
        config['data']['train_degraded'],
        config['data']['train_clean'],
        transform=train_transform,
        size=tuple(config['data']['image_size'])
    )
    
    val_dataset = DocumentDataset(
        config['data']['val_degraded'],
        config['data']['val_clean'],
        transform=val_transform,
        size=tuple(config['data']['image_size'])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Modelo
    model = Restormer(**config['model'])
    model.to(device)
    
    # Optimizador
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    # Funciones de pérdida
    criterion = CharbonnierLoss()
    edge_loss = EdgeLoss()
    
    # Reanudar entrenamiento si se especifica
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
        logger.info(f'Reanudando entrenamiento desde epoch {start_epoch}')
    
    # Entrenamiento
    logger.info('Iniciando entrenamiento...')
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # Entrenar
        train_loss = train_epoch(model, train_loader, optimizer, criterion, edge_loss, device, epoch)
        
        # Validar
        val_loss = validate_epoch(model, val_loader, criterion, edge_loss, device)
        
        # Actualizar scheduler
        scheduler.step()
        
        # Logging
        logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Guardar mejor modelo
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(config['training']['checkpoint_dir'], 'best_model.pth')
            )
        
        # Guardar checkpoint periódico
        if epoch % config['training']['save_freq'] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(config['training']['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
            )
    
    logger.info('Entrenamiento completado')

if __name__ == '__main__':
    main()
