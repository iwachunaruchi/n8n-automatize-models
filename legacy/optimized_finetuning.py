#!/usr/bin/env python3
"""
Fine-tuning OPTIMIZADO del modelo Restormer preentrenado
Implementa múltiples estrategias para mejorar los resultados
"""

import sys
import os
sys.path.append('src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt

# Importar nuestros modelos
from models.restormer import Restormer

class AdvancedDocumentDataset(Dataset):
    """Dataset avanzado con data augmentation para fine-tuning"""
    
    def __init__(self, degraded_dir, clean_dir, img_size=128, augment=True):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.img_size = img_size
        self.augment = augment
        
        # Obtener lista de imágenes
        self.degraded_images = [f for f in os.listdir(degraded_dir) if f.endswith('.png')]
        self.clean_images = [f for f in os.listdir(clean_dir) if f.endswith('.png')]
        
        # Filtrar para tener pares válidos
        self.valid_pairs = []
        for deg_img in self.degraded_images:
            base_name = deg_img.replace('_deg_heavy', '').replace('_deg_medium', '').replace('_deg_light', '')
            base_name = base_name.replace('_var_1', '').replace('_var_2', '').replace('_var_3', '')
            
            for clean_img in self.clean_images:
                if base_name in clean_img or clean_img.replace('val_', '') == base_name:
                    self.valid_pairs.append((deg_img, clean_img))
                    break
        
        print(f"📊 Dataset optimizado: {len(self.valid_pairs)} pares válidos")
        if self.augment:
            print(f"🔄 Data augmentation ACTIVADO - Dataset efectivo: {len(self.valid_pairs) * 4}")
    
    def __len__(self):
        # Si hay augmentation, multiplicamos por 4 las combinaciones
        return len(self.valid_pairs) * (4 if self.augment else 1)
    
    def apply_augmentation(self, degraded, clean, aug_type):
        """Aplicar data augmentation de forma consistente a ambas imágenes"""
        
        if aug_type == 0:
            # Original - sin cambios
            return degraded, clean
        elif aug_type == 1:
            # Flip horizontal
            degraded = cv2.flip(degraded, 1)
            clean = cv2.flip(clean, 1)
        elif aug_type == 2:
            # Flip vertical  
            degraded = cv2.flip(degraded, 0)
            clean = cv2.flip(clean, 0)
        elif aug_type == 3:
            # Rotación 180°
            degraded = cv2.rotate(degraded, cv2.ROTATE_180)
            clean = cv2.rotate(clean, cv2.ROTATE_180)
        
        return degraded, clean
    
    def __getitem__(self, idx):
        # Determinar par base y tipo de augmentation
        if self.augment:
            pair_idx = idx // 4
            aug_type = idx % 4
        else:
            pair_idx = idx
            aug_type = 0
        
        deg_name, clean_name = self.valid_pairs[pair_idx]
        
        # Cargar imágenes
        deg_path = os.path.join(self.degraded_dir, deg_name)
        clean_path = os.path.join(self.clean_dir, clean_name)
        
        degraded = cv2.imread(deg_path)
        clean = cv2.imread(clean_path)
        
        if degraded is None or clean is None:
            degraded = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
            clean = degraded.copy()
        
        # Aplicar augmentation
        degraded, clean = self.apply_augmentation(degraded, clean, aug_type)
        
        # Redimensionar
        degraded = cv2.resize(degraded, (self.img_size, self.img_size))
        clean = cv2.resize(clean, (self.img_size, self.img_size))
        
        # Convertir a RGB y normalizar
        degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Convertir a tensores [C, H, W]
        degraded = torch.from_numpy(degraded).permute(2, 0, 1)
        clean = torch.from_numpy(clean).permute(2, 0, 1)
        
        return degraded, clean

class PerceptualLoss(nn.Module):
    """Loss perceptual para preservar detalles visuales"""
    
    def __init__(self, device):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.device = device
    
    def forward(self, pred, target):
        # Combinar múltiples losses
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        
        # Loss de gradiente para preservar bordes
        grad_loss = self.gradient_loss(pred, target)
        
        # Combinar losses con pesos
        total_loss = 0.7 * l1 + 0.2 * mse + 0.1 * grad_loss
        return total_loss
    
    def gradient_loss(self, pred, target):
        """Loss de gradiente para preservar bordes y detalles"""
        # Calcular gradientes en X e Y
        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        # Loss entre gradientes
        grad_loss_x = self.l1_loss(pred_grad_x, target_grad_x)
        grad_loss_y = self.l1_loss(pred_grad_y, target_grad_y)
        
        return grad_loss_x + grad_loss_y

def download_pretrained_model():
    """Descargar modelo preentrenado si no existe"""
    model_dir = "models/pretrained"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "restormer_denoising.pth")
    
    if os.path.exists(model_path):
        print(f"✅ Modelo preentrenado ya existe")
        return model_path
    
    print("📥 Descargando modelo preentrenado...")
    url = "https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_blind.pth"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Descargando") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ Modelo descargado")
        return model_path
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def load_pretrained_model(device):
    """Cargar modelo preentrenado"""
    model_path = download_pretrained_model()
    if not model_path:
        return None
    
    print("🔧 Cargando modelo preentrenado...")
    
    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias', dual_pixel_task=False
    )
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        print(f"✅ Modelo cargado exitosamente")
        return model
        
    except Exception as e:
        print(f"❌ Error: {e}")
        model = model.to(device)
        return model

def optimized_fine_tuning():
    """Fine-tuning OPTIMIZADO con múltiples estrategias"""
    
    print("🚀 FINE-TUNING OPTIMIZADO")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    
    # ESTRATEGIA 1: Cargar modelo preentrenado
    model = load_pretrained_model(device)
    if model is None:
        return None, None
    
    # ESTRATEGIA 2: Learning rate MUY BAJO + Scheduler agresivo
    learning_rate = 5e-6  # ULTRA BAJO
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                           weight_decay=1e-5, betas=(0.9, 0.999))
    
    # ESTRATEGIA 3: Loss function avanzado
    criterion = PerceptualLoss(device)
    
    # ESTRATEGIA 4: Scheduler más agresivo
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, T_mult=2, eta_min=1e-7
    )
    
    print(f"📚 Configuración optimizada:")
    print(f"   💡 Learning Rate: {learning_rate} (ultra bajo)")
    print(f"   🎯 Loss: Perceptual (L1 + MSE + Gradient)")
    print(f"   📊 Scheduler: CosineAnnealingWarmRestarts")
    print(f"   🔄 Data Augmentation: ACTIVADO")
    
    # ESTRATEGIA 5: Dataset con data augmentation
    train_dataset = AdvancedDocumentDataset(
        degraded_dir="data/train/degraded",
        clean_dir="data/train/clean",
        img_size=128,
        augment=True  # ACTIVAR AUGMENTATION
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  # Batch muy pequeño para máxima estabilidad
        shuffle=True,
        num_workers=0
    )
    
    # ESTRATEGIA 6: MENOS épocas para evitar sobreentrenamiento
    num_epochs = 5  # REDUCIDO de 10 a 5
    
    print(f"\n🏋️ Entrenamiento optimizado:")
    print(f"   🔄 Épocas: {num_epochs} (reducido)")
    print(f"   📊 Batch size: 1")
    print(f"   🎲 Samples totales: {len(train_dataset)}")
    
    model.train()
    epoch_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Optimizado {epoch+1}/{num_epochs}")
        
        for degraded, clean in progress_bar:
            degraded = degraded.to(device)
            clean = clean.to(device)
            
            optimizer.zero_grad()
            
            try:
                restored = model(degraded)
                loss = criterion(restored, clean)
                
                loss.backward()
                
                # ESTRATEGIA 7: Gradient clipping más conservador
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg': f'{running_loss/num_batches:.6f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ CUDA OOM en época {epoch+1}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        epoch_loss = running_loss / num_batches if num_batches > 0 else float('inf')
        epoch_losses.append(epoch_loss)
        
        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"✅ Época {epoch+1}/{num_epochs} - Loss: {epoch_loss:.6f} - LR: {current_lr:.2e}")
        
        # ESTRATEGIA 8: Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            
            # Guardar mejor modelo
            best_model_path = "outputs/checkpoints/optimized_best_model.pth"
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'learning_rate': current_lr
            }, best_model_path)
            print(f"💾 Mejor modelo guardado - Loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= 2:  # Early stopping después de 2 épocas sin mejora
                print(f"🛑 Early stopping activado - Sin mejora por {patience_counter} épocas")
                break
    
    # Guardar modelo final optimizado
    final_model_path = "outputs/checkpoints/optimized_restormer_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_loss': epoch_losses[-1] if epoch_losses else 0,
        'best_loss': best_loss,
        'optimization_config': {
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'data_augmentation': True,
            'loss_function': 'Perceptual',
            'scheduler': 'CosineAnnealingWarmRestarts',
            'early_stopping': True
        }
    }, final_model_path)
    
    print(f"\n🎉 FINE-TUNING OPTIMIZADO COMPLETADO!")
    print(f"💾 Modelo final: {final_model_path}")
    print(f"📉 Mejor loss: {best_loss:.6f}")
    print(f"📉 Loss final: {epoch_losses[-1]:.6f}")
    
    # Gráfico de pérdidas optimizado
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epoch_losses, 'b-', linewidth=2, marker='o')
    plt.title('Fine-tuning Optimizado: Pérdida por Época')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    lrs = [optimizer.param_groups[0]['lr']]  # Simplificado
    plt.plot(lrs, 'r-', linewidth=2)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Época')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = "outputs/analysis/optimized_finetuning_analysis.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Análisis guardado: {plot_path}")
    
    return model, final_model_path

if __name__ == "__main__":
    print("🎯 ESTRATEGIAS DE OPTIMIZACIÓN IMPLEMENTADAS:")
    print("   1. 💡 Learning rate ultra bajo (5e-6)")
    print("   2. 🎯 Loss perceptual (preserva detalles)")
    print("   3. 📊 Scheduler adaptativo")
    print("   4. 🔄 Data augmentation (4x más datos)")
    print("   5. 🔢 Menos épocas (5 en lugar de 10)")
    print("   6. 🛑 Early stopping")
    print("   7. 📉 Gradient clipping conservador")
    print("   8. 💾 Guardado del mejor modelo")
    
    model, model_path = optimized_fine_tuning()
    
    if model and model_path:
        print(f"\n🎯 SIGUIENTE PASO:")
        print(f"   Ejecuta: python test_optimized_model.py")
        print(f"   Para comparar los resultados optimizados")
