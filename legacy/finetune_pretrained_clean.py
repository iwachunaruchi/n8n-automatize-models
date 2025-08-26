#!/usr/bin/env python3
"""
Fine-tuning del modelo Restormer preentrenado con datos específicos de documentos
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

class DocumentDataset(Dataset):
    """Dataset para documentos degradados y limpios"""
    
    def __init__(self, degraded_dir, clean_dir, img_size=128):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.img_size = img_size
        
        # Obtener lista de imágenes
        self.degraded_images = [f for f in os.listdir(degraded_dir) if f.endswith('.png')]
        self.clean_images = [f for f in os.listdir(clean_dir) if f.endswith('.png')]
        
        # Filtrar para tener pares válidos
        self.valid_pairs = []
        for deg_img in self.degraded_images:
            # Buscar imagen limpia correspondiente
            base_name = deg_img.replace('_deg_heavy', '').replace('_deg_medium', '').replace('_deg_light', '')
            base_name = base_name.replace('_var_1', '').replace('_var_2', '').replace('_var_3', '')
            
            # Buscar en imágenes limpias
            for clean_img in self.clean_images:
                if base_name in clean_img or clean_img.replace('val_', '') == base_name:
                    self.valid_pairs.append((deg_img, clean_img))
                    break
        
        print(f"📊 Dataset: {len(self.valid_pairs)} pares válidos encontrados")
        for deg, clean in self.valid_pairs[:5]:
            print(f"   🔗 {deg} ← → {clean}")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        deg_name, clean_name = self.valid_pairs[idx]
        
        # Cargar imágenes
        deg_path = os.path.join(self.degraded_dir, deg_name)
        clean_path = os.path.join(self.clean_dir, clean_name)
        
        # Leer imágenes
        degraded = cv2.imread(deg_path)
        clean = cv2.imread(clean_path)
        
        if degraded is None or clean is None:
            # Imagen de respaldo
            degraded = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
            clean = degraded.copy()
        
        # Redimensionar a tamaño fijo
        degraded = cv2.resize(degraded, (self.img_size, self.img_size))
        clean = cv2.resize(clean, (self.img_size, self.img_size))
        
        # Convertir a RGB y normalizar
        degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Convertir a tensores [C, H, W]
        degraded = torch.from_numpy(degraded).permute(2, 0, 1)
        clean = torch.from_numpy(clean).permute(2, 0, 1)
        
        return degraded, clean

def download_pretrained_model():
    """Descargar modelo preentrenado si no existe"""
    
    model_dir = "models/pretrained"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "restormer_denoising.pth")
    
    if os.path.exists(model_path):
        print(f"✅ Modelo preentrenado ya existe: {model_path}")
        return model_path
    
    print("📥 Descargando modelo Restormer preentrenado...")
    
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
        
        print(f"✅ Modelo descargado: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ Error descargando modelo: {e}")
        return None

def load_pretrained_model(device):
    """Cargar modelo preentrenado y adaptarlo para fine-tuning"""
    
    # Descargar modelo si es necesario
    model_path = download_pretrained_model()
    if not model_path:
        return None
    
    print("🔧 Cargando modelo preentrenado...")
    
    # Crear modelo con arquitectura compatible
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,  # Dimensión del modelo preentrenado
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False
    )
    
    try:
        # Cargar pesos preentrenados
        checkpoint = torch.load(model_path, map_location=device)
        
        # El checkpoint puede tener diferentes estructuras
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
        
        # Cargar pesos
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        
        print(f"✅ Modelo preentrenado cargado exitosamente")
        print(f"📊 Parámetros: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error cargando modelo preentrenado: {e}")
        print("🔄 Creando modelo desde cero...")
        
        model = model.to(device)
        return model

def fine_tune_model():
    """Fine-tuning del modelo preentrenado"""
    
    print("🚀 INICIANDO FINE-TUNING DEL MODELO PREENTRENADO")
    print("=" * 60)
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    
    if device.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Cargar modelo preentrenado
    model = load_pretrained_model(device)
    if model is None:
        print("❌ No se pudo cargar el modelo preentrenado")
        return None, None
    
    # Configurar optimizador con learning rate bajo para fine-tuning
    learning_rate = 1e-5  # Muy bajo para fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Loss function
    criterion = nn.L1Loss()
    
    # Scheduler para reducir learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    print(f"📚 Learning Rate: {learning_rate}")
    print(f"🎯 Loss Function: L1Loss")
    
    # Dataset
    train_dataset = DocumentDataset(
        degraded_dir="data/train/degraded",
        clean_dir="data/train/clean",
        img_size=128  # Tamaño más pequeño para fine-tuning rápido
    )
    
    if len(train_dataset) == 0:
        print("❌ No se encontraron datos de entrenamiento")
        return None, None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,  # Batch pequeño para fine-tuning
        shuffle=True,
        num_workers=0  # 0 para Windows
    )
    
    # Entrenamiento
    print(f"\n🏋️ Iniciando fine-tuning...")
    print(f"📊 Batch size: 2")
    print(f"🔄 Épocas: 10 (fine-tuning rápido)")
    
    model.train()
    epoch_losses = []
    
    for epoch in range(10):  # Pocas épocas para fine-tuning
        running_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/10")
        
        for degraded, clean in progress_bar:
            degraded = degraded.to(device)
            clean = clean.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            try:
                restored = model(degraded)
                loss = criterion(restored, clean)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg': f'{running_loss/num_batches:.6f}'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ CUDA out of memory en época {epoch+1}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Calcular loss promedio de la época
        epoch_loss = running_loss / num_batches if num_batches > 0 else float('inf')
        epoch_losses.append(epoch_loss)
        
        # Scheduler step
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"✅ Época {epoch+1}/10 - Loss: {epoch_loss:.6f} - LR: {current_lr:.2e}")
        
        # Guardar checkpoint cada 3 épocas
        if (epoch + 1) % 3 == 0:
            checkpoint_path = f"outputs/checkpoints/finetuned_restormer_epoch_{epoch+1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'learning_rate': current_lr
            }, checkpoint_path)
            
            print(f"💾 Checkpoint guardado: {checkpoint_path}")
    
    # Guardar modelo final
    final_model_path = "outputs/checkpoints/finetuned_restormer_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_loss': epoch_losses[-1] if epoch_losses else 0,
        'config': {
            'inp_channels': 3,
            'out_channels': 3,
            'dim': 48,
            'num_blocks': [4, 6, 6, 8],
            'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8],
            'ffn_expansion_factor': 2.66,
            'bias': False,
            'LayerNorm_type': 'WithBias',
            'dual_pixel_task': False
        }
    }, final_model_path)
    
    print(f"\n🎉 FINE-TUNING COMPLETADO!")
    print(f"💾 Modelo final guardado: {final_model_path}")
    print(f"📉 Loss final: {epoch_losses[-1]:.6f}")
    
    # Gráfico de pérdidas
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, 'b-', linewidth=2, marker='o')
    plt.title('Fine-tuning: Pérdida por Época')
    plt.xlabel('Época')
    plt.ylabel('Loss (L1)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = "outputs/analysis/finetuning_loss.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Gráfico de pérdidas guardado: {plot_path}")
    
    return model, final_model_path

if __name__ == "__main__":
    model, model_path = fine_tune_model()
    
    if model and model_path:
        print(f"\n🎯 SIGUIENTE PASO:")
        print(f"   Ejecuta: python test_finetuned_model.py")
        print(f"   Para probar el modelo con fine-tuning")
