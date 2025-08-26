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
import hashlib
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
        
        print(f"� Dataset: {len(self.valid_pairs)} pares válidos encontrados")
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
    
    print("� Descargando modelo Restormer preentrenado...")
    
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
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,  # Configuración estándar
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False
    ).to(device)
    
    # Cargar modelo preentrenado
    pretrained_path = "models/pretrained/restormer_denoising.pth"
    print(f"📥 Cargando modelo preentrenado: {pretrained_path}")
    
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print("✅ Modelo preentrenado cargado como punto de partida!")
    else:
        print("❌ No se encontró modelo preentrenado. Entrenando desde cero...")
    
    # Optimizador y pérdida para fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
    
    # Crear directorio para modelos fine-tuned
    os.makedirs("outputs/checkpoints/finetuned", exist_ok=True)
    
    print(f"\n🎯 Iniciando fine-tuning...")
    print("-" * 50)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Fine-tune {epoch+1}/{num_epochs}")
        
        for batch_idx, (degraded, clean) in enumerate(progress_bar):
            degraded = degraded.to(device)
            clean = clean.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            restored = model(degraded)
            
            # Calcular pérdida
            loss = criterion(restored, clean)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Actualizar progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.6f}",
                'Avg': f"{epoch_loss/(batch_idx+1):.6f}"
            })
        
        # Pérdida promedio de la época
        avg_loss = epoch_loss / len(dataloader)
        
        print(f"📊 Epoch {epoch+1:2d} | Loss: {avg_loss:.6f}")
        
        # Guardar mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = "outputs/checkpoints/finetuned/best_finetuned_restormer.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'base_model': 'pretrained_denoising'
            }, checkpoint_path)
            print(f"💾 Nuevo mejor modelo fine-tuned guardado!")
    
    print("\n" + "=" * 50)
    print("🎉 ¡FINE-TUNING COMPLETADO!")
    print(f"✅ Mejor pérdida: {best_loss:.6f}")
    print(f"📁 Modelo guardado en: outputs/checkpoints/finetuned/")
    print("\n🔍 Para probar el modelo fine-tuned:")
    print("   python test_finetuned_model.py")

if __name__ == "__main__":
    finetune_pretrained_model()
