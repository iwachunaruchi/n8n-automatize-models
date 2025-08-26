#!/usr/bin/env python3
"""
Script de entrenamiento adaptado para datos sintéticos
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
from tqdm import tqdm
import yaml
import argparse
from models.restormer import Restormer

class SyntheticDocumentDataset(Dataset):
    """Dataset adaptado para datos sintéticos"""
    
    def __init__(self, clean_dir, degraded_dir, size=(256, 256)):
        self.clean_dir = clean_dir
        self.degraded_dir = degraded_dir
        self.size = size
        
        # Cargar pares sintéticos
        self.pairs = []
        
        # Obtener archivos limpios
        clean_files = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for clean_file in clean_files:
            clean_path = os.path.join(clean_dir, clean_file)
            base_name = os.path.splitext(clean_file)[0]
            
            # Buscar archivos degradados correspondientes
            degraded_files = [f for f in os.listdir(degraded_dir) 
                            if f.startswith(base_name) and f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for degraded_file in degraded_files:
                degraded_path = os.path.join(degraded_dir, degraded_file)
                self.pairs.append((clean_path, degraded_path))
        
        print(f"✅ Dataset cargado: {len(self.pairs)} pares de entrenamiento")
        
        # Mostrar algunos ejemplos
        for i, (clean, degraded) in enumerate(self.pairs[:3]):
            clean_name = os.path.basename(clean)
            degraded_name = os.path.basename(degraded)
            print(f"   📝 Par {i+1}: {clean_name} ↔ {degraded_name}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        clean_path, degraded_path = self.pairs[idx]
        
        # Cargar imágenes
        clean_img = cv2.imread(clean_path)
        degraded_img = cv2.imread(degraded_path)
        
        if clean_img is None or degraded_img is None:
            raise ValueError(f"Error cargando par {idx}: {clean_path}, {degraded_path}")
        
        # Convertir BGR → RGB
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        degraded_img = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar si es necesario
        if clean_img.shape[:2] != self.size:
            clean_img = cv2.resize(clean_img, self.size, interpolation=cv2.INTER_AREA)
        if degraded_img.shape[:2] != self.size:
            degraded_img = cv2.resize(degraded_img, self.size, interpolation=cv2.INTER_AREA)
        
        # Normalizar [0, 1]
        clean_img = clean_img.astype(np.float32) / 255.0
        degraded_img = degraded_img.astype(np.float32) / 255.0
        
        # Convertir a tensor [C, H, W]
        clean_tensor = torch.from_numpy(clean_img).permute(2, 0, 1)
        degraded_tensor = torch.from_numpy(degraded_img).permute(2, 0, 1)
        
        return degraded_tensor, clean_tensor

def train_restormer():
    """Entrenar el modelo Restormer"""
    
    print("🚀 INICIANDO ENTRENAMIENTO DE RESTORMER")
    print("=" * 60)
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Dispositivo: {device}")
    
    # Hiperparámetros optimizados para GTX 1650 (4GB VRAM)
    batch_size = 1  # Reducido para ahorrar memoria
    learning_rate = 1e-4
    num_epochs = 30  # Reducido para pruebas más rápidas
    image_size = (128, 128)  # Reducido de 256x256 para ahorrar memoria
    
    print(f"📊 Configuración:")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • Epochs: {num_epochs}")
    print(f"   • Image size: {image_size}")
    
    # Dataset
    print(f"\n📁 Cargando dataset...")
    dataset = SyntheticDocumentDataset(
        clean_dir="data/train/clean",
        degraded_dir="data/train/degraded", 
        size=image_size
    )
    
    if len(dataset) == 0:
        print("❌ Error: No se encontraron datos de entrenamiento")
        return
    
    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Modelo Restormer optimizado para GTX 1650
    print(f"\n🧠 Inicializando modelo Restormer optimizado...")
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=32,  # Reducido de 48 a 32
        num_blocks=[2, 4, 4, 6],  # Reducido de [4, 6, 6, 8]
        num_refinement_blocks=2,  # Reducido de 4 a 2
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.0,  # Reducido de 2.66 a 2.0
        bias=False
    ).to(device)
    
    # Optimizador y pérdida
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()  # L1 loss para mejor preservación de detalles
    
    # Crear directorio de checkpoints
    os.makedirs("outputs/checkpoints", exist_ok=True)
    
    print(f"\n🎯 Iniciando entrenamiento...")
    print("-" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
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
                'Loss': f"{loss.item():.4f}",
                'Avg': f"{epoch_loss/(batch_idx+1):.4f}"
            })
        
        # Pérdida promedio de la época
        avg_loss = epoch_loss / len(dataloader)
        
        print(f"📊 Epoch {epoch+1:3d} | Loss: {avg_loss:.6f}")
        
        # Guardar mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = "outputs/checkpoints/best_restormer.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"💾 Nuevo mejor modelo guardado: {checkpoint_path}")
        
        # Guardar checkpoint cada 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"outputs/checkpoints/restormer_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"💾 Checkpoint guardado: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("🎉 ¡ENTRENAMIENTO COMPLETADO!")
    print(f"✅ Mejor pérdida: {best_loss:.6f}")
    print(f"📁 Modelo guardado en: outputs/checkpoints/best_restormer.pth")
    print("\n🔍 Para probar el modelo entrenado:")
    print("   python demo_complete.py --demo")

def test_dataset():
    """Probar que el dataset se carga correctamente"""
    print("🧪 PROBANDO DATASET")
    print("=" * 40)
    
    try:
        dataset = SyntheticDocumentDataset(
            clean_dir="data/train/clean",
            degraded_dir="data/train/degraded",
            size=(256, 256)
        )
        
        if len(dataset) > 0:
            print(f"\n🔍 Probando carga de datos...")
            degraded, clean = dataset[0]
            print(f"   📐 Degraded shape: {degraded.shape}")
            print(f"   📐 Clean shape: {clean.shape}")
            print(f"   📊 Degraded range: [{degraded.min():.3f}, {degraded.max():.3f}]")
            print(f"   📊 Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
            print("✅ Dataset funciona correctamente!")
        else:
            print("❌ Dataset vacío")
            
    except Exception as e:
        print(f"❌ Error en dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar modelo Restormer')
    parser.add_argument('--test', action='store_true', help='Solo probar el dataset')
    
    args = parser.parse_args()
    
    if args.test:
        test_dataset()
    else:
        train_restormer()
