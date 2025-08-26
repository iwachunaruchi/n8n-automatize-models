#!/usr/bin/env python3
"""
ENTRENAMIENTO CON TRANSFER LEARNING GRADUAL
Versión optimizada y estructurada
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
sys.path.append('src')
sys.path.append('models')

from src.models.restormer import Restormer
from src.utils import load_config

class DocumentDataset(Dataset):
    """Dataset optimizado para documentos con data augmentation"""
    
    def __init__(self, degraded_dir, clean_dir, img_size=128, augment=True):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.img_size = img_size
        self.augment = augment
        
        # Buscar imágenes pareadas
        degraded_files = set(os.listdir(degraded_dir))
        clean_files = set(os.listdir(clean_dir))
        
        # Solo tomar imágenes que existen en ambos directorios
        self.paired_files = sorted(list(degraded_files.intersection(clean_files)))
        
        # Si hay pocas imágenes, crear múltiples versiones con augmentation
        if len(self.paired_files) < 50 and augment:
            augmented_files = []
            for file in self.paired_files:
                for i in range(4):  # 4 versiones por imagen
                    augmented_files.append((file, i))
            self.file_list = augmented_files
        else:
            self.file_list = [(file, 0) for file in self.paired_files]
        
        print(f"📊 Dataset creado: {len(self.file_list)} samples de {len(self.paired_files)} imágenes únicas")
    
    def __len__(self):
        return len(self.file_list)
    
    def apply_augmentation(self, degraded, clean, aug_type):
        """Aplicar data augmentation"""
        if aug_type == 1:  # Flip horizontal
            degraded = cv2.flip(degraded, 1)
            clean = cv2.flip(clean, 1)
        elif aug_type == 2:  # Flip vertical
            degraded = cv2.flip(degraded, 0)
            clean = cv2.flip(clean, 0)
        elif aug_type == 3:  # Rotar 90°
            degraded = cv2.rotate(degraded, cv2.ROTATE_90_CLOCKWISE)
            clean = cv2.rotate(clean, cv2.ROTATE_90_CLOCKWISE)
        
        return degraded, clean
    
    def __getitem__(self, idx):
        filename, aug_type = self.file_list[idx]
        
        # Cargar imágenes
        degraded_path = os.path.join(self.degraded_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)
        
        degraded = cv2.imread(degraded_path)
        clean = cv2.imread(clean_path)
        
        if degraded is None or clean is None:
            # Imagen de respaldo
            degraded = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            clean = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Aplicar augmentation si es necesario
        if aug_type > 0 and self.augment:
            degraded, clean = self.apply_augmentation(degraded, clean, aug_type)
        
        # Redimensionar
        degraded = cv2.resize(degraded, (self.img_size, self.img_size))
        clean = cv2.resize(clean, (self.img_size, self.img_size))
        
        # Convertir a tensor
        degraded_tensor = torch.from_numpy(degraded).permute(2, 0, 1).float() / 255.0
        clean_tensor = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        
        return degraded_tensor, clean_tensor

class GradualTransferLearning:
    """
    Implementación del Transfer Learning Gradual
    
    ESTRATEGIA:
    Stage 1: Solo output layers (1 época) - LR 1e-5
    Stage 2: + refinement blocks (2 épocas) - LR 5e-6  
    Stage 3: + últimos transformer blocks (3 épocas) - LR 2e-6
    Stage 4: Modelo completo (4 épocas) - LR 1e-6
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.stage_configs = [
            {
                'name': 'Stage 1: Output Layers',
                'epochs': 1,
                'lr': 1e-5,
                'frozen_components': ['layers', 'refinement_head', 'patch_embed']
            },
            {
                'name': 'Stage 2: + Refinement Blocks', 
                'epochs': 2,
                'lr': 5e-6,
                'frozen_components': ['layers']
            },
            {
                'name': 'Stage 3: + Last Transformer Blocks',
                'epochs': 3, 
                'lr': 2e-6,
                'frozen_components': ['layers.0', 'layers.1', 'layers.2']
            },
            {
                'name': 'Stage 4: Full Model',
                'epochs': 4,
                'lr': 1e-6,
                'frozen_components': []
            }
        ]
        
        self.training_history = []
    
    def freeze_layers_by_name(self, frozen_components):
        """Congelar componentes específicos del modelo"""
        # Primero descongelar todo
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Luego congelar componentes específicos
        for name, param in self.model.named_parameters():
            for frozen_component in frozen_components:
                if frozen_component in name:
                    param.requires_grad = False
                    break
    
    def setup_stage(self, stage_idx):
        """Configurar una etapa específica"""
        config = self.stage_configs[stage_idx]
        
        print(f"\n🎯 {config['name']}")
        print(f"   ⏱️  Épocas: {config['epochs']}")
        print(f"   🎚️  Learning Rate: {config['lr']}")
        
        # Configurar congelamiento
        self.freeze_layers_by_name(config['frozen_components'])
        
        # Contar parámetros entrenables
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"   🔧 Parámetros entrenables: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
        
        # Crear optimizador para esta etapa
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['lr'],
            weight_decay=1e-6
        )
        
        return optimizer, config
    
    def train_stage(self, stage_idx, dataloader, criterion):
        """Entrenar una etapa específica"""
        optimizer, config = self.setup_stage(stage_idx)
        
        stage_losses = []
        
        for epoch in range(config['epochs']):
            epoch_loss = 0.0
            num_batches = 0
            
            # Barra de progreso
            pbar = tqdm(dataloader, desc=f"Época {epoch+1}/{config['epochs']}")
            
            self.model.train()
            for degraded, clean in pbar:
                degraded = degraded.to(self.device)
                clean = clean.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                restored = self.model(degraded)
                loss = criterion(restored, clean)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Estadísticas
                epoch_loss += loss.item()
                num_batches += 1
                
                # Actualizar barra de progreso
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
            avg_loss = epoch_loss / num_batches
            stage_losses.append(avg_loss)
            print(f"   📊 Época {epoch+1} - Loss promedio: {avg_loss:.6f}")
        
        # Guardar checkpoint de la etapa
        checkpoint_path = f"outputs/checkpoints/gradual_stage_{stage_idx+1}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'stage': stage_idx + 1,
            'loss': stage_losses[-1],
            'config': config
        }, checkpoint_path)
        print(f"   💾 Checkpoint guardado: {checkpoint_path}")
        
        self.training_history.extend(stage_losses)
        return stage_losses

def download_pretrained_model():
    """Descargar modelo preentrenado si no existe"""
    model_dir = "models/pretrained"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "restormer_denoising.pth")
    
    if os.path.exists(model_path):
        return model_path
    
    print("📥 Descargando modelo preentrenado Restormer...")
    
    import urllib.request
    url = "https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_blind.pth"
    
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"✅ Modelo descargado: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Error descargando modelo: {e}")
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
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
        model.to(device)
        print("✅ Modelo preentrenado cargado exitosamente")
        return model
    except Exception as e:
        print(f"❌ Error cargando modelo preentrenado: {e}")
        return None

def create_training_plot(history):
    """Crear gráfico del entrenamiento"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history, 'b-', linewidth=2)
    plt.title('Transfer Learning Gradual - Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Añadir líneas verticales para marcar las etapas
    stage_boundaries = [1, 3, 6]  # 1, 1+2, 1+2+3
    for boundary in stage_boundaries:
        if boundary < len(history):
            plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    stages = ['Stage 1\n(Output)', 'Stage 2\n(+Refinement)', 'Stage 3\n(+Blocks)', 'Stage 4\n(Full)']
    stage_losses = [
        history[0] if len(history) > 0 else 0,
        np.mean(history[1:3]) if len(history) > 2 else 0,
        np.mean(history[3:6]) if len(history) > 5 else 0,
        np.mean(history[6:10]) if len(history) > 9 else 0
    ]
    
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']
    bars = plt.bar(stages, stage_losses, color=colors, alpha=0.7)
    plt.title('Loss por Etapa')
    plt.ylabel('Loss Promedio')
    plt.xticks(rotation=45)
    
    # Añadir valores sobre las barras
    for bar, loss in zip(bars, stage_losses):
        if loss > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{loss:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Guardar
    output_path = "outputs/analysis/gradual_transfer_learning.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Gráfico guardado: {output_path}")

def main():
    """Función principal de entrenamiento"""
    print("🚀 TRANSFER LEARNING GRADUAL PARA RESTORMER")
    print("=" * 60)
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    
    # Directorios
    degraded_dir = "data/train/degraded"
    clean_dir = "data/train/clean" 
    
    if not os.path.exists(degraded_dir) or not os.path.exists(clean_dir):
        print(f"❌ Directorios de datos no encontrados")
        print(f"   Degraded: {degraded_dir}")
        print(f"   Clean: {clean_dir}")
        return
    
    # Dataset
    print("\n📁 Preparando dataset...")
    dataset = DocumentDataset(degraded_dir, clean_dir, img_size=128, augment=True)
    
    if len(dataset) == 0:
        print("❌ Dataset vacío")
        return
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    # Cargar modelo preentrenado
    print("\n🔧 Cargando modelo preentrenado...")
    model = load_pretrained_model(device)
    if model is None:
        return
    
    # Crear directorio de checkpoints
    os.makedirs("outputs/checkpoints", exist_ok=True)
    
    # Configurar Transfer Learning Gradual
    gradual_trainer = GradualTransferLearning(model, device)
    criterion = nn.L1Loss()
    
    print(f"\n🎯 INICIANDO TRANSFER LEARNING GRADUAL")
    print(f"📊 Total etapas: {len(gradual_trainer.stage_configs)}")
    print(f"📁 Samples de entrenamiento: {len(dataset)}")
    
    # Entrenar por etapas
    for stage_idx in range(len(gradual_trainer.stage_configs)):
        print(f"\n{'='*60}")
        stage_losses = gradual_trainer.train_stage(stage_idx, dataloader, criterion)
        print(f"✅ Etapa {stage_idx+1} completada - Loss final: {stage_losses[-1]:.6f}")
    
    # Guardar modelo final
    final_path = "outputs/checkpoints/gradual_transfer_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': gradual_trainer.training_history,
        'final_loss': gradual_trainer.training_history[-1],
        'method': 'Transfer Learning Gradual',
        'total_epochs': sum(config['epochs'] for config in gradual_trainer.stage_configs)
    }, final_path)
    
    print(f"\n💾 Modelo final guardado: {final_path}")
    
    # Crear visualización
    create_training_plot(gradual_trainer.training_history)
    
    # Resumen final
    print(f"\n🎉 ENTRENAMIENTO COMPLETADO")
    print(f"📊 Total épocas: {sum(config['epochs'] for config in gradual_trainer.stage_configs)}")
    print(f"📉 Loss inicial: {gradual_trainer.training_history[0]:.6f}")
    print(f"📈 Loss final: {gradual_trainer.training_history[-1]:.6f}")
    print(f"🔽 Mejora: {(gradual_trainer.training_history[0] - gradual_trainer.training_history[-1]):.6f}")
    
    print(f"\n🎯 PRÓXIMOS PASOS:")
    print(f"   1. Ejecuta: python main_pipeline.py")
    print(f"   2. Revisa: outputs/analysis/gradual_transfer_learning.png")
    print(f"   3. Compara con: python evaluation/compare_models.py")

if __name__ == "__main__":
    main()
