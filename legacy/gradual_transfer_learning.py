#!/usr/bin/env python3
"""
Implementación de Transfer Learning Gradual para Restormer
Estrategia avanzada de fine-tuning por etapas
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
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt

# Importar nuestros modelos
from models.restormer import Restormer

class GradualTransferLearning:
    """
    Implementa Transfer Learning Gradual para Restormer
    
    ETAPAS:
    1. Congelar todo → Entrenar solo clasificador final
    2. Descongelar últimas capas → Entrenar refinement blocks  
    3. Descongelar capas medias → Entrenar algunos transformer blocks
    4. Descongelar todo → Fine-tuning completo con LR muy bajo
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.stage = 0
        self.stage_configs = [
            {
                'name': 'ETAPA 1: Solo Refinement Blocks',
                'frozen_params': ['encoder', 'decoder', 'bottleneck'],
                'trainable_params': ['refinement'],
                'learning_rate': 1e-4,
                'epochs': 3,
                'description': 'Entrena solo los bloques de refinamiento final'
            },
            {
                'name': 'ETAPA 2: Decoder + Refinement',
                'frozen_params': ['encoder', 'bottleneck'],
                'trainable_params': ['decoder', 'refinement'],
                'learning_rate': 5e-5,
                'epochs': 3,
                'description': 'Añade el decoder manteniendo encoder congelado'
            },
            {
                'name': 'ETAPA 3: Bottleneck + Decoder + Refinement',
                'frozen_params': ['encoder'],
                'trainable_params': ['bottleneck', 'decoder', 'refinement'],
                'learning_rate': 2e-5,
                'epochs': 3,
                'description': 'Añade bottleneck, mantiene encoder congelado'
            },
            {
                'name': 'ETAPA 4: Fine-tuning Completo',
                'frozen_params': [],
                'trainable_params': ['encoder', 'bottleneck', 'decoder', 'refinement'],
                'learning_rate': 1e-5,
                'epochs': 5,
                'description': 'Entrena todo el modelo con LR ultra bajo'
            }
        ]
    
    def freeze_layers_by_name(self, frozen_components):
        """Congelar componentes específicos del modelo"""
        
        total_params = 0
        frozen_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            
            # Determinar si este parámetro debe estar congelado
            should_freeze = False
            for component in frozen_components:
                if component in name.lower():
                    should_freeze = True
                    break
            
            param.requires_grad = not should_freeze
            if should_freeze:
                frozen_params += param.numel()
        
        trainable_params = total_params - frozen_params
        print(f"   🧊 Parámetros congelados: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"   🔥 Parámetros entrenables: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        
        return trainable_params
    
    def setup_stage(self, stage_idx):
        """Configurar una etapa específica del transfer learning gradual"""
        
        if stage_idx >= len(self.stage_configs):
            print(f"❌ Etapa {stage_idx} no existe")
            return None
        
        config = self.stage_configs[stage_idx]
        
        print(f"\n🎯 {config['name']}")
        print("=" * 60)
        print(f"📝 {config['description']}")
        print(f"🔥 Learning Rate: {config['learning_rate']}")
        print(f"🔄 Épocas: {config['epochs']}")
        
        # Configurar qué parámetros entrenar
        trainable_params = self.freeze_layers_by_name(config['frozen_params'])
        
        # Configurar optimizador para esta etapa
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['learning_rate'],
            weight_decay=1e-4
        )
        
        # Scheduler específico para cada etapa
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['learning_rate']/10
        )
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'config': config,
            'trainable_params': trainable_params
        }
    
    def print_model_structure(self):
        """Mostrar estructura del modelo para entender las capas"""
        
        print(f"\n🔍 ESTRUCTURA DEL MODELO RESTORMER:")
        print("=" * 50)
        
        component_params = {}
        
        for name, param in self.model.named_parameters():
            # Clasificar parámetros por componente
            if 'encoder' in name.lower():
                component = 'Encoder'
            elif 'decoder' in name.lower():
                component = 'Decoder'
            elif 'bottleneck' in name.lower():
                component = 'Bottleneck'
            elif 'refinement' in name.lower():
                component = 'Refinement'
            else:
                component = 'Other'
            
            if component not in component_params:
                component_params[component] = 0
            component_params[component] += param.numel()
        
        total_params = sum(component_params.values())
        
        for component, params in component_params.items():
            percentage = params / total_params * 100
            print(f"📊 {component}: {params:,} parámetros ({percentage:.1f}%)")

class AdvancedDocumentDataset(Dataset):
    """Dataset optimizado para transfer learning gradual"""
    
    def __init__(self, degraded_dir, clean_dir, img_size=128, augment=True):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.img_size = img_size
        self.augment = augment
        
        # Obtener pares válidos
        self.valid_pairs = []
        degraded_images = [f for f in os.listdir(degraded_dir) if f.endswith('.png')]
        clean_images = [f for f in os.listdir(clean_dir) if f.endswith('.png')]
        
        for deg_img in degraded_images:
            base_name = deg_img.replace('_deg_heavy', '').replace('_deg_medium', '').replace('_deg_light', '')
            base_name = base_name.replace('_var_1', '').replace('_var_2', '').replace('_var_3', '')
            
            for clean_img in clean_images:
                if base_name in clean_img or clean_img.replace('val_', '') == base_name:
                    self.valid_pairs.append((deg_img, clean_img))
                    break
        
        print(f"📊 Dataset: {len(self.valid_pairs)} pares válidos")
    
    def __len__(self):
        return len(self.valid_pairs) * (4 if self.augment else 1)
    
    def __getitem__(self, idx):
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
        
        # Aplicar augmentation según tipo
        if aug_type == 1:
            degraded = cv2.flip(degraded, 1)
            clean = cv2.flip(clean, 1)
        elif aug_type == 2:
            degraded = cv2.flip(degraded, 0)
            clean = cv2.flip(clean, 0)
        elif aug_type == 3:
            degraded = cv2.rotate(degraded, cv2.ROTATE_180)
            clean = cv2.rotate(clean, cv2.ROTATE_180)
        
        # Procesamiento
        degraded = cv2.resize(degraded, (self.img_size, self.img_size))
        clean = cv2.resize(clean, (self.img_size, self.img_size))
        
        degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        degraded = torch.from_numpy(degraded).permute(2, 0, 1)
        clean = torch.from_numpy(clean).permute(2, 0, 1)
        
        return degraded, clean

def download_pretrained_model():
    """Descargar modelo preentrenado"""
    model_dir = "models/pretrained"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "restormer_denoising.pth")
    
    if os.path.exists(model_path):
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
        
        return model_path
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def load_pretrained_model(device):
    """Cargar modelo preentrenado"""
    model_path = download_pretrained_model()
    if not model_path:
        return None
    
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
        return model
        
    except Exception as e:
        print(f"❌ Error: {e}")
        model = model.to(device)
        return model

def gradual_transfer_learning():
    """Implementar transfer learning gradual completo"""
    
    print("🚀 TRANSFER LEARNING GRADUAL")
    print("=" * 60)
    print("📚 CONCEPTO: Entrenar por etapas, desbloqueando progresivamente las capas")
    print("🎯 OBJETIVO: Preservar conocimiento preentrenado + adaptarse a tus datos")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Dispositivo: {device}")
    
    # Cargar modelo preentrenado
    model = load_pretrained_model(device)
    if model is None:
        return None
    
    # Crear gestor de transfer learning gradual
    gtl = GradualTransferLearning(model, device)
    
    # Mostrar estructura del modelo
    gtl.print_model_structure()
    
    # Dataset
    dataset = AdvancedDocumentDataset(
        degraded_dir="data/train/degraded",
        clean_dir="data/train/clean",
        img_size=128,
        augment=True
    )
    
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0
    )
    
    # Loss function
    criterion = nn.L1Loss()
    
    # Historial de entrenamiento
    training_history = {
        'stages': [],
        'losses': [],
        'learning_rates': []
    }
    
    print(f"\n🎭 INICIANDO TRANSFER LEARNING GRADUAL POR ETAPAS")
    print("=" * 60)
    
    # Ejecutar cada etapa
    for stage_idx in range(len(gtl.stage_configs)):
        
        # Configurar etapa
        stage_setup = gtl.setup_stage(stage_idx)
        if stage_setup is None:
            continue
        
        optimizer = stage_setup['optimizer']
        scheduler = stage_setup['scheduler']
        config = stage_setup['config']
        
        model.train()
        stage_losses = []
        
        # Entrenar esta etapa
        for epoch in range(config['epochs']):
            running_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f"Etapa {stage_idx+1} - Época {epoch+1}/{config['epochs']}")
            
            for degraded, clean in progress_bar:
                degraded = degraded.to(device)
                clean = clean.to(device)
                
                optimizer.zero_grad()
                
                try:
                    restored = model(degraded)
                    loss = criterion(restored, clean)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), 
                        max_norm=0.5
                    )
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
                        print(f"⚠️ CUDA OOM en etapa {stage_idx+1}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            # Calcular loss de la época
            epoch_loss = running_loss / num_batches if num_batches > 0 else float('inf')
            stage_losses.append(epoch_loss)
            
            # Scheduler step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"✅ Etapa {stage_idx+1} - Época {epoch+1}/{config['epochs']} - Loss: {epoch_loss:.6f} - LR: {current_lr:.2e}")
        
        # Guardar progreso de la etapa
        training_history['stages'].append(config['name'])
        training_history['losses'].append(stage_losses)
        training_history['learning_rates'].append([optimizer.param_groups[0]['lr']])
        
        # Guardar checkpoint de etapa
        stage_checkpoint_path = f"outputs/checkpoints/gradual_stage_{stage_idx+1}.pth"
        os.makedirs(os.path.dirname(stage_checkpoint_path), exist_ok=True)
        
        torch.save({
            'stage': stage_idx + 1,
            'model_state_dict': model.state_dict(),
            'stage_config': config,
            'stage_losses': stage_losses,
            'trainable_params': stage_setup['trainable_params']
        }, stage_checkpoint_path)
        
        print(f"💾 Checkpoint etapa {stage_idx+1} guardado")
        print(f"📉 Loss promedio etapa: {np.mean(stage_losses):.6f}")
    
    # Guardar modelo final
    final_model_path = "outputs/checkpoints/gradual_transfer_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': training_history,
        'method': 'gradual_transfer_learning',
        'stages_completed': len(gtl.stage_configs)
    }, final_model_path)
    
    print(f"\n🎉 TRANSFER LEARNING GRADUAL COMPLETADO!")
    print(f"💾 Modelo final: {final_model_path}")
    
    # Crear gráfico de progreso por etapas
    create_gradual_learning_plot(training_history)
    
    return model, final_model_path

def create_gradual_learning_plot(history):
    """Crear gráfico mostrando el progreso por etapas"""
    
    plt.figure(figsize=(15, 10))
    
    # Plot principal: Loss por etapa
    plt.subplot(2, 2, 1)
    all_losses = []
    stage_boundaries = []
    current_epoch = 0
    
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for i, (stage_name, stage_losses) in enumerate(zip(history['stages'], history['losses'])):
        epochs = list(range(current_epoch, current_epoch + len(stage_losses)))
        plt.plot(epochs, stage_losses, 'o-', color=colors[i % len(colors)], 
                label=f'Etapa {i+1}', linewidth=2, markersize=4)
        
        all_losses.extend(stage_losses)
        stage_boundaries.append(current_epoch + len(stage_losses))
        current_epoch += len(stage_losses)
    
    # Líneas verticales para separar etapas
    for boundary in stage_boundaries[:-1]:
        plt.axvline(x=boundary-0.5, color='gray', linestyle='--', alpha=0.7)
    
    plt.title('Transfer Learning Gradual: Pérdida por Etapa')
    plt.xlabel('Época Global')
    plt.ylabel('Loss (L1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Comparación promedio por etapa
    plt.subplot(2, 2, 2)
    stage_avg_losses = [np.mean(losses) for losses in history['losses']]
    stage_names = [f'Etapa {i+1}' for i in range(len(stage_avg_losses))]
    
    bars = plt.bar(stage_names, stage_avg_losses, color=colors[:len(stage_avg_losses)])
    plt.title('Loss Promedio por Etapa')
    plt.ylabel('Loss Promedio')
    plt.xticks(rotation=45)
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Subplot 3: Descripción del método
    plt.subplot(2, 2, 3)
    plt.text(0.1, 0.9, "🎯 TRANSFER LEARNING GRADUAL", fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, "Estrategia por etapas:", fontsize=12, transform=plt.gca().transAxes)
    
    descriptions = [
        "Etapa 1: Solo refinement blocks",
        "Etapa 2: + Decoder",
        "Etapa 3: + Bottleneck", 
        "Etapa 4: Modelo completo"
    ]
    
    for i, desc in enumerate(descriptions):
        plt.text(0.1, 0.7 - i*0.1, f"{i+1}. {desc}", fontsize=10, transform=plt.gca().transAxes)
    
    plt.text(0.1, 0.2, "✅ Preserva conocimiento preentrenado", fontsize=10, color='green', transform=plt.gca().transAxes)
    plt.text(0.1, 0.1, "✅ Adaptación gradual a tus datos", fontsize=10, color='green', transform=plt.gca().transAxes)
    
    plt.axis('off')
    
    # Subplot 4: Métricas finales
    plt.subplot(2, 2, 4)
    final_loss = history['losses'][-1][-1] if history['losses'] else 0
    total_epochs = sum(len(losses) for losses in history['losses'])
    
    plt.text(0.1, 0.8, "📊 MÉTRICAS FINALES", fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Loss final: {final_loss:.6f}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"Épocas totales: {total_epochs}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Etapas completadas: {len(history['stages'])}", fontsize=12, transform=plt.gca().transAxes)
    
    plt.text(0.1, 0.2, "🎯 Ventajas del método gradual:", fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.1, "• Estabilidad en entrenamiento", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.05, "• Preservación de características", fontsize=10, transform=plt.gca().transAxes)
    
    plt.axis('off')
    
    plt.tight_layout()
    
    plot_path = "outputs/analysis/gradual_transfer_learning.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Análisis gradual guardado: {plot_path}")

if __name__ == "__main__":
    print("🎓 TRANSFER LEARNING GRADUAL")
    print("=" * 50)
    print("📚 CONCEPTO:")
    print("   • Entrenar por etapas progresivas")
    print("   • Desbloquear capas gradualmente") 
    print("   • Preservar conocimiento preentrenado")
    print("   • Adaptarse suavemente a datos específicos")
    print()
    print("🔄 PROCESO:")
    print("   1. Congelar encoder → Entrenar solo refinement")
    print("   2. Añadir decoder → Mantener encoder congelado")
    print("   3. Añadir bottleneck → Encoder aún congelado")
    print("   4. Desbloquear todo → Fine-tuning completo")
    
    model, model_path = gradual_transfer_learning()
    
    if model and model_path:
        print(f"\n🎯 SIGUIENTE PASO:")
        print(f"   Ejecuta: python test_gradual_model.py")
        print(f"   Para comparar transfer learning gradual vs otros métodos")
