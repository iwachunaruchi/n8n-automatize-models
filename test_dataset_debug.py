#!/usr/bin/env python3
"""
Script de debug para probar el DocumentDataset independientemente
"""
import sys
import os
sys.path.append('.')
sys.path.append('./layers/train-layers/')

from train_layer_2 import DocumentDataset
import torch
from torch.utils.data import DataLoader

def test_dataset():
    """Probar el DocumentDataset directamente"""
    print("🔧 Probando DocumentDataset...")
    
    try:
        # Crear dataset de prueba
        print("📦 Creando dataset...")
        dataset = DocumentDataset(
            max_pairs=3,
            patch_size=128,
            use_training_bucket=True
        )
        
        print(f"📊 Dataset creado con {len(dataset)} elementos")
        
        # Probar acceso directo
        print("🔍 Probando acceso directo al dataset...")
        if len(dataset) > 0:
            item = dataset[0]
            print(f"✅ Item 0 tipo: {type(item)}")
            if isinstance(item, tuple):
                print(f"✅ Item es tupla con {len(item)} elementos")
                print(f"   - Elemento 0 tipo: {type(item[0])}, shape: {item[0].shape if hasattr(item[0], 'shape') else 'N/A'}")
                print(f"   - Elemento 1 tipo: {type(item[1])}, shape: {item[1].shape if hasattr(item[1], 'shape') else 'N/A'}")
            else:
                print(f"❌ Item no es tupla: {item}")
        
        # Probar DataLoader
        print("🔍 Probando DataLoader...")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        for i, batch in enumerate(dataloader):
            print(f"🔍 Batch {i} tipo: {type(batch)}")
            print(f"🔍 Batch {i} contenido: {batch if isinstance(batch, (str, int, float)) else 'tensor_data'}")
            
            if isinstance(batch, (list, tuple)):
                print(f"🔍 Batch es lista/tupla con {len(batch)} elementos")
                for j, elem in enumerate(batch):
                    print(f"   - Elemento {j} tipo: {type(elem)}, shape: {elem.shape if hasattr(elem, 'shape') else 'N/A'}")
            
            break  # Solo probar el primer batch
            
        print("✅ DataLoader funciona correctamente")
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
