#!/usr/bin/env python3
"""
Test rápido para verificar que el código local está arreglado
"""
import sys
import os
sys.path.append('.')
sys.path.append('./layers/train-layers/')

from train_layer_2 import Layer2Trainer, DocumentDataset
import torch
from torch.utils.data import DataLoader

def test_quick_verification():
    """Test rápido para verificar correcciones"""
    print("🔧 VERIFICACIÓN RÁPIDA DEL CÓDIGO LOCAL")
    print("=" * 50)
    
    try:
        # 1. Test Dataset
        print("📦 Probando DocumentDataset...")
        dataset = DocumentDataset(max_pairs=2, patch_size=128)
        print(f"✅ Dataset creado con {len(dataset)} elementos")
        
        if len(dataset) > 0:
            # Probar acceso directo
            item = dataset[0]
            print(f"✅ Item [0] tipo: {type(item)} - {'Tupla' if isinstance(item, tuple) else 'Otro'}")
            
            # Probar DataLoader
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            for i, batch in enumerate(dataloader):
                print(f"✅ Batch tipo: {type(batch)} - {'Lista' if isinstance(batch, list) else 'Otro'}")
                
                # Simular el manejo de batch como en train_epoch
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    degraded, clean = batch[0], batch[1]
                    print(f"✅ Batch handling OK - degraded shape: {degraded.shape}, clean shape: {clean.shape}")
                else:
                    print(f"❌ Batch handling FAIL - batch: {batch}")
                    return False
                break
        
        # 2. Test Trainer
        print("\n🚀 Probando Layer2Trainer...")
        trainer = Layer2Trainer()
        print(f"✅ Trainer creado en dispositivo: {trainer.device}")
        
        # 3. Test setup_finetuning_params
        print("\n🎯 Probando setup_finetuning_params...")
        from train_layer_2 import setup_finetuning_params
        
        result = setup_finetuning_params(
            model=trainer.nafnet,
            freeze_backbone=False,
            learning_rate_factor=0.1
        )
        
        if isinstance(result, list) and len(result) == 2:
            print(f"✅ setup_finetuning_params OK - {len(result)} grupos de parámetros")
            for i, group in enumerate(result):
                if isinstance(group, dict) and 'params' in group and 'lr_factor' in group:
                    print(f"   - Grupo {i}: ✅ Formato correcto")
                else:
                    print(f"   - Grupo {i}: ❌ Formato incorrecto - {group}")
                    return False
        else:
            print(f"❌ setup_finetuning_params FAIL - resultado: {result}")
            return False
        
        print(f"\n✅ TODAS LAS VERIFICACIONES PASARON")
        return True
        
    except Exception as e:
        print(f"❌ Error en verificación: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quick_verification()
    print(f"\n{'🎉 CÓDIGO LOCAL ESTÁ COMPLETAMENTE ARREGLADO' if success else '💥 CÓDIGO LOCAL TIENE PROBLEMAS'}")
