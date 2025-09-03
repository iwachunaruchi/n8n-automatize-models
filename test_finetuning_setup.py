#!/usr/bin/env python3
"""
Test simple para verificar setup_finetuning_params
"""
import sys
sys.path.append('/app/layers/train-layers')

from train_layer_2 import setup_finetuning_params, NAFNet, SimpleDocUNet
import torch

def test_setup_finetuning():
    print("🔧 Probando setup_finetuning_params...")
    
    try:
        # Crear modelo simple
        model = NAFNet(width=64)
        print(f"✅ Modelo NAFNet creado")
        
        # Probar setup_finetuning_params
        result = setup_finetuning_params(
            model=model,
            freeze_backbone=False,
            learning_rate_factor=0.1
        )
        
        print(f"🔍 Resultado tipo: {type(result)}")
        print(f"🔍 Resultado contenido: {result}")
        
        if isinstance(result, list):
            print(f"✅ Es lista con {len(result)} elementos")
            for i, group in enumerate(result):
                print(f"   - Grupo {i}: tipo {type(group)}, keys: {group.keys() if isinstance(group, dict) else 'No es dict'}")
        else:
            print(f"❌ No es lista: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_setup_finetuning()
