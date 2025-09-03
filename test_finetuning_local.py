#!/usr/bin/env python3
"""
Script para probar setup_finetuning_params localmente con Poetry
"""
import sys
import os
sys.path.append('.')
sys.path.append('./layers/train-layers/')

import torch
from train_layer_2 import setup_finetuning_params, setup_pretrained_nafnet

def test_finetuning_setup():
    """Probar la función setup_finetuning_params"""
    print("🔧 Probando setup_finetuning_params...")
    
    try:
        # Crear modelo NAFNet simple para testing
        print("📦 Creando modelo NAFNet...")
        nafnet = setup_pretrained_nafnet(width=64, use_pretrained=False)
        print(f"✅ Modelo creado: {type(nafnet)}")
        
        # Probar la función de fine-tuning
        print("🔍 Probando setup_finetuning_params...")
        result = setup_finetuning_params(
            nafnet,
            freeze_backbone=False,
            learning_rate_factor=0.1
        )
        
        print(f"✅ Resultado tipo: {type(result)}")
        print(f"✅ Resultado contenido: {result}")
        
        if isinstance(result, list):
            print(f"✅ Es lista con {len(result)} elementos")
            for i, item in enumerate(result):
                print(f"   - Elemento {i}: tipo {type(item)}")
                if isinstance(item, dict):
                    print(f"     - Keys: {list(item.keys())}")
                    if 'params' in item:
                        print(f"     - Params: {len(list(item['params']))} parámetros")
                    if 'lr_factor' in item:
                        print(f"     - LR factor: {item['lr_factor']}")
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_finetuning_setup()
