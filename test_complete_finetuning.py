#!/usr/bin/env python3
"""
Test completo del sistema de fine-tuning de Layer 2 localmente
"""
import sys
import os
sys.path.append('.')
sys.path.append('./layers/train-layers/')

from train_layer_2 import Layer2Trainer
import torch

def test_complete_finetuning():
    """Test completo del sistema de fine-tuning"""
    print("🔧 PROBANDO SISTEMA COMPLETO DE FINE-TUNING LAYER 2")
    print("=" * 60)
    
    try:
        # 1. Crear el trainer
        print("📦 Inicializando Layer2Trainer...")
        trainer = Layer2Trainer()
        print(f"✅ Trainer creado - Dispositivo: {trainer.device}")
        
        # 2. Probar configuración de fine-tuning
        print("\n🎯 Probando configuración de fine-tuning...")
        
        # Simular parámetros de entrenamiento
        epochs = 2
        max_pairs = 5
        batch_size = 1
        use_finetuning = True
        freeze_backbone = False
        finetuning_lr_factor = 0.1
        
        print(f"   - Épocas: {epochs}")
        print(f"   - Pares máximos: {max_pairs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Fine-tuning: {use_finetuning}")
        print(f"   - Congelar backbone: {freeze_backbone}")
        print(f"   - Factor LR: {finetuning_lr_factor}")
        
        # 3. Simular la función de entrenamiento principal
        print(f"\n🚀 Simulando entrenamiento con fine-tuning...")
        
        result = trainer.train(
            num_epochs=epochs,
            max_pairs=max_pairs,
            batch_size=batch_size,
            use_finetuning=use_finetuning,
            freeze_backbone=freeze_backbone,
            finetuning_lr_factor=finetuning_lr_factor
        )
        
        print(f"✅ Entrenamiento completado exitosamente!")
        print(f"📊 Resultado: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en test completo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_finetuning()
    print(f"\n{'✅ CÓDIGO LOCAL ARREGLADO COMPLETAMENTE' if success else '❌ CÓDIGO LOCAL TIENE PROBLEMAS'}")
