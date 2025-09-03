#!/usr/bin/env python3
"""
Script simple para investigar la estructura del modelo NAFNet descargado
"""
import sys
sys.path.append('.')

from api.services.minio_service import minio_service
import torch
import os

def investigate_nafnet_model():
    """Investigar estructura del modelo NAFNet"""
    print("🔍 Investigando estructura del modelo NAFNet...")
    
    try:
        # Descargar modelo
        print("📥 Descargando modelo...")
        model_data = minio_service.download_file(
            bucket='models',
            filename='pretrained_models/layer_2/nafnet/NAFNet-SIDD-width64.pth'
        )
        
        # Guardar temporalmente
        temp_path = './temp_investigate.pth'
        with open(temp_path, 'wb') as f:
            f.write(model_data)
        
        # Cargar con PyTorch
        print("🔍 Cargando checkpoint...")
        checkpoint = torch.load(temp_path, map_location='cpu')
        
        print(f"Tipo de checkpoint: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Claves principales: {list(checkpoint.keys())}")
            
            for key, value in checkpoint.items():
                print(f"\nClave '{key}':")
                print(f"  Tipo: {type(value)}")
                
                if hasattr(value, 'keys') and callable(value.keys):
                    print(f"  Sub-claves: {list(value.keys())[:10]}...")  # Primeras 10
                elif hasattr(value, '__len__'):
                    print(f"  Longitud: {len(value)}")
                
                if key == 'params' and hasattr(value, 'keys'):
                    print(f"  Parámetros en 'params':")
                    for i, (param_name, param_value) in enumerate(value.items()):
                        if i < 10:  # Primeros 10 parámetros
                            if hasattr(param_value, 'shape'):
                                print(f"    {param_name}: {param_value.shape}")
                            else:
                                print(f"    {param_name}: {type(param_value)}")
        
        # Limpiar
        os.remove(temp_path)
        print("\n✅ Investigación completada")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate_nafnet_model()
