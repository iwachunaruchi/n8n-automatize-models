#!/usr/bin/env python3
"""
Script simple para probar la carga de modelo preentrenado NAFNet
"""
import sys
import os
sys.path.append('.')
import torch
import torch.nn as nn

from api.services.minio_service import minio_service

# Replicar clases necesarias directamente
class SimpleChannelAttention(nn.Module):
    """Atención de canal simplificada"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class NAFBlock(nn.Module):
    """Bloque NAF simplificado"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 2, 1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.ca = SimpleChannelAttention(channels)
        self.norm = nn.LayerNorm([channels, 1, 1])
        
    def forward(self, x):
        residual = x
        
        # Feature processing
        x = self.conv1(x)
        x = self.conv2(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2  # Simple gate
        x = self.conv3(x)
        
        # Attention
        x = self.ca(x)
        
        return residual + x

class SimpleNAFNet(nn.Module):
    """NAFNet simplificado para entrenamiento rápido"""
    def __init__(self, in_channels=3, width=64, num_blocks=4):
        super().__init__()
        
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList([
            NAFBlock(width),
            NAFBlock(width * 2),
        ])
        self.downs = nn.ModuleList([
            nn.Conv2d(width, width * 2, 2, stride=2),
        ])
        
        # Middle
        self.middle = nn.Sequential(*[NAFBlock(width * 2) for _ in range(num_blocks)])
        
        # Decoder
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(width * 2, width, 2, stride=2),
        ])
        self.decoder = nn.ModuleList([
            NAFBlock(width),
        ])
        
        self.ending = nn.Conv2d(width, in_channels, 3, padding=1)
        
    def forward(self, x):
        # Intro
        x = self.intro(x)
        
        # Encoder
        skips = []
        for i, (enc, down) in enumerate(zip(self.encoder, self.downs)):
            x = enc(x)
            skips.append(x)
            x = down(x)
        
        # Middle
        x = self.middle(x)
        
        # Decoder
        for i, (up, dec) in enumerate(zip(self.ups, self.decoder)):
            x = up(x)
            x = x + skips[-(i+1)]
            x = dec(x)
        
        # Output
        x = self.ending(x)
        return x

def load_pretrained_nafnet_simple(model_path: str, target_model: SimpleNAFNet) -> bool:
    """Cargar pesos preentrenados en un modelo NAFNet"""
    try:
        print(f"🔍 Cargando modelo preentrenado desde: {model_path}")
        
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extraer state_dict
        if isinstance(checkpoint, dict) and 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint
        
        # Obtener estado actual del modelo
        model_dict = target_model.state_dict()
        compatible_dict = {}
        
        print(f"📊 Parámetros en checkpoint: {len(state_dict)}")
        print(f"📊 Parámetros en modelo objetivo: {len(model_dict)}")
        
        # Mapeo básico - principalmente intro y ending que son compatibles
        for name, param in state_dict.items():
            if name in model_dict and param.shape == model_dict[name].shape:
                compatible_dict[name] = param
                print(f"✅ Capa compatible: {name} {param.shape}")
        
        if compatible_dict:
            # Cargar capas compatibles
            model_dict.update(compatible_dict)
            target_model.load_state_dict(model_dict, strict=False)
            print(f"✅ Transferencia exitosa: {len(compatible_dict)} capas")
            return True
        else:
            print("❌ No se encontraron capas compatibles")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_simple():
    """Prueba simple de carga de modelo"""
    print("🧪 Prueba simple de modelo NAFNet preentrenado")
    print("=" * 50)
    
    try:
        # Descargar modelo
        print("📥 Descargando modelo desde MinIO...")
        model_data = minio_service.download_file(
            bucket='models',
            filename='pretrained_models/layer_2/nafnet/NAFNet-SIDD-width64.pth'
        )
        
        # Guardar temporalmente
        temp_path = './temp_nafnet_simple.pth'
        with open(temp_path, 'wb') as f:
            f.write(model_data)
        
        print(f"💾 Modelo guardado temporalmente ({len(model_data)/(1024*1024):.1f} MB)")
        
        # Crear modelo
        print("🔧 Creando modelo SimpleNAFNet...")
        model = SimpleNAFNet(width=64)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Parámetros totales: {total_params:,}")
        
        # Intentar cargar preentrenado
        print("\n🎯 Intentando cargar pesos preentrenados...")
        success = load_pretrained_nafnet_simple(temp_path, model)
        
        if success:
            print("\n🧪 Probando inferencia...")
            dummy_input = torch.randn(1, 3, 128, 128)
            with torch.no_grad():
                output = model(dummy_input)
            print(f"✅ Inferencia exitosa: {dummy_input.shape} -> {output.shape}")
        
        # Limpiar
        os.remove(temp_path)
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple()
    if success:
        print("\n🎉 ¡Prueba exitosa! El sistema de modelos preentrenados funciona.")
    else:
        print("\n⚠️ La prueba falló. Revisar errores arriba.")
