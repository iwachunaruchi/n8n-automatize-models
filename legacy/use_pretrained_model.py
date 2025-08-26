#!/usr/bin/env python3
"""
Script para descargar y usar modelo Restormer preentrenado
"""

import os
import sys
import torch
import requests
from tqdm import tqdm
import cv2
import numpy as np

sys.path.append('src')
from pipeline import DocumentRestorationPipeline

def download_pretrained_model():
    """Descargar modelo Restormer preentrenado"""
    
    print("🔽 DESCARGANDO MODELO RESTORMER PREENTRENADO")
    print("=" * 55)
    
    # URLs de modelos preentrenados
    models = {
        'denoising': {
            'url': 'https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_blind.pth',
            'description': 'Eliminación de ruido (recomendado para documentos)'
        },
        'deraining': {
            'url': 'https://github.com/swz30/Restormer/releases/download/v1.0/deraining.pth', 
            'description': 'Eliminación de lluvia/artefactos'
        },
        'motion_deblur': {
            'url': 'https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth',
            'description': 'Eliminación de motion blur'
        }
    }
    
    # Crear directorio para modelos preentrenados
    pretrained_dir = "models/pretrained"
    os.makedirs(pretrained_dir, exist_ok=True)
    
    print("📋 Modelos disponibles:")
    for key, info in models.items():
        print(f"   • {key}: {info['description']}")
    
    # Descargar modelo de denoising (mejor para documentos)
    model_name = 'denoising'
    model_info = models[model_name]
    model_path = os.path.join(pretrained_dir, f"restormer_{model_name}.pth")
    
    if os.path.exists(model_path):
        print(f"\n✅ Modelo ya existe: {model_path}")
        return model_path
    
    print(f"\n🔽 Descargando modelo: {model_name}")
    print(f"📁 Destino: {model_path}")
    
    try:
        response = requests.get(model_info['url'], stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f, tqdm(
            desc="Descarga",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Descarga completada: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ Error descargando modelo: {e}")
        return None

def create_pretrained_pipeline():
    """Crear pipeline con modelo preentrenado"""
    
    print("\n🔧 CONFIGURANDO PIPELINE CON MODELO PREENTRENADO")
    print("=" * 55)
    
    # Descargar modelo preentrenado
    model_path = download_pretrained_model()
    if not model_path:
        print("❌ No se pudo descargar el modelo preentrenado")
        return None
    
    # Configurar pipeline
    pipeline = DocumentRestorationPipeline()
    pipeline.config['processing']['use_esrgan'] = False
    
    # Usar configuración estándar para modelo preentrenado
    pipeline.config['restormer'] = {
        'inp_channels': 3,
        'out_channels': 3,
        'dim': 48,  # Configuración estándar del modelo preentrenado
        'num_blocks': [4, 6, 6, 8],
        'num_refinement_blocks': 4,
        'heads': [1, 2, 4, 8],
        'ffn_expansion_factor': 2.66,
        'bias': False
    }
    
    print("🔄 Inicializando pipeline...")
    pipeline.initialize_models()
    
    print(f"📥 Cargando modelo preentrenado: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=pipeline.device)
        
        # Los modelos preentrenados pueden tener diferentes estructuras de checkpoint
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        pipeline.restormer.load_state_dict(state_dict, strict=False)
        print("✅ Modelo preentrenado cargado exitosamente!")
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        print("🔄 Usando tu modelo entrenado como alternativa...")
        
        # Fallback a tu modelo entrenado
        trained_model = "outputs/checkpoints/best_restormer.pth"
        if os.path.exists(trained_model):
            checkpoint = torch.load(trained_model)
            pipeline.restormer.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Usando tu modelo entrenado: {trained_model}")
            return pipeline
        
        return None

def test_pretrained_model():
    """Probar modelo preentrenado con tus documentos"""
    
    print("\n🧪 PROBANDO MODELO PREENTRENADO")
    print("=" * 40)
    
    # Crear pipeline preentrenado
    pipeline = create_pretrained_pipeline()
    if not pipeline:
        print("❌ No se pudo crear el pipeline")
        return
    
    # Probar con tus imágenes
    degraded_dir = "data/train/degraded"
    output_dir = "outputs/samples/pretrained_model"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener imágenes de prueba
    images = [f for f in os.listdir(degraded_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"📸 Probando con {min(3, len(images))} imágenes...")
    print("-" * 40)
    
    for i, image_name in enumerate(images[:3]):
        input_path = os.path.join(degraded_dir, image_name)
        output_path = os.path.join(output_dir, f"pretrained_{image_name}")
        
        print(f"🔍 Procesando {i+1}/3: {image_name}")
        
        try:
            # Cargar imagen
            image = cv2.imread(input_path)
            if image is None:
                print(f"   ❌ Error cargando imagen")
                continue
            
            h, w = image.shape[:2]
            print(f"   📐 Dimensiones: {w}x{h}")
            
            # Restaurar con modelo preentrenado
            print("   🧠 Aplicando modelo preentrenado...")
            restored = pipeline.restore_document(image)
            
            # Guardar resultado
            cv2.imwrite(output_path, restored)
            print(f"   ✅ Guardado: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    print("=" * 40)
    print("🎉 ¡PRUEBA COMPLETADA!")
    print(f"📁 Resultados en: {output_dir}")
    print("\n📊 Compara los modelos:")
    print("   • Tu modelo entrenado: outputs/samples/trained_model/")
    print("   • Modelo preentrenado: outputs/samples/pretrained_model/")

if __name__ == "__main__":
    test_pretrained_model()
