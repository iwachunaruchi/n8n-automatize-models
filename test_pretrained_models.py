#!/usr/bin/env python3
"""
Script de prueba para verificar carga de modelos preentrenados NAFNet
"""
import sys
import os
sys.path.append('.')

# Configurar logging
import logging
logging.basicConfig(level=logging.INFO)

def test_pretrained_model_download():
    """Probar descarga de modelo preentrenado"""
    print("🧪 Probando descarga de modelo preentrenado...")
    
    try:
        # Importar funciones
        from api.services.minio_service import minio_service
        
        # Probar descarga directa desde MinIO
        print("📥 Intentando descargar modelo desde MinIO...")
        model_data = minio_service.download_file(
            bucket='models',
            filename='pretrained_models/layer_2/nafnet/NAFNet-SIDD-width64.pth'
        )
        
        print(f"✅ Modelo descargado exitosamente!")
        print(f"   Tamaño: {len(model_data) / (1024*1024):.2f} MB")
        
        # Guardar temporalmente para verificar
        temp_path = './temp_nafnet_test.pth'
        with open(temp_path, 'wb') as f:
            f.write(model_data)
        
        print(f"💾 Modelo guardado temporalmente en: {temp_path}")
        
        # Verificar que se puede cargar con PyTorch
        import torch
        checkpoint = torch.load(temp_path, map_location='cpu')
        print(f"🔍 Tipo de checkpoint: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"🔍 Claves en checkpoint: {list(checkpoint.keys())}")
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        print(f"🔍 Número de parámetros: {len(state_dict)}")
        print(f"🔍 Primeras 5 capas:")
        for i, (name, param) in enumerate(state_dict.items()):
            if i < 5:
                print(f"   {name}: {param.shape}")
        
        # Limpiar archivo temporal
        os.remove(temp_path)
        print("🧹 Archivo temporal eliminado")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

def test_nafnet_architecture():
    """Probar arquitectura NAFNet"""
    print("\n🧪 Probando arquitectura NAFNet...")
    
    try:
        # Importar desde el archivo (usando exec porque tiene guión)
        spec_path = 'layers/train-layers/train_layer_2.py'
        with open(spec_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Crear namespace y ejecutar código
        namespace = {}
        exec(code, namespace)
        
        # Extraer clases necesarias
        SimpleNAFNet = namespace['SimpleNAFNet']
        setup_pretrained_nafnet = namespace['setup_pretrained_nafnet']
        
        print("✅ Clases importadas exitosamente")
        
        # Crear modelo simple
        print("🔧 Creando modelo NAFNet simple...")
        model_simple = SimpleNAFNet(width=32)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in model_simple.parameters())
        print(f"   Parámetros totales (width=32): {total_params:,}")
        
        # Crear modelo con setup de preentrenado (pero sin cargar)
        print("🔧 Creando modelo NAFNet con setup preentrenado...")
        model_pretrained = setup_pretrained_nafnet(width=64, use_pretrained=False)
        
        total_params_64 = sum(p.numel() for p in model_pretrained.parameters())
        print(f"   Parámetros totales (width=64): {total_params_64:,}")
        
        # Probar forward pass
        import torch
        dummy_input = torch.randn(1, 3, 128, 128)
        
        print("🧪 Probando forward pass...")
        with torch.no_grad():
            output_simple = model_simple(dummy_input)
            output_pretrained = model_pretrained(dummy_input)
        
        print(f"   Output shape (width=32): {output_simple.shape}")
        print(f"   Output shape (width=64): {output_pretrained.shape}")
        
        print("✅ Arquitectura NAFNet funciona correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba de arquitectura: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pretrained_loading():
    """Probar carga completa de modelo preentrenado"""
    print("\n🧪 Probando carga completa de modelo preentrenado...")
    
    try:
        # Importar desde el archivo
        spec_path = 'layers/train-layers/train_layer_2.py'
        with open(spec_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        namespace = {}
        exec(code, namespace)
        
        setup_pretrained_nafnet = namespace['setup_pretrained_nafnet']
        
        print("🔧 Intentando crear modelo con pesos preentrenados...")
        model = setup_pretrained_nafnet(width=64, use_pretrained=True)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Parámetros totales: {total_params:,}")
        
        # Probar forward pass
        import torch
        dummy_input = torch.randn(1, 3, 256, 256)
        
        print("🧪 Probando inferencia...")
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        print("✅ Carga de modelo preentrenado exitosa")
        return True
        
    except Exception as e:
        print(f"❌ Error en carga de preentrenado: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("🚀 Iniciando pruebas de modelos preentrenados NAFNet")
    print("=" * 60)
    
    results = []
    
    # Prueba 1: Descarga desde MinIO
    results.append(test_pretrained_model_download())
    
    # Prueba 2: Arquitectura NAFNet
    results.append(test_nafnet_architecture())
    
    # Prueba 3: Carga completa
    results.append(test_pretrained_loading())
    
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE PRUEBAS:")
    
    tests = [
        "Descarga desde MinIO",
        "Arquitectura NAFNet", 
        "Carga modelo preentrenado"
    ]
    
    for i, (test_name, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {test_name}: {status}")
    
    if all(results):
        print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
        print("   El sistema de modelos preentrenados está listo para usar.")
    else:
        print("\n⚠️ Algunas pruebas fallaron. Revisar errores arriba.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
