#!/usr/bin/env python3
"""
Script de prueba para verificar la generación de pares de entrenamiento
"""

import asyncio
import sys
from pathlib import Path

# Agregar el path del API
api_path = Path(__file__).parent / "api"
sys.path.append(str(api_path))

async def test_training_pairs():
    """Probar la generación de pares de entrenamiento"""
    try:
        # Importar el servicio
        from services.training_service import training_service
        
        print("🔄 Iniciando prueba de generación de pares de entrenamiento...")
        
        # Verificar estado actual
        print("\n📊 Verificando estado actual de datos...")
        data_status = training_service.check_layer2_data_status()
        print(f"Estado actual: {data_status}")
        
        if data_status["success"]:
            current_pairs = data_status["statistics"]["valid_pairs"]
            print(f"Pares actuales: {current_pairs}")
            
            # Intentar generar 5 pares adicionales
            target_pairs = current_pairs + 5
            print(f"\n🎯 Generando datos para alcanzar {target_pairs} pares...")
            
            result = await training_service.prepare_layer2_data(target_pairs, "document-clean")
            print(f"Resultado: {result}")
            
            # Verificar estado después
            print("\n🔍 Verificando estado después de la generación...")
            new_status = training_service.check_layer2_data_status()
            if new_status["success"]:
                new_pairs = new_status["statistics"]["valid_pairs"]
                print(f"Pares después: {new_pairs}")
                print(f"Diferencia: +{new_pairs - current_pairs}")
            
        else:
            print(f"❌ Error verificando estado: {data_status['error']}")
    
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_training_pairs())
