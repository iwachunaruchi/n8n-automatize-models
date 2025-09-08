#!/usr/bin/env python3
"""
🧪 TEST DE LA NUEVA ESTRUCTURA MODULAR
======================================
Script para probar la nueva organización de workers.
"""

import sys
import os

# Agregar paths necesarios
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

def test_imports():
    """Probar que todas las importaciones funcionen"""
    print("🧪 TESTING NUEVA ESTRUCTURA MODULAR")
    print("=" * 50)
    
    try:
        # Test 1: Importar utilidades
        print("📦 Test 1: Importando utilidades...")
        from workers.utils.rq_utils import RQJobProgressTracker, setup_job_environment
        print("✅ Utilidades importadas correctamente")
        
        # Test 2: Importar tasks individuales
        print("\n📦 Test 2: Importando tasks individuales...")
        from workers.tasks.test_tasks import simple_test_job, math_calculation_job
        print("✅ Test tasks importadas correctamente")
        
        from workers.tasks.training_tasks import layer2_training_job
        print("✅ Training tasks importadas correctamente")
        
        from workers.tasks.synthetic_data_tasks import generate_synthetic_data_job
        print("✅ Synthetic data tasks importadas correctamente")
        
        from workers.tasks.restoration_tasks import single_document_restoration_job
        print("✅ Restoration tasks importadas correctamente")
        
        # Test 3: Registry de tasks
        print("\n📦 Test 3: Probando registry...")
        from workers.tasks import AVAILABLE_TASKS, list_available_tasks, get_task_info
        
        total_tasks = len(AVAILABLE_TASKS)
        available = list_available_tasks()
        info = get_task_info()
        
        print(f"📊 Total tasks registradas: {total_tasks}")
        print(f"📋 Tasks disponibles: {len(available)}")
        print(f"🏷️ Categorías: {len(info)}")
        
        # Test 4: Ejecutar una task simple
        print("\n📦 Test 4: Ejecutando task de prueba...")
        
        # Ejecutar test job directamente
        result = simple_test_job(message="Test estructura modular", duration=2)
        print(f"✅ Task ejecutada: {result['status']}")
        
        print("\n🎉 ¡Todos los tests pasaron! Estructura modular funcionando correctamente.")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_task_categories():
    """Probar las categorías de tasks"""
    try:
        from workers.tasks import get_task_info
        
        print("\n📋 CATEGORÍAS DE TASKS:")
        print("-" * 30)
        
        info = get_task_info()
        for category, tasks in info.items():
            print(f"\n🔸 {category.upper().replace('_', ' ')}:")
            for task_name, description in tasks.items():
                print(f"  • {task_name}: {description}")
        
        return True
    except Exception as e:
        print(f"❌ Error mostrando categorías: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        test_task_categories()
    
    print(f"\n{'✅ ÉXITO' if success else '❌ FALLO'}: Testing de estructura modular {'completado' if success else 'falló'}")
