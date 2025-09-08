#!/usr/bin/env python3
"""
🎯 JOBS SIMPLES PARA TESTS
==========================
Funciones de jobs que pueden ser procesadas por RQ workers.
"""

import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def simple_test_job(message="Test job", duration=3, **kwargs):
    """Job simple para pruebas"""
    # Ignorar parámetros adicionales como timeout
    logger.info(f"🚀 Job iniciado: {message}")
    
    for i in range(duration):
        logger.info(f"📈 Progreso: {i+1}/{duration}")
        time.sleep(1)
    
    result = {
        'message': message,
        'duration': duration,
        'completed_at': datetime.now().isoformat(),
        'status': 'completed'
    }
    
    logger.info(f"✅ Job completado: {result}")
    return result

def math_job(operation="add", a=10, b=5, **kwargs):
    """Job matemático simple"""
    # Ignorar parámetros adicionales como timeout
    logger.info(f"🔢 Operación matemática: {operation}({a}, {b})")
    
    if operation == "add":
        result = a + b
    elif operation == "multiply":
        result = a * b
    elif operation == "power":
        result = a ** b
    else:
        result = 0
    
    time.sleep(2)  # Simular procesamiento
    
    response = {
        'operation': operation,
        'a': a,
        'b': b,
        'result': result,
        'completed_at': datetime.now().isoformat()
    }
    
    logger.info(f"✅ Operación completada: {response}")
    return response

def simulate_training_job(epochs=5, batch_size=32, **kwargs):
    """Simular job de entrenamiento"""
    # Ignorar parámetros adicionales como timeout
    logger.info(f"🧠 Simulando entrenamiento: {epochs} épocas")
    
    results = []
    
    for epoch in range(epochs):
        # Simular época de entrenamiento
        loss = 1.0 - (epoch * 0.15)  # Loss decreciente
        accuracy = 0.5 + (epoch * 0.1)  # Accuracy creciente
        
        epoch_result = {
            'epoch': epoch + 1,
            'loss': round(loss, 4),
            'accuracy': round(accuracy, 4),
            'batch_size': batch_size
        }
        
        results.append(epoch_result)
        logger.info(f"📊 Época {epoch + 1}/{epochs}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
        time.sleep(1)  # Simular tiempo de entrenamiento
    
    final_result = {
        'status': 'completed',
        'total_epochs': epochs,
        'batch_size': batch_size,
        'training_history': results,
        'final_loss': results[-1]['loss'],
        'final_accuracy': results[-1]['accuracy'],
        'completed_at': datetime.now().isoformat()
    }
    
    logger.info(f"✅ Entrenamiento completado: {final_result['final_accuracy']:.1%} accuracy")
    return final_result
