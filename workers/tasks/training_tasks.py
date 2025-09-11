#!/usr/bin/env python3
"""
🧠 TRAINING TASKS - RQ SPECIALIZED
==================================
Tasks especializadas para entrenamiento de modelos usando RQ.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Importar utilidades RQ
from ..utils.rq_utils import RQJobProgressTracker, setup_job_environment, execute_with_progress

logger = logging.getLogger(__name__)

def layer2_training_job(job_type: str = None,
                       parameters: Dict[str, Any] = None,
                       created_at: str = None,
                       status: str = None,
                       num_epochs: int = None, 
                       batch_size: int = None, 
                       max_pairs: int = None,
                       use_training_bucket: bool = None,
                       use_finetuning: bool = None,
                       freeze_backbone: bool = None,
                       finetuning_lr_factor: float = None,
                       **kwargs) -> Dict[str, Any]:
    """
    🧠 Job de entrenamiento Layer 2
    
    Args:
        job_type: Tipo de job
        parameters: Diccionario con parámetros de entrenamiento
        created_at: Timestamp de creación
        status: Estado inicial del job
        num_epochs: Número de épocas de entrenamiento (legacy)
        batch_size: Tamaño del batch (legacy)
        max_pairs: Máximo número de pares de entrenamiento (legacy)
        use_training_bucket: Usar bucket de entrenamiento (legacy)
        use_finetuning: Usar fine-tuning con modelo preentrenado
        freeze_backbone: Congelar backbone durante fine-tuning
        finetuning_lr_factor: Factor de learning rate para fine-tuning
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultados del entrenamiento
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    # Extraer parámetros del diccionario si están disponibles
    if parameters:
        num_epochs = parameters.get('num_epochs', num_epochs or 10)
        batch_size = parameters.get('batch_size', batch_size or 8)
        max_pairs = parameters.get('max_pairs', max_pairs or 1000)
        use_training_bucket = parameters.get('use_training_bucket', use_training_bucket or True)
        use_finetuning = parameters.get('use_finetuning', use_finetuning or True)
        freeze_backbone = parameters.get('freeze_backbone', freeze_backbone or False)
        finetuning_lr_factor = parameters.get('finetuning_lr_factor', finetuning_lr_factor or 0.1)
    else:
        # Valores por defecto si no hay diccionario de parámetros
        num_epochs = num_epochs or 10
        batch_size = batch_size or 8
        max_pairs = max_pairs or 1000
        use_training_bucket = use_training_bucket or True
        use_finetuning = use_finetuning or True
        freeze_backbone = freeze_backbone or False
        finetuning_lr_factor = finetuning_lr_factor or 0.1
    
    logger.info(f"🧠 Iniciando Layer 2 training: {num_epochs} épocas, batch_size: {batch_size}, fine-tuning: {use_finetuning}")
    
    try:
        # Importar servicios necesarios
        tracker.update_progress(5, "Cargando servicios de entrenamiento...")
        from api.services.training_service import training_service
        
        # Configurar parámetros de entrenamiento
        training_params = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'max_pairs': max_pairs,
            'use_training_bucket': use_training_bucket,
            'use_finetuning': use_finetuning,
            'freeze_backbone': freeze_backbone,
            'finetuning_lr_factor': finetuning_lr_factor
        }
        
        tracker.update_progress(10, "Configuración completada, iniciando entrenamiento...")
        
        # Callback de progreso para el entrenamiento
        def training_progress_callback(progress: float, message: str):
            # Mapear progreso del entrenamiento (10-90% del job total)
            job_progress = int(10 + (progress * 0.8))
            tracker.update_progress(job_progress, f"Entrenamiento: {message}")
        
        # Generar un job_id interno para el servicio de entrenamiento
        import uuid
        internal_job_id = str(uuid.uuid4())[:8]
        
        # Ejecutar entrenamiento con parámetros de fine-tuning
        import asyncio
        training_result = asyncio.run(training_service.start_layer2_training(
            job_id=internal_job_id,
            num_epochs=num_epochs,
            batch_size=batch_size,
            max_pairs=max_pairs,
            use_training_bucket=use_training_bucket,
            use_finetuning=use_finetuning,
            freeze_backbone=freeze_backbone,
            finetuning_lr_factor=finetuning_lr_factor
        ))
        
        tracker.update_progress(95, "Finalizando y guardando resultados...")
        
        # Preparar resultado final
        final_result = {
            'status': 'completed',
            'training_result': training_result,
            'parameters': training_params,
            'completed_at': datetime.now().isoformat(),
            'job_type': 'layer2_training'
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, "Entrenamiento Layer 2 completado exitosamente")
        
        logger.info(f"✅ Layer 2 training completado: {final_result['status']}")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "Layer 2 Training")
        logger.error(f"❌ Error en Layer 2 training: {e}")
        raise

def layer1_training_job(job_type: str = None,
                       parameters: Dict[str, Any] = None,
                       created_at: str = None,
                       status: str = None,
                       model_type: str = None,
                       num_epochs: int = None,
                       batch_size: int = None,
                       max_images: int = None,
                       **kwargs) -> Dict[str, Any]:
    """
    🔧 Job de entrenamiento Layer 1
    
    Args:
        job_type: Tipo de job
        parameters: Diccionario con parámetros de entrenamiento
        created_at: Timestamp de creación
        status: Estado inicial del job
        model_type: Tipo de modelo (restormer, nafnet, etc.) (legacy)
        num_epochs: Número de épocas (legacy)
        batch_size: Tamaño del batch (legacy)
        max_images: Máximo número de imágenes para evaluación
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultados del entrenamiento
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    # Extraer parámetros del diccionario si están disponibles
    if parameters:
        model_type = parameters.get('model_type', model_type or "restormer")
        num_epochs = parameters.get('num_epochs', num_epochs or 20)
        batch_size = parameters.get('batch_size', batch_size or 4)
        max_images = parameters.get('max_images', max_images or 30)
    else:
        # Valores por defecto si no hay diccionario de parámetros
        model_type = model_type or "restormer"
        num_epochs = num_epochs or 20
        batch_size = batch_size or 4
        max_images = max_images or 30
    
    logger.info(f"🔧 Iniciando Layer 1 training: {model_type}, {num_epochs} épocas, max_images: {max_images}")
    
    try:
        tracker.update_progress(5, "Preparando evaluación Layer 1...")
        
        # Importar servicio de entrenamiento
        from api.services.training_service import training_service
        
        training_params = {
            'max_images': max_images
        }
        
        tracker.update_progress(10, f"Iniciando evaluación Layer 1 con {max_images} imágenes...")
        
        # Generar un job_id interno para el servicio de entrenamiento
        import uuid
        internal_job_id = str(uuid.uuid4())[:8]
        
        # Ejecutar evaluación Layer 1 usando el servicio
        import asyncio
        training_result = asyncio.run(training_service.start_layer1_evaluation(
            job_id=internal_job_id,
            max_images=max_images
        ))
        
        final_result = {
            'status': 'completed',
            'evaluation_result': training_result,
            'parameters': training_params,
            'completed_at': datetime.now().isoformat(),
            'job_type': 'layer1_evaluation'
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, f"Evaluación Layer 1 completada con {max_images} imágenes")
        
        logger.info(f"✅ Layer 1 evaluation completado: {max_images} imágenes procesadas")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "Layer 1 Training")
        logger.error(f"❌ Error en Layer 1 training: {e}")
        raise

def fine_tuning_job(base_model: str,
                   dataset_path: str,
                   learning_rate: float = 1e-4,
                   num_epochs: int = 15,
                   **kwargs) -> Dict[str, Any]:
    """
    🎯 Job de fine-tuning de modelos
    
    Args:
        base_model: Modelo base para fine-tuning
        dataset_path: Ruta del dataset
        learning_rate: Tasa de aprendizaje
        num_epochs: Número de épocas
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultados del fine-tuning
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    logger.info(f"🎯 Iniciando Fine-tuning: {base_model}")
    
    try:
        tracker.update_progress(5, "Preparando fine-tuning...")
        
        # Simular fine-tuning (implementar lógica real aquí)
        for epoch in range(num_epochs):
            progress = int(10 + ((epoch / num_epochs) * 85))
            tracker.update_progress(progress, f"Fine-tuning época {epoch+1}/{num_epochs}")
            
            # Simular tiempo de entrenamiento
            import time
            time.sleep(2)
        
        final_result = {
            'status': 'completed',
            'base_model': base_model,
            'dataset_path': dataset_path,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'completed_at': datetime.now().isoformat(),
            'job_type': 'fine_tuning'
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, "Fine-tuning completado")
        
        logger.info(f"✅ Fine-tuning completado: {base_model}")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "Fine-tuning")
        logger.error(f"❌ Error en fine-tuning: {e}")
        raise
