#!/usr/bin/env python3
"""
🎨 SYNTHETIC DATA TASKS - RQ SPECIALIZED
=======================================
Tasks especializadas para generación de datos sintéticos usando RQ.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Importar utilidades RQ
from ..utils.rq_utils import RQJobProgressTracker, setup_job_environment

logger = logging.getLogger(__name__)

def generate_synthetic_data_job(num_pairs: int = 100,
                               degradation_types: List[str] = None,
                               output_bucket: str = "document-training",
                               **kwargs) -> Dict[str, Any]:
    """
    🎨 Job de generación de datos sintéticos
    
    Args:
        num_pairs: Número de pares de imágenes a generar
        degradation_types: Tipos de degradación a aplicar
        output_bucket: Bucket de destino
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultados de la generación
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    if degradation_types is None:
        degradation_types = ["blur", "noise", "compression", "fading"]
    
    logger.info(f"🎨 Iniciando generación de datos sintéticos: {num_pairs} pares")
    
    try:
        tracker.update_progress(5, "Preparando generación de datos sintéticos...")
        
        # Importar servicio de datos sintéticos
        from api.services.synthetic_data_service import synthetic_data_service
        
        generation_params = {
            'num_pairs': num_pairs,
            'degradation_types': degradation_types,
            'output_bucket': output_bucket
        }
        
        tracker.update_progress(10, "Configuración completada, iniciando generación...")
        
        # Callback de progreso para la generación
        def generation_progress_callback(current: int, total: int, message: str):
            progress = int(10 + ((current / total) * 80))
            tracker.update_progress(progress, f"Generando: {current}/{total} - {message}")
        
        # Ejecutar generación de datos sintéticos
        generation_result = synthetic_data_service.generate_training_pairs(
            num_pairs=num_pairs,
            degradation_types=degradation_types,
            output_bucket=output_bucket,
            progress_callback=generation_progress_callback
        )
        
        tracker.update_progress(95, "Finalizando y guardando metadatos...")
        
        # Preparar resultado final
        final_result = {
            'status': 'completed',
            'generation_result': generation_result,
            'parameters': generation_params,
            'completed_at': datetime.now().isoformat(),
            'job_type': 'synthetic_data_generation',
            'generated_pairs': num_pairs,
            'degradation_types': degradation_types
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, f"Generación de {num_pairs} pares completada")
        
        logger.info(f"✅ Generación de datos sintéticos completada: {num_pairs} pares")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "Synthetic Data Generation")
        logger.error(f"❌ Error en generación de datos sintéticos: {e}")
        raise

def augment_dataset_job(source_bucket: str = "document-clean",
                       output_bucket: str = "document-training",
                       augmentation_factor: int = 3,
                       **kwargs) -> Dict[str, Any]:
    """
    📈 Job de aumento de dataset
    
    Args:
        source_bucket: Bucket fuente con imágenes limpias
        output_bucket: Bucket de destino
        augmentation_factor: Factor de aumento (ej: 3 = 3x más imágenes)
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultados del aumento
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    logger.info(f"📈 Iniciando aumento de dataset: factor {augmentation_factor}x")
    
    try:
        tracker.update_progress(5, "Preparando aumento de dataset...")
        
        # Importar servicios necesarios
        from api.services.synthetic_data_service import synthetic_data_service
        from api.services.minio_service import minio_service
        
        # Listar imágenes fuente
        tracker.update_progress(10, "Obteniendo lista de imágenes fuente...")
        source_images = minio_service.list_files(source_bucket)
        
        total_images = len(source_images)
        total_to_generate = total_images * augmentation_factor
        
        augmentation_params = {
            'source_bucket': source_bucket,
            'output_bucket': output_bucket,
            'augmentation_factor': augmentation_factor,
            'total_source_images': total_images,
            'total_to_generate': total_to_generate
        }
        
        tracker.update_progress(15, f"Procesando {total_images} imágenes fuente...")
        
        generated_count = 0
        
        for i, image_name in enumerate(source_images):
            # Generar variaciones aumentadas para cada imagen
            for aug_variant in range(augmentation_factor):
                # Callback de progreso
                progress = int(15 + ((generated_count / total_to_generate) * 75))
                tracker.update_progress(
                    progress, 
                    f"Augmentando {image_name} (variante {aug_variant+1}/{augmentation_factor})"
                )
                
                # Aplicar aumentos (rotación, escala, brillo, etc.)
                augmented_result = synthetic_data_service.apply_augmentation(
                    source_bucket=source_bucket,
                    image_name=image_name,
                    output_bucket=output_bucket,
                    variant_id=aug_variant
                )
                
                generated_count += 1
        
        final_result = {
            'status': 'completed',
            'parameters': augmentation_params,
            'source_images_count': total_images,
            'generated_images_count': generated_count,
            'augmentation_factor': augmentation_factor,
            'completed_at': datetime.now().isoformat(),
            'job_type': 'dataset_augmentation'
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, f"Aumento de dataset completado: {generated_count} imágenes")
        
        logger.info(f"✅ Aumento de dataset completado: {generated_count} imágenes generadas")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "Dataset Augmentation")
        logger.error(f"❌ Error en aumento de dataset: {e}")
        raise

def validate_synthetic_data_job(bucket_name: str = "document-training",
                               sample_size: int = 50,
                               **kwargs) -> Dict[str, Any]:
    """
    ✅ Job de validación de datos sintéticos
    
    Args:
        bucket_name: Bucket con datos sintéticos
        sample_size: Tamaño de muestra para validación
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultados de la validación
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    logger.info(f"✅ Iniciando validación de datos sintéticos en {bucket_name}")
    
    try:
        tracker.update_progress(5, "Preparando validación...")
        
        # Importar servicios
        from api.services.minio_service import minio_service
        from api.services.image_analysis_service import image_analysis_service
        
        # Obtener muestra de imágenes
        tracker.update_progress(10, "Obteniendo muestra de datos...")
        all_files = minio_service.list_files(bucket_name)
        
        import random
        sample_files = random.sample(all_files, min(sample_size, len(all_files)))
        
        validation_results = {
            'total_files': len(all_files),
            'sample_size': len(sample_files),
            'valid_images': 0,
            'invalid_images': 0,
            'quality_scores': [],
            'format_issues': [],
            'size_issues': []
        }
        
        # Validar cada imagen de la muestra
        for i, file_name in enumerate(sample_files):
            progress = int(10 + ((i / len(sample_files)) * 80))
            tracker.update_progress(progress, f"Validando {file_name}")
            
            try:
                # Analizar imagen
                analysis_result = image_analysis_service.analyze_image_quality(
                    bucket_name, file_name
                )
                
                if analysis_result['is_valid']:
                    validation_results['valid_images'] += 1
                    validation_results['quality_scores'].append(analysis_result['quality_score'])
                else:
                    validation_results['invalid_images'] += 1
                    if 'format_error' in analysis_result:
                        validation_results['format_issues'].append(file_name)
                    if 'size_error' in analysis_result:
                        validation_results['size_issues'].append(file_name)
                        
            except Exception as img_error:
                logger.warning(f"Error validando {file_name}: {img_error}")
                validation_results['invalid_images'] += 1
        
        # Calcular estadísticas finales
        validation_results['average_quality'] = (
            sum(validation_results['quality_scores']) / len(validation_results['quality_scores'])
            if validation_results['quality_scores'] else 0
        )
        validation_results['validation_rate'] = (
            validation_results['valid_images'] / len(sample_files) * 100
            if sample_files else 0
        )
        
        final_result = {
            'status': 'completed',
            'validation_results': validation_results,
            'bucket_name': bucket_name,
            'sample_size': sample_size,
            'completed_at': datetime.now().isoformat(),
            'job_type': 'synthetic_data_validation'
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, f"Validación completada: {validation_results['validation_rate']:.1f}% válidas")
        
        logger.info(f"✅ Validación completada: {validation_results['validation_rate']:.1f}% imágenes válidas")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "Synthetic Data Validation")
        logger.error(f"❌ Error en validación de datos sintéticos: {e}")
        raise
