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

def generate_synthetic_data_job(clean_bucket: str,
                               requested_count: int,
                               job_type: str = "training_pairs_generation",
                               **kwargs) -> Dict[str, Any]:
    """
    🎨 Job de generación de datos sintéticos
    
    Args:
        clean_bucket: Bucket con imágenes limpias
        requested_count: Número de pares a generar
        job_type: Tipo de trabajo
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultados de la generación
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    logger.info(f"🎨 Iniciando generación de datos sintéticos: {requested_count} pares desde {clean_bucket}")
    
    try:
        tracker.update_progress(5, "Preparando generación de datos sintéticos...")
        
        # Importar servicio de datos sintéticos
        from api.services.synthetic_data_service import synthetic_data_service
        
        tracker.update_progress(10, "Configuración completada, iniciando generación...")
        
        # Ejecutar generación de datos sintéticos con los parámetros correctos
        generation_result = synthetic_data_service.generate_training_pairs(
            clean_bucket=clean_bucket,
            count=requested_count
        )
        
        tracker.update_progress(95, "Finalizando y guardando metadatos...")
        
        # Preparar resultado final
        final_result = {
            'status': 'completed',
            'generation_result': generation_result,
            'clean_bucket': clean_bucket,
            'requested_count': requested_count,
            'completed_at': datetime.now().isoformat(),
            'job_type': job_type,
            'generated_pairs': requested_count
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, f"Generación de {requested_count} pares completada")
        
        logger.info(f"✅ Generación de datos sintéticos completada: {requested_count} pares")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "Synthetic Data Generation")
        logger.error(f"❌ Error en generación de datos sintéticos: {e}")
        raise

def augment_dataset_job(bucket: str,
                       target_count: int,
                       job_type: str = "dataset_augmentation",
                       **kwargs) -> Dict[str, Any]:
    """
    📈 Job de aumento de dataset
    
    Args:
        bucket: Bucket con imágenes a aumentar
        target_count: Número objetivo de imágenes
        job_type: Tipo de trabajo
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultados del aumento
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    logger.info(f"📈 Iniciando aumento de dataset: objetivo {target_count} imágenes desde {bucket}")
    
    try:
        tracker.update_progress(5, "Preparando aumento de dataset...")
        
        # Importar servicio de datos sintéticos
        from api.services.synthetic_data_service import synthetic_data_service
        
        tracker.update_progress(10, "Configuración completada, iniciando augmentación...")
        
        # Ejecutar augmentación usando el servicio
        augmentation_result = synthetic_data_service.augment_dataset(
            bucket=bucket,
            target_count=target_count
        )
        
        tracker.update_progress(95, "Finalizando augmentación...")
        
        final_result = {
            'status': 'completed',
            'augmentation_result': augmentation_result,
            'bucket': bucket,
            'target_count': target_count,
            'completed_at': datetime.now().isoformat(),
            'job_type': job_type
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, f"Aumento de dataset completado: {target_count} objetivo")
        
        logger.info(f"✅ Aumento de dataset completado para bucket {bucket}")
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

def generate_nafnet_dataset_job(source_bucket: str,
                               count: int,
                               task: str = "SIDD-width64",
                               train_val_split: bool = True,
                               **kwargs) -> Dict[str, Any]:
    """
    🎯 Job de generación de dataset NAFNet estructurado
    
    Args:
        source_bucket: Bucket con imágenes fuente
        count: Número total de pares a generar
        task: Tarea NAFNet específica
        train_val_split: Si dividir en train/val
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultados de la generación NAFNet
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    logger.info(f"🎯 Iniciando generación de dataset NAFNet: {count} pares para tarea '{task}'")
    
    try:
        tracker.update_progress(5, f"Preparando generación dataset NAFNet para {task}...")
        
        # Importar servicio de datos sintéticos
        from api.services.synthetic_data_service import synthetic_data_service
        
        tracker.update_progress(10, "Configuración completada, iniciando generación NAFNet...")
        
        # Ejecutar generación de dataset NAFNet estructurado
        generation_result = synthetic_data_service.generate_nafnet_training_dataset(
            source_bucket=source_bucket,
            count=count,
            task=task,
            train_val_split=train_val_split
        )
        
        tracker.update_progress(95, "Finalizando generación NAFNet...")
        
        # Preparar resultado final
        final_result = {
            'status': 'completed',
            'generation_result': generation_result,
            'source_bucket': source_bucket,
            'count': count,
            'task': task,
            'train_val_split': train_val_split,
            'completed_at': datetime.now().isoformat(),
            'job_type': 'nafnet_dataset_generation',
            'dataset_structure': generation_result.get('dataset_structure', 'N/A'),
            'total_generated': generation_result.get('total_generated', 0)
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, f"Dataset NAFNet generado: {generation_result.get('total_generated', 0)} pares")
        
        logger.info(f"✅ Dataset NAFNet '{task}' completado: {generation_result.get('total_generated', 0)} pares")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "NAFNet Dataset Generation")
        logger.error(f"❌ Error en generación dataset NAFNet: {e}")
        raise

def validate_nafnet_dataset_job(task: str = "SIDD-width64",
                               **kwargs) -> Dict[str, Any]:
    """
    ✅ Job de validación de dataset NAFNet
    
    Args:
        task: Tarea NAFNet a validar
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultados de la validación NAFNet
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    logger.info(f"✅ Iniciando validación de dataset NAFNet para tarea '{task}'")
    
    try:
        tracker.update_progress(5, f"Preparando validación NAFNet para {task}...")
        
        # Importar servicios
        from api.services.synthetic_data_service import synthetic_data_service
        from api.services.minio_service import minio_service
        
        tracker.update_progress(10, "Obteniendo información del dataset...")
        
        # Obtener información del dataset NAFNet
        dataset_info = synthetic_data_service.get_nafnet_dataset_info(task)
        
        if dataset_info["status"] != "success":
            raise Exception(f"Error obteniendo info dataset: {dataset_info['message']}")
        
        dataset_stats = dataset_info["dataset_info"]
        
        tracker.update_progress(30, "Validando estructura de directorios...")
        
        # Validar estructura completa
        validation_results = {
            'task': task,
            'structure_valid': True,
            'train_pairs': dataset_stats['structure']['complete_pairs']['train'],
            'val_pairs': dataset_stats['structure']['complete_pairs']['val'],
            'total_pairs': dataset_stats['total_pairs'],
            'issues': [],
            'recommendations': []
        }
        
        # Verificar que hay pares suficientes
        if validation_results['train_pairs'] < 10:
            validation_results['issues'].append("Muy pocos pares de entrenamiento (< 10)")
            validation_results['recommendations'].append("Generar más pares de entrenamiento")
        
        if validation_results['val_pairs'] < 2:
            validation_results['issues'].append("Muy pocos pares de validación (< 2)")
            validation_results['recommendations'].append("Generar más pares de validación")
        
        # Verificar balance train/val
        total_pairs = validation_results['train_pairs'] + validation_results['val_pairs']
        if total_pairs > 0:
            val_ratio = validation_results['val_pairs'] / total_pairs
            if val_ratio < 0.1 or val_ratio > 0.3:
                validation_results['issues'].append(f"Ratio de validación no óptimo: {val_ratio:.2f}")
                validation_results['recommendations'].append("Ratio recomendado: 0.15-0.25")
        
        tracker.update_progress(80, "Validando integridad de archivos...")
        
        # Aquí se pueden agregar más validaciones específicas
        # Por ejemplo, verificar que las imágenes lq y gt coinciden en número
        
        validation_results['structure_valid'] = len(validation_results['issues']) == 0
        validation_results['health_score'] = max(0, 100 - len(validation_results['issues']) * 20)
        
        final_result = {
            'status': 'completed',
            'validation_results': validation_results,
            'task': task,
            'completed_at': datetime.now().isoformat(),
            'job_type': 'nafnet_dataset_validation'
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, f"Validación NAFNet completada: {validation_results['health_score']}% saludable")
        
        logger.info(f"✅ Validación NAFNet '{task}' completada: {validation_results['health_score']}% saludable")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "NAFNet Dataset Validation")
        logger.error(f"❌ Error en validación NAFNet: {e}")
        raise
