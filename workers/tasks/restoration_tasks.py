#!/usr/bin/env python3
"""
🔧 RESTORATION TASKS - RQ SPECIALIZED
====================================
Tasks especializadas para restauración de documentos usando RQ.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Importar utilidades RQ
from ..utils.rq_utils import RQJobProgressTracker, setup_job_environment

logger = logging.getLogger(__name__)

def single_document_restoration_job(input_bucket: str,
                                  input_filename: str,
                                  output_bucket: str = "document-restored",
                                  model_type: str = "layer2",
                                  **kwargs) -> Dict[str, Any]:
    """
    🔧 Job de restauración de documento individual
    
    Args:
        input_bucket: Bucket con documento degradado
        input_filename: Nombre del archivo a restaurar
        output_bucket: Bucket de destino
        model_type: Tipo de modelo a usar (layer1, layer2, hybrid)
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultados de la restauración
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    logger.info(f"🔧 Iniciando restauración: {input_filename} usando {model_type}")
    
    try:
        tracker.update_progress(5, f"Preparando restauración de {input_filename}...")
        
        # Importar servicio de restauración
        from api.services.restoration_service import restoration_service
        
        restoration_params = {
            'input_bucket': input_bucket,
            'input_filename': input_filename,
            'output_bucket': output_bucket,
            'model_type': model_type
        }
        
        tracker.update_progress(10, "Cargando modelo de restauración...")
        
        # Callback de progreso para la restauración
        def restoration_progress_callback(stage: str, progress: float):
            job_progress = int(10 + (progress * 0.8))
            tracker.update_progress(job_progress, f"Restaurando: {stage}")
        
        # Ejecutar restauración
        restoration_result = restoration_service.restore_document(
            input_bucket=input_bucket,
            input_filename=input_filename,
            output_bucket=output_bucket,
            model_type=model_type,
            progress_callback=restoration_progress_callback
        )
        
        tracker.update_progress(95, "Guardando resultado...")
        
        # Preparar resultado final
        final_result = {
            'status': 'completed',
            'restoration_result': restoration_result,
            'parameters': restoration_params,
            'output_filename': restoration_result.get('output_filename'),
            'quality_improvement': restoration_result.get('quality_metrics', {}),
            'completed_at': datetime.now().isoformat(),
            'job_type': 'single_document_restoration'
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, f"Restauración de {input_filename} completada")
        
        logger.info(f"✅ Restauración completada: {input_filename}")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "Document Restoration")
        logger.error(f"❌ Error en restauración de {input_filename}: {e}")
        raise

def batch_restoration_job(input_bucket: str,
                         file_pattern: str = "*.jpg",
                         output_bucket: str = "document-restored",
                         model_type: str = "layer2",
                         max_files: int = 100,
                         **kwargs) -> Dict[str, Any]:
    """
    📦 Job de restauración en lotes
    
    Args:
        input_bucket: Bucket con documentos degradados
        file_pattern: Patrón de archivos a procesar
        output_bucket: Bucket de destino
        model_type: Tipo de modelo a usar
        max_files: Máximo número de archivos a procesar
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con resultados de la restauración en lotes
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    logger.info(f"📦 Iniciando restauración en lotes: {file_pattern} en {input_bucket}")
    
    try:
        tracker.update_progress(5, "Preparando restauración en lotes...")
        
        # Importar servicios
        from api.services.restoration_service import restoration_service
        from api.services.minio_service import minio_service
        
        # Obtener lista de archivos
        tracker.update_progress(10, "Obteniendo lista de archivos...")
        all_files = minio_service.list_files(input_bucket, pattern=file_pattern)
        files_to_process = all_files[:max_files]
        
        batch_params = {
            'input_bucket': input_bucket,
            'file_pattern': file_pattern,
            'output_bucket': output_bucket,
            'model_type': model_type,
            'total_files': len(files_to_process)
        }
        
        tracker.update_progress(15, f"Procesando {len(files_to_process)} archivos...")
        
        # Resultados del lote
        batch_results = {
            'processed_files': [],
            'successful_restorations': 0,
            'failed_restorations': 0,
            'errors': []
        }
        
        # Procesar cada archivo
        for i, filename in enumerate(files_to_process):
            progress = int(15 + ((i / len(files_to_process)) * 75))
            tracker.update_progress(progress, f"Restaurando {filename} ({i+1}/{len(files_to_process)})")
            
            try:
                # Restaurar archivo individual
                file_result = restoration_service.restore_document(
                    input_bucket=input_bucket,
                    input_filename=filename,
                    output_bucket=output_bucket,
                    model_type=model_type
                )
                
                batch_results['processed_files'].append({
                    'filename': filename,
                    'status': 'success',
                    'output_filename': file_result.get('output_filename'),
                    'quality_metrics': file_result.get('quality_metrics', {})
                })
                batch_results['successful_restorations'] += 1
                
            except Exception as file_error:
                logger.warning(f"Error restaurando {filename}: {file_error}")
                batch_results['processed_files'].append({
                    'filename': filename,
                    'status': 'error',
                    'error': str(file_error)
                })
                batch_results['failed_restorations'] += 1
                batch_results['errors'].append({
                    'filename': filename,
                    'error': str(file_error)
                })
        
        # Calcular estadísticas finales
        success_rate = (batch_results['successful_restorations'] / len(files_to_process) * 100) if files_to_process else 0
        
        final_result = {
            'status': 'completed',
            'batch_results': batch_results,
            'parameters': batch_params,
            'success_rate': success_rate,
            'completed_at': datetime.now().isoformat(),
            'job_type': 'batch_restoration'
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, f"Lote completado: {success_rate:.1f}% exitoso")
        
        logger.info(f"✅ Restauración en lotes completada: {success_rate:.1f}% exitoso")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "Batch Restoration")
        logger.error(f"❌ Error en restauración en lotes: {e}")
        raise

def quality_assessment_job(restored_bucket: str = "document-restored",
                          original_bucket: str = "document-clean",
                          sample_size: int = 20,
                          **kwargs) -> Dict[str, Any]:
    """
    📊 Job de evaluación de calidad de restauraciones
    
    Args:
        restored_bucket: Bucket con documentos restaurados
        original_bucket: Bucket con documentos originales (para comparación)
        sample_size: Número de documentos a evaluar
        **kwargs: Parámetros adicionales de RQ
    
    Returns:
        Dict con métricas de calidad
    """
    setup_job_environment()
    tracker = RQJobProgressTracker()
    
    logger.info(f"📊 Iniciando evaluación de calidad: {sample_size} muestras")
    
    try:
        tracker.update_progress(5, "Preparando evaluación de calidad...")
        
        # Importar servicios
        from api.services.minio_service import minio_service
        from api.services.image_analysis_service import image_analysis_service
        
        # Obtener muestras de archivos
        tracker.update_progress(10, "Obteniendo muestras de archivos...")
        restored_files = minio_service.list_files(restored_bucket)
        
        import random
        sample_files = random.sample(restored_files, min(sample_size, len(restored_files)))
        
        quality_metrics = {
            'total_evaluated': len(sample_files),
            'average_psnr': 0,
            'average_ssim': 0,
            'average_lpips': 0,
            'quality_scores': [],
            'detailed_results': []
        }
        
        # Evaluar cada archivo
        for i, filename in enumerate(sample_files):
            progress = int(10 + ((i / len(sample_files)) * 80))
            tracker.update_progress(progress, f"Evaluando calidad de {filename}")
            
            try:
                # Buscar archivo original correspondiente
                original_filename = filename.replace('_restored', '').replace('_fixed', '')
                
                # Calcular métricas de calidad
                metrics = image_analysis_service.compare_image_quality(
                    restored_bucket, filename,
                    original_bucket, original_filename
                )
                
                quality_metrics['quality_scores'].append(metrics.get('overall_score', 0))
                quality_metrics['detailed_results'].append({
                    'filename': filename,
                    'metrics': metrics
                })
                
            except Exception as eval_error:
                logger.warning(f"Error evaluando {filename}: {eval_error}")
                quality_metrics['detailed_results'].append({
                    'filename': filename,
                    'error': str(eval_error)
                })
        
        # Calcular promedios
        if quality_metrics['quality_scores']:
            quality_metrics['average_quality'] = sum(quality_metrics['quality_scores']) / len(quality_metrics['quality_scores'])
        
        final_result = {
            'status': 'completed',
            'quality_metrics': quality_metrics,
            'restored_bucket': restored_bucket,
            'original_bucket': original_bucket,
            'sample_size': len(sample_files),
            'completed_at': datetime.now().isoformat(),
            'job_type': 'quality_assessment'
        }
        
        tracker.set_result(final_result)
        tracker.update_progress(100, f"Evaluación completada: {quality_metrics.get('average_quality', 0):.2f} score promedio")
        
        logger.info(f"✅ Evaluación de calidad completada")
        return final_result
        
    except Exception as e:
        tracker.log_error(e, "Quality Assessment")
        logger.error(f"❌ Error en evaluación de calidad: {e}")
        raise
