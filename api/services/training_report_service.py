"""
Servicio para generar reportes de entrenamiento en formato TXT
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

# Importar servicios
try:
    from .minio_service import minio_service
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from minio_service import minio_service

logger = logging.getLogger(__name__)

class TrainingReportService:
    """Servicio para generar reportes detallados de entrenamiento"""
    
    def __init__(self):
        self.reports_bucket = "models"
        self.reports_prefix = "reports/"
    
    def generate_training_report(self, job_data: Dict[str, Any], 
                               model_info: Dict[str, Any] = None,
                               training_metrics: Dict[str, Any] = None,
                               data_statistics: Dict[str, Any] = None) -> str:
        """
        Generar reporte completo de entrenamiento
        
        Args:
            job_data: Información del trabajo de entrenamiento
            model_info: Información del modelo generado
            training_metrics: Métricas de entrenamiento (loss, accuracy, etc.)
            data_statistics: Estadísticas de los datos utilizados
            
        Returns:
            Contenido del reporte en formato texto
        """
        try:
            # Generar contenido del reporte
            report_content = self._create_report_content(
                job_data, model_info, training_metrics, data_statistics
            )
            
            # Generar nombre de archivo único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = job_data.get('job_id', 'unknown')[:8]
            layer = job_data.get('layer', 'unknown')
            
            report_filename = f"training_report_layer{layer}_{job_id}_{timestamp}.txt"
            report_path = f"{self.reports_prefix}{report_filename}"
            
            # Guardar reporte en MinIO
            report_bytes = report_content.encode('utf-8')
            minio_service.client.put_object(
                Bucket=self.reports_bucket,
                Key=report_path,
                Body=report_bytes,
                ContentType='text/plain'
            )
            
            logger.info(f"Reporte de entrenamiento guardado: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generando reporte de entrenamiento: {e}")
            return None
    
    def _create_report_content(self, job_data: Dict[str, Any], 
                             model_info: Dict[str, Any] = None,
                             training_metrics: Dict[str, Any] = None,
                             data_statistics: Dict[str, Any] = None) -> str:
        """Crear el contenido del reporte en formato texto"""
        
        report_lines = []
        
        # Encabezado del reporte
        report_lines.extend([
            "=" * 80,
            "🎯 REPORTE DE ENTRENAMIENTO - DOCUMENT RESTORATION API",
            "=" * 80,
            "",
            f"📅 Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"🆔 Job ID: {job_data.get('job_id', 'N/A')}",
            f"🏷️ Tipo de entrenamiento: {job_data.get('type', 'N/A')}",
            f"🔢 Capa: Layer {job_data.get('layer', 'N/A')}",
            "",
        ])
        
        # Información básica del entrenamiento
        report_lines.extend([
            "📊 INFORMACIÓN DEL ENTRENAMIENTO",
            "-" * 40,
            f"Inicio: {job_data.get('start_time', 'N/A')}",
            f"Fin: {job_data.get('end_time', 'N/A')}",
            f"Duración total: {job_data.get('duration', 'N/A')}",
            f"Estado final: {job_data.get('status', 'N/A')}",
            f"Progreso: {job_data.get('progress', 0)}%",
            "",
        ])
        
        # Parámetros de entrenamiento
        training_info = job_data.get('training_info', {})
        report_lines.extend([
            "⚙️ PARÁMETROS DE ENTRENAMIENTO",
            "-" * 40,
            f"Número de épocas: {training_info.get('num_epochs', 'N/A')}",
            f"Épocas completadas: {training_info.get('current_epoch', 'N/A')}",
            f"Batch size: {training_info.get('batch_size', 'N/A')}",
            f"Máximo de pares: {training_info.get('max_pairs', 'N/A')}",
            f"Learning rate: {training_info.get('learning_rate', 'N/A')}",
            "",
        ])
        
        # Datos utilizados
        results = job_data.get('results', {})
        if data_statistics:
            report_lines.extend([
                "📦 ESTADÍSTICAS DE DATOS",
                "-" * 40,
                f"Total de archivos: {data_statistics.get('total_files', 'N/A')}",
                f"Archivos clean: {data_statistics.get('clean_files', 'N/A')}",
                f"Archivos degraded: {data_statistics.get('degraded_files', 'N/A')}",
                f"Pares válidos: {data_statistics.get('valid_pairs', 'N/A')}",
                f"Pares utilizados: {results.get('pairs_used', 'N/A')}",
                f"Bucket de origen: {results.get('data_source', 'document-training')}",
                "",
            ])
        
        # Métricas de entrenamiento
        final_metrics = results.get('final_metrics', {})
        if training_metrics or final_metrics:
            report_lines.extend([
                "📈 MÉTRICAS DE ENTRENAMIENTO",
                "-" * 40,
            ])
            
            # Métricas finales
            if final_metrics:
                report_lines.extend([
                    f"Loss final: {final_metrics.get('loss', 'N/A'):.4f}" if isinstance(final_metrics.get('loss'), (int, float)) else f"Loss final: {final_metrics.get('loss', 'N/A')}",
                    f"Accuracy final: {final_metrics.get('accuracy', 'N/A'):.3f}" if isinstance(final_metrics.get('accuracy'), (int, float)) else f"Accuracy final: {final_metrics.get('accuracy', 'N/A')}",
                    f"PSNR final: {final_metrics.get('psnr', 'N/A'):.2f} dB" if isinstance(final_metrics.get('psnr'), (int, float)) else f"PSNR final: {final_metrics.get('psnr', 'N/A')}",
                    f"SSIM final: {final_metrics.get('ssim', 'N/A'):.3f}" if isinstance(final_metrics.get('ssim'), (int, float)) else f"SSIM final: {final_metrics.get('ssim', 'N/A')}",
                ])
            
            # Métricas por época (si están disponibles)
            if training_metrics and 'epoch_metrics' in training_metrics:
                report_lines.append("\n📊 Progreso por época:")
                for epoch, metrics in training_metrics['epoch_metrics'].items():
                    loss = metrics.get('loss', 'N/A')
                    accuracy = metrics.get('accuracy', 'N/A')
                    psnr = metrics.get('psnr', 'N/A')
                    ssim = metrics.get('ssim', 'N/A')
                    
                    # Formatear valores numéricos
                    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
                    acc_str = f"{accuracy:.3f}" if isinstance(accuracy, (int, float)) else str(accuracy)
                    psnr_str = f"PSNR={psnr:.2f}" if isinstance(psnr, (int, float)) else f"PSNR={psnr}"
                    ssim_str = f"SSIM={ssim:.3f}" if isinstance(ssim, (int, float)) else f"SSIM={ssim}"
                    
                    report_lines.append(f"  Época {epoch}: Loss={loss_str}, Acc={acc_str}, {psnr_str}, {ssim_str}")
                
                # Análisis de mejora
                if len(training_metrics['epoch_metrics']) > 1:
                    first_epoch = list(training_metrics['epoch_metrics'].values())[0]
                    last_epoch = list(training_metrics['epoch_metrics'].values())[-1]
                    
                    report_lines.append("\n🔄 Análisis de mejora:")
                    
                    for metric in ['loss', 'accuracy', 'psnr', 'ssim']:
                        first_val = first_epoch.get(metric)
                        last_val = last_epoch.get(metric)
                        
                        if isinstance(first_val, (int, float)) and isinstance(last_val, (int, float)):
                            if metric == 'loss':
                                improvement = ((first_val - last_val) / first_val) * 100
                                report_lines.append(f"  - {metric.upper()}: {first_val:.4f} → {last_val:.4f} (-{improvement:.1f}%)")
                            else:
                                improvement = ((last_val - first_val) / first_val) * 100
                                unit = " dB" if metric == 'psnr' else ""
                                report_lines.append(f"  - {metric.upper()}: {first_val:.3f}{unit} → {last_val:.3f}{unit} (+{improvement:.1f}%)")
            
            report_lines.append("")
        
        # Información del modelo generado
        model_saved = results.get('model_saved', {})
        if model_info or model_saved:
            report_lines.extend([
                "🤖 INFORMACIÓN DEL MODELO GENERADO",
                "-" * 40,
                f"Nombre del modelo: {model_saved.get('model_name', model_info.get('filename', 'N/A'))}",
                f"Ruta en MinIO: {model_saved.get('model_path', 'N/A')}",
                f"Tamaño: {self._format_bytes(model_saved.get('size_bytes', model_info.get('size', 0)))}",
                f"Capa: {model_saved.get('layer', 'N/A')}",
                f"Arquitectura: {training_info.get('architecture', 'Layer2_Restormer')}",
                "",
            ])
        
        # Comparación con modelo anterior (si existe)
        previous_model_comparison = self._get_previous_model_comparison(job_data.get('layer'))
        if previous_model_comparison:
            report_lines.extend([
                "🔄 COMPARACIÓN CON MODELO ANTERIOR",
                "-" * 40,
                previous_model_comparison,
                "",
            ])
        
        # Configuración del entorno
        report_lines.extend([
            "🔧 CONFIGURACIÓN DEL ENTORNO",
            "-" * 40,
            f"Python version: 3.11",
            f"Framework: PyTorch",
            f"Dispositivo: {job_data.get('device', 'CPU')}",
            f"Contenedor: Docker",
            f"Arquitectura API: Modular",
            "",
        ])
        
        # Errores o advertencias
        if job_data.get('error'):
            report_lines.extend([
                "⚠️ ERRORES O ADVERTENCIAS",
                "-" * 40,
                f"Error: {job_data.get('error')}",
                "",
            ])
        
        # Recomendaciones
        report_lines.extend([
            "💡 RECOMENDACIONES",
            "-" * 40,
            self._generate_recommendations(job_data, data_statistics, final_metrics),
            "",
        ])
        
        # Pie de página
        report_lines.extend([
            "=" * 80,
            "📝 Este reporte fue generado automáticamente por el sistema",
            "🔗 Document Restoration API - n8n Integration",
            f"🕐 Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
        ])
        
        return "\n".join(report_lines)
    
    def _format_bytes(self, bytes_size: int) -> str:
        """Formatear tamaño en bytes a formato legible"""
        if bytes_size < 1024:
            return f"{bytes_size} bytes"
        elif bytes_size < 1024 * 1024:
            return f"{bytes_size / 1024:.2f} KB"
        elif bytes_size < 1024 * 1024 * 1024:
            return f"{bytes_size / (1024 * 1024):.2f} MB"
        else:
            return f"{bytes_size / (1024 * 1024 * 1024):.2f} GB"
    
    def _get_previous_model_comparison(self, layer: str) -> str:
        """Obtener comparación con modelo anterior de la misma capa"""
        try:
            # Listar modelos de la capa
            models = minio_service.list_models(layer)
            
            if len(models) < 2:
                return "Este es el primer modelo de esta capa o no hay modelos anteriores para comparar."
            
            # Ordenar por fecha (más reciente primero)
            models_sorted = sorted(models, key=lambda x: x['last_modified'], reverse=True)
            current_model = models_sorted[0]
            previous_model = models_sorted[1]
            
            comparison = []
            comparison.append(f"Modelo anterior: {previous_model['filename']}")
            comparison.append(f"Modelo actual: {current_model['filename']}")
            comparison.append(f"Diferencia de tamaño: {current_model['size'] - previous_model['size']} bytes")
            comparison.append(f"Tiempo entre modelos: {current_model['last_modified']} vs {previous_model['last_modified']}")
            comparison.append("Nota: Métricas detalladas de comparación requieren evaluación específica.")
            
            return "\n".join(comparison)
            
        except Exception as e:
            return f"No se pudo obtener comparación con modelo anterior: {e}"
    
    def _generate_recommendations(self, job_data: Dict[str, Any], 
                                data_statistics: Dict[str, Any] = None,
                                metrics: Dict[str, Any] = None) -> str:
        """Generar recomendaciones basadas en los resultados"""
        recommendations = []
        
        # Análisis del estado del entrenamiento
        status = job_data.get('status', '')
        if status == 'completed':
            recommendations.append("✅ Entrenamiento completado exitosamente")
        elif status == 'failed':
            recommendations.append("❌ Entrenamiento falló - revisar logs para más detalles")
        
        # Recomendaciones basadas en datos
        if data_statistics:
            valid_pairs = data_statistics.get('valid_pairs', 0)
            if valid_pairs < 50:
                recommendations.append("⚠️ Pocos datos disponibles - considerar generar más datos sintéticos (recomendado: >200 pares)")
            elif valid_pairs < 200:
                recommendations.append("💡 Cantidad de datos moderada - podría mejorarse para mejor rendimiento (óptimo: >500 pares)")
            else:
                recommendations.append("✅ Cantidad de datos adecuada para entrenamiento")
        
        # Recomendaciones basadas en métricas finales
        if metrics:
            loss = metrics.get('loss')
            accuracy = metrics.get('accuracy')
            psnr = metrics.get('psnr')
            ssim = metrics.get('ssim')
            
            if isinstance(loss, (int, float)):
                if loss > 0.1:
                    recommendations.append("⚠️ Loss elevado - considerar más épocas o ajustar learning rate")
                elif loss < 0.01:
                    recommendations.append("✅ Loss excelente - modelo bien entrenado")
                else:
                    recommendations.append("✅ Loss dentro del rango aceptable")
            
            if isinstance(accuracy, (int, float)):
                if accuracy < 0.7:
                    recommendations.append("⚠️ Accuracy bajo - revisar calidad de datos o arquitectura del modelo")
                elif accuracy > 0.9:
                    recommendations.append("✅ Accuracy excelente - modelo muy preciso")
                else:
                    recommendations.append("✅ Accuracy aceptable - modelo funcional")
            
            if isinstance(psnr, (int, float)):
                if psnr < 25:
                    recommendations.append("⚠️ PSNR bajo - calidad de reconstrucción mejorable")
                elif psnr > 30:
                    recommendations.append("✅ PSNR excelente - alta calidad de reconstrucción")
                else:
                    recommendations.append("✅ PSNR aceptable para aplicaciones generales")
            
            if isinstance(ssim, (int, float)):
                if ssim < 0.8:
                    recommendations.append("⚠️ SSIM bajo - similitud estructural mejorable")
                elif ssim > 0.9:
                    recommendations.append("✅ SSIM excelente - alta similitud estructural")
                else:
                    recommendations.append("✅ SSIM aceptable - buena preservación de estructura")
        
        # Recomendaciones basadas en parámetros de entrenamiento
        training_info = job_data.get('training_info', {})
        epochs = training_info.get('num_epochs', 0)
        batch_size = training_info.get('batch_size', 0)
        
        if epochs > 0:
            if epochs < 10:
                recommendations.append("💡 Considerar aumentar el número de épocas para mejor convergencia")
            elif epochs > 50:
                recommendations.append("💡 Muchas épocas utilizadas - verificar si hay overfitting")
        
        if batch_size > 0:
            if batch_size == 1:
                recommendations.append("💡 Batch size muy pequeño - considerar aumentar si hay suficiente memoria")
            elif batch_size > 8:
                recommendations.append("💡 Batch size grande - puede acelerar entrenamiento pero usar más memoria")
        
        # Recomendaciones basadas en duración
        duration = job_data.get('duration', '')
        if 'minutos' in str(duration):
            try:
                minutes = int(duration.split()[0])
                if minutes > 60:
                    recommendations.append("⏱️ Entrenamiento largo - considerar usar GPU o optimizar parámetros")
                elif minutes < 5:
                    recommendations.append("⏱️ Entrenamiento muy rápido - verificar que se completó correctamente")
            except:
                pass
        
        # Recomendaciones sobre hardware
        device = job_data.get('device', 'CPU')
        if device == 'CPU':
            recommendations.append("🚀 Considerar usar GPU para entrenamientos más rápidos y eficientes")
        
        # Recomendación general si no hay observaciones específicas
        if len(recommendations) == 0:
            recommendations.append("✅ El entrenamiento parece haber completado exitosamente sin observaciones especiales")
        
        return "\n".join(recommendations)
    
    def list_training_reports(self, layer: str = None) -> List[Dict[str, Any]]:
        """Listar reportes de entrenamiento disponibles"""
        try:
            prefix = f"{self.reports_prefix}"
            if layer:
                prefix += f"training_report_layer{layer}_"
            
            response = minio_service.client.list_objects_v2(
                Bucket=self.reports_bucket,
                Prefix=prefix
            )
            
            reports = []
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.txt'):
                    reports.append({
                        'filename': obj['Key'].split('/')[-1],
                        'path': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'layer': self._extract_layer_from_filename(obj['Key'])
                    })
            
            return sorted(reports, key=lambda x: x['last_modified'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listando reportes: {e}")
            return []
    
    def _extract_layer_from_filename(self, filename: str) -> str:
        """Extraer número de capa del nombre de archivo"""
        try:
            if 'layer' in filename:
                start = filename.find('layer') + 5
                end = filename.find('_', start)
                return filename[start:end] if end != -1 else filename[start:start+1]
        except:
            pass
        return "unknown"
    
    def download_report(self, report_path: str) -> str:
        """Descargar contenido de un reporte"""
        try:
            response = minio_service.client.get_object(
                Bucket=self.reports_bucket,
                Key=report_path
            )
            return response['Body'].read().decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error descargando reporte {report_path}: {e}")
            return None

# Instancia global del servicio
training_report_service = TrainingReportService()
