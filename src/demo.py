"""
Script de demostración del pipeline completo de restauración
"""

import os
import argparse
import sys
import cv2
import numpy as np
from PIL import Image

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from pipeline import DocumentRestorationPipeline

def main():
    parser = argparse.ArgumentParser(description='Demostración del pipeline de restauración de documentos')
    parser.add_argument('--input', type=str, required=True,
                       help='Ruta de la imagen de entrada o directorio')
    parser.add_argument('--output', type=str, required=True,
                       help='Ruta de salida')
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml',
                       help='Archivo de configuración del pipeline')
    parser.add_argument('--restormer_model', type=str, default='outputs/checkpoints/best_model.pth',
                       help='Ruta al modelo Restormer entrenado')
    parser.add_argument('--esrgan_model', type=str, default=None,
                       help='Ruta al modelo ESRGAN (opcional)')
    parser.add_argument('--batch', action='store_true',
                       help='Procesamiento en lote (input debe ser un directorio)')
    parser.add_argument('--show_intermediate', action='store_true',
                       help='Mostrar resultado intermedio de Restormer')
    parser.add_argument('--no_esrgan', action='store_true',
                       help='Deshabilitar ESRGAN')
    
    args = parser.parse_args()
    
    print("=== Pipeline de Restauración de Documentos ===")
    print("Restormer + ESRGAN")
    print("=" * 50)
    
    # Crear pipeline
    config_path = args.config if os.path.exists(args.config) else None
    pipeline = DocumentRestorationPipeline(config_path=config_path)
    
    # Deshabilitar ESRGAN si se solicita
    if args.no_esrgan:
        pipeline.config['processing']['use_esrgan'] = False
        print("ESRGAN deshabilitado")
    
    # Inicializar modelos
    pipeline.initialize_models(
        restormer_path=args.restormer_model,
        esrgan_path=args.esrgan_model
    )
    
    if args.batch:
        # Procesamiento en lote
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} no es un directorio")
            return
        
        pipeline.batch_restore(args.input, args.output)
    
    else:
        # Procesamiento de imagen única
        if not os.path.exists(args.input):
            print(f"Error: {args.input} no existe")
            return
        
        # Restaurar imagen
        if args.show_intermediate:
            result, intermediate = pipeline.restore_document(
                args.input, 
                save_path=args.output, 
                return_intermediate=True
            )
            
            # Guardar resultado intermedio
            intermediate_path = args.output.replace('.', '_restormer_only.')
            cv2.imwrite(intermediate_path, cv2.cvtColor(intermediate, cv2.COLOR_RGB2BGR))
            print(f"Resultado intermedio guardado en: {intermediate_path}")
            
        else:
            result = pipeline.restore_document(args.input, save_path=args.output)
        
        print(f"Resultado final guardado en: {args.output}")
        
        # Mostrar comparación
        try:
            import matplotlib.pyplot as plt
            
            # Cargar imágenes
            original = cv2.imread(args.input)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            
            # Crear figura
            if args.show_intermediate:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(original)
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                axes[1].imshow(intermediate)
                axes[1].set_title('Restormer')
                axes[1].axis('off')
                
                axes[2].imshow(result)
                axes[2].set_title('Final (Restormer + ESRGAN)')
                axes[2].axis('off')
            else:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(original)
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                axes[1].imshow(result)
                axes[1].set_title('Restaurado')
                axes[1].axis('off')
            
            plt.tight_layout()
            
            # Guardar comparación
            comparison_path = args.output.replace('.', '_comparison.')
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            print(f"Comparación guardada en: {comparison_path}")
            
            # Mostrar si es posible
            try:
                plt.show()
            except:
                print("No se puede mostrar la visualización en este entorno")
            
        except ImportError:
            print("Matplotlib no disponible, omitiendo visualización")

def create_demo_data():
    """Crear datos de demostración con documentos sintéticamente degradados"""
    print("Creando datos de demostración...")
    
    # Crear directorios
    demo_dir = "demo_data"
    os.makedirs(f"{demo_dir}/original", exist_ok=True)
    os.makedirs(f"{demo_dir}/degraded", exist_ok=True)
    os.makedirs(f"{demo_dir}/results", exist_ok=True)
    
    # Crear imagen de documento sintético
    def create_document_image(width=800, height=600):
        # Crear imagen blanca
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Agregar texto simulado
        cv2.rectangle(img, (50, 50), (width-50, 100), (0, 0, 0), 2)
        cv2.putText(img, "DOCUMENTO DE PRUEBA", (60, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Agregar párrafos simulados
        for i in range(5):
            y_start = 150 + i * 80
            cv2.rectangle(img, (50, y_start), (width-50, y_start + 60), (50, 50, 50), -1)
            
        return img
    
    # Crear imagen original
    original = create_document_image()
    cv2.imwrite(f"{demo_dir}/original/document.png", original)
    
    # Crear versión degradada
    degraded = original.copy()
    
    # Agregar ruido
    noise = np.random.normal(0, 25, degraded.shape).astype(np.uint8)
    degraded = cv2.add(degraded, noise)
    
    # Agregar blur
    degraded = cv2.GaussianBlur(degraded, (5, 5), 0)
    
    # Reducir calidad
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    _, encoded_img = cv2.imencode('.jpg', degraded, encode_param)
    degraded = cv2.imdecode(encoded_img, 1)
    
    cv2.imwrite(f"{demo_dir}/degraded/document.png", degraded)
    
    print(f"Datos de demostración creados en: {demo_dir}/")
    print("Usa: python demo.py --input demo_data/degraded/document.png --output demo_data/results/restored.png")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'create_demo':
        create_demo_data()
    else:
        main()
