"""
Servicio para manejo del modelo de restauración
"""
import torch
import logging
import sys
import os

# Agregar paths necesarios
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('src')

try:
    from config.settings import model_state
except ImportError:
    # Fallback state
    model_state = {'model': None, 'device': None, 'loaded': False}

try:
    from src.models.restormer import Restormer
except ImportError:
    logging.warning("No se pudo importar Restormer - funcionalidad limitada")

logger = logging.getLogger(__name__)

class ModelService:
    """Servicio para operaciones con el modelo de restauración"""
    
    def load_model(self):
        """Cargar modelo de restauración"""
        if model_state['loaded']:
            return model_state['model']
        
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Cargando modelo en dispositivo: {device}")
            
            # Verificar si Restormer está disponible
            if 'Restormer' not in globals():
                logger.warning("Restormer no disponible - saltando carga de modelo")
                model_state['loaded'] = False
                return None
            
            # Cargar el mejor modelo disponible
            model_paths = [
                "outputs/checkpoints/gradual_transfer_final.pth",
                "outputs/checkpoints/optimized_restormer_final.pth", 
                "outputs/checkpoints/finetuned_restormer_final.pth"
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                raise Exception("No se encontró ningún modelo entrenado")
            
            logger.info(f"Modelo cargado desde: {model_path}")
            
            # Crear modelo Restormer
            model = Restormer(
                inp_channels=3,
                out_channels=3,
                dim=48,
                num_blocks=[4,6,6,8],
                num_refinement_blocks=4,
                heads=[1,2,4,8],
                ffn_expansion_factor=2.66,
                bias=False,
                LayerNorm_type='WithBias',
                dual_pixel_task=False
            )
            
            # Cargar pesos del modelo
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            
            # Actualizar estado global
            model_state['model'] = model
            model_state['device'] = device
            model_state['loaded'] = True
            
            logger.info("Modelo cargado exitosamente")
            return model
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise e
    
    def predict(self, image_tensor):
        """Realizar predicción con el modelo"""
        if not model_state['loaded']:
            self.load_model()
        
        model = model_state['model']
        device = model_state['device']
        
        with torch.no_grad():
            # Mover tensor al dispositivo
            if image_tensor.device != device:
                image_tensor = image_tensor.to(device)
            
            # Realizar predicción
            output = model(image_tensor)
            
            return output

# Instancia global del servicio
model_service = ModelService()
