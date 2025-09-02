# 🧹 PLAN DE LIMPIEZA DEL PROYECTO

## 📊 ANÁLISIS DE ARCHIVOS

### ✅ ARCHIVOS NECESARIOS (MANTENER)

#### API Core (En uso activo)
- `api/` - Todo el directorio de la API
- `docker-compose.yml` - Configuración de contenedores
- `Dockerfile` - Imagen de la API
- `requirements.txt` - Dependencias base del proyecto

#### Configuración
- `config/` - Configuraciones yaml necesarias
- `.env.example` - Template de variables de entorno

#### Modelos y Data
- `src/models/` - Modelos usados por la API (Restormer)
- `src/utils.py` - Utilities usadas por model_service
- `layers/` - Módulos de entrenamiento Layer 1 y 2 (usados por training_service)
- `models/pretrained/` - Modelo preentrenado base
- `data/` - Dataset necesario para entrenamiento

#### N8N Workflows
- `n8n/` - Workflows y helpers para n8n

### ❌ ARCHIVOS LEGACY/DUPLICADOS (ELIMINAR)

#### Scripts Legacy (Ya no se usan con la API)
- `main_pipeline.py` - Pipeline principal legacy (reemplazado por API)
- `quick_evaluation.py` - Evaluación legacy (reemplazado por API endpoints)
- `setup_api_environment.py` - Setup manual (reemplazado por Docker)
- `setup_project.py` - Setup manual (reemplazado por Docker)
- `test_corrections.py` - Script de pruebas obsoleto

#### Directorios Legacy
- `legacy/` - Todo el directorio (código anterior al refactor)
- `demos/` - Demos obsoletos (reemplazados por API)
- `evaluation/` - Evaluación manual (reemplazada por API)
- `data_generation/` - Generación manual (reemplazada por API)
- `training/` - Entrenamiento manual (reemplazado por API)

#### Archivos de documentación obsoletos
- `MEMORY_OPTIMIZATION_GUIDE.md` - Guía obsoleta (implementado en API)

### 🔄 ARCHIVOS A REVISAR

#### API Backup Files
- `api/main_backup.py` - Revisar si se necesita como backup
- `api/main_modular.py` - Revisar si se necesita
- `api/client.py` - Cliente de prueba (posiblemente mantener)
- `api/synthetic_data_client.py` - Cliente específico (revisar uso)

## 🗂️ ESTRUCTURA FINAL LIMPIA

```
n8n-automatize-models/
├── api/                    # ✅ API FastAPI completa
├── config/                 # ✅ Configuraciones
├── data/                   # ✅ Dataset de entrenamiento
├── src/                    # ✅ Solo models/ y utils necesarios
│   ├── models/
│   │   └── restormer.py
│   └── utils.py
├── layers/                 # ✅ Módulos de entrenamiento
├── models/                 # ✅ Modelos preentrenados
├── n8n/                    # ✅ Workflows
├── outputs/                # ✅ Resultados (checkpoints, etc.)
├── docker-compose.yml      # ✅ Docker setup
├── Dockerfile             # ✅ API container
├── requirements.txt       # ✅ Dependencias base
└── README.md              # ✅ Documentación actualizada
```

## 📋 ACCIONES A REALIZAR

1. **Eliminar directorios legacy**
2. **Eliminar scripts obsoletos en raíz**
3. **Limpiar archivos de backup en API** (después de confirmar)
4. **Actualizar documentación**
5. **Verificar que todo funcione después de la limpieza**
