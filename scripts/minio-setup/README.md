# 🚀 MinIO Setup Scripts

Scripts para configurar y preparar el entorno MinIO con la estructura necesaria para el sistema de modelos preentrenados.

## 📋 Scripts Incluidos

### `setup_pretrained_folders.py`
Script principal para crear la estructura organizacional completa en MinIO.

**Funcionalidades:**
- ✅ Crea estructura de carpetas por layers
- ✅ Genera marcadores de directorio en S3/MinIO
- ✅ Crea documentación README automática
- ✅ Valida conexión a MinIO
- ✅ Manejo robusto de errores

**Estructura creada:**
```
models/
└── pretrained_models/
    ├── layer_1/
    │   └── restormer/
    ├── layer_2/
    │   ├── nafnet/
    │   └── docunet/
    └── general/
```

## 🔧 Uso

### Configuración Inicial
```bash
# Navegar al directorio
cd scripts/minio-setup

# Ejecutar setup
python setup_pretrained_folders.py
```

### Requisitos Previos
- MinIO ejecutándose en `localhost:9000`
- Credenciales configuradas:
  - **Usuario**: minio
  - **Contraseña**: minio123
- Bucket `models` existente

## 📁 Estructura Resultante

El script crea una organización completa con:

### Layer 1 - Primera Etapa
```
pretrained_models/layer_1/
└── restormer/           # Modelos Restormer
    └── README.md        # Documentación específica
```

### Layer 2 - Segunda Etapa
```
pretrained_models/layer_2/
├── nafnet/              # Modelos NAFNet
│   └── README.md
└── docunet/             # Modelos DocUNet
    └── README.md
```

### General
```
pretrained_models/general/
└── README.md            # Modelos auxiliares
```

## 🎯 Beneficios para Deployment

### Máquinas Nuevas
- **Setup automático**: Un comando crea toda la estructura
- **Documentación incluida**: Cada carpeta tiene su README explicativo
- **Validación**: Verifica que todo esté configurado correctamente

### Escalabilidad
- **Fácil expansión**: Agregar nuevos layers es trivial
- **Mantenimiento**: Estructura clara y documentada
- **Recuperación**: Recrea estructura en caso de pérdida

## 🔄 Configuración Adaptable

### Para Nuevos Layers
Edita la lista `PRETRAINED_FOLDERS` en el script:
```python
PRETRAINED_FOLDERS = [
    'pretrained_models/',
    'pretrained_models/layer_1/',
    'pretrained_models/layer_2/',
    'pretrained_models/layer_3/',    # Nuevo layer
    # ... más layers
]
```

### Para Nuevos Tipos de Modelos
Agrega subdirectorios específicos:
```python
'pretrained_models/layer_2/new_model_type/',
```

## 🌐 Integración con Docker

Este setup es compatible con:
- **Docker Compose**: Configuración de MinIO containerizada
- **Volúmenes persistentes**: Los datos sobreviven reinicios
- **Redes Docker**: Comunicación entre servicios

## 📋 Checklist de Verificación

Después de ejecutar el script, verifica:
- [ ] MinIO accesible en http://localhost:9000
- [ ] Bucket `models` existe
- [ ] Carpetas `pretrained_models/` creadas
- [ ] READMEs generados en cada carpeta
- [ ] No errores en la ejecución

## 🛠️ Solución de Problemas

### Error: Bucket no encontrado
```bash
# Crear bucket manualmente en MinIO console
# O usar mc (MinIO Client)
mc mb minio/models
```

### Error: Conexión rechazada
```bash
# Verificar que MinIO esté ejecutándose
docker-compose ps

# Levantar MinIO si está parado
docker-compose up -d minio
```

### Error: Credenciales inválidas
Verifica la configuración en `.env.docker`:
```bash
MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=minio123
```

## 🔗 Scripts Relacionados

Después de ejecutar este setup, puedes usar:
- `../model-management/layer-1/download_restormer_models.py`
- `../model-management/layer-2/download_nafnet_models.py`
- `../model-management/*/upload_local_models.py`

## 🚀 Automatización Futura

Este script puede integrarse en:
- **Scripts de deployment**: Setup automático en producción
- **CI/CD pipelines**: Preparación de entornos
- **Docker init scripts**: Configuración en container startup
