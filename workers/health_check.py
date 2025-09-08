#!/usr/bin/env python3
"""
Health check script para el worker modular
Crea un archivo de salud que indica que el worker está funcionando
"""

import os
import time
from pathlib import Path

def create_health_file():
    """Crear archivo de salud para el health check de Docker"""
    health_file = Path("/app/workers/temp/worker_health.txt")
    health_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(health_file, 'w') as f:
        f.write(f"Worker healthy at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    create_health_file()
