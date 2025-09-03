#!/usr/bin/env python3
"""
Script para iniciar la API con Poetry
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """Iniciar la API usando uvicorn"""
    project_root = Path(__file__).parent.parent
    api_path = project_root / "api"
    
    # Cambiar al directorio del proyecto
    os.chdir(project_root)
    
    # Detectar si estamos en un entorno virtual de Poetry
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        python_cmd = str(venv_python)
        uvicorn_cmd = str(project_root / ".venv" / "Scripts" / "uvicorn.exe")
    else:
        python_cmd = "python"
        uvicorn_cmd = "uvicorn"
    
    # Comando para iniciar la API
    cmd = [
        uvicorn_cmd,
        "api.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload",
        "--reload-dir", "api",
        "--reload-dir", "src"
    ]
    
    print("🚀 Iniciando API de restauración...")
    print(f"📁 Directorio: {project_root}")
    print(f"🌐 URL: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"🐍 Python: {python_cmd}")
    print(f"⚡ Uvicorn: {uvicorn_cmd}")
    
    try:
        # Verificar que uvicorn existe
        if not Path(uvicorn_cmd).exists() and uvicorn_cmd != "uvicorn":
            print("❌ Uvicorn no encontrado. Usando poetry run...")
            cmd = ["poetry", "run"] + cmd[1:]  # Usar poetry run en su lugar
        
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  API detenida por el usuario")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al iniciar la API: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ Archivo no encontrado: {e}")
        print("💡 Intenta usar: poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
        sys.exit(1)

if __name__ == "__main__":
    main()
