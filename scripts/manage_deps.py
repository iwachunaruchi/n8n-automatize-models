#!/usr/bin/env python3
"""
Script para gestionar dependencias con Poetry
"""
import subprocess
import sys
import argparse
from pathlib import Path

def run_command(command):
    """Ejecutar comando y mostrar output"""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def list_dependencies():
    """Listar todas las dependencias"""
    print("📦 Dependencias instaladas:")
    run_command("poetry show")

def add_dependency(package, dev=False):
    """Agregar una dependencia"""
    group_flag = "--group dev" if dev else ""
    print(f"➕ Agregando {'dependencia de desarrollo' if dev else 'dependencia'}: {package}")
    run_command(f"poetry add {group_flag} {package}")

def remove_dependency(package):
    """Remover una dependencia"""
    print(f"➖ Removiendo dependencia: {package}")
    run_command(f"poetry remove {package}")

def update_dependencies():
    """Actualizar todas las dependencias"""
    print("🔄 Actualizando dependencias...")
    run_command("poetry update")

def check_outdated():
    """Verificar dependencias desactualizadas"""
    print("🔍 Verificando dependencias desactualizadas...")
    run_command("poetry show --outdated")

def export_requirements():
    """Exportar requirements.txt para compatibilidad"""
    print("📄 Exportando requirements.txt...")
    run_command("poetry export -f requirements.txt --output requirements.txt --without-hashes")
    run_command("poetry export -f requirements.txt --output requirements-dev.txt --with dev --without-hashes")
    print("✅ Archivos generados: requirements.txt, requirements-dev.txt")

def main():
    parser = argparse.ArgumentParser(description="Gestionar dependencias con Poetry")
    parser.add_argument("action", choices=["list", "add", "remove", "update", "outdated", "export"], 
                       help="Acción a realizar")
    parser.add_argument("package", nargs="?", help="Nombre del paquete (para add/remove)")
    parser.add_argument("--dev", action="store_true", help="Agregar como dependencia de desarrollo")
    
    args = parser.parse_args()
    
    # Cambiar al directorio del proyecto
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    if args.action == "list":
        list_dependencies()
    elif args.action == "add":
        if not args.package:
            print("❌ Especifica el nombre del paquete")
            sys.exit(1)
        add_dependency(args.package, args.dev)
    elif args.action == "remove":
        if not args.package:
            print("❌ Especifica el nombre del paquete")
            sys.exit(1)
        remove_dependency(args.package)
    elif args.action == "update":
        update_dependencies()
    elif args.action == "outdated":
        check_outdated()
    elif args.action == "export":
        export_requirements()

if __name__ == "__main__":
    main()
