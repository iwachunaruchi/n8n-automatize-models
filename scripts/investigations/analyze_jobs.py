import os
import re
import sys

def analyze_job_functions():
    api_dir = 'api'
    function_count = 0
    functions_with_job_props = []
    
    # Patterns to identify job property usage
    job_property_patterns = [
        r'jobs_state\[',
        r'job\[[\'\"]\w+[\'\"]\]',  # job['property']
        r'job\.get\(',
        r'job\.__dict__',
        r'\.job_id',
        r'\.status',
        r'\.progress', 
        r'\.created_at',
        r'\.results',
        r'\.start_time',
        r'\.type'
    ]
    
    # Walk through all Python files in api directory
    for root, dirs, files in os.walk(api_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Find all function definitions
                    function_matches = re.finditer(r'def\s+(\w+)\s*\(', content)
                    
                    for func_match in function_matches:
                        func_name = func_match.group(1)
                        func_start = func_match.start()
                        
                        # Find the end of this function (next function or end of file)
                        next_func = re.search(r'\ndef\s+\w+\s*\(', content[func_start + 1:])
                        if next_func:
                            func_end = func_start + next_func.start() + 1
                        else:
                            func_end = len(content)
                        
                        func_content = content[func_start:func_end]
                        
                        # Check if this function uses job properties
                        uses_job_props = False
                        for pattern in job_property_patterns:
                            if re.search(pattern, func_content):
                                uses_job_props = True
                                break
                        
                        if uses_job_props:
                            function_count += 1
                            relative_path = filepath.replace('api/', '').replace('api\\', '')
                            functions_with_job_props.append(f'{relative_path}::{func_name}')
                            
                except Exception as e:
                    print(f'Error reading {filepath}: {e}')
    
    return function_count, functions_with_job_props

if __name__ == "__main__":
    count, functions = analyze_job_functions()
    print(f'TOTAL DE FUNCIONES QUE USAN PROPIEDADES DE JOBS: {count}')
    print()
    print('Lista de funciones que usan propiedades de jobs:')
    print('=' * 50)
    for i, func in enumerate(functions, 1):
        print(f'{i:2d}. {func}')
