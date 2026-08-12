import py_compile
import sys
import os
errors=0
for root, dirs, files in os.walk('.'):
    if '.git' in root or 'venv' in root or 'env' in root:
        continue
    for f in files:
        if f.endswith('.py'):
            path=os.path.join(root,f)
            try:
                py_compile.compile(path, doraise=True)
            except Exception as e:
                print(f"{path}: {e}")
                errors+=1
if errors:
    sys.exit(1)
else:
    print('All .py files compiled successfully')
