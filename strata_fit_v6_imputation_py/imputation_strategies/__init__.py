import importlib
from pathlib import Path

# Dynamically import all modules in the current folder
package_dir = Path(__file__).parent

for file in package_dir.glob("*.py"):
    if file.name in ("__init__.py", "base.py"):
        continue
    module_name = f".{file.stem}"
    importlib.import_module(module_name, package=__name__)
