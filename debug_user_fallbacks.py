import os
import sys

# Inject the user's cloned repositories into the path
sys.path.insert(0, r"D:\PADDLEOCR")
sys.path.insert(0, r"D:\ultralytics")

import subprocess
from pathlib import Path

# We run the internal test pipeline script with these paths
ROOT = Path(r"D:\TraduzAi")
os.environ["PYTHONPATH"] = f"D:\\PADDLEOCR;D:\\ultralytics;{os.environ.get('PYTHONPATH', '')}"

print("--> Executando debug pipeline no debug_test.jpg...")
try:
    cmd = [sys.executable, str(ROOT / "run_full_debug.py")]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    
    print("--- STDOUT ---")
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)
    
except Exception as e:
    print(f"Erro ao executar: {e}")
