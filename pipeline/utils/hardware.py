import os
import sys
import psutil
import platform
import json
def get_hardware_facts():
    """Detecta as especificidades do hardware atual."""
    import torch
    facts = {
        "cpu_name": platform.processor() or "Generic CPU",
        "cpu_cores": psutil.cpu_count(logical=False) or 4,
        "cpu_threads": psutil.cpu_count(logical=True) or 8,
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "gpu_available": False,
        "gpu_name": "Nenhuma",
        "gpu_vram_gb": None
    }

    # Tenta detecção via CUDA
    try:
        if torch.cuda.is_available():
            facts["gpu_available"] = True
            facts["gpu_name"] = torch.cuda.get_device_name(0)
            # VRAM em GB
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            facts["gpu_vram_gb"] = round(vram_bytes / (1024**3), 1)
    except Exception:
        pass

    return facts

if __name__ == "__main__":
    print(json.dumps(get_hardware_facts(), ensure_ascii=False))
