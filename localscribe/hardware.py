"""Detect GPU/CPU/RAM and pick runtime options that fit the machine."""
import subprocess
import sys
from dataclasses import dataclass

import psutil

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass
class Hardware:
    gpu_name: str
    vram_total_mb: int
    ram_gb: float
    physical_cores: int

    @property
    def has_cuda(self) -> bool:
        return self.vram_total_mb > 0


def _nvidia_smi(query: str):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return [part.strip() for part in out.stdout.splitlines()[0].split(",")]


def vram_free_mb() -> int:
    row = _nvidia_smi("memory.free")
    try:
        return int(row[0]) if row else 0
    except ValueError:
        return 0


def detect() -> Hardware:
    row = _nvidia_smi("name,memory.total")
    gpu_name, vram = "", 0
    if row and len(row) >= 2:
        gpu_name = row[0]
        try:
            vram = int(float(row[1]))
        except ValueError:
            vram = 0
    return Hardware(
        gpu_name=gpu_name,
        vram_total_mb=vram,
        ram_gb=psutil.virtual_memory().total / 2**30,
        physical_cores=psutil.cpu_count(logical=False) or 4,
    )


def cuda_usable() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def whisper_runtime(device: str, compute_type: str):
    """Resolve 'auto' settings to a concrete (device, compute_type)."""
    if device == "auto":
        device = "cuda" if cuda_usable() else "cpu"
    if compute_type == "auto":
        # int8 weights + fp16 math: least VRAM on small GPUs, near-fp16 accuracy.
        compute_type = "int8_float16" if device == "cuda" else "int8"
    return device, compute_type
