"""Windows DLL-search fixes, applied before llama_cpp or PySide6 are imported.

1. CUDA: llama-cpp-python loads its DLLs with ``winmode=0``, which ignores
   ``os.add_dll_directory`` and searches only PATH. An activated conda env puts
   ``<env>/Library/bin`` (cudart, cublas, cudnn) on PATH; a desktop shortcut or
   a plain ``python.exe`` launch does not, and llama.dll then fails to load.

2. ICU: Qt6Core.dll imports ``icuuc.dll`` expecting Windows' own copy in
   System32. Conda Python searches ``<env>/Library/bin`` first, and a conda-forge
   ``icu`` package there exports versioned symbols (``ucnv_open_78``), so Qt
   fails with "The specified procedure could not be found". Loading the System32
   copy first makes Qt bind to it, since Windows reuses an already-loaded module.
"""
import ctypes
import os
import sys
from pathlib import Path


def ensure() -> None:
    if sys.platform != "win32":
        return
    _cuda_on_path()
    _preload_system_icu()


def _cuda_on_path() -> None:
    lib_bin = Path(sys.prefix) / "Library" / "bin"
    if not lib_bin.is_dir():
        return
    path = os.environ.get("PATH", "")
    entries = {p.rstrip("\\/").lower() for p in path.split(os.pathsep) if p}
    if str(lib_bin).lower() not in entries:
        os.environ["PATH"] = str(lib_bin) + os.pathsep + path
    os.environ.setdefault("CUDA_PATH", str(lib_bin.parent))
    os.add_dll_directory(str(lib_bin))


def _preload_system_icu() -> None:
    system_icu = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "icuuc.dll"
    if system_icu.exists():
        try:
            ctypes.WinDLL(str(system_icu))
        except OSError:
            pass
