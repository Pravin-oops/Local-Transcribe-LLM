"""Find the models that are actually on this machine, for the model pickers.

Whisper models live in the Hugging Face cache (downloaded by faster-whisper);
LLMs are .gguf files in Documents/LocalScribe/models (plus the one selected in
settings, wherever it is). Nothing here imports faster-whisper or llama.cpp,
so the window stays quick to open.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path

from localscribe.config import models_dir

# faster-whisper names -> Hugging Face repo, and approximate download size (MB).
WHISPER_CATALOG = {
    "large-v3-turbo": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", 1620),
    "distil-large-v3": ("Systran/faster-distil-whisper-large-v3", 1510),
    "large-v3": ("Systran/faster-whisper-large-v3", 3090),
    "medium.en": ("Systran/faster-whisper-medium.en", 1530),
    "small.en": ("Systran/faster-whisper-small.en", 484),
    "base.en": ("Systran/faster-whisper-base.en", 145),
    "tiny.en": ("Systran/faster-whisper-tiny.en", 75),
}
WHISPER_NOTES = {
    "large-v3-turbo": "best balance", "distil-large-v3": "fast, English", "large-v3": "most accurate, slow",
    "medium.en": "good, English", "small.en": "fast, English", "base.en": "fastest usable",
    "tiny.en": "draft quality",
}


@dataclass
class ModelInfo:
    key: str           # Whisper model name, or path to a .gguf file
    label: str
    size_bytes: int
    note: str = ""

    @property
    def size(self) -> str:
        return pretty_size(self.size_bytes)


def pretty_size(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    return f"{max(1, round(n / 1e6))} MB"


def hf_cache_dir() -> Path:
    try:
        from huggingface_hub import constants
        return Path(constants.HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _whisper_weights(repo: str, cache: Path):
    snapshots = cache / f"models--{repo.replace('/', '--')}" / "snapshots"
    for snap in snapshots.glob("*") if snapshots.is_dir() else []:
        weights = snap / "model.bin"
        if weights.exists():
            return weights
    return None


def installed_whisper(cache: Path = None) -> list:
    """Whisper models that are downloaded and ready, in catalog order."""
    cache = cache or hf_cache_dir()
    found = []
    for name, (repo, _) in WHISPER_CATALOG.items():
        weights = _whisper_weights(repo, cache)
        if weights:
            found.append(ModelInfo(name, name, weights.stat().st_size, WHISPER_NOTES.get(name, "")))
    return found


def downloadable_whisper(cache: Path = None) -> list:
    """Catalog models not downloaded yet, with their approximate download size."""
    have = {m.key for m in installed_whisper(cache)}
    return [ModelInfo(name, name, mb * 1_000_000, WHISPER_NOTES.get(name, ""))
            for name, (_, mb) in WHISPER_CATALOG.items() if name not in have]


_META_CACHE = {}


def _gguf_label(path: Path) -> str:
    """'Meta Llama 3.1 8B Instruct · Q4_K_M' from the GGUF header, falling back to the file name."""
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _META_CACHE:
        name = ""
        try:
            from localscribe.summarizer import read_gguf_metadata
            name = str(read_gguf_metadata(str(path)).get("general.name", "")).strip()
        except Exception:
            pass
        if not name:
            name = re.sub(r"[-_.]?(Q\d\w*|IQ\d\w*|F16|BF16|F32)$", "", path.stem, flags=re.I).replace("-", " ")
        quant = re.search(r"(I?Q\d[\w]*|BF16|F16|F32)(?=$)", path.stem, flags=re.I)
        _META_CACHE[key] = f"{name} · {quant.group(1).upper()}" if quant else name
    return _META_CACHE[key]


def installed_llms(selected: str = "", folder: Path = None) -> list:
    """GGUF models in the models folder, plus the selected file if it lives elsewhere."""
    folder = folder or models_dir()
    paths = list(folder.glob("*.gguf")) if folder.is_dir() else []
    found = sorted((ModelInfo(str(p), _gguf_label(p), p.stat().st_size) for p in paths),
                   key=lambda m: m.label.lower())
    if selected and Path(selected).is_file() and Path(selected).resolve() not in {p.resolve() for p in paths}:
        found.append(ModelInfo(selected, _gguf_label(Path(selected)), Path(selected).stat().st_size))
    return found


def open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.startfile(path)
