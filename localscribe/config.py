"""User settings, persisted as JSON in %APPDATA%/LocalScribe/settings.json."""
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path


def app_data_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "LocalScribe"


def default_output_dir() -> Path:
    return Path.home() / "Documents" / "LocalScribe"


@dataclass
class Settings:
    # Transcription
    whisper_model: str = "large-v3-turbo"
    language: str = "en"
    device: str = "auto"          # auto | cuda | cpu
    compute_type: str = "auto"    # auto | float16 | int8_float16 | int8 | float32
    beam_size: int = 5
    batch_size: int = 0           # >1 uses faster-whisper's batched pipeline (faster, more VRAM)
    chunk_minutes: float = 10.0

    # Summarization
    summarize: bool = True
    llm_path: str = ""            # GGUF instruct model
    llm_ctx: int = 8192
    llm_gpu_layers: int = -1      # -1 = auto-fit to free VRAM, 0 = CPU only

    output_dir: str = str(default_output_dir())

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


def settings_path() -> Path:
    return app_data_dir() / "settings.json"


def load_settings(path: Path = None) -> Settings:
    path = path or settings_path()
    try:
        return Settings.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return Settings()


def save_settings(settings: Settings, path: Path = None) -> None:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, path)
