"""User settings, persisted as JSON in %APPDATA%/LocalScribe/settings.json."""
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# LLM outputs a job can produce on top of the transcript. Only the selected
# ones are computed, so a movie retelling never pays for study notes.
OUTPUTS = {
    "summary": "Summary",          # overview, key points, timestamped chapters
    "notes": "Study notes",        # key concepts, takeaways, examples, review questions
    "story": "Engaging Story",     # narrator script retelling the whole recording
}

# Content-type presets: (label, outputs, detect speakers)
PRESETS = {
    "lecture": ("Lecture / class", ["summary", "notes"], False),
    "podcast": ("Podcast / talk / interview", ["summary"], True),
    "movie": ("Movie / story", ["story"], True),
}

STORY_TONES = {
    "auto": "Match the film's genre",
    "suspense": "Suspense / thriller",
    "epic": "Epic adventure",
    "drama": "Emotional drama",
    "comedy": "Light-hearted",
    "dark": "Dark / eerie",
    "bedtime": "Gentle bedtime story",
    "documentary": "Documentary recap",
}

# Story length is chosen as reading time; the story is written for reading aloud.
STORY_MINUTES = (5, 10, 15, 20, 30)
SPOKEN_WPM = 165
_LEGACY_LENGTHS = {"short": 5, "medium": 15, "long": 30}  # v2 "story_length" values


def story_minutes(value) -> int:
    """Minutes from an int or a v2 length name ('short', 'medium', 'long')."""
    if isinstance(value, str) and not value.isdigit():
        return _LEGACY_LENGTHS.get(value, 15)
    return max(1, int(value))


def app_data_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "LocalScribe"


def default_output_dir() -> Path:
    return Path.home() / "Documents" / "LocalScribe"


def models_dir() -> Path:
    return default_output_dir() / "models"


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

    # What to produce
    content_type: str = "lecture"                                     # a PRESETS key or "custom"
    outputs: list = field(default_factory=lambda: ["summary", "notes"])
    diarize: bool = False                                             # label speakers
    num_speakers: int = 0                                             # 0 = detect automatically
    story_tone: str = "auto"
    story_minutes: int = 15
    translate_to: str = ""        # "" or a translator.LANGUAGES code ("ta"): also write the story in it

    # Local LLM
    llm_path: str = ""            # GGUF instruct model
    llm_ctx: int = 8192
    llm_gpu_layers: int = -1      # -1 = auto-fit to free VRAM, 0 = CPU only

    output_dir: str = str(default_output_dir())

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        data = dict(data)
        if "outputs" not in data and data.get("summarize") is False:  # v1 settings file
            data["outputs"] = []
        if "story_minutes" not in data and "story_length" in data:  # v2 settings file
            data["story_minutes"] = story_minutes(data["story_length"])
        known = {f.name for f in fields(cls)}
        settings = cls(**{k: v for k, v in data.items() if k in known})
        settings.outputs = [o for o in settings.outputs if o in OUTPUTS]
        return settings

    def to_dict(self) -> dict:
        return asdict(self)

    def apply_preset(self, name: str) -> None:
        _, outputs, diarize = PRESETS[name]
        self.content_type = name
        self.outputs = list(outputs)
        self.diarize = diarize


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
