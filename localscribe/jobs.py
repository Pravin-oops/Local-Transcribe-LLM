"""Per-file job folders with a JSON manifest, so interrupted work can resume.

Layout of ``<output_dir>/jobs/<name>-<hash>/``::

    manifest.json        source info, chunk plan, settings used, stage flags
    audio.pcm            decoded 16 kHz mono int16 audio
    parts/0000.json ...  transcribed segments per chunk (absolute timestamps)
    notes/0000.json ...  per-block summary notes (map step)
    transcript.txt/.srt/.vtt/.json, chapters.md, notes.md, summary.md
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


def write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def fingerprint(source: Path) -> str:
    st = source.stat()
    key = f"{source.name}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def _safe_name(stem: str) -> str:
    cleaned = re.sub(r"[^\w\- ]+", "_", stem).strip(" ._") or "audio"
    return cleaned[:60]


class Job:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def root_for(source: Path, output_dir: Path) -> Path:
        """Where the job for ``source`` lives (without creating it)."""
        source = Path(source).resolve()
        return Path(output_dir) / "jobs" / f"{_safe_name(source.stem)}-{fingerprint(source)}"

    @classmethod
    def for_source(cls, source: Path, output_dir: Path) -> "Job":
        source = Path(source).resolve()
        job = cls(cls.root_for(source, output_dir))
        if not job.manifest_path.exists():
            job.save_manifest({
                "version": 1,
                "source": str(source),
                "source_size": source.stat().st_size,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "stages": {},
            })
        return job

    # paths
    manifest_path = property(lambda self: self.root / "manifest.json")
    pcm_path = property(lambda self: self.root / "audio.pcm")
    parts_dir = property(lambda self: self.root / "parts")
    notes_dir = property(lambda self: self.root / "notes")

    def output(self, name: str) -> Path:
        return self.root / name

    def part_path(self, index: int) -> Path:
        return self.parts_dir / f"{index:04d}.json"

    def note_path(self, index: int) -> Path:
        return self.notes_dir / f"{index:04d}.json"

    # manifest
    def load_manifest(self) -> dict:
        return read_json(self.manifest_path, {}) or {}

    def save_manifest(self, manifest: dict) -> None:
        write_json_atomic(self.manifest_path, manifest)

    def update_manifest(self, **changes) -> dict:
        manifest = self.load_manifest()
        manifest.update(changes)
        self.save_manifest(manifest)
        return manifest

    def set_stage(self, stage: str, done: bool) -> None:
        manifest = self.load_manifest()
        manifest.setdefault("stages", {})[stage] = done
        self.save_manifest(manifest)

    def stage_done(self, stage: str) -> bool:
        return bool(self.load_manifest().get("stages", {}).get(stage))

    # parts
    def read_part(self, index: int):
        return read_json(self.part_path(index))

    def write_part(self, index: int, segments: list) -> None:
        write_json_atomic(self.part_path(index), segments)

    def clear_dir(self, directory: Path) -> None:
        if directory.is_dir():
            for f in directory.glob("*.json"):
                f.unlink()
