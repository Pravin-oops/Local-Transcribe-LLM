"""Speaker diarization ("who spoke when") with sherpa-onnx.

Uses pyannote segmentation-3.0 and NVIDIA's English TitaNet-small speaker
embeddings, both as ONNX files run on the CPU through sherpa-onnx: no PyTorch,
no Hugging Face token. The models (~47 MB) are downloaded once from the
sherpa-onnx GitHub releases into Documents/LocalScribe/models/diarization.

TitaNet-small was chosen over CAM++ (VoxCeleb) after a two-voice test drama:
2 speakers found with 99% of lines attributed correctly, versus 10 speakers
(5 after merging) and 90% for CAM++, at the same speed.
"""
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from localscribe.config import models_dir
from localscribe.transcriber import Cancelled

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMBEDDING_URL = f"{_RELEASES}/speaker-recongition-models/nemo_en_titanet_small.onnx"
ENGINE = "sherpa-onnx/pyannote-seg-3.0+titanet-small-en/shift0.5+merge"  # recorded in the cache key


def model_paths():
    root = models_dir() / "diarization"
    return root / "pyannote-segmentation-3.0.onnx", root / "titanet-small-en.onnx"


def _download(url: str, dest: Path, on_status) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if on_status:
        on_status(f"Downloading speaker model {dest.name} (first use only)...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "LocalScribe"})
    with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    os.replace(tmp, dest)


def ensure_models(on_status=None):
    """Return (segmentation, embedding) model paths, downloading them if missing."""
    segmentation, embedding = model_paths()
    if not segmentation.exists():
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "segmentation.tar.bz2"
            _download(SEGMENTATION_URL, archive, on_status)
            with tarfile.open(archive) as tar:
                member = next(m for m in tar.getmembers()
                              if m.isfile() and m.name.endswith("/model.onnx"))
                member.name = segmentation.name  # flatten; never trust archive paths
                tar.extract(member, segmentation.parent, filter="data")
    if not embedding.exists():
        _download(EMBEDDING_URL, embedding, on_status)
    return segmentation, embedding


def diarize(samples, num_speakers: int = 0, threads: int = 4, on_progress=None, cancel=None,
            threshold: float = 0.5, window_shift: float = 0.5, on_status=None) -> list:
    """Return speaker turns [{"start", "end", "speaker": int}] for float32 16 kHz audio.

    ``num_speakers`` > 0 fixes the number of speakers; 0 lets clustering decide
    (``threshold``: lower finds more speakers), followed by :func:`merge_speakers`.
    ``window_shift`` is the step of the 10 s analysis window as a fraction of
    it. 0.5 runs about 4x faster than sherpa-onnx's 0.1 at equal accuracy on
    our tests (~6 min per 3 h of audio on an 8-core laptop CPU).
    """
    import sherpa_onnx

    segmentation, embedding = ensure_models(on_status)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation), window_shift_ratio=window_shift),
            num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers > 0 else -1, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    def callback(done, total):
        if on_progress and total:
            on_progress(done / total)
        return 1 if cancel is not None and cancel.is_set() else 0

    result = sd.process(samples, callback=callback)
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    turns = [{"start": round(r.start, 2), "end": round(r.end, 2), "speaker": int(r.speaker)}
             for r in result.sort_by_start_time()]
    if num_speakers > 0:
        return turns
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding), num_threads=threads))

    def embed(clip):
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=clip)
        stream.input_finished()
        return extractor.compute(stream)

    return merge_speakers(samples, turns, embed)


def _unit(v):
    import numpy as np
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) or 1.0)


def merge_speakers(samples, turns: list, embed, threshold: float = 0.75, min_seconds: float = 3.0,
                   clip_seconds: float = 30.0) -> list:
    """Second pass over automatic clustering, which tends to split one voice into
    several speakers and to leave tiny fragment clusters.

    Each speaker gets one voice embedding (from up to ``clip_seconds`` of their
    longest turns). Speakers whose voices are at least ``threshold`` cosine-
    similar are merged, then speakers with under ``min_seconds`` of speech in
    total are absorbed into the most similar remaining speaker.
    """
    import numpy as np

    sr = 16000
    by_spk = {}
    for t in turns:
        by_spk.setdefault(t["speaker"], []).append(t)
    if len(by_spk) < 2:
        return turns
    groups = {}
    for spk, ts in by_spk.items():
        longest = sorted(ts, key=lambda t: t["end"] - t["start"], reverse=True)
        clip = np.concatenate([samples[int(t["start"] * sr):int(t["end"] * sr)] for t in longest])
        groups[spk] = {"members": {spk}, "seconds": sum(t["end"] - t["start"] for t in ts),
                       "emb": _unit(embed(clip[: int(clip_seconds * sr)]))}

    def merge(a, b):
        ga, gb = groups[a], groups.pop(b)
        total = ga["seconds"] + gb["seconds"]
        ga["emb"] = _unit(ga["emb"] * ga["seconds"] + gb["emb"] * gb["seconds"])
        ga["members"] |= gb["members"]
        ga["seconds"] = total

    while len(groups) > 1:
        ids = list(groups)
        a, b = max(((x, y) for i, x in enumerate(ids) for y in ids[i + 1:]),
                   key=lambda p: float(groups[p[0]]["emb"] @ groups[p[1]]["emb"]))
        if float(groups[a]["emb"] @ groups[b]["emb"]) < threshold:
            break
        merge(*sorted((a, b), key=lambda s: -groups[s]["seconds"]))
    for small in sorted(groups, key=lambda s: groups[s]["seconds"]):
        if len(groups) < 2 or groups[small]["seconds"] >= min_seconds:
            continue
        target = max((s for s in groups if s != small),
                     key=lambda s: float(groups[s]["emb"] @ groups[small]["emb"]))
        merge(target, small)

    owner = {m: g for g, info in groups.items() for m in info["members"]}
    return [{**t, "speaker": owner[t["speaker"]]} for t in turns]


def assign_speakers(segments: list, turns: list, max_distance: float = 2.0) -> list:
    """Label each transcript segment with the speaker who overlaps it most.

    Speakers are renamed "Speaker 1", "Speaker 2", ... in order of first
    appearance. A segment with no overlapping turn takes the nearest turn
    within ``max_distance`` seconds, otherwise the previous segment's speaker.
    """
    names, out, previous = {}, [], None
    for seg in segments:
        overlap = {}
        for t in turns:
            o = min(seg["end"], t["end"]) - max(seg["start"], t["start"])
            if o > 0:
                overlap[t["speaker"]] = overlap.get(t["speaker"], 0) + o
        if overlap:
            spk = max(overlap, key=overlap.get)
        else:
            mid = (seg["start"] + seg["end"]) / 2
            near = min(turns, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])), default=None)
            close = near and min(abs(mid - near["start"]), abs(mid - near["end"])) <= max_distance
            spk = near["speaker"] if close else previous
        if spk is not None and spk not in names:
            names[spk] = f"Speaker {len(names) + 1}"
        labelled = dict(seg)
        if spk is not None:
            labelled["speaker"] = names[spk]
        out.append(labelled)
        previous = spk
    return out
