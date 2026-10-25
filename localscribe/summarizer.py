"""Map-reduce summarization of long transcripts with a local GGUF model.

Map:    the transcript is cut into blocks that fit the context window; each
        block yields chapters (with timestamps) and study notes, saved to
        ``notes/NNNN.json`` so an interrupted run resumes.
Reduce: block notes are condensed in groups until they fit, then merged into
        the final study notes. Chapters are concatenated in time order.
"""
import gc
import json
import os
import re
import struct
from collections import Counter
from pathlib import Path

from localscribe import prompts
from localscribe.jobs import read_json, write_json_atomic
from localscribe.merger import group_segments
from localscribe.timefmt import hms, parse_hms
from localscribe.transcriber import Cancelled

MAP_OUTPUT_TOKENS = 1600  # a full ~12 min block asks for ~18 points + chapters + concepts
CONDENSE_OUTPUT_TOKENS = 1500
NOTES_OUTPUT_TOKENS = 2000
PROMPT_OVERHEAD_TOKENS = 600


# --------------------------------------------------------------------------- backend

_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf_metadata(path: str) -> dict:
    """Read scalar key/values from a GGUF header without loading the model."""
    def read(fmt):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    def read_str():
        return f.read(read("<Q")).decode("utf-8", "replace")

    def skip_value(vtype):
        if vtype == 8:
            f.seek(read("<Q"), 1)
        elif vtype == 9:
            etype, count = read("<I"), read("<Q")
            if etype in _GGUF_SCALARS:
                f.seek(struct.calcsize(_GGUF_SCALARS[etype]) * count, 1)
            else:
                for _ in range(count):
                    skip_value(etype)
        else:
            f.seek(struct.calcsize(_GGUF_SCALARS[vtype]), 1)

    meta = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        if read("<I") < 2:
            raise ValueError("GGUF v1 files are not supported")
        read("<Q")  # tensor count
        for _ in range(read("<Q")):
            key, vtype = read_str(), read("<I")
            if vtype in _GGUF_SCALARS:
                meta[key] = read(_GGUF_SCALARS[vtype])
            elif vtype == 8:
                meta[key] = read_str()
            else:
                skip_value(vtype)
    return meta


def auto_gpu_layers(model_path: str, n_ctx: int, free_vram_mb: int, reserve_mb: int = 700) -> int:
    """Estimate how many transformer layers fit in free VRAM (0 = CPU only).

    Per layer we count its share of the weights plus its fp16 KV cache.
    """
    if free_vram_mb <= reserve_mb:
        return 0
    meta = read_gguf_metadata(model_path)
    arch = meta.get("general.architecture", "llama")
    layers = int(meta.get(f"{arch}.block_count", 32))
    embd = int(meta.get(f"{arch}.embedding_length", 4096))
    heads = int(meta.get(f"{arch}.attention.head_count", 32))
    kv_heads = int(meta.get(f"{arch}.attention.head_count_kv", heads))
    head_dim = int(meta.get(f"{arch}.attention.key_length", embd // heads))

    file_mb = os.path.getsize(model_path) / 2**20
    weights_per_layer = file_mb / (layers + 2)  # +2: token embeddings and output head
    kv_per_layer = 2 * n_ctx * kv_heads * head_dim * 2 / 2**20
    fit = int((free_vram_mb - reserve_mb) // (weights_per_layer + kv_per_layer))
    return -1 if fit >= layers + 1 else max(0, min(fit, layers))


class LlamaBackend:
    def __init__(self, model_path: str, n_ctx: int, gpu_layers: int, threads: int, on_status=None):
        from llama_cpp import Llama

        self.name = Path(model_path).stem
        self.n_ctx = n_ctx
        attempts = [gpu_layers] + ([gpu_layers // 2, 0] if gpu_layers != 0 else [])
        last_error = None
        for layers in dict.fromkeys(attempts):
            try:
                self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=layers,
                                 n_threads=threads, n_batch=512, offload_kqv=True, verbose=False)
                self.gpu_layers = layers
                break
            except Exception as exc:  # out of VRAM -> retry with fewer layers
                last_error = exc
                if on_status:
                    on_status(f"Could not load model with {layers} GPU layers, retrying with fewer")
        else:
            raise RuntimeError(f"Could not load {model_path}: {last_error}")

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int,
                      cancel=None, on_token=None) -> str:
        stream = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object", "schema": schema},
            temperature=0.2, top_p=0.9, max_tokens=max_tokens, stream=True,
        )
        parts, n = [], 0
        for chunk in stream:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            delta = chunk["choices"][0].get("delta", {}).get("content")
            if delta:
                parts.append(delta)
                n += 1
                if on_token and n % 20 == 0:
                    on_token(n)
        return "".join(parts)

    def close(self) -> None:
        self.llm = None
        gc.collect()


# --------------------------------------------------------------------------- pure helpers

def split_blocks(lines: list, count_tokens, max_tokens: int) -> list:
    """Greedily pack timestamped lines into blocks of at most ``max_tokens``."""
    blocks, current, used = [], [], 0
    for line in lines:
        cost = count_tokens(format_line(line)) + 1
        if current and used + cost > max_tokens:
            blocks.append(current)
            current, used = [], 0
        current.append(line)
        used += cost
    if current:
        blocks.append(current)
    return blocks


def format_line(line: dict) -> str:
    return f"[{hms(line['start'])}] {line['text']}"


def _words(text: str) -> set:
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w[:-1] if w.endswith("s") and len(w) > 4 else w for w in words}  # 'matters' ~ 'matter'


def refine_start(line: dict, title: str, segments: list, floor: float, distinctive: set,
                 lookback: float = 90.0) -> float:
    """Move a chapter start to the first segment that mentions a distinctive
    word of its title, searching from up to ``lookback`` seconds before the
    chosen ~30 s line to its end, but never before ``floor``.

    Small models reliably name the topic but tend to pick a line after the
    topic begins ('Condensation' at 1:20 when 'The second stage is
    condensation' is said at 1:37 is the same failure in the other direction).
    """
    keywords = _words(title) & distinctive
    if keywords:
        lo = max(floor, line["start"] - lookback)
        for seg in segments:
            if seg["start"] < lo - 0.01:
                continue
            if seg["start"] > line["end"]:
                break
            if keywords & _words(seg["text"]):
                return seg["start"]
    return max(floor, line["start"])


def clean_chapters(raw: list, block: list, segments: list = None) -> list:
    """Validate model-written chapter starts against the block's real timestamps.

    A start outside the block (or unparseable) falls back to the block start;
    every start is snapped to the nearest actual line start, then refined to a
    segment start when ``segments`` are given.
    """
    lo, hi = block[0]["start"], block[-1]["end"]
    chosen = []
    for ch in raw or []:
        title = str(ch.get("title", "")).strip()
        if not title:
            continue
        t = parse_hms(str(ch.get("start", "")))
        if t is None or not (lo - 1 <= t <= hi + 1):
            t = lo
        line = min(block, key=lambda l: abs(l["start"] - t))
        chosen.append((line, title, str(ch.get("summary", "")).strip()))
    chosen.sort(key=lambda c: c[0]["start"])

    distinctive = set()
    if segments:
        in_block = [s for s in segments if lo - 0.01 <= s["start"] <= hi]
        df = Counter(w for s in in_block for w in _words(s["text"]))
        distinctive = {w for w, n in df.items() if n <= max(2, 0.25 * len(in_block))}

    out, floor = [], lo
    for line, title, summary in chosen:
        t = refine_start(line, title, segments, floor, distinctive) if segments else line["start"]
        out.append({"start": t, "title": title, "summary": summary})
        floor = t + 1.0
    out.sort(key=lambda c: c["start"])
    if not out:
        out.append({"start": lo, "title": f"Part starting {hms(lo)}", "summary": ""})
    return out


def merge_chapters(chapters: list) -> list:
    """Drop a chapter that repeats the previous title (common at block edges)."""
    merged = []
    for ch in sorted(chapters, key=lambda c: c["start"]):
        if merged and ch["title"].lower() == merged[-1]["title"].lower():
            continue
        if merged and ch["start"] == merged[-1]["start"]:
            continue
        merged.append(ch)
    return merged


def notes_digest(note_sets: list) -> str:
    """Render block notes as compact text for the reduce prompts."""
    chunks = []
    for item in note_sets:
        lines = [f"## {item.get('label', 'Notes')}"]
        for p in item.get("key_points", []):
            lines.append(f"- {p}")
        for c in item.get("concepts", []):
            lines.append(f"- Concept: {c.get('term', '')}: {c.get('explanation', '')}")
        for e in item.get("examples", []):
            lines.append(f"- Example: {e}")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)


def _parse_json(text: str):
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except ValueError:
        return None


def _complete(backend, user: str, schema: dict, max_tokens: int, cancel, on_token, warn):
    """Ask for JSON; on truncated/invalid output retry once with more room."""
    for attempt, budget in enumerate((max_tokens, int(max_tokens * 1.6))):
        prompt = user if attempt == 0 else user + "\nKeep every item brief."
        data = _parse_json(backend.complete_json(prompts.SYSTEM, prompt, schema, budget,
                                                 cancel=cancel, on_token=on_token))
        if data is not None:
            return data
    warn("The model returned unusable output for one step; that part is left out.")
    return None


# --------------------------------------------------------------------------- map-reduce

def summarize(segments: list, backend, job, title: str, on_progress, cancel=None, warn=print) -> dict:
    """Run map-reduce over ``segments``; returns {"chapters": [...], "notes": {...}}.

    ``on_progress(fraction, message)`` reports progress within summarization.
    """
    lines = group_segments(segments, max_span=30.0, max_gap=3.0)
    if not lines:
        return {"chapters": [], "notes": None}

    block_budget = min(3000, backend.n_ctx - MAP_OUTPUT_TOKENS - PROMPT_OVERHEAD_TOKENS)
    blocks = split_blocks(lines, backend.count_tokens, block_budget)
    total = len(blocks)

    chapters, note_sets = [], []
    for i, block in enumerate(blocks):
        key = {"model": backend.name, "prompts": prompts.VERSION, "start": block[0]["start"],
               "end": block[-1]["end"], "lines": len(block)}
        cached = read_json(job.note_path(i))
        if cached and cached.get("key") == key and "raw" in cached:
            data = cached["raw"]
        else:
            message = f"Summarizing part {i + 1}/{total}"
            on_progress(i / (total + 1), message)
            minutes = (block[-1]["end"] - block[0]["start"]) / 60
            user = prompts.map_prompt("\n".join(format_line(l) for l in block),
                                      hms(block[0]["start"]), hms(block[-1]["end"]), i + 1, total,
                                      minutes)
            data = _complete(backend, user, prompts.MAP_SCHEMA, MAP_OUTPUT_TOKENS, cancel,
                             lambda n: on_progress(i / (total + 1), f"{message} ({n} tokens)"),
                             warn) or {}
            # Raw model output is cached; cleaning runs on every assembly so fixes
            # to timestamp handling also apply to resumed jobs.
            write_json_atomic(job.note_path(i), {"key": key, "raw": data})
        result = {
            "chapters": clean_chapters(data.get("chapters"), block, segments),
            "key_points": data.get("key_points", []),
            "concepts": data.get("concepts", []),
            "examples": data.get("examples", []),
        }
        chapters.extend(result["chapters"])
        note_sets.append({**result, "label": f"Part {i + 1} [{hms(block[0]['start'])}-{hms(block[-1]['end'])}]"})

    on_progress(total / (total + 1), "Writing study notes")
    notes = _reduce(note_sets, backend, title, cancel, warn,
                    lambda n: on_progress(total / (total + 1), f"Writing study notes ({n} tokens)"))
    on_progress(1.0, "Summary complete")
    return {"chapters": merge_chapters(chapters), "notes": notes}


def _reduce(note_sets: list, backend, title: str, cancel, warn, on_token):
    budget = backend.n_ctx - NOTES_OUTPUT_TOKENS - PROMPT_OVERHEAD_TOKENS
    rounds = 0
    while backend.count_tokens(notes_digest(note_sets)) > budget and rounds < 4:
        rounds += 1
        groups, current = [], []
        for item in note_sets:
            candidate = current + [item]
            if current and backend.count_tokens(notes_digest(candidate)) > budget:
                groups.append(current)
                current = [item]
            else:
                current = candidate
        groups.append(current)
        condensed = []
        for g in groups:
            data = _complete(backend, prompts.condense_prompt(notes_digest(g)),
                             prompts.CONDENSE_SCHEMA, CONDENSE_OUTPUT_TOKENS, cancel, on_token, warn)
            if data:
                label = g[0]["label"] if len(g) == 1 else f"{g[0]['label']} to {g[-1]['label']}"
                condensed.append({**data, "label": label})
        note_sets = condensed or note_sets
    return _complete(backend, prompts.notes_prompt(notes_digest(note_sets), title),
                     prompts.NOTES_SCHEMA, NOTES_OUTPUT_TOKENS, cancel, on_token, warn)


# --------------------------------------------------------------------------- rendering

def render_chapters(chapters: list) -> str:
    out = ["## Chapters", ""]
    for ch in chapters:
        summary = f" — {ch['summary']}" if ch.get("summary") else ""
        out.append(f"- **[{hms(ch['start'])}] {ch['title']}**{summary}")
    return "\n".join(out) + "\n"


def render_notes(notes: dict) -> str:
    if not notes:
        return "## Study Notes\n\n_Study notes could not be generated._\n"
    out = ["## Study Notes", "", "### Overview", "", notes.get("overview", "").strip(), ""]
    if notes.get("key_concepts"):
        out += ["### Key Concepts", ""]
        out += [f"- **{c.get('term', '').strip()}**: {c.get('explanation', '').strip()}"
                for c in notes["key_concepts"]]
        out.append("")
    for heading, key in (("Key Takeaways", "key_takeaways"), ("Examples", "examples")):
        if notes.get(key):
            out += [f"### {heading}", ""] + [f"- {x}" for x in notes[key]] + [""]
    if notes.get("review_questions"):
        out += ["### Review Questions", ""]
        out += [f"{i}. {q}" for i, q in enumerate(notes["review_questions"], 1)] + [""]
    return "\n".join(out)
