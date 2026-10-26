"""Local GGUF backend plus map-reduce for the Summary and Study notes outputs.

Map:    the transcript is cut into blocks that fit the context window; each
        block yields only the fields the selected outputs need (chapters, key
        points, concepts, examples), saved to ``notes/NNNN.json`` so an
        interrupted run resumes.
Reduce: block notes are condensed in groups until they fit, then turned into
        the summary and/or study notes. Chapters are concatenated in time order.
"""
import gc
import hashlib
import json
import os
import re
import struct
from collections import Counter
from pathlib import Path

from localscribe import prompts
from localscribe.jobs import read_json, write_json_atomic
from localscribe.merger import group_segments, labelled
from localscribe.timefmt import hms, parse_hms
from localscribe.transcriber import Cancelled

MAP_OUTPUT_TOKENS = 1600  # a full ~12 min block asks for ~18 points + chapters + concepts
CONDENSE_OUTPUT_TOKENS = 1500
REDUCE_OUTPUT_TOKENS = 2000
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
        """Schema-constrained JSON (llama.cpp grammar), low temperature."""
        return self._stream(system, user, max_tokens, 0.2, cancel, on_token,
                            response_format={"type": "json_object", "schema": schema})

    def complete_text(self, system: str, user: str, max_tokens: int, cancel=None, on_token=None,
                      temperature: float = 0.8) -> str:
        """Free-form prose (used for the story script)."""
        return self._stream(system, user, max_tokens, temperature, cancel, on_token,
                            repeat_penalty=1.1)

    def _stream(self, system, user, max_tokens, temperature, cancel, on_token, **options) -> str:
        stream = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature, top_p=0.9, max_tokens=max_tokens, stream=True, **options,
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
    return f"[{hms(line['start'])}] {labelled(line)}"


def transcript_blocks(segments: list, backend, output_tokens: int) -> list:
    """~30 s timestamped lines packed into blocks that fit the context window."""
    lines = group_segments(segments, max_span=30.0, max_gap=3.0)
    budget = min(3000, backend.n_ctx - output_tokens - PROMPT_OVERHEAD_TOKENS)
    return split_blocks(lines, backend.count_tokens, budget)


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def block_key(backend, block: list, **extra) -> dict:
    """Cache key for a block's LLM result: changes when model, prompts or text
    (including speaker labels) change."""
    return {"model": backend.name, "prompts": prompts.VERSION, "start": block[0]["start"],
            "end": block[-1]["end"], "text": digest("\n".join(format_line(l) for l in block)), **extra}


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


def complete_structured(backend, user: str, schema: dict, max_tokens: int, cancel, on_token, warn,
                        system: str = prompts.SYSTEM):
    """Ask for JSON; on truncated/invalid output retry once with more room."""
    for attempt, budget in enumerate((max_tokens, int(max_tokens * 1.6))):
        prompt = user if attempt == 0 else user + "\nKeep every item brief."
        data = _parse_json(backend.complete_json(system, prompt, schema, budget,
                                                 cancel=cancel, on_token=on_token))
        if data is not None:
            return data
    warn("The model returned unusable output for one step; that part is left out.")
    return None


# --------------------------------------------------------------------------- map-reduce

def summarize(segments: list, backend, job, title: str, outputs, on_progress, cancel=None,
              warn=print) -> dict:
    """Map-reduce for the "summary" and/or "notes" outputs.

    The map step asks only for the fields the selected outputs need (chapters
    and key points for a summary; key points, concepts and examples for study
    notes). Returns {"summary", "chapters", "notes"} with None for outputs
    not requested. ``on_progress(fraction, message)`` reports progress.
    """
    outputs = [o for o in ("summary", "notes") if o in outputs]
    result = {"summary": None, "chapters": [], "notes": None}
    fields = prompts.fields_for(outputs)
    if not fields:
        return result
    blocks = transcript_blocks(segments, backend, MAP_OUTPUT_TOKENS)
    if not blocks:
        return result
    total, steps = len(blocks), len(blocks) + len(outputs)

    chapters, note_sets = [], []
    for i, block in enumerate(blocks):
        key = block_key(backend, block)
        cached = read_json(job.note_path(i))
        # A cached result is reusable if it has every field we need now (e.g. a
        # summary+notes run serves a later summary-only run).
        if cached and cached.get("key") == key and set(fields) <= set(cached.get("fields", [])):
            data = cached["raw"]
        else:
            message = f"Reading part {i + 1}/{total}"
            on_progress(i / steps, message)
            minutes = (block[-1]["end"] - block[0]["start"]) / 60
            user = prompts.map_prompt("\n".join(format_line(l) for l in block),
                                      hms(block[0]["start"]), hms(block[-1]["end"]), i + 1, total,
                                      minutes, fields)
            data = complete_structured(backend, user, prompts.map_schema(fields), MAP_OUTPUT_TOKENS,
                                       cancel, lambda n: on_progress(i / steps, f"{message} ({n} tokens)"),
                                       warn) or {}
            # Raw model output is cached; cleaning runs on every assembly so fixes
            # to timestamp handling also apply to resumed jobs.
            write_json_atomic(job.note_path(i), {"key": key, "fields": fields, "raw": data})
        if "summary" in outputs:
            chapters.extend(clean_chapters(data.get("chapters"), block, segments))
        note_sets.append({**{f: data.get(f, []) for f in fields if f != "chapters"},
                          "label": f"Part {i + 1} [{hms(block[0]['start'])}-{hms(block[-1]['end'])}]"})

    note_sets = _fit_notes(note_sets, fields, backend, cancel, warn,
                           lambda n: on_progress(total / steps, f"Condensing notes ({n} tokens)"))
    digest = notes_digest(note_sets)
    done = total
    for output, label, prompt_fn, schema in (
            ("summary", "Writing summary", prompts.summary_prompt, prompts.SUMMARY_SCHEMA),
            ("notes", "Writing study notes", prompts.notes_prompt, prompts.NOTES_SCHEMA)):
        if output not in outputs:
            continue
        on_progress(done / steps, label)
        result[output] = complete_structured(
            backend, prompt_fn(digest, title), schema, REDUCE_OUTPUT_TOKENS, cancel,
            lambda n, label=label, done=done: on_progress(done / steps, f"{label} ({n} tokens)"), warn)
        done += 1
    result["chapters"] = merge_chapters(chapters)
    on_progress(1.0, "Notes complete")
    return result


def _fit_notes(note_sets: list, fields, backend, cancel, warn, on_token) -> list:
    """Condense block notes in groups until they fit the reduce prompt."""
    budget = backend.n_ctx - REDUCE_OUTPUT_TOKENS - PROMPT_OVERHEAD_TOKENS
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
            data = complete_structured(backend, prompts.condense_prompt(notes_digest(g), fields),
                                       prompts.condense_schema(fields), CONDENSE_OUTPUT_TOKENS,
                                       cancel, on_token, warn)
            if data:
                label = g[0]["label"] if len(g) == 1 else f"{g[0]['label']} to {g[-1]['label']}"
                condensed.append({**data, "label": label})
        note_sets = condensed or note_sets
    return note_sets


# --------------------------------------------------------------------------- rendering

def render_summary(summary: dict, chapters: list) -> str:
    out = ["## Summary", ""]
    if summary:
        out += [summary.get("overview", "").strip(), ""]
        if summary.get("key_points"):
            out += ["### Key Points", ""] + [f"- {p}" for p in summary["key_points"]] + [""]
    else:
        out += ["*The summary could not be generated.*", ""]
    if chapters:
        out += ["### Chapters", ""]
        for ch in chapters:
            line = f" — {ch['summary']}" if ch.get("summary") else ""
            out.append(f"- **[{hms(ch['start'])}] {ch['title']}**{line}")
        out.append("")
    return "\n".join(out)


def render_notes(notes: dict) -> str:
    if not notes:
        return "## Study Notes\n\n*Study notes could not be generated.*\n"
    out = ["## Study Notes", ""]
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
