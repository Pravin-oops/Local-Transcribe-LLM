"""Translate the finished Engaging Story into another language, offline.

The last stage of the pipeline: once ``story.md`` exists (and the LLM has been
unloaded) the story is translated sentence by sentence with Meta's NLLB-200
(distilled, 1.3B) through CTranslate2, the engine faster-whisper already
uses, and written next to the original as ``story.<lang>.md``.

A dedicated translation model is used because small instruct LLMs write poor
Tamil. The model (1.3 GB, int8) is downloaded once from Hugging Face into
Documents/LocalScribe/models/translation. Its licence is CC-BY-NC 4.0:
non-commercial use only.

Only prose is translated: Markdown structure, character names in the cast list
and the technical details line are kept, so both files line up.
"""
import re
from pathlib import Path

from localscribe.config import models_dir
from localscribe.transcriber import Cancelled

# code -> (name, NLLB code, {English label: translated label})
LANGUAGES = {
    "ta": ("Tamil", "tam_Taml", {"Cast": "கதாபாத்திரங்கள்"}),
}
SOURCE_LANGUAGE = "eng_Latn"
MODEL_REPO = "OpenNMT/nllb-200-distilled-1.3B-ct2-int8"
MODEL_NAME = "NLLB-200 1.3B"
MODEL_FILES = ("model.bin", "config.json", "shared_vocabulary.json", "tokenizer.json")
BATCH = 16  # sentences per call: progress and cancel are checked between calls

_HEADING = re.compile(r"^(#{1,6}\s+)(.+)$")
_CAST = re.compile(r"^(- \*\*.+?\*\* — )(.+)$")
_BOLD = re.compile(r"^\*\*(.+)\*\*$")
_ITALIC = re.compile(r"^\*(?!\*)(.+)\*$")
_DETAILS = re.compile(r"^\*Genre: .*\*$")  # genre, word count, source file, model: kept as written
_SENTENCE_END = re.compile(r'(?<=[.!?…])\s+|(?<=[.!?…]["”’\')])\s+')


def language_name(code: str) -> str:
    return LANGUAGES[code][0] if code in LANGUAGES else ""


def output_name(code: str) -> str:
    return f"story.{code}.md"


def model_dir() -> Path:
    return models_dir() / "translation" / MODEL_REPO.split("/")[-1]


def model_ready(folder: Path = None) -> bool:
    folder = folder or model_dir()
    return all((folder / name).is_file() for name in MODEL_FILES)


def ensure_model(on_status=None) -> Path:
    """The model folder, downloading the model on first use (resumes if interrupted)."""
    folder = model_dir()
    if not model_ready(folder):
        if on_status:
            on_status(f"Downloading the translation model {MODEL_NAME} (1.3 GB, first use only)...")
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_REPO, local_dir=str(folder), allow_patterns=list(MODEL_FILES))
    return folder


def split_sentences(text: str) -> list:
    """Sentences of a paragraph. A lower-case continuation stays with the quote
    before it ('"Is anyone here?" she called.'): the model needs that context."""
    out = []
    for piece in _SENTENCE_END.split(text.strip()):
        if out and piece[:1].islower():
            out[-1] += " " + piece
        elif piece:
            out.append(piece)
    return out


def _tidy(source: str, translated: str) -> str:
    """Repair two NLLB habits: a colon written as the Tamil letter aytham ("கேட்டார்ஃ"),
    and one of a sentence's quotation marks dropped."""
    if ":" in source:
        translated = re.sub(r'ஃ(?=\s|"|$)', ":", translated)
    if source.count('"') % 2 == 0 and translated.count('"') % 2 == 1:
        if source.endswith('"') and not translated.endswith('"'):
            return translated + '"'
        if source.startswith('"') and not translated.startswith('"'):
            return '"' + translated
    return translated


class NllbTranslator:
    name = MODEL_NAME

    def __init__(self, folder: Path, device: str = "cpu", compute_type: str = "int8", threads: int = 4):
        import ctranslate2
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(Path(folder) / "tokenizer.json"))
        self.model = ctranslate2.Translator(str(folder), device=device, compute_type=compute_type,
                                            intra_threads=threads)
        self.device = device

    def translate(self, texts: list, language: str, on_progress=None, cancel=None) -> list:
        """Translate English sentences to ``language`` (a LANGUAGES key), in order."""
        target = LANGUAGES[language][1]
        out = []
        for i in range(0, len(texts), BATCH):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            batch = texts[i:i + BATCH]
            # NLLB reads "<source language> tokens </s>" and starts its answer with the target language.
            source = [[SOURCE_LANGUAGE, *self.tokenizer.encode(t, add_special_tokens=False).tokens, "</s>"]
                      for t in batch]
            results = self.model.translate_batch(
                source, target_prefix=[[target]] * len(batch), beam_size=4, repetition_penalty=1.1,
                max_input_length=400, max_decoding_length=400)
            for text, result in zip(batch, results):
                ids = [self.tokenizer.token_to_id(t) for t in result.hypotheses[0][1:]]
                translated = self.tokenizer.decode([i for i in ids if i is not None]).strip()
                # An empty answer keeps the original sentence.
                out.append(_tidy(text, translated) if translated else text)
            if on_progress:
                on_progress(len(out) / len(texts))
        return out

    def close(self) -> None:
        self.model.unload_model()
        self.model = None


def translate_markdown(text: str, translate, labels: dict = None, note: str = "") -> str:
    """Translate the prose of a story file, keeping its Markdown structure.

    ``translate(list of str) -> list of str`` is called once with every sentence
    and heading. ``labels`` gives fixed translations for bold labels such as
    "Cast"; ``note`` is appended to the technical details line.
    """
    labels = labels or {}
    lines, wanted = [], []   # each line: literal str, or (prefix, [indexes into wanted], suffix)

    def ask(pieces: list) -> list:
        wanted.extend(pieces)
        return list(range(len(wanted) - len(pieces), len(wanted)))

    for raw in text.split("\n"):
        line = raw.strip()
        if not line or set(line) <= {"-"}:
            lines.append(raw)
        elif _DETAILS.match(line):
            lines.append(f"{line[:-1]} · {note}*" if note else line)
        elif m := _HEADING.match(line):
            lines.append((m.group(1), ask([m.group(2)]), ""))
        elif m := _CAST.match(line):
            lines.append((m.group(1), ask(split_sentences(m.group(2))), ""))
        elif m := _BOLD.match(line):
            lines.append(f"**{labels[m.group(1)]}**" if m.group(1) in labels
                         else ("**", ask([m.group(1)]), "**"))
        elif m := _ITALIC.match(line):
            lines.append(("*", ask(split_sentences(m.group(1))), "*"))
        else:
            lines.append(("", ask(split_sentences(line)), ""))
    done = translate(wanted) if wanted else []
    return "\n".join(l if isinstance(l, str) else l[0] + " ".join(done[i] for i in l[1]) + l[2]
                     for l in lines)
