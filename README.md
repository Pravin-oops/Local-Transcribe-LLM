# LocalScribe

Offline Windows app that transcribes long English audio (3+ hours) and turns it into the output
you choose, entirely on your own machine:

| Output | What you get | Typical content |
|---|---|---|
| **Summary** | Overview, key points, timestamped chapters | Lectures, podcasts, talks |
| **Study notes** | Key concepts, takeaways, examples, review questions | Lectures, classes |
| **Engaging Story** | A captivating narrator script retelling the whole recording, with a hook, parts that end on cliffhangers, cast list and genre-matched narration | Movies, audio dramas, story podcasts |

Pick a **content type** (Lecture → Summary + Study notes, Podcast → Summary, Movie → Engaging
Story) or tick outputs yourself. Only the selected outputs are computed: a movie retelling never
pays for study notes, and a summary-only run asks the LLM for fewer fields.

- Transcription: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`large-v3-turbo` by default) on the GPU, CPU fallback
- Speakers ("who said what"): [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with pyannote segmentation-3.0 + TitaNet-small embeddings, ONNX on the CPU; no PyTorch, no account
- Writing: any GGUF instruct model through [llama.cpp](https://github.com/abetlen/llama-cpp-python), split across GPU and CPU when it does not fit in VRAM
- Long files are processed **part by part** and every finished step is saved, so a crash, a cancel or closing the app never loses work: press Start again and it resumes. Adding an output later (say, a story for a finished lecture) computes only that output.

## Architecture

```
PySide6 window ──start/cancel──► worker process (spawned)
      ▲                               │
      └──── events via mp.Queue ◄─────┤
                                      ▼
  pipeline.run_job
   1. prepare     audio.py       PyAV decodes any audio/video → 16 kHz mono int16 PCM (memory-mapped)
                                 plan_chunks: ~10 min parts, cut at the quietest 0.5 s near each boundary
   2. transcribe  transcriber.py faster-whisper per part (VAD on), previous part's tail as prompt
                                 → parts/NNNN.json after each part (resume point)
   3. speakers    diarizer.py    (optional) sherpa-onnx over the whole file → speakers.json,
                                 each segment labelled "Speaker N" by largest overlap
   4. merge       merger.py      transcript.txt / .srt / .vtt / .json
   5. unload Whisper (frees VRAM), load the LLM once for all selected outputs
   6a. summary/notes  summarizer.py  map: ~3k-token blocks → only the fields the outputs need
                                     (JSON-schema constrained) → notes/NNNN.json
                                     reduce: condense until it fits → summary.md, notes.md
   6b. story          story.py       beats per block (scenes, cast, lines) → story/beats_NNNN.json
                                     outline (title, logline, genre, cast) → story/outline.json
                                     script written part by part with "story so far" and the
                                     previous part's ending → story/part_NN.json → story.md
```

Each input file gets a job folder: `Documents\LocalScribe\jobs\<name>-<hash>\` containing
`manifest.json`, the decoded audio, per-step results, outputs and `log.txt`. Changing the Whisper
model or chunk length invalidates only what depends on it; changing the story tone or length
rewrites only the script.

| Module | Role |
|---|---|
| `localscribe/pipeline.py` | Orchestration, stages, resume logic, progress/ETA, GPU→CPU fallback |
| `localscribe/audio.py` | Decoding (PyAV) and silence-aware chunk planning |
| `localscribe/transcriber.py` | faster-whisper wrapper, hallucination and sound-tag filter |
| `localscribe/diarizer.py` | Speaker detection (sherpa-onnx) and segment labelling |
| `localscribe/summarizer.py` | llama.cpp backend, VRAM-aware layer offload, summary/notes map-reduce |
| `localscribe/story.py` | Engaging Story: beats, outline, part-by-part narrator script |
| `localscribe/prompts.py` | Prompts and JSON schemas |
| `localscribe/config.py` | Settings, outputs, content-type presets, story tones and lengths |
| `localscribe/jobs.py` | Job folders, manifest, atomic writes |
| `localscribe/ui/` | Main window, output picker, settings dialog, worker-process bridge |
| `localscribe/_win_dlls.py` | Windows DLL-search fixes for CUDA and Qt (see Troubleshooting) |

## Setup (conda env `local-llm`)

```powershell
conda activate local-llm
conda install -c conda-forge cuda-cudart=12.4 libcublas=12.4 cudnn=9
pip install pyside6 faster-whisper sherpa-onnx psutil pytest
pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Models (one-time downloads):
- Whisper `large-v3-turbo` (1.6 GB) downloads automatically on first run into the Hugging Face cache.
- Speaker models (~47 MB) download automatically the first time speakers are detected, into
  `Documents\LocalScribe\models\diarization`.
- LLM: put a GGUF instruct model in `Documents\LocalScribe\models\` and select it in
  **Settings**. Tested with `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (4.9 GB).

## Usage

```powershell
python -m localscribe                                   # desktop app
python -m localscribe info                              # detected GPU/CPU and current settings
python -m localscribe run lecture.mp3 --preset lecture  # summary + study notes
python -m localscribe run episode.mp3 --preset podcast  # summary, speakers labelled
python -m localscribe run film.mkv --preset movie --tone suspense --length medium
python -m localscribe run talk.mp3 --outputs summary,story --speakers 2
python -m localscribe run lecture.mp3 --transcript-only --chunk-minutes 15
python -m localscribe run lecture.mp3 --resummarize     # redo all LLM outputs
```

Story tones: `auto` (match the film), `suspense`, `epic`, `drama`, `comedy`, `dark`, `bedtime`,
`documentary`. Lengths: `short` (~1,000 words), `medium` (~2,500), `long` (~5,000). With little
source material the story comes out shorter rather than padded with invented events.

Ctrl+C (CLI) or Cancel (app) stops after the current step; run again to resume.

Settings are stored in `%APPDATA%\LocalScribe\settings.json`.

## Tests

```powershell
python -m pytest -q
```

The tests use fake Whisper/diarizer/LLM backends (`tests/fakes.py`) plus real audio decoding, and
cover chunk planning, transcript formats, speaker labelling, field-selective map-reduce, story
parts and caching, presets, cancel → resume, and cache invalidation.

## Troubleshooting

- **`llama.dll` (or one of its dependencies) not found**: the CUDA runtime DLLs are not on PATH.
  The app adds `<env>\Library\bin` itself; for other scripts, activate the env first.
- **`DLL load failed while importing QtCore: The specified procedure could not be found`**:
  a conda-forge `icu` package shadows Windows' `icuuc.dll`. The app preloads the System32 copy
  (`_win_dlls.py`); scripts that import PySide6 directly should `import localscribe` first.
- **Slow summaries or stories**: an 8B model on a 4 GB GPU runs partly on the CPU. Use a 3-4B
  model for speed, or lower **Context size** to 4096 so more layers fit on the GPU.
- **Speakers merged or split**: automatic detection merges near-identical voices and tiny
  fragments; if people are still merged or split, set **Number of speakers** in Settings.
