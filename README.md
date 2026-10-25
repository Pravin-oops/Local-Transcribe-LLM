# LocalScribe

Offline Windows app that transcribes long English audio (3+ hours) and turns it into
**timestamped chapters** and **study notes**, entirely on your own machine.

- Transcription: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`large-v3-turbo` by default) on the GPU, CPU fallback
- Summaries: any GGUF instruct model through [llama.cpp](https://github.com/abetlen/llama-cpp-python), split across GPU and CPU when it does not fit in VRAM
- Long files are processed **part by part** and every finished part is saved, so a crash, a cancel or closing the app never loses work: press Start again and it resumes

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
   3. merge       merger.py      transcript.txt / .srt / .vtt / .json
   4. unload Whisper (frees VRAM)
   5. summarize   summarizer.py  map:    ~3k-token blocks → chapters + notes (JSON-schema constrained)
                                         → notes/NNNN.json after each block (resume point)
                                 reduce: condense notes in groups until they fit → final study notes
   6. render      chapters.md, notes.md, summary.md
```

Each input file gets a job folder: `Documents\LocalScribe\jobs\<name>-<hash>\` containing
`manifest.json`, the decoded audio, per-part results, outputs and `log.txt`. Changing the Whisper
model or chunk length invalidates only what depends on it.

| Module | Role |
|---|---|
| `localscribe/pipeline.py` | Orchestration, resume logic, progress/ETA, GPU→CPU fallback |
| `localscribe/audio.py` | Decoding (PyAV) and silence-aware chunk planning |
| `localscribe/transcriber.py` | faster-whisper wrapper (sequential or batched) |
| `localscribe/summarizer.py` | llama.cpp backend, VRAM-aware layer offload, map-reduce |
| `localscribe/prompts.py` | Prompts and JSON schemas |
| `localscribe/jobs.py` | Job folders, manifest, atomic writes |
| `localscribe/ui/` | Main window, settings dialog, worker-process bridge |
| `localscribe/_win_dlls.py` | Windows DLL-search fixes for CUDA and Qt (see Troubleshooting) |

## Setup (conda env `local-llm`)

```powershell
conda activate local-llm
conda install -c conda-forge cuda-cudart=12.4 libcublas=12.4 cudnn=9
pip install pyside6 faster-whisper psutil pytest
pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Models (one-time downloads):
- Whisper `large-v3-turbo` (1.6 GB) downloads automatically on first run into the Hugging Face cache.
- Summary model: put a GGUF instruct model in `Documents\LocalScribe\models\` and select it in
  **Settings**. Tested with `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (4.9 GB).

## Usage

```powershell
python -m localscribe                       # desktop app
python -m localscribe info                  # detected GPU/CPU and current settings
python -m localscribe run lecture.mp3       # command line, prints progress
python -m localscribe run lecture.mp3 --no-summary --chunk-minutes 15
python -m localscribe run lecture.mp3 --resummarize   # redo only the summary
```

Ctrl+C (CLI) or Cancel (app) stops after the current step; run again to resume.

Settings are stored in `%APPDATA%\LocalScribe\settings.json`.

## Tests

```powershell
python -m pytest -q
```

The tests use fake Whisper/LLM backends (`tests/fakes.py`) plus real audio decoding, and cover
chunk planning, transcript formats, map-reduce with condensing, cancel → resume, and cache
invalidation.

## Troubleshooting

- **`llama.dll` (or one of its dependencies) not found**: the CUDA runtime DLLs are not on PATH.
  The app adds `<env>\Library\bin` itself; for other scripts, activate the env first.
- **`DLL load failed while importing QtCore: The specified procedure could not be found`**:
  a conda-forge `icu` package shadows Windows' `icuuc.dll`. The app preloads the System32 copy
  (`_win_dlls.py`); scripts that import PySide6 directly should `import localscribe` first.
- **Slow summaries**: an 8B model on a 4 GB GPU runs partly on the CPU. Use a 3-4B model for
  speed, or lower **Context size** to 4096 so more layers fit on the GPU.
