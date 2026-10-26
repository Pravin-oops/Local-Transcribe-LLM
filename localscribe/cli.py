"""Command line: `python -m localscribe [gui | run FILE | info]`."""
import argparse
import sys
import threading

from localscribe import __version__
from localscribe.config import OUTPUTS, PRESETS, STORY_TONES, load_settings, story_minutes
from localscribe.subtitles import is_subtitle
from localscribe.timefmt import hms
from localscribe.transcriber import Cancelled


def _cmd_info(_args) -> int:
    from localscribe import hardware

    hw = hardware.detect()
    settings = load_settings()
    device, compute = hardware.whisper_runtime(settings.device, settings.compute_type)
    print(f"LocalScribe {__version__}")
    print(f"GPU:      {hw.gpu_name or 'none'} ({hw.vram_total_mb} MB VRAM, {hardware.vram_free_mb()} MB free)")
    print(f"CPU/RAM:  {hw.physical_cores} cores, {hw.ram_gb:.1f} GB RAM")
    print(f"Whisper:  {settings.whisper_model} on {device} ({compute})")
    print(f"LLM:      {settings.llm_path or '(not set)'}")
    print(f"Outputs:  {', '.join(OUTPUTS[o] for o in settings.outputs) or 'transcript only'}"
          f"{' + speaker labels' if settings.diarize else ''}")
    print(f"Output:   {settings.output_dir}")
    return 0


def _cmd_run(args) -> int:
    from localscribe.pipeline import run_job

    settings = load_settings()
    if args.preset:
        settings.apply_preset(args.preset)
    elif args.outputs is None and is_subtitle(args.file):
        settings.apply_preset("movie")  # subtitles default to the Engaging Story
        print("Subtitle file: using the movie preset (Engaging Story); pass --outputs to change.")
    for attr, value in (("whisper_model", args.whisper_model), ("llm_path", args.llm),
                        ("chunk_minutes", args.chunk_minutes), ("output_dir", args.output),
                        ("device", args.device), ("batch_size", args.batch_size),
                        ("story_tone", args.tone),
                        ("story_minutes", story_minutes(args.length) if args.length else args.minutes),
                        ("num_speakers", args.speakers), ("diarize", args.diarize)):
        if value is not None:
            setattr(settings, attr, value)
    if args.outputs is not None:
        settings.outputs = [o.strip() for o in args.outputs.split(",") if o.strip() in OUTPUTS]
    if args.transcript_only:
        settings.outputs = []

    def emit(event):
        kind = event["type"]
        if kind == "progress":
            eta = f" · ETA {hms(event['eta'])}" if event.get("eta") else ""
            line = f"{event['fraction'] * 100:5.1f}%  {event['message']}{eta}"
            sys.stdout.write("\r" + line[:110].ljust(110))
            sys.stdout.flush()
        elif kind == "log":
            sys.stdout.write("\r" + " " * 110 + "\r")
            print(f"[{event['level']}] {event['message']}")
        elif kind == "job":
            print(f"Job folder: {event['job_dir']}")
        elif kind == "done":
            print("\nOutputs:")
            for name, path in event["files"].items():
                print(f"  {name:16} {path}")

    cancel = threading.Event()
    try:
        run_job(args.file, settings, emit, cancel, resummarize=args.resummarize)
    except (KeyboardInterrupt, Cancelled):
        cancel.set()
        print("\nStopped. Finished parts are saved; run the same command again to resume.")
        return 130
    return 0


def _cmd_gui(_args) -> int:
    from localscribe.ui.app import main as gui_main
    return gui_main()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="localscribe",
                                     description="Offline transcription and summarization")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("gui", help="open the desktop app (default)")
    sub.add_parser("info", help="show detected hardware and settings")
    run = sub.add_parser("run", help="transcribe one file and produce the selected outputs")
    run.add_argument("file", help="audio/video file, or .srt/.vtt subtitles (skips transcription)")
    run.add_argument("--preset", choices=list(PRESETS),
                     help="lecture = summary + notes, podcast = summary, movie = Engaging Story")
    run.add_argument("--outputs", help=f"comma-separated, any of: {','.join(OUTPUTS)}")
    run.add_argument("--transcript-only", "--no-summary", action="store_true",
                     help="skip all LLM outputs")
    run.add_argument("--tone", choices=list(STORY_TONES), help="Engaging Story narration tone")
    run.add_argument("--minutes", type=int, help="Engaging Story length as reading time in minutes")
    run.add_argument("--length", choices=["short", "medium", "long"],
                     help="shortcut for --minutes 5 / 15 / 30")
    run.add_argument("--diarize", action=argparse.BooleanOptionalAction, default=None,
                     help="label speakers (on by default for podcast and movie presets)")
    run.add_argument("--speakers", type=int, help="number of speakers, 0 = detect")
    run.add_argument("--whisper-model")
    run.add_argument("--llm", help="path to a GGUF instruct model")
    run.add_argument("--resummarize", action="store_true", help="redo all LLM outputs from scratch")
    run.add_argument("--chunk-minutes", type=float)
    run.add_argument("--batch-size", type=int)
    run.add_argument("--device", choices=["auto", "cuda", "cpu"])
    run.add_argument("--output", help="output folder (jobs are created inside it)")
    args = parser.parse_args(argv)
    handlers = {"run": _cmd_run, "info": _cmd_info, "gui": _cmd_gui, None: _cmd_gui}
    return handlers[args.command](args)
