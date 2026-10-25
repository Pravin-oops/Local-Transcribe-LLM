"""Command line: `python -m localscribe [gui | run FILE | info]`."""
import argparse
import sys
import threading

from localscribe import __version__
from localscribe.config import load_settings
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
    print(f"Output:   {settings.output_dir}")
    return 0


def _cmd_run(args) -> int:
    from localscribe.pipeline import run_job

    settings = load_settings()
    for attr, value in (("whisper_model", args.whisper_model), ("llm_path", args.llm),
                        ("chunk_minutes", args.chunk_minutes), ("output_dir", args.output),
                        ("device", args.device), ("batch_size", args.batch_size)):
        if value is not None:
            setattr(settings, attr, value)
    if args.no_summary:
        settings.summarize = False

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
    run = sub.add_parser("run", help="transcribe (and summarize) one file")
    run.add_argument("file")
    run.add_argument("--whisper-model")
    run.add_argument("--llm", help="path to a GGUF summary model")
    run.add_argument("--no-summary", action="store_true")
    run.add_argument("--resummarize", action="store_true", help="redo the summary from scratch")
    run.add_argument("--chunk-minutes", type=float)
    run.add_argument("--batch-size", type=int)
    run.add_argument("--device", choices=["auto", "cuda", "cpu"])
    run.add_argument("--output", help="output folder (jobs are created inside it)")
    args = parser.parse_args(argv)
    handlers = {"run": _cmd_run, "info": _cmd_info, "gui": _cmd_gui, None: _cmd_gui}
    return handlers[args.command](args)
