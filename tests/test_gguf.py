import struct

from localscribe.summarizer import auto_gpu_layers, read_gguf_metadata


def _gguf_string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _write_gguf(path, kv: list, pad_mb: int = 0):
    """kv: list of (key, type, packed_value_bytes)."""
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for key, vtype, value in kv:
        body += _gguf_string(key) + struct.pack("<I", vtype) + value
    path.write_bytes(body + b"\0" * (pad_mb * 2**20))


def _llama_like(path, pad_mb):
    tokens = struct.pack("<IQ", 8, 3) + b"".join(_gguf_string(t) for t in ("<s>", "a", "b"))
    _write_gguf(path, [
        ("general.architecture", 8, _gguf_string("llama")),
        ("tokenizer.ggml.tokens", 9, tokens),                       # string array is skipped
        ("tokenizer.ggml.scores", 9, struct.pack("<IQ", 6, 3) + struct.pack("<3f", 0, 0, 0)),
        ("llama.block_count", 4, struct.pack("<I", 32)),
        ("llama.embedding_length", 4, struct.pack("<I", 4096)),
        ("llama.attention.head_count", 4, struct.pack("<I", 32)),
        ("llama.attention.head_count_kv", 4, struct.pack("<I", 8)),
    ], pad_mb)


def test_reads_scalars_and_skips_arrays(tmp_path):
    path = tmp_path / "m.gguf"
    _llama_like(path, 0)
    meta = read_gguf_metadata(str(path))
    assert meta["general.architecture"] == "llama"
    assert meta["llama.block_count"] == 32
    assert meta["llama.attention.head_count_kv"] == 8
    assert "tokenizer.ggml.tokens" not in meta


def test_auto_gpu_layers_scales_with_vram(tmp_path):
    path = tmp_path / "m.gguf"
    _llama_like(path, 340)  # ~10 MB/layer -> tiny stand-in for a real model
    assert auto_gpu_layers(str(path), 512, free_vram_mb=600) == 0           # below reserve
    partial = auto_gpu_layers(str(path), 512, free_vram_mb=900)
    assert 0 < partial < 32
    assert auto_gpu_layers(str(path), 512, free_vram_mb=100_000) == -1     # everything fits
