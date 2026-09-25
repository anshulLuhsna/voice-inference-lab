"""voice-inference-lab -- the experiment logic for the Moshi milestones.

This module has no provider imports. It runs on any Linux host with an NVIDIA
GPU and the pinned runtime installed, whether that host is a managed container
or a raw EC2 instance. Provider lifecycle stays outside it: the one operation
that only a provider can define, making writes durable, arrives as a `commit`
callable and defaults to a no-op, which is correct for a local filesystem.

Four operations, in the order they are run:

    inspect_gpu      probe the host GPU and runtime
    cache_weights    download the pinned checkpoint and fixture, no GPU needed
    load_model       load Moshi and Mimi onto the GPU
    stream_session   run one duplex streaming session and write the tracks

The storage root comes from VOICE_LAB_ROOT and defaults to /cache, which is the
mount point used by the Modal runner.
"""

import os

# Everything the experiments write lives under one root, so a provider only has
# to decide where that root is. The HF cache, the fixture, the audio tracks and
# the sustained series CSV are all derived from it.
DATA_ROOT = os.environ.get("VOICE_LAB_ROOT", "/cache")
HF_HOME = f"{DATA_ROOT}/hf"

# ---------------------------------------------------------------------------
# Pins. The whole point of this project is that none of these drift.
#
# Weights are pinned to an immutable revision SHA: a cached copy can never go
# stale, and a later run can never be pulled toward a newer upstream commit.
#
# Code is pinned because it has to be. moshi 0.2.13 declares
# `torch<2.10,>=2.2.0` and its README says it is tested on PyTorch 2.2/2.4.
# The GPU probe deliberately floats torch so it reports whatever the host hands
# out by default; everything else uses the pinned version.
# ---------------------------------------------------------------------------
MOSHI_REPO = "kyutai/moshiko-pytorch-bf16"
MOSHI_REVISION = "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"  # 7.69B params, ~15.8 GB

# The Moshi language model, the Mimi codec, and the text tokenizer all ship
# inside MOSHI_REPO. The load and stream operations read those filenames from
# moshi's own constants, so the pairing cannot drift.
#
# Caution: `kyutai/mimi` holds the Hugging Face `transformers` build of Mimi.
# That build uses different `state_dict` keys, and the native moshi loader
# rejects it. Do not point this file at that repo.
MOSHI_PACKAGE = "moshi==0.2.13"
TORCH_PACKAGE = "torch==2.4.1"
PYTHON_VERSION = "3.12"

# The fixed audio fixture, and where the generated audio is written.
#
# The fixture is the one Kyutai reference in their own sphn README. It lives
# under the data root rather than in the repo, and the cache operation records
# its SHA-256 so later runs are comparable.
#
# The file is about 45 seconds of narration, which is far too long for a
# conversational test, so we take a fixed window from it. Both numbers are
# constants, so the window is identical on every run.
FIXTURE_URL = "https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3"
FIXTURE_PATH = f"{DATA_ROOT}/fixtures/bria.mp3"
FIXTURE_START_SECONDS = 0.0
FIXTURE_SECONDS = 8.0

# Moshi is full-duplex: it never decides that a turn has ended. After the clip
# finishes we keep feeding silence, so the model has room to speak.
TAIL_SECONDS = 5.0

# Session length in frames. The sustained milestone runs past Moshi's 3000 frame
# context window to find out what happens at the wrap. The fixed clip is
# repeated to fill the session, because content is irrelevant to a memory
# lifetime test; only the frame clock matters.
SUSTAINED_FRAMES = 3400  # 272 seconds, about 32 seconds past the window

# Moshi declares a one frame delay on its outputs, and the first decoded frame
# arrives on loop iteration 1. Each decoded frame is therefore placed on the
# shared timeline at the iteration that produced it. Set this to 0 if the mix
# sounds 80 ms early.
OUTPUT_OFFSET_FRAMES = 1

# Each track is attenuated before the sum, so mixed.wav cannot clip and neither
# voice is favoured.
MIX_SCALE = 0.5

# Synthetic silent frames pushed through the full path before any real audio, to
# move the one-time CUDA graph capture and lazy kernel setup off the first real
# frame's critical path. Set this to 0 to measure the unwarmed behaviour.
#
# The warm-up must share one streaming context with the real session. The graph
# wrappers are built when the context is entered, so a second context would
# rebuild them and throw the warm-up away.
EXPLICIT_WARMUP_FRAMES = 5

# The first real frame and the next five are reported on their own, so a warm-up
# that failed to settle cannot hide inside an average.
HEADLINE_FRAMES = 6

# This milestone sustains the session past the context window, so it writes under
# its own prefix. The shorter verified tracks in `outputs/` are left untouched.
HUMAN_PATH = f"{DATA_ROOT}/outputs/sustained/human.wav"
MOSHI_PATH = f"{DATA_ROOT}/outputs/sustained/moshi.wav"
MIXED_PATH = f"{DATA_ROOT}/outputs/sustained/mixed.wav"
SERIES_PATH = f"{DATA_ROOT}/outputs/sustained/series.csv"


def _noop() -> None:
    """Default commit. A plain filesystem needs nothing to make writes durable."""


def _snapshot(repo: str, revision: str, offline: bool) -> str:
    """Return the local path of a pinned snapshot.

    With ``offline=True`` huggingface_hub resolves purely from the local cache.
    If anything is missing it raises instead of quietly reaching for the
    network, which is what makes the cache guarantee real rather than assumed.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo, revision=revision, local_files_only=offline)


def _resolve_weights(loaders) -> dict:
    """Resolve the pinned weight files inside the cached snapshot.

    Raises with a clear message when moshi expects a different repo, or when a
    file is absent. Offline resolution is set by the caller's environment, so a
    cache miss surfaces here instead of silently fetching.
    """
    from pathlib import Path

    if loaders.DEFAULT_REPO != MOSHI_REPO:
        raise RuntimeError(
            f"moshi expects {loaders.DEFAULT_REPO!r}, but this file pins {MOSHI_REPO!r}."
        )

    snapshot = _snapshot(MOSHI_REPO, MOSHI_REVISION, offline=True)
    weights = {
        "moshi": f"{snapshot}/{loaders.MOSHI_NAME}",
        "mimi": f"{snapshot}/{loaders.MIMI_NAME}",
        "tokenizer": f"{snapshot}/{loaders.TEXT_TOKENIZER_NAME}",
    }
    missing = [name for name, path in weights.items() if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"Missing {', '.join(missing)} in {snapshot}; cache incomplete.")
    return weights


def _json(report: dict) -> str:
    """Serialise a report as a string.

    The operations return text rather than Python objects so a driver process
    never needs torch installed just to read a result.
    """
    import json

    text = json.dumps(report, indent=2)
    print(text)
    return text


def _write_wav(path: str, samples, sample_rate: int) -> None:
    """Write one mono 16-bit WAV. ``samples`` is a 1-D torch tensor in [-1, 1]."""
    import wave
    from pathlib import Path

    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pcm = (samples.clamp(-1, 1) * 32767).to(torch.int16).numpy().tobytes()
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def inspect_gpu() -> str:
    """Probe the host: proves CUDA is reachable and that a real tensor op runs.

    Certifies the environment only. It says nothing about whether a given voice
    model will fit or run here.
    """
    import platform
    import shutil
    import subprocess
    import time

    import torch

    report = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }

    if shutil.which("nvidia-smi"):
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        report["nvidia_smi"] = smi.stdout.strip()

    if report["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        start = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        report.update(
            {
                "device_count": torch.cuda.device_count(),
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "vram_total_gb": round(props.total_memory / 1e9, 1),
                "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                "matmul_ok": tuple(c.shape) == (512, 512),
                "matmul_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        )

    return _json(report)


def cache_weights(commit=_noop) -> str:
    """Download the pinned snapshot and the fixture. No GPU is required.

    The transfer is deliberately separable from any GPU work, so bytes are never
    moved on a billed GPU. Reports whether each item was already present.
    """
    import hashlib
    import urllib.request
    from pathlib import Path

    # Reconstruct the cache location huggingface_hub will use, so we can tell
    # "was already here" from "we just downloaded it".
    snapshot_dir = (
        Path(HF_HOME)
        / "hub"
        / ("models--" + MOSHI_REPO.replace("/", "--"))
        / "snapshots"
        / MOSHI_REVISION
    )
    already_present = snapshot_dir.is_dir() and any(snapshot_dir.iterdir())

    path = _snapshot(MOSHI_REPO, MOSHI_REVISION, offline=False)
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())

    # The fixture is fetched here rather than during a GPU run, for the same
    # reason as the weights.
    fixture = Path(FIXTURE_PATH)
    fixture.parent.mkdir(parents=True, exist_ok=True)
    if not fixture.is_file():
        urllib.request.urlretrieve(FIXTURE_URL, fixture)
    fixture_bytes = fixture.read_bytes()

    report = {
        "checkpoint": {
            "repo": MOSHI_REPO,
            "status": "already present" if already_present else "downloaded",
            "revision": MOSHI_REVISION,
            "path": path,
            "size_gb": round(size / 1e9, 2),
        },
        "fixture": {
            "url": FIXTURE_URL,
            "path": str(fixture),
            "size_mb": round(len(fixture_bytes) / 1e6, 2),
            "sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        },
    }

    # Nothing written above is durable until the provider makes it so.
    commit()
    report["volume_committed"] = True
    return _json(report)


def load_model() -> str:
    """Load the weights from the local copy and record what it costs.

    Records model initialization time and GPU memory either side of each
    component. This operation never commits anything, so the cached weights are
    left untouched.
    """
    import time
    import traceback

    import torch
    from moshi.models import loaders

    report = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not report["cuda_available"]:
        report["error"] = "CUDA is unavailable; refusing to fall back to CPU."
        return _json(report)

    def vram() -> dict:
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "free_gb": round(free / 1e9, 2),
            "total_gb": round(total / 1e9, 2),
        }

    def timed(label: str, load):
        """Run one component's load and record how long it took and its VRAM."""
        start = time.perf_counter()
        result = load()
        torch.cuda.synchronize()
        report[label] = {
            "init_ms": round((time.perf_counter() - start) * 1000),
            "vram_after": vram(),
        }
        return result

    try:
        weights = _resolve_weights(loaders)
        report["weights"] = weights

        torch.cuda.reset_peak_memory_stats()
        report["vram_baseline"] = vram()

        mimi = timed("mimi", lambda: loaders.get_mimi(weights["mimi"], device="cuda"))
        moshi = timed("moshi", lambda: loaders.get_moshi_lm(weights["moshi"], device="cuda"))

        report["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        report["parameters"] = {
            "mimi": sum(p.numel() for p in mimi.parameters()),
            "moshi": sum(p.numel() for p in moshi.parameters()),
        }
        report["loaded"] = True
    except Exception:
        report["loaded"] = False
        report["error"] = traceback.format_exc()

    return _json(report)


def stream_session(commit=_noop) -> str:
    """Stream the fixture through Moshi as two synchronized tracks.

    Moshi is not an input/output API. It runs two continuous streams at once:
    human audio flows in, and model audio flows out, on the same clock. This
    function preserves that. One loop drives one frame counter, so frame ``i``
    holds the input chunk submitted at that point and the output frame produced
    at that point.

    Each frame does exactly three things, in the official order:

      1. ``mimi.encode`` turns 1920 input samples into 8 codebooks.
      2. ``lm_gen.step`` advances the Moshi language model by one frame.
      3. ``mimi.decode`` turns the model's 8 output codebooks back into audio.

    The loop returns nothing on the first frame, because Moshi's output streams
    are delayed against its inputs. After the clip ends we keep feeding silence
    for ``TAIL_SECONDS``, so the model has room to speak.

    Three files are written, all on one timeline and all the same length:

      ``human.wav``  the input clip, in place, silence elsewhere
      ``moshi.wav``  the decoded output, placed at ``OUTPUT_OFFSET_FRAMES``
      ``mixed.wav``  the two tracks summed

    Before the real session, ``EXPLICIT_WARMUP_FRAMES`` synthetic silent frames
    go through the same path, and then the streaming state is reset. Set that
    constant to 0 to measure the unwarmed behaviour instead.
    """
    import time
    import traceback
    from pathlib import Path

    import numpy as np
    import sphn
    import torch
    from moshi.models import LMGen, loaders

    sample_rate = loaders.SAMPLE_RATE
    frame_rate = loaders.FRAME_RATE

    report = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not report["cuda_available"]:
        report["error"] = "CUDA is unavailable; refusing to fall back to CPU."
        return _json(report)

    def vram() -> dict:
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "free_gb": round(free / 1e9, 2),
            "total_gb": round(total / 1e9, 2),
        }

    outputs = []
    wav = None
    track_samples = 0
    try:
        run_start = time.perf_counter()
        weights = _resolve_weights(loaders)

        load_start = time.perf_counter()
        mimi = loaders.get_mimi(weights["mimi"], device="cuda")
        moshi = loaders.get_moshi_lm(weights["moshi"], device="cuda")
        torch.cuda.synchronize()
        report["model_load_ms"] = round((time.perf_counter() - load_start) * 1000)

        frame_size = mimi.frame_size

        # Decode the fixture. Moshi wants 24 kHz mono, so resample only when the
        # fixture disagrees, and report the source rate either way.
        fixture = Path(FIXTURE_PATH)
        if not fixture.is_file():
            report["error"] = f"Fixture missing at {fixture}. Run cache_weights first."
            return _json(report)

        audio_np, source_rate = sphn.read(str(fixture))
        wav = torch.as_tensor(np.asarray(audio_np)).float()
        if wav.dim() == 1:
            wav = wav[None]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Take the fixed window before resampling, so the window is defined on
        # the source timeline and does not shift if the resample changes.
        begin = int(FIXTURE_START_SECONDS * source_rate)
        wav = wav[:, begin : begin + int(FIXTURE_SECONDS * source_rate)]
        input_seconds = round(wav.shape[-1] / source_rate, 2)

        if source_rate != sample_rate:
            target = int(round(wav.shape[-1] * sample_rate / source_rate))
            wav = torch.nn.functional.interpolate(
                wav[None], size=target, mode="linear", align_corners=False
            )[0]

        # Moshi requires whole frames. Pad the end of the clip with silence.
        padding = (-wav.shape[-1]) % frame_size
        if padding:
            wav = torch.cat([wav, torch.zeros(1, padding)], dim=-1)
        clip = [wav[:, start : start + frame_size] for start in range(0, wav.shape[-1], frame_size)]
        silence = torch.zeros(1, frame_size)
        tail_frames = int(round(TAIL_SECONDS * frame_rate))
        total_frames = SUSTAINED_FRAMES
        clip_frames = total_frames - tail_frames

        # The tracks span the session plus the output offset, so no decoded
        # frame is dropped and every file covers the same span of time.
        track_frames = total_frames + OUTPUT_OFFSET_FRAMES
        track_samples = track_frames * frame_size

        report["fixture"] = {
            "path": str(fixture),
            "source_sample_rate": source_rate,
            "resampled": source_rate != sample_rate,
            "window_seconds": [FIXTURE_START_SECONDS, FIXTURE_START_SECONDS + FIXTURE_SECONDS],
            "input_seconds": input_seconds,
            "padded_samples": padding,
        }
        report["frame"] = {
            "frame_size_samples": frame_size,
            "frame_rate_hz": frame_rate,
            "session_seconds": round(total_frames / frame_rate, 1),
            "clip_seconds": input_seconds,
            "clip_frames": clip_frames,
            "clip_repeats": -(-clip_frames // len(clip)),
            "tail_frames": tail_frames,
            "total_frames": total_frames,
            "track_frames": track_frames,
            "output_offset_frames": OUTPUT_OFFSET_FRAMES,
        }

        # The peak spans the warm-up and the session, because that is the
        # memory the host actually needs.
        torch.cuda.reset_peak_memory_stats()
        report["vram_after_load"] = vram()
        warmup_min_free = None
        session_min_free = None

        lm_gen = LMGen(moshi, temp=0.8, temp_text=0.7)

        init_start = time.perf_counter()
        # One streaming context for both phases. The CUDA graph wrappers live on
        # the streaming state and are built when the context is entered, so a
        # second context would rebuild them and discard the warm-up.
        with torch.no_grad(), lm_gen.streaming(1), mimi.streaming(1):
            torch.cuda.synchronize()
            init_ms = (time.perf_counter() - init_start) * 1000

            # PHASE A -- explicit warm-up, before any real audio. Silent frames
            # go through the full path so the graph capture lands here.
            warmup_ms_list = []
            warmup_start = time.perf_counter()
            warmup_frame_start = warmup_start
            for _ in range(EXPLICIT_WARMUP_FRAMES):
                codes = mimi.encode(silence.unsqueeze(0).cuda())
                tokens = lm_gen.step(codes)
                if tokens is not None:
                    mimi.decode(tokens[:, 1:])
                free, _ = torch.cuda.mem_get_info()
                warmup_min_free = free if warmup_min_free is None else min(warmup_min_free, free)
                torch.cuda.synchronize()
                now = time.perf_counter()
                warmup_ms_list.append((now - warmup_frame_start) * 1000)
                warmup_frame_start = now
            warmup_ms = (time.perf_counter() - warmup_start) * 1000
            report["vram_after_warmup"] = vram()

            # Discard every trace of the warm-up before real audio arrives. This
            # clears offsets and the step counter but keeps the graph wrappers,
            # so the first real frame should again be the frame that yields no
            # output.
            lm_gen.reset_streaming()
            mimi.reset_streaming()

            # PHASE B -- the real fixture, on the already captured graphs.
            start = time.perf_counter()
            frame_start = start
            series = []
            first_output_frame = None
            first_model_output_ms = None

            for index in range(total_frames):
                source = clip[index % len(clip)] if index < clip_frames else silence

                codes = mimi.encode(source.unsqueeze(0).cuda())
                tokens = lm_gen.step(codes)

                if tokens is not None:
                    if first_output_frame is None:
                        first_output_frame = index
                        torch.cuda.synchronize()
                        first_model_output_ms = (time.perf_counter() - start) * 1000
                    decoded = mimi.decode(tokens[:, 1:])[0, 0]
                    outputs.append((index, decoded.float().cpu()))

                # Synchronising inside the loop is what makes a per frame number
                # honest, and it serialises the CPU against the GPU. The cost is
                # disclosed rather than hidden.
                torch.cuda.synchronize()
                now = time.perf_counter()
                frame_ms = (now - frame_start) * 1000

                # Memory is sampled after the timestamp is taken, so these three
                # queries cannot inflate the per frame compute number. The clock
                # restarts after the sampling for the same reason.
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                free, _ = torch.cuda.mem_get_info()
                series.append((index, frame_ms, allocated / 1e9, reserved / 1e9, free / 1e9))
                session_min_free = free if session_min_free is None else min(session_min_free, free)
                frame_start = time.perf_counter()

            torch.cuda.synchronize()
            total_ms = (time.perf_counter() - start) * 1000

        real_frame_ms = [row[1] for row in series]
        settled = real_frame_ms[HEADLINE_FRAMES:]
        frees = [value for value in (warmup_min_free, session_min_free) if value is not None]

        # The full series goes to a file. The report keeps only the windows that
        # answer the question, because thousands of rows do not belong in a
        # console.
        series_path = Path(SERIES_PATH)
        series_path.parent.mkdir(parents=True, exist_ok=True)
        with series_path.open("w") as handle:
            handle.write("frame,ms,allocated_gb,reserved_gb,free_gb\n")
            for row in series:
                handle.write(f"{row[0]},{row[1]:.2f},{row[2]:.4f},{row[3]:.4f},{row[4]:.4f}\n")

        # Read from the model rather than trusting a hardcoded 3000.
        context_frames = moshi.context

        def summarise(values):
            if not values:
                return None
            ordered = sorted(values)
            return {
                "frames": len(values),
                "mean_ms": round(sum(values) / len(values), 1),
                "median_ms": round(ordered[len(ordered) // 2], 1),
                "max_ms": round(max(values), 1),
            }

        def at(index):
            row = series[min(max(index, 0), len(series) - 1)]
            return {
                "frame": row[0],
                "ms": round(row[1], 1),
                "allocated_gb": round(row[2], 3),
                "reserved_gb": round(row[3], 3),
                "free_gb": round(row[4], 2),
            }

        report["context"] = {
            "context_frames": context_frames,
            "context_seconds": round(context_frames / frame_rate, 1),
            "session_frames": len(series),
            "crosses_context": len(series) > context_frames,
        }
        report["series"] = {
            "path": str(series_path),
            "rows": len(series),
            "columns": "frame,ms,allocated_gb,reserved_gb,free_gb",
        }
        report["warmup"] = {
            "frames": len(warmup_ms_list),
            "total_ms": round(warmup_ms),
            "per_frame_ms": [round(value) for value in warmup_ms_list],
            "min_device_free_gb": round(warmup_min_free / 1e9, 2) if warmup_min_free else None,
        }
        report["session"] = {
            "streaming_context_init_ms": round(init_ms),
            "first_model_output_ms": round(first_model_output_ms) if first_model_output_ms else None,
            "first_output_at_frame": first_output_frame,
            "first_real_frame_ms": round(real_frame_ms[0], 1),
            "next_real_frames_ms": [round(value, 1) for value in real_frame_ms[1:HEADLINE_FRAMES]],
            "steady_frames": len(settled),
            "steady_mean_ms": round(sum(settled) / len(settled), 1) if settled else None,
            "steady_median_ms": round(sorted(settled)[len(settled) // 2], 1) if settled else None,
            "steady_max_ms": round(max(settled), 1) if settled else None,
            "total_ms": round(total_ms),
            "unpaced": True,
        }
        report["latency"] = {
            "startup_ms": [round(value, 1) for value in real_frame_ms[:8]],
            "before_wrap": summarise(real_frame_ms[HEADLINE_FRAMES:context_frames]),
            "after_wrap": summarise(real_frame_ms[context_frames + 1:]),
            "last_10_before_wrap_ms": [
                round(value, 1) for value in real_frame_ms[context_frames - 10 : context_frames]
            ],
            "first_10_after_wrap_ms": [
                round(value, 1) for value in real_frame_ms[context_frames : context_frames + 10]
            ],
        }
        report["memory"] = {
            "vram_after_session": vram(),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
            "min_device_free_gb": round(min(frees) / 1e9, 2) if frees else None,
            "session_min_device_free_gb": (
                round(session_min_free / 1e9, 2) if session_min_free else None
            ),
        }
        report["at_frames"] = {
            str(index): at(index)
            for index in sorted(
                {
                    0,
                    1,
                    2,
                    50,
                    100,
                    context_frames - 1,
                    context_frames,
                    context_frames + 1,
                    context_frames + 100,
                    len(series) - 1,
                }
            )
        }
        report["growth"] = {
            "frame_100_to_last": {
                "allocated_gb": round(series[-1][2] - series[100][2], 4) if len(series) > 100 else None,
                "reserved_gb": round(series[-1][3] - series[100][3], 4) if len(series) > 100 else None,
                "free_gb": round(series[-1][4] - series[100][4], 4) if len(series) > 100 else None,
            }
        }
        report["wall_clock_ms"] = round((time.perf_counter() - run_start) * 1000)
        report["oom"] = False
        report["completed"] = True
    except torch.cuda.OutOfMemoryError:
        report["oom"] = True
        report["completed"] = False
        report["error"] = traceback.format_exc()
    except Exception:
        report["oom"] = False
        report["completed"] = False
        report["error"] = traceback.format_exc()

    # Build the three tracks. Each covers the whole session, so a given
    # timestamp means the same moment in every file. A partial run stays
    # inspectable.
    if outputs and wav is not None and track_samples:
        human = torch.zeros(track_samples)
        human[: wav.shape[-1]] = wav[0]

        moshi = torch.zeros(track_samples)
        for index, decoded in outputs:
            start_sample = (index + OUTPUT_OFFSET_FRAMES) * frame_size
            moshi[start_sample : start_sample + decoded.shape[-1]] = decoded

        mixed = (human + moshi) * MIX_SCALE

        _write_wav(HUMAN_PATH, human, sample_rate)
        _write_wav(MOSHI_PATH, moshi, sample_rate)
        _write_wav(MIXED_PATH, mixed, sample_rate)

        # The audio must outlive the process, so the provider is asked to make
        # it durable.
        commit()
        report["audio"] = {
            "human": HUMAN_PATH,
            "moshi": MOSHI_PATH,
            "mixed": MIXED_PATH,
            "track_seconds": round(track_samples / sample_rate, 2),
            "decoded_frames": len(outputs),
            "output_seconds": round(len(outputs) * frame_size / sample_rate, 2),
            "mix_scale": MIX_SCALE,
            "mixed_peak": round(float(mixed.abs().max()), 3),
            "volume_committed": True,
        }

    return _json(report)
