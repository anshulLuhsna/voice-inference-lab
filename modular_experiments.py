"""voice-inference-lab -- the experiment logic for the modular stack.

The second architecture family. Where Moshi is one continuous full-duplex
stream, this is three separate models in a row: Whisper transcribes, Qwen
answers, Kokoro speaks. Nothing here is shared with `moshi_experiments`, and
nothing is abstracted across the two. The shapes are different, and seeing how
they differ is the point of having both.

This module has no provider imports. It runs on any Linux host with an NVIDIA
GPU and the pinned runtime installed. Provider lifecycle stays outside it: the
one operation only a provider can define, making writes durable, arrives as a
`commit` callable and defaults to a no-op, which is correct for a local
filesystem.

Two operations, in the order they are run:

    cache_models     download all three model sets at pinned revisions, no GPU
    residency        load all three onto one GPU and measure what they cost

Both are deliberately one container on one A10. Three containers would add two
cold starts and two network hops, and would put the stage timings in three
clock domains. We are measuring pipeline scheduling, not distributed-service
architecture, so the models stay resident together and the only hop is
localhost. Splitting them is a later experiment, and only measurements can
justify it.

What this module does not do yet: run a turn. Sequential and overlapped turns
come after the memory floor is known, because the smallest vLLM allocation that
leaves adequate KV headroom is an input to those experiments, not an output.
"""

import os
import re

DATA_ROOT = os.environ.get("VOICE_LAB_ROOT", "/cache")
HF_HOME = f"{DATA_ROOT}/hf"
MODELS_DIR = f"{DATA_ROOT}/models"
OUTPUT_DIR = f"{DATA_ROOT}/outputs/modular"

PYTHON_VERSION = "3.12"

# ---------------------------------------------------------------------------
# Pins. None of these may drift, for the same reason as the Moshi pins: a
# cached copy must never go stale, and a later run must never be pulled toward
# a newer upstream commit.
#
# Every model here is loaded from a local directory under MODELS_DIR that
# `cache_models` filled at an immutable revision. The runtime images are forced
# offline, so a missing file is a loud error rather than a silent re-download.
# ---------------------------------------------------------------------------

# STT -- Whisper large-v3-turbo, converted to CTranslate2.
#
# Three separate facts, each of which cost time to establish:
#
# 1. `openai/whisper-large-v3-turbo` cannot be loaded here at all. These are
#    the original weights, and CTranslate2 wants a converted model.
# 2. faster-whisper's own alias table maps both "large-v3-turbo" and "turbo" to
#    `mobiuslabsgmbh/faster-whisper-large-v3-turbo`, and Hugging Face now
#    redirects that repo id to the one pinned below. Loading by alias would
#    therefore depend on a rename redirect continuing to resolve.
# 3. So the canonical repo and an immutable revision are pinned instead. Same
#    weights, different serialization.
WHISPER_REPO = "dropbox-dash/faster-whisper-large-v3-turbo"
WHISPER_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
WHISPER_DIR = f"{MODELS_DIR}/faster-whisper-large-v3-turbo"
FASTER_WHISPER_PACKAGE = "faster-whisper==1.2.1"

# CTranslate2 needs cuBLAS and cuDNN 9 present as libraries. They are not
# bundled, and the dynamic loader reads LD_LIBRARY_PATH once at process start,
# so setting it from inside Python is too late. The image sets it as a real
# environment variable; `residency` reports the paths it actually finds, so a
# wrong assumption here fails visibly instead of silently.
NVIDIA_LIB_PATH = (
    "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:"
    "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib"
)

# 16 kHz mono is what Whisper expects, and one second of silence is enough to
# make the runtime allocate its working buffers.
WHISPER_SAMPLE_RATE = 16000
WHISPER_COMPUTE_TYPE = "float16"

# LLM -- Qwen2.5-3B-Instruct served by vLLM.
#
# vLLM's compiled extension is built against exactly one CUDA major, and the
# whole stack has to agree with it. Both halves of that were measured, in this
# order.
#
# With torch on cu129 and vLLM taken from PyPI, every sweep value died in about
# three seconds:
#
#     ImportError: libcudart.so.13: cannot open shared object file
#
# vLLM 0.30.0's PyPI wheel is built for CUDA 13.0. The release notes say so and
# the error confirms it. The installation guide's prose still claims CUDA 12.9
# by default; that sentence is stale, and the guide's own worked example for a
# specific CUDA version, plus the error, are what to trust.
#
# With torch on cu130 the import was fixed and a worse failure appeared. Whisper
# and Kokoro both *loaded* -- and Kokoro died on its first convolution:
#
#     cudnn_status: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
#
# At that moment the process had both libcublas.so.12 and libcublas.so.13
# mapped, and a single cuDNN 9: the CUDA 12 build, which is on LD_LIBRARY_PATH
# because CTranslate2 requires CUDA 12 and cuDNN 9. cuDNN 9 is split into
# sublibraries that must all come from one build, so torch on cu13 and
# CTranslate2 on cu12 cannot both be satisfied. Two CUDA majors in one process
# is not a risk to keep an eye on, it is broken -- and only the execution test
# showed it, because loading proved nothing.
#
# So the stack is pinned to one major, CUDA 12.9, and vLLM comes from the CUDA
# 12.9 wheel vLLM publishes as a release asset rather than from PyPI. Same vLLM
# version, same code, a different CUDA build. Not a downgrade, and not a
# substituted runtime.
VLLM_CUDA_VERSION = "129"
VLLM_VERSION = "0.30.0"

# One constant drives both the wheel and the torch backend, so they cannot
# disagree. A mismatch between them is precisely the failure described above.
VLLM_TORCH_BACKEND = f"cu{VLLM_CUDA_VERSION}"
VLLM_WHEEL_URL = (
    f"https://github.com/vllm-project/vllm/releases/download/v{VLLM_VERSION}/"
    f"vllm-{VLLM_VERSION}+cu{VLLM_CUDA_VERSION}-cp38-abi3-manylinux_2_28_x86_64.whl"
)
QWEN_REPO = "Qwen/Qwen2.5-3B-Instruct"
QWEN_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
QWEN_DIR = f"{MODELS_DIR}/Qwen2.5-3B-Instruct"

# `--generation-config vllm` is passed at run time. Without it vLLM applies the
# model repo's own generation_config.json, which silently overrides our sampling
# parameters with the model author's; that is the wrong default for a
# measurement we intend to repeat.
VLLM_SERVED_NAME = "qwen2.5-3b-instruct"
VLLM_PORT = 8000

# Context and output are bounded on purpose. The experiment is a handful of
# short conversational turns, and an unbounded context would make the KV cache
# requirement arbitrary. These two numbers, not taste, define what "enough KV
# headroom" means: four concurrent sequences of the full context length.
VLLM_MAX_MODEL_LEN = 4096
VLLM_MAX_NUM_SEQS = 4
VLLM_MAX_OUTPUT_TOKENS = 512

# vLLM takes a fraction of *total* device memory, not of free memory, so the
# value must always be set: its default of 0.90 would claim most of the A10
# regardless of what Whisper and Kokoro already hold.
#
# One value, not a sweep. The residency work settled the memory question: two of
# the three models cost 2.9 GB together against a 23.7 GB card, vLLM reached KV
# sizing comfortably at 0.40, and the startup failure that followed turned out
# to be a FlashInfer JIT compile with nothing to do with memory. Sweeping
# further would answer a question nobody is asking. What remains to establish is
# that all three models execute while simultaneously resident.
#
# It stays a tuple because the harness selects the smallest adequate setting
# from it, and with one entry that selection is trivially that entry.
VLLM_UTILIZATION = 0.40
VLLM_UTILIZATION_SWEEP = (VLLM_UTILIZATION,)

# The probe request, sized to the workload rather than to a benchmark.
VLLM_PROBE_PROMPT = "Reply with one short sentence about the weather."
VLLM_PROBE_MAX_TOKENS = 24

# Where a complete startup log is written. Its own artifact, so diagnosing a
# failure never depends on a terminal's scrollback.
VLLM_LOG_PATH = f"{OUTPUT_DIR}/vllm_startup.log"

# The phases a vLLM start passes through, in order. Each pattern is a line vLLM
# prints when that phase completed, so a phase counts as reached only when its
# marker is present. These are string matches against a log format we do not
# own: the matched line is recorded next to every verdict, because a phase that
# is missing only because vLLM reworded its output looks exactly like a phase
# that never happened, and the line is what tells them apart.
VLLM_PHASES = (
    ("engine_initialization", r"Loading model weights took|Model loading took|init engine"),
    ("memory_profiling", r"[Mm]emory profiling|profiling result"),
    ("kv_cache_creation", r"GPU KV cache size:|Available KV cache memory:"),
    (
        "http_server_startup",
        r"Starting vLLM API server|Application startup complete|Uvicorn running on",
    ),
)

# vLLM settings that are environment variables rather than flags. Passed to the
# child process explicitly, so they show up in the report instead of living only
# in an image definition.
#
# VLLM_USE_FLASHINFER_SAMPLER=0 is the load-bearing one. FlashInfer compiles its
# sampling kernels on first use with ninja and nvcc. This image carries the CUDA
# runtime libraries but no compiler, so the engine core died during kernel
# warmup, on the chain
#
#     flashinfer/sampling.py -> get_sampling_module().build_and_load()
#       -> jit/cpp_ext.py get_cuda_path()
#         RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
#
# vLLM's own issue tracker records this crash and names this variable as the
# workaround. Disabling it also removes a just-in-time compile from startup,
# which matters here: the first request would otherwise pay for a build, and
# this experiment measures latency. The target workload is one user sampling
# greedily, so FlashInfer's optimised sampling kernels buy nothing we measure.
VLLM_ENV_OVERRIDES = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}

# TTS -- Kokoro-82M.
#
# `KPipeline` exposes no revision parameter and no local-path parameter: it
# hands its `repo_id` straight to `hf_hub_download`, which then resolves through
# the Hugging Face cache. That was measured, not assumed. Pointing that loader
# at a directory of ours under offline mode does not resolve, so fetching into a
# directory of ours cannot work -- the files have to go into the cache the
# loader will read.
#
# Offline mode is what makes that cache immutable afterwards: with no network,
# resolution can only return what was fetched, so a moved upstream cannot leak
# in. The fetch therefore asks for the same default ref the loader will ask for,
# and the resolved commit is then checked against the pin below. A repointed
# upstream fails the run loudly instead of silently serving different weights.
#
# espeak-ng is a system dependency, not a Python one. It is the fallback G2P for
# out-of-dictionary English words, and the pipeline degrades quietly without it,
# so the image installs it.
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
KOKORO_PACKAGE = "kokoro==0.9.4"
KOKORO_LANG_CODE = "a"
KOKORO_VOICE = "af_heart"
KOKORO_SAMPLE_RATE = 24000
KOKORO_FILES = ("config.json", "kokoro-v1_0.pth", "voices/af_heart.pt")

# ---------------------------------------------------------------------------
# Experiment 1: the fixture, and the sequential turn.
# ---------------------------------------------------------------------------

# The spoken prompt, synthesised once with the pinned Kokoro voice and then
# frozen. Regenerating it per run would put synthesis inside the measurement
# window and change the input between runs, which is what an experiment must not
# do.
#
# A question rather than narration, deliberately: Whisper has one clear expected
# transcript, Qwen has a bounded task, and the answer naturally contains several
# sentence boundaries -- which is what the later LLM-to-TTS overlap experiment
# needs something real to overlap.
FIXTURE_TEXT = "Give me three practical tips for sleeping better, one short sentence each."
FIXTURE_SPEED = 1
FIXTURE_DIR = f"{DATA_ROOT}/fixtures"
FIXTURE_WAV = f"{FIXTURE_DIR}/exp1_prompt.wav"
FIXTURE_MANIFEST = f"{FIXTURE_DIR}/exp1_prompt.json"

# The turn's prompt template and output bound, fixed so that Experiment 2 can
# change exactly one thing: when synthesis is allowed to begin.
TURN_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer with at most three short sentences."
)
TURN_MAX_OUTPUT_TOKENS = 160

EXP1_DIR = f"{OUTPUT_DIR}/exp1"
EXP2_DIR = f"{OUTPUT_DIR}/exp2"

# Sentence boundaries for the overlapped turn. Deliberately the simplest rule
# that works on this fixture: a full stop, question mark or exclamation mark
# followed by whitespace, plus whatever remains when the stream ends.
#
# No NLP tokenizer. It would add a dependency and a behaviour we would then have
# to characterise, and the fixture does not need one. If the simple rule is seen
# to fail on real speech, that is the evidence that justifies replacing it.
SENTENCE_END = re.compile(r"([.?!])(\s|$)")

# misaki, Kokoro's G2P layer, installs this spaCy model the first time a
# pipeline is built, by shelling out to pip and fetching it from GitHub. That
# was observed in a run, not inferred: the load printed a pip install and a
# 12.8 MB download before any audio was produced.
#
# Two reasons it is pinned into the image instead. It makes the load
# reproducible, because otherwise the runtime resolves an unpinned dependency
# after the experiment starts. And it stops a timed model load from containing
# an install, which would otherwise make the first Kokoro timing meaningless.
SPACY_MODEL_VERSION = "3.8.0"
SPACY_MODEL_PACKAGE = (
    "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/"
    f"en_core_web_sm-{SPACY_MODEL_VERSION}/en_core_web_sm-{SPACY_MODEL_VERSION}-py3-none-any.whl"
)

# How often the background sampler reads device memory. Fast enough to catch a
# transient peak during a model load, cheap enough not to distort the run.
PEAK_SAMPLE_INTERVAL_SECONDS = 0.05

# How long to wait for the vLLM server to answer, and for device memory to come
# back after it is killed. A cold vLLM start includes weight load and CUDA graph
# capture, so the first is generous -- but it is also bounded, because a server
# that hangs rather than exits would otherwise hold the GPU indefinitely.
VLLM_READY_TIMEOUT_SECONDS = 600
MEMORY_RELEASE_TIMEOUT_SECONDS = 120


def _noop() -> None:
    """Default commit. A plain filesystem needs nothing to make writes durable."""


def _json(report: dict) -> str:
    """Serialise a report as a string.

    The operations return text rather than Python objects so a driver process
    never needs torch installed just to read a result.
    """
    import json

    return json.dumps(report, indent=2, sort_keys=False)


def _bytes_on_disk(path: str) -> int:
    """Total size of a file or a directory tree, in bytes."""
    from pathlib import Path

    target = Path(path)
    if target.is_file():
        return target.stat().st_size
    return sum(f.stat().st_size for f in target.rglob("*") if f.is_file())


def _package_versions() -> dict:
    """Resolved versions of the libraries this stack actually runs.

    The pins in this file are what we asked for; this is what the image
    resolved. That is the copy that can drift under us, so it belongs in the
    artifact rather than in a build log.
    """
    from importlib.metadata import PackageNotFoundError, version

    names = (
        "vllm",
        "faster-whisper",
        "ctranslate2",
        "kokoro",
        "misaki",
        "spacy",
        "en-core-web-sm",
        "torch",
        "transformers",
        "huggingface-hub",
        "numpy",
    )
    resolved = {}
    for name in names:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = None
    return resolved


def _write_report(report: dict, filename: str) -> str:
    """Persist a report to the Volume path, returning where it landed.

    Called more than once per run. A returned value only survives in a
    terminal's scrollback, and a long run that is interrupted has to leave
    behind what it did measure rather than nothing.
    """
    from pathlib import Path

    path = Path(OUTPUT_DIR, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(report))
    return str(path)


def _require(path: str, what: str) -> None:
    """Fail loudly when a cached asset is missing.

    The runtime images are offline, so this is the message a user sees instead
    of a silent multi-gigabyte download onto a billed GPU.
    """
    from pathlib import Path

    if not Path(path).exists():
        raise FileNotFoundError(
            f"{what} is not in the cache at {path}. "
            f"Run the cache operation first; it is the only one allowed to use the network."
        )


def _nvidia_lib_paths() -> dict:
    """Check that the CUDA libraries CTranslate2 needs are where we claim.

    An earlier version asked the `nvidia.cublas.lib` module for its `__file__`.
    That reports "unavailable" even when the libraries are present, because
    those packages are not ordinary modules. Reading the directories is a real
    check; asking the import system was not.
    """
    from pathlib import Path

    declared = os.environ.get("LD_LIBRARY_PATH") or ""
    directories = {}
    for entry in filter(None, declared.split(":")):
        directory = Path(entry)
        directories[entry] = {
            "exists": directory.is_dir(),
            "libraries": sorted(p.name for p in directory.glob("*.so*"))
            if directory.is_dir()
            else [],
        }
    return {"LD_LIBRARY_PATH": declared, "directories": directories}


def _snapshot_sha(resolved_path: str):
    """The commit recorded in a resolved Hugging Face cache path, if any."""
    from pathlib import Path

    parts = Path(resolved_path).parts
    if "snapshots" in parts:
        return parts[parts.index("snapshots") + 1]
    return None


# ---------------------------------------------------------------------------
# Device measurement.
#
# Three models, two independent allocators. PyTorch tracks its own allocations
# and CTranslate2 tracks none of them, so the honest measure is the device's own
# free-memory counter: what it says does not care who allocated what. Every
# per-model number below is a delta of that counter, never a sum of parameter
# sizes.
# ---------------------------------------------------------------------------


def _device() -> dict:
    """Device memory as the driver reports it."""
    import torch

    free, total = torch.cuda.mem_get_info()
    return {
        "total_bytes": int(total),
        "free_bytes": int(free),
        "used_bytes": int(total - free),
        "total_gb": round(total / 1e9, 3),
        "free_gb": round(free / 1e9, 3),
        "used_gb": round((total - free) / 1e9, 3),
    }


def _processes() -> list:
    """Per-process device memory, which separates a subprocess from its parent."""
    import re
    import subprocess

    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        pid, used = (part.strip() for part in line.split(","))
        rows.append({"pid": int(pid), "used_mib": int(re.sub(r"[^\d]", "", used) or 0)})
    return rows


def _gpu_query() -> dict:
    """What the platform actually handed us, for the record."""
    import subprocess

    import torch

    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        info["nvidia_smi"] = result.stdout.strip()
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        info["nvidia_smi"] = f"unavailable: {exc}"
    return info


class _PeakSampler:
    """Sample device memory in the background for the length of one stage.

    A single reading before and after a load misses a peak that occurs during
    it, which is exactly where an allocator spike would be. This is a thread
    rather than a hook because CTranslate2 exposes no allocation callback.
    """

    def __init__(self, interval: float = PEAK_SAMPLE_INTERVAL_SECONDS):
        import threading

        import torch

        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        _, self._total = torch.cuda.mem_get_info()
        self.peak_used_bytes = 0
        self.samples = 0

    def _run(self) -> None:
        import torch

        while not self._stop.is_set():
            free, _ = torch.cuda.mem_get_info()
            self.peak_used_bytes = max(self.peak_used_bytes, self._total - free)
            self.samples += 1
            self._stop.wait(self._interval)

    def __enter__(self) -> "_PeakSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        import torch

        self._stop.set()
        self._thread.join(timeout=5)
        free, _ = torch.cuda.mem_get_info()
        self.peak_used_bytes = max(self.peak_used_bytes, self._total - free)


def _run_stage(label: str, action):
    """Run one stage, recording its cost, its effect on the device, and its peak.

    Returns ``(record, result)``. ``delta_used_bytes`` is the honest per-model
    footprint: the change in what the driver reports as used, which counts
    CTranslate2 and PyTorch alike. An exception is captured into the record
    rather than raised, so a failed stage still reports which stage it was.
    """
    import time
    import traceback

    import torch

    before = _device()
    started = time.perf_counter()
    result = None
    error = None
    try:
        with _PeakSampler() as sampler:
            result = action()
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - reported, not swallowed
        error = traceback.format_exc()
    elapsed_ms = (time.perf_counter() - started) * 1000
    after = _device()

    record = {
        "ok": error is None,
        "duration_ms": round(elapsed_ms),
        "device_before_used_gb": before["used_gb"],
        "device_before_free_gb": before["free_gb"],
        "device_after_used_gb": after["used_gb"],
        "device_after_free_gb": after["free_gb"],
        "delta_used_bytes": after["used_bytes"] - before["used_bytes"],
        "delta_used_gb": round((after["used_bytes"] - before["used_bytes"]) / 1e9, 3),
        "peak_used_bytes": sampler.peak_used_bytes if error is None else None,
        "peak_used_gb": round(sampler.peak_used_bytes / 1e9, 3) if error is None else None,
        "samples": sampler.samples if error is None else None,
    }
    if error is not None:
        record["error"] = error
        record["loaded_cuda_libraries"] = _loaded_cuda_libraries()
    # A stage's own return value is often the measurement -- a transcript, a
    # generated string, a time to first token -- so it is kept when it can be
    # serialised into the report.
    if isinstance(result, (dict, list, str, int, float, bool)):
        record["result"] = result
    print(
        f"[residency] {label}: ok={record['ok']} {round(elapsed_ms)} ms,"
        f" device used {before['used_gb']} -> {after['used_gb']} GB",
        flush=True,
    )
    return record, result


# ---------------------------------------------------------------------------
# vLLM as a subprocess.
#
# The official serving path is used rather than an in-process engine, because
# the point of the exercise is the real API: an OpenAI-compatible endpoint whose
# streaming behaviour and time-to-first-token we can measure from the client
# side. Running it as a child process on localhost keeps that API while leaving
# every stage in one container, one clock domain and one billing unit.
# ---------------------------------------------------------------------------


def _vllm_command(utilization: float) -> list:
    return [
        "vllm",
        "serve",
        QWEN_DIR,
        "--served-model-name",
        VLLM_SERVED_NAME,
        "--gpu-memory-utilization",
        str(utilization),
        "--max-model-len",
        str(VLLM_MAX_MODEL_LEN),
        "--max-num-seqs",
        str(VLLM_MAX_NUM_SEQS),
        "--generation-config",
        "vllm",
        "--host",
        "127.0.0.1",
        "--port",
        str(VLLM_PORT),
        # No `--disable-log-requests`: vLLM 0.30.0 removed it, and passing it
        # makes `vllm serve` exit with status 2 before it loads anything --
        #        vllm: error: unrecognized arguments: --disable-log-requests
        # The flag was here to keep request lines out of the captured log. That
        # is cosmetic, and the log is parsed for startup numbers, so losing it
        # costs nothing worth another failed sweep.
    ]


def _http_ready(url: str, timeout: float, process=None) -> tuple:
    """Poll a URL until it answers, the process dies, or the timeout expires.

    Watching the process is the point. A server that exits because it could not
    allocate its KV cache will never answer, and sitting out the whole timeout
    for it bills GPU time to learn nothing. Returns (ready, reason) so a failure
    records why it was abandoned.
    """
    import time
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False, f"process exited with code {process.returncode}"
        try:
            with urllib.request.urlopen(url, timeout=5):
                return True, "ok"
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)
    return False, "timed out waiting for the server to answer"


def _kv_cache_tokens(log_text: str):
    """Read the KV cache capacity vLLM reports at startup, in tokens.

    Returns None rather than a guess when the line is absent or its wording has
    changed, so a missing number is visible as missing.
    """
    import re

    match = re.search(r"KV cache size:\s*([\d,]+)\s*tokens", log_text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def _kv_cache_bytes(log_text: str):
    """KV cache capacity in bytes, as vLLM reports it.

    vLLM prints the pool size in GiB. Kept as bytes so it can be compared with
    the device counters rather than floating around as a differently rounded
    number.
    """
    import re

    match = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", log_text)
    if not match:
        return None
    return int(float(match.group(1)) * 1024**3)


def _max_concurrency(log_text: str):
    """The concurrency vLLM reports for our configured context length."""
    import re

    match = re.search(
        r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x", log_text
    )
    if not match:
        return None
    return float(match.group(1))


def _reported_model_memory(log_text: str):
    """vLLM's own figure for what loading the weights cost, if it prints one."""
    import re

    match = re.search(r"Model loading took\s*([\d.]+)\s*GiB", log_text)
    if not match:
        return None
    return float(match.group(1))


def _loaded_cuda_libraries(pid=None) -> dict:
    """Which CUDA libraries a process has actually mapped into memory.

    Recorded because ``import`` succeeding proves very little: two runtimes on
    different CUDA majors can both import and still fail to share a device. What
    is mapped is evidence; what is installed is not.
    """
    import re
    from pathlib import Path

    maps = Path(f"/proc/{pid or 'self'}/maps")
    if not maps.exists():
        return {"error": f"{maps} is not readable"}

    keys = ("libcudart", "libcublas", "libcudnn", "libcuda.so", "libnvrtc", "ctranslate2", "libtorch")
    found = {}
    for line in maps.read_text().splitlines():
        match = re.search(r"/([^/\s]+\.so[^\s/]*)", line)
        if not match:
            continue
        name = match.group(1)
        if any(key in name for key in keys):
            found[name] = found.get(name, 0) + 1
    return dict(sorted(found.items()))


def _wait_for_release(baseline_used: int, timeout: float) -> dict:
    """Wait for a killed server's device memory to come back.

    CUDA context teardown is not instantaneous, and starting the next sweep
    value while the previous one still holds memory would corrupt the
    measurement.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = _device()
        if current["used_bytes"] <= baseline_used + (256 * 1024 * 1024):
            return {"released": True, "used_gb": current["used_gb"]}
        time.sleep(1.0)
    current = _device()
    return {"released": False, "used_gb": current["used_gb"]}


def _spawn_vllm(utilization: float):
    """Start ``vllm serve`` and stream its output into a list.

    Returns ``(process, logs)``. The caller owns the lifecycle, because the two
    callers want different things from the same output: the sweep keeps a
    bounded tail inside a larger report, while the startup probe keeps every
    line and writes the log out as its own artifact.
    """
    import os
    import subprocess
    import threading

    logs = []
    process = subprocess.Popen(
        _vllm_command(utilization),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, **VLLM_ENV_OVERRIDES},
    )

    def _drain() -> None:
        for line in process.stdout:
            logs.append(line.rstrip())

    threading.Thread(target=_drain, daemon=True).start()
    return process, logs


def _stop_vllm(process) -> dict:
    """Terminate a spawned server and report how it ended.

    An exit code is recorded whether the process died on its own or was
    stopped here, and the two are not the same fact: a server that was still
    running when we stopped it did not fail.
    """
    import subprocess

    if process is None:
        return {"exit_code": None, "stopped_by_us": None}

    stopped_by_us = process.poll() is None
    if stopped_by_us:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    return {"exit_code": process.returncode, "stopped_by_us": stopped_by_us}


def _first_fatal(log_lines: list) -> dict:
    """The first line that looks like the actual error, and its causal line.

    The first, not the last. Once something fatal happens, the rest of the log
    is the shutdown path unwinding, and the final line is almost never the
    cause. Three tiers are tried in order so that a genuine traceback outranks a
    mere log level, and the tier that matched is recorded.

    For a traceback the causal line is its *last* one, so the scan runs past the
    fixed window to reach it. The first real diagnosis was lost because the
    exception sat 35 lines beyond a 60-line window: the report showed the whole
    call chain and not the one line that explained it.
    """
    import re

    causal = re.compile(r"\b\w*(?:Error|Exception):\s")
    tiers = (
        ("traceback", re.compile(r"Traceback \(most recent call last\)")),
        ("exception", re.compile(r"^\s*(?:\w+\.)*\w*(?:Error|Exception)\b")),
        ("error_level", re.compile(r"\bERROR\b")),
    )

    for kind, pattern in tiers:
        for index, line in enumerate(log_lines):
            if not pattern.search(line):
                continue
            end = min(len(log_lines), index + 60)
            causal_index = None
            if kind == "traceback":
                for offset in range(index + 1, min(len(log_lines), index + 400)):
                    if causal.search(log_lines[offset]):
                        causal_index = offset
                        # The context ends at the cause rather than running on
                        # past it. Everything after an exception is the shutdown
                        # path, and the whole log is saved anyway.
                        end = offset + 1
                        break
            return {
                "kind": kind,
                "line_number": index + 1,
                "line": line.strip(),
                "causal_line_number": (
                    causal_index + 1 if causal_index is not None else None
                ),
                "causal_line": (
                    log_lines[causal_index].strip() if causal_index is not None else None
                ),
                "context": log_lines[max(0, index - 3) : end],
            }
    return None


def _detect_phases(log_lines: list, generation=None) -> dict:
    """Which vLLM startup phases were reached, and what proves each one."""
    import re

    phases = {}
    for name, pattern in VLLM_PHASES:
        compiled = re.compile(pattern)
        match = next(
            (
                (index, line)
                for index, line in enumerate(log_lines)
                if compiled.search(line)
            ),
            None,
        )
        phases[name] = (
            {"verdict": "PASS", "line_number": match[0] + 1, "line": match[1].strip()}
            if match
            else {"verdict": "FAIL", "line_number": None, "line": None}
        )

    phases["first_generation"] = {
        "verdict": "PASS" if (generation or {}).get("ok") else "FAIL",
        "evidence": (generation or {}).get("text") or (generation or {}).get("error"),
    }
    # first_generation counts towards the last phase reached. Leaving it out
    # reported a run where generation had just succeeded as having stopped at
    # the HTTP server.
    order = [name for name, _ in VLLM_PHASES] + ["first_generation"]
    reached = [name for name in order if phases[name]["verdict"] == "PASS"]
    return {
        "phases": phases,
        "last_phase_reached": reached[-1] if reached else None,
        "died_after_engine_initialization": phases["engine_initialization"]["verdict"]
        == "PASS",
    }


def _try_vllm(utilization: float, stages) -> dict:
    """Start the server at one memory fraction, run the stages, then stop it.

    ``stages`` is an ordered list of ``(name, callable)``. They run in that order
    while vLLM holds the device, because "all three models loaded" is a weaker
    claim than "all three executed while simultaneously resident", and the order
    is the order the real pipeline will use.

    Each stage is wrapped on its own, so a failure names the stage that produced
    it and records the device state and the mapped libraries at that moment --
    captured before anything is changed in response to it.
    """
    import time
    import traceback

    entry = {
        "gpu_memory_utilization": utilization,
        "started": False,
        "env_overrides": VLLM_ENV_OVERRIDES,
    }
    before = _device()
    logs = []
    process = None
    # Printed as it happens. The sweep is several cold starts, and without
    # progress lines the run is unfollowable: vLLM's own output is captured for
    # the report rather than streamed, so nothing else reaches the container log.
    print(
        f"[residency] vllm gpu_memory_utilization={utilization}: starting", flush=True
    )

    try:
        with _PeakSampler() as sampler:
            started = time.perf_counter()
            process, logs = _spawn_vllm(utilization)
            ready, reason = _http_ready(
                f"http://127.0.0.1:{VLLM_PORT}/v1/models",
                VLLM_READY_TIMEOUT_SECONDS,
                process,
            )
            entry["startup_ms"] = round((time.perf_counter() - started) * 1000)
            entry["started"] = ready
            if not ready:
                entry["failure"] = reason

            entry["stage_sequence"] = []
            if ready:
                for name, action in stages:
                    record, _ = _run_stage(name, action)
                    entry["stage_sequence"].append({"stage": name, **record})

        entry["device_after"] = _device()
        entry["delta_used_bytes"] = entry["device_after"]["used_bytes"] - before["used_bytes"]
        entry["device_total_gb"] = entry["device_after"]["total_gb"]
        entry["device_free_before_vllm_gb"] = before["free_gb"]
        entry["device_free_after_vllm_gb"] = entry["device_after"]["free_gb"]
        entry["peak_used_bytes_whole_start"] = sampler.peak_used_bytes
        entry["peak_used_gb_whole_start"] = round(sampler.peak_used_bytes / 1e9, 3)

        log_text = "\n".join(logs)
        entry["qwen_reported_load_gb"] = _reported_model_memory(log_text)
        entry["kv_cache_tokens"] = _kv_cache_tokens(log_text)
        entry["kv_cache_bytes"] = _kv_cache_bytes(log_text)
        entry["max_concurrency"] = _max_concurrency(log_text)
        entry["loaded_cuda_libraries_main_process"] = _loaded_cuda_libraries()

        outcomes = [stage["ok"] for stage in entry["stage_sequence"]]
        entry["all_stages_ok"] = bool(outcomes) and all(outcomes)
        print(
            f"[residency] vllm gpu_memory_utilization={utilization}:"
            f" started={entry['started']} kv_cache_tokens={entry['kv_cache_tokens']}"
            f" all_stages_ok={entry['all_stages_ok']}"
            f" startup_ms={entry.get('startup_ms')} failure={entry.get('failure')}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        entry["failure"] = f"{type(exc).__name__}: {exc}"
        entry["exception_traceback"] = traceback.format_exc()
        entry["loaded_cuda_libraries_main_process"] = _loaded_cuda_libraries()
    finally:
        entry.update(_stop_vllm(process))

    entry["log_tail"] = logs[-40:]
    entry["log_lines"] = len(logs)
    # Classified from the log itself, and not gated on the server having failed
    # to start: a server that came up and then died during a stage is a
    # different bug from one that never listened, and both leave a log.
    if logs and not entry.get("all_stages_ok"):
        lowered = "\n".join(logs).lower()
        if "cannot open shared object file" in lowered:
            # The compiled extension wants a different CUDA major than the
            # installed torch. Named explicitly so the report diagnoses itself
            # instead of leaving a traceback to be read by hand.
            entry["failure_reason"] = "compiled_extension_needs_another_cuda_major"
        elif "no available memory for the cache blocks" in lowered:
            entry["failure_reason"] = "kv_cache_too_small_for_requested_context"
        elif "out of memory" in lowered:
            entry["failure_reason"] = "device_out_of_memory"
        elif "subprocess" in lowered or "engine core" in lowered:
            entry["failure_reason"] = "engine_subprocess_died"
    entry["memory_after_release"] = _wait_for_release(
        before["used_bytes"], MEMORY_RELEASE_TIMEOUT_SECONDS
    )
    return entry


def _streaming_completion(
    url: str, payload: dict, on_first_chunk=None, on_first_token=None
) -> dict:
    """Send one streaming chat request and time the first token.

    Time to first token is measured on this side rather than read from a server
    metric, because it is the number the pipeline will feel and because a
    client-side measurement cannot drift when the server changes how it reports.
    Two timings are kept: the first SSE chunk, and the first chunk that actually
    carries content. They are not the same event, and the difference is real.

    The optional hooks exist so a caller can stamp those moments on its own
    clock instead of adding an offset to a number measured here.
    """
    import json
    import time
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    result = {
        "ok": False,
        "first_chunk_ms": None,
        "first_token_ms": None,
        "total_ms": None,
        "chunks": 0,
        "text": "",
    }
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                body = line[len("data:") :].strip()
                if body == "[DONE]":
                    break
                chunk = json.loads(body)
                result["chunks"] += 1
                if result["first_chunk_ms"] is None:
                    result["first_chunk_ms"] = round(
                        (time.perf_counter() - started) * 1000
                    )
                    if on_first_chunk is not None:
                        on_first_chunk()
                choices = chunk.get("choices") or []
                piece = (choices[0].get("delta") or {}).get("content") if choices else None
                if piece:
                    if result["first_token_ms"] is None:
                        result["first_token_ms"] = round(
                            (time.perf_counter() - started) * 1000
                        )
                        if on_first_token is not None:
                            on_first_token()
                    result["text"] += piece
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["total_ms"] = round((time.perf_counter() - started) * 1000)
    return result


# ---------------------------------------------------------------------------
# Operations.
# ---------------------------------------------------------------------------


def cache_models(commit=_noop) -> str:
    """Download all three model sets at pinned revisions. No GPU is required.

    Acquisition is separable from any GPU work, exactly as in the Moshi phases:
    moving several gigabytes is not something to bill an A10 for. Reports
    whether each asset was already present, so a re-run is cheap and visible.
    """
    import time
    from pathlib import Path

    from huggingface_hub import hf_hub_download, snapshot_download

    report = {"operation": "cache_models", "assets": {}, "packages": _package_versions()}

    def record(name: str, path: str, existed: bool) -> None:
        report["assets"][name] = {
            "path": path,
            "present_before": existed,
            "bytes": _bytes_on_disk(path),
            "gb": round(_bytes_on_disk(path) / 1e9, 3),
        }

    # STT. `download_model` applies faster-whisper's own allow-list, so only the
    # files CTranslate2 actually reads are fetched.
    from faster_whisper.utils import download_model

    whisper_existed = Path(WHISPER_DIR, "model.bin").exists()
    started = time.perf_counter()
    download_model(WHISPER_REPO, output_dir=WHISPER_DIR, revision=WHISPER_REVISION)
    record("whisper", WHISPER_DIR, whisper_existed)
    report["assets"]["whisper"].update(
        {"repo": WHISPER_REPO, "revision": WHISPER_REVISION, "seconds": round(time.perf_counter() - started)}
    )

    # LLM. Config, tokenizer and weights only; the README is not needed to serve.
    qwen_existed = Path(QWEN_DIR, "config.json").exists()
    started = time.perf_counter()
    snapshot_download(
        QWEN_REPO,
        revision=QWEN_REVISION,
        local_dir=QWEN_DIR,
        allow_patterns=["*.json", "*.safetensors", "*.txt"],
    )
    record("qwen", QWEN_DIR, qwen_existed)
    report["assets"]["qwen"].update(
        {"repo": QWEN_REPO, "revision": QWEN_REVISION, "seconds": round(time.perf_counter() - started)}
    )

    # TTS. Fetched into the Hugging Face cache rather than into a directory of
    # ours, because the cache is the only place the loader will look. The ref
    # requested is the one the loader will ask for offline; the commit it
    # resolved to is then checked against the pin, so a moved upstream is a
    # failed run rather than a silent change of weights.
    def _cached(filename: str) -> bool:
        try:
            hf_hub_download(
                repo_id=KOKORO_REPO, filename=filename, local_files_only=True
            )
            return True
        except Exception:  # noqa: BLE001 - a cache miss is the answer, not an error
            return False

    kokoro_existed = all(_cached(filename) for filename in KOKORO_FILES)
    started = time.perf_counter()
    resolved = {
        filename: hf_hub_download(repo_id=KOKORO_REPO, filename=filename)
        for filename in KOKORO_FILES
    }
    kokoro_shas = {_snapshot_sha(path) for path in resolved.values()}
    record("kokoro", str(Path(resolved["config.json"]).parent), kokoro_existed)
    report["assets"]["kokoro"].update(
        {
            "repo": KOKORO_REPO,
            "pinned_revision": KOKORO_REVISION,
            "resolved_revisions": sorted(sha for sha in kokoro_shas if sha),
            "revision_matches_pin": kokoro_shas == {KOKORO_REVISION},
            "seconds": round(time.perf_counter() - started),
        }
    )

    # Structural check, not a log reading: every file each loader will ask for
    # must exist before we agree that the cache is complete.
    expected = {
        "whisper": [Path(WHISPER_DIR, f) for f in ("model.bin", "config.json", "tokenizer.json")],
        "qwen": [Path(QWEN_DIR, "config.json")],
        "kokoro": [Path(path) for path in resolved.values()],
    }
    missing = [str(p) for paths in expected.values() for p in paths if not p.exists()]
    report["missing"] = missing
    report["complete"] = not missing

    # Written before committing, because a returned value only survives in a
    # terminal's scrollback. The artifact has to outlive the run.
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    report_path = Path(OUTPUT_DIR, "cache.json")
    report_path.write_text(_json(report))
    report["written_to"] = str(report_path)
    commit()
    return _json(report)


def vllm_startup_probe(commit=_noop) -> str:
    """One vLLM start at a fixed fraction, with the complete log kept.

    Nothing else is loaded, and no sweep runs. This exists so a vLLM startup
    failure can be diagnosed without paying for a three-model residency run
    every time, and so the evidence is the whole subprocess output rather than
    an exit code.

    The log is written to the Volume as its own file, because the thing being
    diagnosed is the text, and text that only exists in a terminal is text that
    gets lost.
    """
    import time
    from pathlib import Path

    report = {
        "operation": "vllm_startup_probe",
        "environment": _gpu_query(),
        "packages": _package_versions(),
        "gpu_memory_utilization": VLLM_UTILIZATION,
        "command": _vllm_command(VLLM_UTILIZATION),
        "env_overrides": VLLM_ENV_OVERRIDES,
        "model_path": QWEN_DIR,
        "model_repo": QWEN_REPO,
        "model_revision": QWEN_REVISION,
        "notes": [
            "One start, one setting. The question is what the failure is, not how it "
            "varies with memory, so no other value is tried.",
            "The complete stdout and stderr are joined, written to the path in log_path, "
            "and analysed for the first causal error rather than the last line.",
            "Phase verdicts are string matches against a log format we do not control. "
            "The matched line is recorded beside each verdict, because a phase that is "
            "missing only because vLLM reworded its output looks exactly like one that "
            "never happened.",
        ],
    }

    if not report["environment"]["cuda_available"]:
        report["error"] = "CUDA is unavailable; refusing to fall back to CPU."
        return _json(report)

    _require(QWEN_DIR, "The Qwen weights")
    report["model_files"] = sorted(p.name for p in Path(QWEN_DIR).iterdir())
    report["device_before"] = _device()

    process, logs = _spawn_vllm(VLLM_UTILIZATION)
    ready, reason = _http_ready(
        f"http://127.0.0.1:{VLLM_PORT}/v1/models", VLLM_READY_TIMEOUT_SECONDS, process
    )
    report["http_reachable"] = ready
    report["http_outcome"] = reason

    generation = None
    if ready:
        generation = _streaming_completion(
            f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
            {
                "model": VLLM_SERVED_NAME,
                "messages": [{"role": "user", "content": VLLM_PROBE_PROMPT}],
                "max_tokens": VLLM_PROBE_MAX_TOKENS,
                "temperature": 0,
                "stream": True,
            },
        )
    report["generation"] = generation

    report["exit"] = _stop_vllm(process)
    # The drain thread reads until the pipe closes; give it a moment to finish
    # so the log does not lose its final lines.
    time.sleep(1.5)
    report["device_after"] = _device()

    report["log_lines"] = len(logs)
    report["log_path"] = VLLM_LOG_PATH
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(VLLM_LOG_PATH).write_text("\n".join(logs) + "\n")
    report["log_head"] = logs[:15]
    report["log_tail"] = logs[-15:]

    log_text = "\n".join(logs)
    report["first_fatal"] = _first_fatal(logs)
    report["phase_analysis"] = _detect_phases(logs, generation)
    report["kv_cache_tokens"] = _kv_cache_tokens(log_text)
    report["kv_cache_bytes"] = _kv_cache_bytes(log_text)
    report["max_concurrency"] = _max_concurrency(log_text)
    report["loaded_cuda_libraries_main_process"] = _loaded_cuda_libraries()

    phases = report["phase_analysis"]["phases"]
    print(
        f"[probe] exit={report['exit']} http_reachable={ready} log_lines={len(logs)}",
        flush=True,
    )
    for name in [n for n, _ in VLLM_PHASES] + ["first_generation"]:
        print(f"[probe]   {name}: {phases[name]['verdict']}", flush=True)
    print(
        f"[probe] last phase reached: {report['phase_analysis']['last_phase_reached']}",
        flush=True,
    )
    if report["first_fatal"]:
        fatal = report["first_fatal"]
        print(
            f"[probe] first fatal ({fatal['kind']}) at line {fatal['line_number']}:"
            f" {fatal['line']}",
            flush=True,
        )
        for line in fatal["context"]:
            print(f"[probe]   | {line}", flush=True)
    else:
        print("[probe] no fatal line matched any pattern", flush=True)
    print(f"[probe] full log: {VLLM_LOG_PATH}", flush=True)

    report["written_to"] = _write_report(report, "vllm_startup.json")
    commit()
    return _json(report)


def residency(commit=_noop) -> str:
    """Load all three models into one A10 and measure what they actually cost.

    This is the memory floor for the whole modular stack, so it is measured
    before any turn is built. It answers three questions the turn experiments
    depend on: what each model really holds on the device, whether the three fit
    together on 24 GB, and the smallest vLLM memory fraction that still leaves
    the KV cache room our bounded context needs.

    Order matters. Whisper loads first, then Kokoro, then vLLM, so each stage's
    delta is attributable to the model loaded in it. All three stay resident
    throughout the sweep, because coexistence is the thing being tested.
    """
    import traceback

    report = {
        "operation": "residency",
        "environment": _gpu_query(),
        "packages": _package_versions(),
        "ctranslate2_libs": _nvidia_lib_paths(),
        "notes": [
            "device_baseline is not zero: querying CUDA creates a context in this process, and "
            "that costs device memory before any model is loaded. Per-model deltas are measured "
            "against that baseline, not against an empty device.",
            "Whisper and Kokoro footprints are device-level deltas. CTranslate2 keeps no "
            "allocation statistics, so PyTorch's counters would miss Whisper entirely; the "
            "driver's own free-memory counter is the only measure that sees both allocators.",
            "Qwen runs in a child process, so the per-process listing separates its memory from "
            "this process's. The sweep deltas are still device-level, because coexistence is "
            "what is being tested.",
            "vLLM's fraction is of total device memory, not of free memory. Its default of 0.90 "
            "would claim most of the A10 regardless of what Whisper and Kokoro already hold, "
            "which is why it is set explicitly and swept rather than left alone.",
            "peak_used_bytes comes from a sampling thread, not from an allocation hook. A peak "
            "shorter than the sampling interval can be missed; the sample count is reported so "
            "the resolution is visible.",
            "The Whisper probe transcribes one second of digital silence. Its transcript is "
            "recorded as an observation about the model, not as evidence that the probe worked: "
            "Whisper produces text from silence. That bears directly on how this pipeline should "
            "decide a turn has ended, so it is kept rather than filtered out.",
            "Only the cache operation is allowed to touch the Hugging Face Hub. Every runtime "
            "container sets HF_HUB_OFFLINE=1, and the vLLM child process inherits it, so a cache "
            "miss raises instead of downloading onto a billed GPU. That guarantee is scoped to "
            "huggingface_hub, not to the network: the first Kokoro load was observed installing a "
            "spaCy model from GitHub, which is why that model is now pinned into the image.",
        ],
    }

    if not report["environment"]["cuda_available"]:
        report["error"] = "CUDA is unavailable; refusing to fall back to CPU."
        return _json(report)

    try:
        for path, what in (
            (WHISPER_DIR, "The Whisper conversion"),
            (QWEN_DIR, "The Qwen weights"),
        ):
            _require(path, what)

        report["device_baseline"] = _device()
        report["stages"] = {}
        report["processes_baseline"] = _processes()

        # --- STT ---------------------------------------------------------
        import numpy as np
        from faster_whisper import WhisperModel

        record, whisper = _run_stage(
            "whisper_load",
            lambda: WhisperModel(
                WHISPER_DIR, device="cuda", compute_type=WHISPER_COMPUTE_TYPE
            ),
        )
        report["stages"]["whisper_load"] = record

        silence = np.zeros(WHISPER_SAMPLE_RATE, dtype="float32")

        def _transcribe() -> dict:
            segments, _info = whisper.transcribe(silence, language="en", beam_size=1)
            return {"text": "".join(segment.text for segment in segments).strip()}

        record, report["whisper_probe"] = _run_stage("whisper_probe", _transcribe)
        report["stages"]["whisper_probe"] = record
        report["device_after_whisper"] = _device()

        # --- TTS ---------------------------------------------------------
        from kokoro import KPipeline

        # The repo id is passed, not a directory: resolving through the cache is
        # the only route this loader supports, which was measured rather than
        # assumed. The cache operation filled that cache, and the image keeps
        # it offline so nothing else can appear in it.
        record, kokoro = _run_stage(
            "kokoro_load",
            lambda: KPipeline(
                lang_code=KOKORO_LANG_CODE, repo_id=KOKORO_REPO, device="cuda"
            ),
        )
        report["stages"]["kokoro_load"] = record

        def _synthesise() -> dict:
            for result in kokoro("Hello there.", voice=KOKORO_VOICE):
                if result.audio is not None:
                    return {
                        "samples": int(result.audio.shape[-1]),
                        "sample_rate": KOKORO_SAMPLE_RATE,
                        "seconds": round(result.audio.shape[-1] / KOKORO_SAMPLE_RATE, 3),
                    }
            return {"samples": 0}

        record, report["kokoro_probe"] = _run_stage("kokoro_probe", _synthesise)
        report["stages"]["kokoro_probe"] = record
        report["device_after_kokoro"] = _device()
        report["processes_after_kokoro"] = _processes()

        # Both must actually load before coexistence means anything. Loading is
        # captured per stage rather than raised, so a failure is written to the
        # report before this stops the run.
        if not (
            report["stages"]["whisper_load"]["ok"]
            and report["stages"]["kokoro_load"]["ok"]
        ):
            raise RuntimeError(
                "a model failed to load, so coexistence cannot be tested; "
                "see stages.whisper_load and stages.kokoro_load"
            )

        # --- LLM ---------------------------------------------------------
        # Whisper and Kokoro are already resident and stay resident, so every
        # vLLM measurement below is made with the other two holding memory.

        def _qwen_generate() -> dict:
            return _streaming_completion(
                f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
                {
                    "model": VLLM_SERVED_NAME,
                    "messages": [{"role": "user", "content": VLLM_PROBE_PROMPT}],
                    "max_tokens": VLLM_PROBE_MAX_TOKENS,
                    "temperature": 0,
                    "stream": True,
                },
            )

        # Run in the order the real pipeline will use, at every setting. "All
        # three models loaded" is a much weaker claim than "all three executed
        # while simultaneously resident", and only the second one is worth
        # anything for a pipeline.
        coexistence_stages = [
            ("whisper_transcribe", _transcribe),
            ("qwen_generate", _qwen_generate),
            ("kokoro_synthesise", _synthesise),
        ]

        # Written and committed after every value, so an interrupted sweep still
        # leaves behind what it measured. A long run has already ended twice
        # with nothing to read; this is the fix for that.
        sweep = []
        for value in VLLM_UTILIZATION_SWEEP:
            sweep.append(_try_vllm(value, coexistence_stages))
            report["vllm_sweep"] = sweep
            report["written_to"] = _write_report(report, "residency.json")
            commit()

        report["kv_required_tokens"] = VLLM_MAX_MODEL_LEN * VLLM_MAX_NUM_SEQS
        # Adequate means all three conditions, not just a big KV cache: the
        # server started, every stage executed, and the cache holds a full
        # context for every sequence we allow.
        adequate = [
            entry
            for entry in sweep
            if entry.get("started")
            and entry.get("all_stages_ok")
            and (entry.get("kv_cache_tokens") or 0) >= report["kv_required_tokens"]
        ]
        smallest = (
            min(adequate, key=lambda e: e["gpu_memory_utilization"])
            if adequate
            else None
        )
        report["selected"] = (
            {
                "gpu_memory_utilization": smallest["gpu_memory_utilization"],
                "kv_cache_tokens": smallest["kv_cache_tokens"],
                "kv_cache_bytes": smallest["kv_cache_bytes"],
                "device_free_after_gb": smallest["device_free_after_vllm_gb"],
                "criterion": (
                    f"smallest fraction where the server started, all three stages ran, "
                    f"and the KV cache holds {VLLM_MAX_MODEL_LEN} tokens x "
                    f"{VLLM_MAX_NUM_SEQS} sequences with Whisper and Kokoro resident"
                ),
            }
            if smallest
            else None
        )

        # One repeat at the chosen value. It answers "is this the final
        # configuration" by measurement rather than by inheriting a sweep row,
        # and it doubles as a check that the same value behaves the same twice.
        if smallest:
            report["final_residency"] = _try_vllm(
                smallest["gpu_memory_utilization"], coexistence_stages
            )
            report["written_to"] = _write_report(report, "residency.json")
            commit()

        report["device_after_all"] = _device()
        report["processes_after_all"] = _processes()
        report["resident_total_bytes"] = (
            report["device_after_all"]["used_bytes"]
            - report["device_baseline"]["used_bytes"]
        )
        report["loaded"] = True

    except Exception:  # noqa: BLE001 - reported, not swallowed
        report["loaded"] = False
        report["error"] = traceback.format_exc()

    report["written_to"] = _write_report(report, "residency.json")
    commit()
    return _json(report)


# ---------------------------------------------------------------------------
# Fixture preparation.
#
# Generating the spoken prompt is deliberately separate from measuring the
# pipeline, and it runs with no GPU attached. The boundary matters: the fixture
# is an input to the experiment, not a stage of it, so its cost never appears in
# a reported latency.
# ---------------------------------------------------------------------------


def _audio_to_numpy(audio):
    """Kokoro hands back a torch tensor; the WAV writer wants an array."""
    import numpy as np

    if hasattr(audio, "detach"):
        return np.asarray(audio.detach().cpu()).reshape(-1)
    return np.asarray(audio).reshape(-1)


def _split_sentences(buffer: str) -> tuple:
    """Take completed sentences off the front of a streaming buffer.

    Returns ``(completed, remainder)``. A sentence ends at a full stop, question
    mark or exclamation mark followed by whitespace, or at the end of the
    buffer -- the caller flushes the remainder when the stream closes, which is
    what covers the final sentence when no trailing space follows it.

    The text is preserved as it arrived apart from surrounding whitespace, so
    what Kokoro is given is what the model actually produced.
    """
    completed = []
    start = 0
    for match in SENTENCE_END.finditer(buffer):
        sentence = buffer[start : match.end(1)].strip()
        if sentence:
            completed.append(sentence)
        start = match.end()
    return completed, buffer[start:]


def _first_offsets(events: list) -> dict:
    """The first occurrence of each event name, as a turn-relative offset."""
    offsets = {}
    for event in events:
        offsets.setdefault(event["event"], event["t_ms"])
    return offsets


def _span(offsets: dict, first: str, second: str):
    """One span, read off the trace so the summary cannot disagree with it."""
    if first in offsets and second in offsets:
        return round(offsets[second] - offsets[first], 3)
    return None


def _sha256_file(path: str) -> str:
    import hashlib
    from pathlib import Path

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_fixture(commit=_noop) -> str:
    """Synthesise the spoken prompt once, with the pinned Kokoro voice.

    Fixture preparation, not measurement. It runs before a benchmark and never
    during one. Two boundaries are recorded rather than left implicit:

    Kokoro produced this input, but that does not make Kokoro part of the
    measured input path. The fixture is a fixed WAV, and the measured pipeline
    starts at that WAV.

    Synthetic speech is cleaner than a real microphone, so this is suitable for
    a controlled scheduling experiment and is not an ASR quality benchmark. The
    live browser run is the real-human-speech check.
    """
    import time
    from pathlib import Path

    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    Path(FIXTURE_DIR).mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    pipeline = KPipeline(lang_code=KOKORO_LANG_CODE, repo_id=KOKORO_REPO, device="cpu")
    chunks = [
        _audio_to_numpy(result.audio)
        for result in pipeline(FIXTURE_TEXT, voice=KOKORO_VOICE, speed=FIXTURE_SPEED)
        if result.audio is not None
    ]
    audio = (
        np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
    ).astype("float32")
    synthesis_ms = round((time.perf_counter() - started) * 1000)

    sf.write(FIXTURE_WAV, audio, KOKORO_SAMPLE_RATE, subtype="PCM_16")
    digest = _sha256_file(FIXTURE_WAV)

    manifest = {
        "source_text": FIXTURE_TEXT,
        "model_repo": KOKORO_REPO,
        "model_revision": KOKORO_REVISION,
        "voice": KOKORO_VOICE,
        "speed": FIXTURE_SPEED,
        "generated_sample_rate": KOKORO_SAMPLE_RATE,
        "channels": 1,
        "subtype": "PCM_16",
        "duration_seconds": round(audio.shape[0] / KOKORO_SAMPLE_RATE, 3),
        "samples": int(audio.shape[0]),
        "sha256": digest,
        "wav_path": FIXTURE_WAV,
        "synthesis_ms": synthesis_ms,
        "generation_device": "cpu",
        "limitations": [
            "Synthetic speech is cleaner than real microphone speech. Suitable for a "
            "controlled scheduling experiment; not an ASR quality benchmark. The live "
            "browser run provides the real-human-speech check.",
            "Kokoro produced this input, but that does not make Kokoro part of the "
            "measured input path. Fixture preparation happens once, offline, and its "
            "latency is excluded from every reported turn.",
        ],
    }
    Path(FIXTURE_MANIFEST).write_text(_json(manifest) + "\n")

    print(
        f"[fixture] {FIXTURE_WAV}: {manifest['duration_seconds']}s,"
        f" sha256 {digest[:16]}..., synthesis {synthesis_ms} ms",
        flush=True,
    )
    commit()
    return _json(manifest)


def _to_whisper_rate(path: str) -> tuple:
    """Derive the 16 kHz mono float input Whisper needs from the saved fixture.

    Derived rather than synthesised a second time, so the 16 kHz input and the
    24 kHz fixture are provably the same recording. 24000:16000 is exactly 3:2,
    so no fractional rate is approximated.

    PyAV is used because faster-whisper already depends on it: no new dependency,
    and it is the same decoder the library would have used had it been handed the
    file. The transformation is returned beside the samples so it lands in the
    report instead of staying implicit.
    """
    import av
    import numpy as np

    blocks = []
    with av.open(path) as container:
        stream = container.streams.audio[0]
        source_rate = stream.rate
        source_samples = 0
        resampler = av.AudioResampler(
            format="fltp", layout="mono", rate=WHISPER_SAMPLE_RATE
        )
        for frame in container.decode(stream):
            source_samples += frame.samples
            blocks.extend(resampler.resample(frame))
        try:
            blocks.extend(resampler.resample(None))
        except Exception:  # noqa: BLE001 - older builds have no flush signature
            pass

    samples = np.concatenate([block.to_ndarray() for block in blocks], axis=1)
    samples = samples.reshape(-1).astype("float32")
    return samples, {
        "method": "av.AudioResampler(format=fltp, layout=mono, rate=16000)",
        "source_path": path,
        "source_rate": source_rate,
        "target_rate": WHISPER_SAMPLE_RATE,
        "ratio": f"{source_rate}:{WHISPER_SAMPLE_RATE}",
        "source_samples": int(source_samples),
        "source_seconds": round(source_samples / source_rate, 3),
        "target_samples": int(samples.shape[0]),
        "target_seconds": round(samples.shape[0] / WHISPER_SAMPLE_RATE, 3),
        "dtype": str(samples.dtype),
    }


# ---------------------------------------------------------------------------
# One implementation of each policy, shared by every operation that runs them.
#
# The two arms of the comparison must not be two copies of the same logic: a
# change to one copy would silently make the experiment compare two different
# pipelines. They live here, once, and the standalone operations and the paired
# comparison all call the same code.
# ---------------------------------------------------------------------------

# Paid before any measured turn in the paired run. Its cost is large enough to
# dominate a time-to-first-audio number.
WARMUP_PHRASE = "Warming up the speech models."

# How many sequential/overlapped pairs the paired run measures. The turns are a
# few seconds each, so repetition costs little and gives a spread to judge
# against the effect.
PAIRED_PAIRS = 3

PAIRED_DIR = f"{OUTPUT_DIR}/paired"


def _stt_stage(whisper, emit) -> dict:
    """Transcribe the fixture. Identical in both arms, deliberately."""
    emit("stt_start", "stt")
    samples, resample = _to_whisper_rate(FIXTURE_WAV)
    segments, info = whisper.transcribe(samples, language="en", beam_size=5)
    segment_list = [
        {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
        for s in segments
    ]
    transcript = "".join(segment["text"] for segment in segment_list).strip()
    emit(
        "stt_final",
        "stt",
        transcript=transcript,
        segments=len(segment_list),
        audio_seconds=resample["source_seconds"],
    )
    return {
        "transcript": transcript,
        "segments": segment_list,
        "resample": resample,
        "language": {
            "code": info.language,
            "probability": round(info.language_probability, 4),
        },
    }


def _turn_payload(transcript: str) -> dict:
    """The request, built in one place so both arms send the same thing."""
    return {
        "model": VLLM_SERVED_NAME,
        "messages": [
            {"role": "system", "content": TURN_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        "max_tokens": TURN_MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "stream": True,
    }


def _arm_sequential(kokoro, payload, emit) -> dict:
    """Generate the whole reply, then synthesise it. Experiment 1's policy."""
    import numpy as np

    chunks = []
    parts = []
    with _PeakSampler() as sampler:
        emit(
            "llm_request_start",
            "llm",
            prompt_chars=len(payload["messages"][1]["content"]),
        )
        generation = _streaming_completion(
            f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
            payload,
            on_first_token=lambda: emit("llm_ttft", "llm"),
        )
        if not generation.get("ok"):
            raise RuntimeError(f"generation failed: {generation.get('error')}")
        reply = generation["text"].strip()
        emit("llm_done", "llm", chars=len(reply), chunks=generation["chunks"])

        emit("tts_start", "tts", chars=len(reply))
        for index, result in enumerate(kokoro(reply, voice=KOKORO_VOICE)):
            if result.audio is None:
                continue
            if not chunks:
                emit("tts_first_audio", "tts")
            part = _audio_to_numpy(result.audio)
            parts.append(part)
            chunks.append(
                {
                    "index": index,
                    "graphemes": result.graphemes,
                    "samples": int(part.shape[0]),
                    "seconds": round(part.shape[0] / KOKORO_SAMPLE_RATE, 3),
                }
            )
        response = (
            np.concatenate(parts) if parts else np.zeros(0, dtype="float32")
        ).astype("float32")
        emit(
            "tts_done",
            "tts",
            chunks=len(chunks),
            seconds=round(response.shape[0] / KOKORO_SAMPLE_RATE, 3),
        )

    return {
        "reply": reply,
        "generation": generation,
        "tts_chunks": chunks,
        "response": response,
        "device_peak_bytes": sampler.peak_used_bytes,
        "device_peak_samples": sampler.samples,
    }


def _arm_overlapped(kokoro, payload, emit, now_ms) -> dict:
    """Synthesise each completed sentence while the model continues.

    Experiment 2's policy. The stream is read on its own thread: if synthesis
    ran on the reading thread, this process would stop reading while it
    synthesised, tokens would pile up in the socket, and llm_done would report
    when the consumer got round to them rather than when the model finished. The
    overlap would look real while being an artifact of a slow reader.
    """
    import json
    import queue
    import threading
    import urllib.request

    import numpy as np

    pending = queue.Queue()
    reply_parts = []
    token_timeline = []

    def _reader() -> None:
        buffer = ""
        clause_id = 0
        first_token = True
        request = urllib.request.Request(
            f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[len("data:") :].strip()
                    if body == "[DONE]":
                        break
                    chunk = json.loads(body)
                    choices = chunk.get("choices") or []
                    piece = (
                        (choices[0].get("delta") or {}).get("content")
                        if choices
                        else None
                    )
                    if not piece:
                        continue
                    token_timeline.append({"t_ms": now_ms(), "chars": len(piece)})
                    reply_parts.append(piece)
                    if first_token:
                        first_token = False
                        emit("llm_ttft", "llm")
                    buffer += piece
                    completed, buffer = _split_sentences(buffer)
                    for sentence in completed:
                        emit(
                            "llm_sentence_ready",
                            "llm",
                            clause_id=clause_id,
                            text=sentence,
                        )
                        pending.put({"clause_id": clause_id, "text": sentence})
                        clause_id += 1
            if buffer.strip():
                emit(
                    "llm_sentence_ready",
                    "llm",
                    clause_id=clause_id,
                    text=buffer.strip(),
                )
                pending.put({"clause_id": clause_id, "text": buffer.strip()})
            emit("llm_done", "llm")
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            pending.put({"error": f"{type(exc).__name__}: {exc}"})
        finally:
            pending.put({"end": True})

    emit(
        "llm_request_start",
        "llm",
        prompt_chars=len(payload["messages"][1]["content"]),
    )
    threading.Thread(target=_reader, daemon=True).start()

    clauses = []
    clause_audios = []
    with _PeakSampler() as sampler:
        while True:
            try:
                item = pending.get(timeout=600)
            except queue.Empty:
                raise RuntimeError("the generation stream stopped producing sentences")
            if item.get("end"):
                break
            if "error" in item:
                raise RuntimeError(f"generation stream failed: {item['error']}")

            clause_id, text = item["clause_id"], item["text"]
            emit("tts_clause_start", "tts", clause_id=clause_id, text=text)
            parts = []
            for result in kokoro(text, voice=KOKORO_VOICE):
                if result.audio is None:
                    continue
                if not parts:
                    emit("tts_first_audio", "tts", clause_id=clause_id)
                parts.append(_audio_to_numpy(result.audio))

            clause_audio = (
                np.concatenate(parts) if parts else np.zeros(0, dtype="float32")
            ).astype("float32")
            clause_audios.append(clause_audio)
            clauses.append(
                {
                    "clause_id": clause_id,
                    "clause_text": text,
                    "chars": len(text),
                    "samples": int(clause_audio.shape[0]),
                    "seconds": round(clause_audio.shape[0] / KOKORO_SAMPLE_RATE, 3),
                }
            )
            emit(
                "tts_clause_done",
                "tts",
                clause_id=clause_id,
                seconds=clauses[-1]["seconds"],
            )
            # There is no downstream socket in this harness, so "sent" means the
            # first clause's audio is complete and queued. Availability and
            # handoff are the same instant here, and both are recorded rather
            # than one being inferred from the other.
            if clause_id == 0:
                emit("first_audio_sent", "tts", clause_id=0)

    response = (
        np.concatenate(clause_audios) if clause_audios else np.zeros(0, dtype="float32")
    ).astype("float32")
    return {
        "reply": "".join(reply_parts).strip(),
        "clauses": clauses,
        "clause_audios": clause_audios,
        "response": response,
        "token_timeline": token_timeline,
        "device_peak_bytes": sampler.peak_used_bytes,
        "device_peak_samples": sampler.samples,
    }


def _warm_up(whisper, kokoro) -> dict:
    """Pay both models' first-call costs before any measured turn.

    The first call through either model is far more expensive than later ones.
    Kokoro compiles CUDA graphs and lazily initialises kernels; CTranslate2 does
    the same for Whisper. Measured, that cost landed on the first synthesis of
    the first turn, where it dominated the very number these experiments
    compare -- a first clause costing 2,851 ms against 145 ms and 133 ms for the
    identical work that followed.

    Paying it once, here, is the same move the Moshi work made with its explicit
    warm-up. It is outside every measured turn and reported separately.
    """
    import time

    result = {"phrase": WARMUP_PHRASE, "note": "Outside every measured turn."}

    started = time.perf_counter()
    samples, _ = _to_whisper_rate(FIXTURE_WAV)
    segments, _info = whisper.transcribe(samples, language="en", beam_size=5)
    "".join(segment.text for segment in segments)
    result["whisper_ms"] = round((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    for _ in kokoro(WARMUP_PHRASE, voice=KOKORO_VOICE):
        pass
    result["kokoro_ms"] = round((time.perf_counter() - started) * 1000)

    print(
        f"[paired] warm-up: whisper {result['whisper_ms']} ms,"
        f" kokoro {result['kokoro_ms']} ms",
        flush=True,
    )
    return result


def sequential_turn(commit=_noop) -> str:
    """Experiment 1 -- the fully sequential baseline.

    Fixed WAV, Whisper full transcription, Qwen full generation, Kokoro full
    synthesis of the whole reply, saved output WAV. Nothing overlaps: each stage
    begins only once the previous one has finished. That is the point of it.
    Experiment 2 changes exactly one thing -- when synthesis is allowed to
    begin -- and this is the baseline it changes from.

    One clock throughout: this process's monotonic counter, stamped relative to
    turn_start. No wall clock and no browser clock are mixed in, so every span is
    exact and immune to the host clock moving underneath it.

    Model loading happens before turn_start and is reported separately. A turn is
    the pipeline running with its models resident; folding a cold start into the
    turn would describe a different experiment.
    """
    import json
    import time
    from pathlib import Path

    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    from kokoro import KPipeline

    Path(EXP1_DIR).mkdir(parents=True, exist_ok=True)

    clock = time.perf_counter_ns
    started_ns = clock()
    events = []

    def emit(name: str, stage: str, **meta) -> None:
        offset = clock() - started_ns
        events.append(
            {
                "event": name,
                "stage": stage,
                "t_ns": offset,
                "t_ms": round(offset / 1e6, 3),
                "meta": meta,
            }
        )

    summary = {
        "operation": "sequential_turn",
        "experiment": 1,
        "session_id": f"exp1-{int(time.time())}",
        "clock": (
            "perf_counter_ns in one process. t_ms is an offset from the start of the "
            "operation; turn_start is an event inside the trace and marks the start of "
            "the measured turn, which is the origin every reported span uses."
        ),
        "environment": _gpu_query(),
        "packages": _package_versions(),
        "config": {
            "fixture_wav": FIXTURE_WAV,
            "fixture_text": FIXTURE_TEXT,
            "whisper_dir": WHISPER_DIR,
            "whisper_compute_type": WHISPER_COMPUTE_TYPE,
            "whisper_input_rate": WHISPER_SAMPLE_RATE,
            "qwen_repo": QWEN_REPO,
            "qwen_revision": QWEN_REVISION,
            "qwen_served_name": VLLM_SERVED_NAME,
            "qwen_max_model_len": VLLM_MAX_MODEL_LEN,
            "qwen_max_output_tokens": TURN_MAX_OUTPUT_TOKENS,
            "qwen_temperature": 0,
            "qwen_system_prompt": TURN_SYSTEM_PROMPT,
            "gpu_memory_utilization": VLLM_UTILIZATION,
            "env_overrides": VLLM_ENV_OVERRIDES,
            "kokoro_repo": KOKORO_REPO,
            "kokoro_revision": KOKORO_REVISION,
            "kokoro_voice": KOKORO_VOICE,
            "kokoro_sample_rate": KOKORO_SAMPLE_RATE,
            "vllm_command": _vllm_command(VLLM_UTILIZATION),
        },
    }

    _require(FIXTURE_WAV, "The Experiment 1 fixture")
    _require(QWEN_DIR, "The Qwen weights")

    manifest = json.loads(Path(FIXTURE_MANIFEST).read_text())
    observed_sha = _sha256_file(FIXTURE_WAV)
    summary["fixture"] = {
        **manifest,
        "observed_sha256": observed_sha,
        "matches_manifest": observed_sha == manifest["sha256"],
    }
    if observed_sha != manifest["sha256"]:
        raise RuntimeError(
            f"fixture sha256 {observed_sha} does not match the manifest "
            f"{manifest['sha256']}. Regenerating the fixture invalidates comparison "
            f"with earlier runs, so this stops instead of continuing quietly."
        )

    process, logs = None, []
    try:
        # Models first, then the turn. Loading is not part of the measurement.
        record, whisper = _run_stage(
            "whisper_load",
            lambda: WhisperModel(
                WHISPER_DIR, device="cuda", compute_type=WHISPER_COMPUTE_TYPE
            ),
        )
        summary["whisper_load"] = record

        record, kokoro = _run_stage(
            "kokoro_load",
            lambda: KPipeline(
                lang_code=KOKORO_LANG_CODE, repo_id=KOKORO_REPO, device="cuda"
            ),
        )
        summary["kokoro_load"] = record

        process, logs = _spawn_vllm(VLLM_UTILIZATION)
        ready, reason = _http_ready(
            f"http://127.0.0.1:{VLLM_PORT}/v1/models",
            VLLM_READY_TIMEOUT_SECONDS,
            process,
        )
        summary["vllm"] = {"started": ready, "outcome": reason}
        if not ready:
            raise RuntimeError(f"vLLM did not start: {reason}")

        emit("turn_start", "turn", fixture_sha256=observed_sha)

        stage = _stt_stage(whisper, emit)
        summary["transcript"] = stage["transcript"]
        summary["transcript_segments"] = stage["segments"]
        summary["whisper_language"] = stage["language"]
        summary["resample"] = stage["resample"]

        arm = _arm_sequential(kokoro, _turn_payload(stage["transcript"]), emit)
        summary["reply"] = arm["reply"]
        summary["generation"] = arm["generation"]
        summary["tts_chunks"] = arm["tts_chunks"]
        summary["device_peak_gb"] = round(arm["device_peak_bytes"] / 1e9, 3)

        response = arm["response"]
        response_wav = str(Path(EXP1_DIR, "response.wav"))
        sf.write(response_wav, response, KOKORO_SAMPLE_RATE, subtype="PCM_16")
        summary["response_wav"] = response_wav
        summary["response_seconds"] = round(response.shape[0] / KOKORO_SAMPLE_RATE, 3)

        emit("turn_done", "turn")
    finally:
        summary["vllm_exit"] = _stop_vllm(process)
        if logs:
            Path(EXP1_DIR, "vllm_startup.log").write_text("\n".join(logs) + "\n")

    # Spans are read off the trace rather than measured a second time, so the
    # summary and the events cannot disagree.
    offsets = _first_offsets(events)
    summary["timings_ms"] = {
        "stt": _span(offsets, "stt_start", "stt_final"),
        "llm_ttft": _span(offsets, "llm_request_start", "llm_ttft"),
        "llm_total": _span(offsets, "llm_request_start", "llm_done"),
        "tts_first_audio": _span(offsets, "tts_start", "tts_first_audio"),
        "tts_total": _span(offsets, "tts_start", "tts_done"),
        "end_to_end": _span(offsets, "turn_start", "turn_done"),
    }
    summary["events"] = events

    trace_path = Path(EXP1_DIR, "trace.jsonl")
    with trace_path.open("w") as handle:
        handle.write(
            _json(
                {
                    "trace_header": {
                        "session_id": summary["session_id"],
                        "experiment": 1,
                        "clock": summary["clock"],
                        "event_order": [event["event"] for event in events],
                    }
                }
            )
            + "\n"
        )
        for event in events:
            handle.write(json.dumps(event) + "\n")

    # The input is copied in beside the output so one directory holds the whole
    # turn: what went in, what came out, and the trace that connects them.
    Path(EXP1_DIR, "input.wav").write_bytes(Path(FIXTURE_WAV).read_bytes())

    summary["artifacts"] = {
        "input_wav": str(Path(EXP1_DIR, "input.wav")),
        "response_wav": response_wav,
        "trace_jsonl": str(trace_path),
        "vllm_log": str(Path(EXP1_DIR, "vllm_startup.log")),
    }
    summary["written_to"] = _write_report(summary, "exp1/sequential.json")

    print(f"[exp1] transcript: {summary.get('transcript')!r}", flush=True)
    print(f"[exp1] reply: {summary.get('reply')!r}", flush=True)
    print(f"[exp1] timings_ms: {json.dumps(summary['timings_ms'])}", flush=True)
    print(f"[exp1] response seconds: {summary.get('response_seconds')}", flush=True)
    commit()
    return _json(summary)


def _normalise_for_comparison(text: str) -> str:
    """Lower-case words only, so two transcripts can be compared fairly.

    Punctuation and capitalisation differ between what a model writes and what a
    second model hears back, and neither difference means the audio was wrong.
    """
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def overlap_turn(commit=_noop) -> str:
    """Experiment 2 -- the same turn, with one policy changed.

    Experiment 1 waited for the whole response before synthesising. This one
    synthesises each completed sentence while the model is still generating the
    next, so Qwen and Kokoro share the A10 for part of the turn.

    That sharing is the experiment, not noise to be suppressed. If Kokoro slows
    Qwen's decoding, or Qwen slows Kokoro, it shows up as shifted token arrivals
    and lengthened clause times, and those are reported rather than smoothed.

    Everything except the scheduling policy is identical to Experiment 1: same
    fixture, same Whisper, same transcript, same prompt and generation
    parameters, same Kokoro voice, same GPU, same container.

    The stream is read on its own thread. That is not incidental. If synthesis
    ran on the reading thread, this process would stop reading while it
    synthesised, tokens would pile up in the socket, and llm_done would report
    when the consumer got around to them rather than when the model finished.
    The overlap would look real and be an artifact of a slow reader. One thread
    reads, the other synthesises, and the trace can tell the difference.
    """
    import json
    import time
    import traceback
    from pathlib import Path

    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    from kokoro import KPipeline

    Path(EXP2_DIR).mkdir(parents=True, exist_ok=True)

    clock = time.perf_counter_ns
    started_ns = clock()
    events = []

    def emit(name: str, stage: str, **meta) -> None:
        offset = clock() - started_ns
        events.append(
            {
                "event": name,
                "stage": stage,
                "t_ns": offset,
                "t_ms": round(offset / 1e6, 3),
                "meta": meta,
            }
        )

    summary = {
        "operation": "overlap_turn",
        "experiment": 2,
        "ok": True,
        "session_id": f"exp2-{int(time.time())}",
        "clock": (
            "perf_counter_ns in one process. t_ms is an offset from the start of the "
            "operation; turn_start is an event inside the trace and marks the start of "
            "the measured turn, which is the origin every reported span uses."
        ),
        "policy": (
            "synthesis begins at each completed sentence rather than after the whole "
            "response; identical to Experiment 1 in every other respect"
        ),
        "environment": _gpu_query(),
        "packages": _package_versions(),
        "config": {
            "fixture_wav": FIXTURE_WAV,
            "fixture_text": FIXTURE_TEXT,
            "whisper_dir": WHISPER_DIR,
            "whisper_compute_type": WHISPER_COMPUTE_TYPE,
            "whisper_input_rate": WHISPER_SAMPLE_RATE,
            "qwen_repo": QWEN_REPO,
            "qwen_revision": QWEN_REVISION,
            "qwen_served_name": VLLM_SERVED_NAME,
            "qwen_max_model_len": VLLM_MAX_MODEL_LEN,
            "qwen_max_output_tokens": TURN_MAX_OUTPUT_TOKENS,
            "qwen_temperature": 0,
            "qwen_system_prompt": TURN_SYSTEM_PROMPT,
            "gpu_memory_utilization": VLLM_UTILIZATION,
            "env_overrides": VLLM_ENV_OVERRIDES,
            "kokoro_repo": KOKORO_REPO,
            "kokoro_revision": KOKORO_REVISION,
            "kokoro_voice": KOKORO_VOICE,
            "kokoro_sample_rate": KOKORO_SAMPLE_RATE,
            "sentence_rule": SENTENCE_END.pattern,
            "vllm_command": _vllm_command(VLLM_UTILIZATION),
        },
    }

    _require(FIXTURE_WAV, "The Experiment 1 fixture")
    _require(QWEN_DIR, "The Qwen weights")

    manifest = json.loads(Path(FIXTURE_MANIFEST).read_text())
    observed_sha = _sha256_file(FIXTURE_WAV)
    summary["fixture"] = {
        **manifest,
        "observed_sha256": observed_sha,
        "matches_manifest": observed_sha == manifest["sha256"],
    }
    if observed_sha != manifest["sha256"]:
        raise RuntimeError(
            f"fixture sha256 {observed_sha} does not match the manifest "
            f"{manifest['sha256']}. Regenerating the fixture invalidates comparison "
            f"with Experiment 1, so this stops instead of continuing quietly."
        )

    process, logs = None, []
    try:
        record, whisper = _run_stage(
            "whisper_load",
            lambda: WhisperModel(
                WHISPER_DIR, device="cuda", compute_type=WHISPER_COMPUTE_TYPE
            ),
        )
        summary["whisper_load"] = record

        record, kokoro = _run_stage(
            "kokoro_load",
            lambda: KPipeline(
                lang_code=KOKORO_LANG_CODE, repo_id=KOKORO_REPO, device="cuda"
            ),
        )
        summary["kokoro_load"] = record

        process, logs = _spawn_vllm(VLLM_UTILIZATION)
        ready, reason = _http_ready(
            f"http://127.0.0.1:{VLLM_PORT}/v1/models",
            VLLM_READY_TIMEOUT_SECONDS,
            process,
        )
        summary["vllm"] = {"started": ready, "outcome": reason}
        if not ready:
            raise RuntimeError(f"vLLM did not start: {reason}")

        emit("turn_start", "turn", fixture_sha256=observed_sha)

        stage = _stt_stage(whisper, emit)
        summary["transcript"] = stage["transcript"]
        summary["transcript_segments"] = stage["segments"]
        summary["whisper_language"] = stage["language"]
        summary["resample"] = stage["resample"]

        arm = _arm_overlapped(
            kokoro,
            _turn_payload(stage["transcript"]),
            emit,
            lambda: round((clock() - started_ns) / 1e6, 3),
        )
        clauses = arm["clauses"]
        response = arm["response"]
        response_wav = str(Path(EXP2_DIR, "response.wav"))
        sf.write(response_wav, response, KOKORO_SAMPLE_RATE, subtype="PCM_16")
        for clause, clause_audio in zip(clauses, arm["clause_audios"]):
            sf.write(
                str(Path(EXP2_DIR, f"clause-{clause['clause_id']:03d}.wav")),
                clause_audio,
                KOKORO_SAMPLE_RATE,
                subtype="PCM_16",
            )

        summary["reply"] = arm["reply"]
        summary["clauses"] = clauses
        summary["token_timeline"] = arm["token_timeline"]
        summary["device_peak_gb"] = round(arm["device_peak_bytes"] / 1e9, 3)
        summary["device_peak_samples"] = arm["device_peak_samples"]
        summary["response_wav"] = response_wav
        summary["response_seconds"] = round(response.shape[0] / KOKORO_SAMPLE_RATE, 3)
        summary["clause_seconds_sum"] = round(
            sum(clause["seconds"] for clause in clauses), 3
        )
        summary["response_peak_amplitude"] = (
            round(float(np.max(np.abs(response))), 4) if response.size else None
        )
        emit("turn_done", "turn")

        # --- verification, after the turn and outside every measured span --
        # The objective check available for the question the ear would answer:
        # does the concatenated audio still say what the model wrote? If a clause
        # boundary dropped a word or doubled one, this is where it shows.
        roundtrip_samples, roundtrip_resample = _to_whisper_rate(response_wav)
        roundtrip_segments, _ = whisper.transcribe(
            roundtrip_samples, language="en", beam_size=5
        )
        roundtrip = "".join(segment.text for segment in roundtrip_segments).strip()
        summary["roundtrip"] = {
            "transcript": roundtrip,
            "resample": roundtrip_resample,
            "matches_reply_normally": _normalise_for_comparison(roundtrip)
            == _normalise_for_comparison(summary.get("reply", "")),
            "note": (
                "Post-turn verification. Not part of any measured span, and not an ASR "
                "quality judgement: it only checks that concatenating clauses did not "
                "drop, duplicate or reorder words."
            ),
        }
    except Exception:  # noqa: BLE001 - reported, not swallowed
        summary["ok"] = False
        summary["error"] = traceback.format_exc()
    finally:
        summary["vllm_exit"] = _stop_vllm(process)
        if logs:
            Path(EXP2_DIR, "vllm_startup.log").write_text("\n".join(logs) + "\n")

    offsets = _first_offsets(events)
    summary["timings_ms"] = {
        "stt": _span(offsets, "stt_start", "stt_final"),
        "llm_ttft": _span(offsets, "llm_request_start", "llm_ttft"),
        "llm_total": _span(offsets, "llm_request_start", "llm_done"),
        "first_audio_ready": _span(offsets, "turn_start", "tts_first_audio"),
        "first_audio_sent": _span(offsets, "turn_start", "first_audio_sent"),
        "end_to_end": _span(offsets, "turn_start", "turn_done"),
    }

    # The sentence timeline is the evidence for overlap: when each sentence
    # became ready, when synthesis was allowed to start on it, and how long it
    # then waited for the synthesiser to be free.
    ready, starts, firsts, dones = {}, {}, {}, {}
    for event in events:
        clause_id = event["meta"].get("clause_id")
        if clause_id is None:
            continue
        if event["event"] == "llm_sentence_ready":
            ready.setdefault(clause_id, event["t_ms"])
        elif event["event"] == "tts_clause_start":
            starts.setdefault(clause_id, event["t_ms"])
        elif event["event"] == "tts_first_audio":
            firsts.setdefault(clause_id, event["t_ms"])
        elif event["event"] == "tts_clause_done":
            dones.setdefault(clause_id, event["t_ms"])

    # The clause timeline is expressed relative to turn_start, the same origin
    # as timings_ms. Reporting it against process start made two sets of numbers
    # that describe the same events look unrelated.
    turn_offset = offsets.get("turn_start")

    def relative(value):
        if value is None or turn_offset is None:
            return value
        return round(value - turn_offset, 3)

    summary["clause_timeline"] = [
        {
            "clause_id": clause_id,
            "sentence_ready_ms": relative(ready.get(clause_id)),
            "tts_start_ms": relative(starts.get(clause_id)),
            "tts_first_audio_ms": relative(firsts.get(clause_id)),
            "tts_done_ms": relative(dones.get(clause_id)),
            "waited_for_synthesiser_ms": (
                round(starts[clause_id] - ready[clause_id], 3)
                if clause_id in starts and clause_id in ready
                else None
            ),
            "tts_ms": (
                round(dones[clause_id] - starts[clause_id], 3)
                if clause_id in starts and clause_id in dones
                else None
            ),
        }
        for clause_id in sorted(set(ready) | set(starts))
    ]
    summary["clause_timeline_note"] = (
        "All offsets are relative to turn_start, like timings_ms. The first run "
        "reported these against process start instead, which made them look "
        "unrelated to the spans beside them."
    )

    summary["events"] = events

    # --- the comparison, read from Experiment 1's own trace ---------------
    exp1_path = Path(OUTPUT_DIR, "exp1/sequential.json")
    baseline = None
    if exp1_path.exists():
        exp1 = json.loads(exp1_path.read_text())
        exp1_offsets = _first_offsets(exp1["events"])
        baseline = {
            "source": str(exp1_path),
            "container": "a previous run in a different container",
            "time_to_first_audio_ms": _span(exp1_offsets, "turn_start", "tts_first_audio"),
            "llm_ttft_ms": exp1["timings_ms"]["llm_ttft"],
            "llm_total_ms": exp1["timings_ms"]["llm_total"],
            "end_to_end_ms": exp1["timings_ms"]["end_to_end"],
            "output_seconds": exp1.get("response_seconds"),
        }
    summary["sequential_baseline"] = baseline

    overlapped = summary["timings_ms"]["first_audio_ready"]
    if baseline and baseline["time_to_first_audio_ms"] and overlapped:
        improvement = baseline["time_to_first_audio_ms"] - overlapped
        summary["comparison"] = {
            "metric": "server-side time to first audio: turn_start to first audio available",
            "sequential_ms": baseline["time_to_first_audio_ms"],
            "overlapped_ms": overlapped,
            "absolute_improvement_ms": round(improvement, 3),
            "percentage_improvement": round(
                100 * improvement / baseline["time_to_first_audio_ms"], 1
            ),
            "end_to_end_sequential_ms": baseline["end_to_end_ms"],
            "end_to_end_overlapped_ms": summary["timings_ms"]["end_to_end"],
            "caveat": (
                "the baseline ran in a different container, so treat the deltas as an "
                "indication rather than a controlled A/B"
            ),
        }

    trace_path = Path(EXP2_DIR, "trace.jsonl")
    with trace_path.open("w") as handle:
        handle.write(
            _json(
                {
                    "trace_header": {
                        "session_id": summary["session_id"],
                        "experiment": 2,
                        "clock": summary["clock"],
                        "policy": summary["policy"],
                        "event_order": [event["event"] for event in events],
                    }
                }
            )
            + "\n"
        )
        for event in events:
            handle.write(json.dumps(event) + "\n")

    Path(EXP2_DIR, "input.wav").write_bytes(Path(FIXTURE_WAV).read_bytes())
    summary["artifacts"] = {
        "input_wav": str(Path(EXP2_DIR, "input.wav")),
        "response_wav": response_wav,
        "clause_wavs": [
            str(Path(EXP2_DIR, f"clause-{clause['clause_id']:03d}.wav"))
            for clause in summary.get("clauses", [])
        ],
        "trace_jsonl": str(trace_path),
        "vllm_log": str(Path(EXP2_DIR, "vllm_startup.log")),
    }
    summary["written_to"] = _write_report(summary, "exp2/overlap.json")

    print(f"[exp2] ok={summary['ok']} transcript: {summary.get('transcript')!r}", flush=True)
    print(f"[exp2] timings_ms: {json.dumps(summary['timings_ms'])}", flush=True)
    print(f"[exp2] clause_timeline: {json.dumps(summary['clause_timeline'])}", flush=True)
    print(f"[exp2] comparison: {json.dumps(summary.get('comparison'))}", flush=True)
    if summary.get("roundtrip"):
        print(
            f"[exp2] roundtrip matches: {summary['roundtrip']['matches_reply_normally']}"
            f" | {summary['roundtrip']['transcript']!r}",
            flush=True,
        )
    commit()
    return _json(summary)


def _spread(values: list) -> dict:
    """Min, max and their ratio, so comparability can be judged at a glance."""
    clean = [value for value in values if value is not None]
    if not clean:
        return {"count": 0}
    return {
        "count": len(clean),
        "min_ms": round(min(clean), 3),
        "max_ms": round(max(clean), 3),
        "max_over_min": round(max(clean) / min(clean), 3) if min(clean) else None,
    }


def paired_turns(commit=_noop) -> str:
    """Both policies in one container, alternating, for a valid comparison.

    The first two experiments ran in separate containers, and the difference
    between those containers -- visible on work the scheduling policy cannot
    touch, such as transcription time -- was larger than the effect being
    measured. Here the models are loaded once, both models' first-call costs are
    paid before any measured turn, and the two policies alternate in one
    session. The policy is then the only thing that differs between the arms.

    Each turn gets its own clock, so every turn's events start at zero. The
    report carries the per-turn numbers, the per-pair deltas, and the spread of
    the policy-independent metrics -- which is what says whether the comparison
    is worth reading at all.
    """
    import json
    import time
    import traceback
    from pathlib import Path

    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    from kokoro import KPipeline

    Path(PAIRED_DIR).mkdir(parents=True, exist_ok=True)

    summary = {
        "operation": "paired_turns",
        "experiment": 3,
        "ok": True,
        "session_id": f"paired-{int(time.time())}",
        "question": "does letting synthesis begin at each sentence move time to first audio?",
        "policy_under_test": "when synthesis is allowed to begin",
        "clock": (
            "perf_counter_ns. Each turn has its own clock, so a turn's events start at "
            "zero and every reported span is a difference inside that turn."
        ),
        "pairs_per_policy": PAIRED_PAIRS,
        "warmup_note": (
            "Model loading and both first-call costs are paid before any measured turn "
            "and reported separately, because Kokoro's first synthesis otherwise "
            "dominates time to first audio."
        ),
        "environment": _gpu_query(),
        "packages": _package_versions(),
        "config": {
            "fixture_wav": FIXTURE_WAV,
            "fixture_text": FIXTURE_TEXT,
            "whisper_dir": WHISPER_DIR,
            "whisper_compute_type": WHISPER_COMPUTE_TYPE,
            "whisper_input_rate": WHISPER_SAMPLE_RATE,
            "qwen_repo": QWEN_REPO,
            "qwen_revision": QWEN_REVISION,
            "qwen_served_name": VLLM_SERVED_NAME,
            "qwen_max_model_len": VLLM_MAX_MODEL_LEN,
            "qwen_max_output_tokens": TURN_MAX_OUTPUT_TOKENS,
            "qwen_temperature": 0,
            "qwen_system_prompt": TURN_SYSTEM_PROMPT,
            "gpu_memory_utilization": VLLM_UTILIZATION,
            "env_overrides": VLLM_ENV_OVERRIDES,
            "kokoro_repo": KOKORO_REPO,
            "kokoro_revision": KOKORO_REVISION,
            "kokoro_voice": KOKORO_VOICE,
            "kokoro_sample_rate": KOKORO_SAMPLE_RATE,
            "sentence_rule": SENTENCE_END.pattern,
            "vllm_command": _vllm_command(VLLM_UTILIZATION),
        },
    }

    _require(FIXTURE_WAV, "The Experiment 1 fixture")
    _require(QWEN_DIR, "The Qwen weights")

    manifest = json.loads(Path(FIXTURE_MANIFEST).read_text())
    observed_sha = _sha256_file(FIXTURE_WAV)
    summary["fixture"] = {
        **manifest,
        "observed_sha256": observed_sha,
        "matches_manifest": observed_sha == manifest["sha256"],
    }
    if observed_sha != manifest["sha256"]:
        raise RuntimeError(
            f"fixture sha256 {observed_sha} does not match the manifest "
            f"{manifest['sha256']}. A changed fixture invalidates the comparison, so "
            f"this stops instead of continuing quietly."
        )

    def _run_turn(policy: str, index: int, whisper, kokoro) -> dict:
        """One measured turn on its own clock, using the shared arm."""
        turn_start_ns = time.perf_counter_ns()
        events = []

        def emit(name: str, stage: str, **meta) -> None:
            offset = time.perf_counter_ns() - turn_start_ns
            events.append(
                {
                    "event": name,
                    "stage": stage,
                    "t_ns": offset,
                    "t_ms": round(offset / 1e6, 3),
                    "meta": meta,
                }
            )

        def now_ms() -> float:
            return round((time.perf_counter_ns() - turn_start_ns) / 1e6, 3)

        emit("turn_start", "turn", policy=policy, fixture_sha256=observed_sha)
        stage = _stt_stage(whisper, emit)
        payload = _turn_payload(stage["transcript"])
        arm = (
            _arm_sequential(kokoro, payload, emit)
            if policy == "sequential"
            else _arm_overlapped(kokoro, payload, emit, now_ms)
        )
        emit("turn_done", "turn")

        offsets = _first_offsets(events)
        return {
            "policy": policy,
            "turn_index": index,
            "transcript": stage["transcript"],
            "reply": arm["reply"],
            "response_seconds": round(
                arm["response"].shape[0] / KOKORO_SAMPLE_RATE, 3
            ),
            "device_peak_gb": round(arm["device_peak_bytes"] / 1e9, 3),
            "device_peak_samples": arm["device_peak_samples"],
            "timings_ms": {
                "stt": _span(offsets, "stt_start", "stt_final"),
                "llm_ttft": _span(offsets, "llm_request_start", "llm_ttft"),
                "llm_total": _span(offsets, "llm_request_start", "llm_done"),
                "time_to_first_audio": _span(offsets, "turn_start", "tts_first_audio"),
                "first_audio_sent": _span(offsets, "turn_start", "first_audio_sent"),
                "end_to_end": _span(offsets, "turn_start", "turn_done"),
            },
            "events": events,
            "audio": arm["response"],
        }

    process, logs = None, []
    turns = []
    try:
        record, whisper = _run_stage(
            "whisper_load",
            lambda: WhisperModel(
                WHISPER_DIR, device="cuda", compute_type=WHISPER_COMPUTE_TYPE
            ),
        )
        summary["whisper_load"] = record

        record, kokoro = _run_stage(
            "kokoro_load",
            lambda: KPipeline(
                lang_code=KOKORO_LANG_CODE, repo_id=KOKORO_REPO, device="cuda"
            ),
        )
        summary["kokoro_load"] = record

        process, logs = _spawn_vllm(VLLM_UTILIZATION)
        ready, reason = _http_ready(
            f"http://127.0.0.1:{VLLM_PORT}/v1/models",
            VLLM_READY_TIMEOUT_SECONDS,
            process,
        )
        summary["vllm"] = {"started": ready, "outcome": reason}
        if not ready:
            raise RuntimeError(f"vLLM did not start: {reason}")

        summary["warmup"] = _warm_up(whisper, kokoro)

        for pair_index in range(PAIRED_PAIRS):
            for policy in ("sequential", "overlapped"):
                result = _run_turn(policy, len(turns), whisper, kokoro)
                audio = result.pop("audio")
                wav = str(
                    Path(PAIRED_DIR, f"turn-{result['turn_index']:02d}-{policy}.wav")
                )
                sf.write(wav, audio, KOKORO_SAMPLE_RATE, subtype="PCM_16")
                result["wav"] = wav
                turns.append(result)
                print(
                    f"[paired] turn {result['turn_index']} {policy}:"
                    f" {json.dumps(result['timings_ms'])}",
                    flush=True,
                )
                summary["turns"] = turns
                summary["written_to"] = _write_report(summary, "paired/paired.json")
                commit()
    except Exception:  # noqa: BLE001 - reported, not swallowed
        summary["ok"] = False
        summary["error"] = traceback.format_exc()
    finally:
        summary["vllm_exit"] = _stop_vllm(process)
        if logs:
            Path(PAIRED_DIR, "vllm_startup.log").write_text("\n".join(logs) + "\n")

    # --- the comparison ---------------------------------------------------
    def median(values):
        ordered = sorted(value for value in values if value is not None)
        if not ordered:
            return None
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return round(ordered[mid], 3)
        return round((ordered[mid - 1] + ordered[mid]) / 2, 3)

    by_policy = {
        policy: [turn for turn in turns if turn["policy"] == policy]
        for policy in ("sequential", "overlapped")
    }
    summary["medians_ms"] = {
        policy: {
            key: median([turn["timings_ms"][key] for turn in group])
            for key in (
                "stt",
                "llm_ttft",
                "llm_total",
                "time_to_first_audio",
                "end_to_end",
            )
        }
        for policy, group in by_policy.items()
    }

    pairs = []
    for pair_index in range(PAIRED_PAIRS):
        group = [
            turn for turn in turns if turn["turn_index"] in (pair_index * 2, pair_index * 2 + 1)
        ]
        if len(group) != 2:
            continue
        sequential, overlapped = group
        pairs.append(
            {
                "pair": pair_index,
                "sequential_time_to_first_audio_ms": sequential["timings_ms"][
                    "time_to_first_audio"
                ],
                "overlapped_time_to_first_audio_ms": overlapped["timings_ms"][
                    "time_to_first_audio"
                ],
                "delta_ms": round(
                    sequential["timings_ms"]["time_to_first_audio"]
                    - overlapped["timings_ms"]["time_to_first_audio"],
                    3,
                ),
                "sequential_end_to_end_ms": sequential["timings_ms"]["end_to_end"],
                "overlapped_end_to_end_ms": overlapped["timings_ms"]["end_to_end"],
                "sequential_llm_total_ms": sequential["timings_ms"]["llm_total"],
                "overlapped_llm_total_ms": overlapped["timings_ms"]["llm_total"],
            }
        )
    summary["pairs"] = pairs

    deltas = [pair["delta_ms"] for pair in pairs]
    summary["result"] = {
        "metric": "server-side time to first audio, turn_start to first audio available",
        "per_pair_delta_ms": deltas,
        "median_delta_ms": median(deltas),
        "sign_convention": "positive means the overlapped policy produced audio earlier",
        "note": (
            "Time to first audio and total completion answer different questions, and "
            "neither policy may be called faster without saying which one is meant."
        ),
    }

    summary["validity"] = {
        "policy_independent_spread": {
            "stt_ms": _spread([turn["timings_ms"]["stt"] for turn in turns]),
            "llm_ttft_ms": _spread([turn["timings_ms"]["llm_ttft"] for turn in turns]),
        },
        "why_this_matters": (
            "Transcription and time to first token cannot be affected by the scheduling "
            "policy. Their spread across the turns bounds how much of any difference in "
            "time to first audio can reasonably be attributed to the policy."
        ),
    }

    trace_path = Path(PAIRED_DIR, "trace.jsonl")
    with trace_path.open("w") as handle:
        handle.write(
            _json(
                {
                    "trace_header": {
                        "session_id": summary["session_id"],
                        "experiment": 3,
                        "clock": summary["clock"],
                        "turn_order": [
                            f"{turn['turn_index']}:{turn['policy']}" for turn in turns
                        ],
                    }
                }
            )
            + "\n"
        )
        for turn in turns:
            for event in turn["events"]:
                handle.write(
                    json.dumps({"turn": turn["turn_index"], "policy": turn["policy"], **event})
                    + "\n"
                )

    summary["artifacts"] = {
        "turn_wavs": [turn.get("wav") for turn in turns],
        "trace_jsonl": str(trace_path),
        "vllm_log": str(Path(PAIRED_DIR, "vllm_startup.log")),
    }
    summary["written_to"] = _write_report(summary, "paired/paired.json")

    print(f"[paired] ok={summary['ok']}", flush=True)
    print(f"[paired] medians_ms: {json.dumps(summary['medians_ms'])}", flush=True)
    print(f"[paired] pairs: {json.dumps(pairs)}", flush=True)
    print(f"[paired] result: {json.dumps(summary['result'])}", flush=True)
    print(f"[paired] validity: {json.dumps(summary['validity'])}", flush=True)
    commit()
    return _json(summary)
