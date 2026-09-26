"""voice-inference-lab -- the experiment logic for the Qwen Omni phase.

The third architecture family, and the third answer to the same question.

Moshi is one continuous full-duplex stream inside one model. The modular trio is
three independent models in three containers. Qwen3-Omni is one model that the
serving stack runs as a **three-stage pipeline** -- Thinker, Talker, Code2Wav --
so an "end-to-end" speech model turns out to be a multi-stage inference system
internally, and where those stages are placed is a deployment decision rather
than a property of the model.

This module has no provider imports. Provider lifecycle stays outside it: the
one operation only a provider can define, making writes durable, arrives as a
`commit` callable and defaults to a no-op.

Two operations:

    cache_model     download the pinned checkpoint, no GPU needed
    omni_turn       serve it with vLLM-Omni and run one audio-to-audio turn

The runtime is vLLM-Omni, which is the officially documented path for audio
output. Upstream vLLM's own serve path for Qwen3-Omni supports only the thinker,
which returns text and no speech.
"""

import os

DATA_ROOT = os.environ.get("VOICE_LAB_ROOT", "/cache")
HF_HOME = f"{DATA_ROOT}/hf"
MODELS_DIR = f"{DATA_ROOT}/models"
OMNI_DIR = f"{DATA_ROOT}/outputs/omni"
OMNI_TRACE = f"{OMNI_DIR}/omni_turn.json"
OMNI_AUDIO = f"{OMNI_DIR}/response.wav"
OMNI_SERVER_LOG = f"{OMNI_DIR}/vllm_omni_server.log"
OMNI_DEPLOY_CONFIG_COPY = f"{OMNI_DIR}/qwen3_omni_moe.deploy.yaml"

PYTHON_VERSION = "3.12"

# ---------------------------------------------------------------------------
# Pins.
#
# The checkpoint is pinned by *recorded resolution* rather than by a requested
# revision, because the loaders hear accept neither a revision nor a local path
# in a way we can enforce. Offline mode is what makes the cache immutable
# afterwards: with no network, resolution can only return what was fetched, so a
# moved upstream cannot leak in. The resolved commit is recorded in the report
# and checked on every run, which is the same guarantee the Kokoro weights get
# and for the same reason.
#
# vLLM and vLLM-Omni must share a major *and* minor version. vLLM owns the CLI
# entrypoint and delegates `--omni` to vLLM-Omni, so a mismatch fails in a way
# that points nowhere useful. That pairing is the reason this image has its own
# version constant rather than reusing the modular stack's.
# ---------------------------------------------------------------------------
OMNI_MODEL_REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
OMNI_MODEL_DIR = f"{MODELS_DIR}/Qwen3-Omni-30B-A3B-Instruct"
OMNI_SERVED_NAME = OMNI_MODEL_REPO

# The real reason this does not serve on vLLM-Omni 0.28.0, established by
# research rather than by inference from our own failures.
#
# vLLM-Omni 0.30.0's release notes describe parallel stage initialization with
# device-aware admission and locking. The RFC behind that work states the old
# path used **process-scoped NVML memory estimation** for LLM stages, which let
# colocated stages profile and allocate against the whole device rather than
# against their own share. The replacement sums stage budgets, graph reserves and
# safety margin and admits them against the physical GPU before launching, with
# per-device locks around profiling.
#
# That fits every measurement we took:

#   - "Available KV cache memory: 125.82 GiB (process-scoped)" on a 141 GB card
#     whose Process 1 already held 139.04 GiB, then a 2.62 GiB allocation OOM.
#   - 80 GB and 141 GB failing with byte-identical 458-line logs. A capacity
#     problem cannot produce that; a per-process accounting problem can.
#   - v0.28.0 has a separate, reproduced bug where shared-GPU stages hit a
#     spawn-lock/device-lock inversion during initialization even when the
#     budget sums are valid.
#
# So 0.28.0 was the problem, not the deploy config. The config's fractions are
# per *device*, not per stage: 0.9 on GPU 0 and 0.6 + 0.1 = 0.7 on GPU 1, which
# is comfortably under 1.0 on each. They only overcommit if every stage is forced
# onto one card.

# Switched back to the 30B. The 3B was tried on the belief that the constraint
# was the model's size, which the research above shows it was not: the 3B failed
# the same way on one A10, so the size was never the variable.

# The pairing, and why it is not simply "the newest of each".
#
# vLLM-Omni publishes stable releases on every EVEN-numbered upstream vLLM
# minor. An earlier pin used 0.28.0, on the reading that only a release candidate
# existed for 0.30; v0.30.0 has since shipped, and its device-aware stage
# admission is the fix, so both sides now sit at 0.30.0.
#
# Given the documented rule that the two must share a major and a minor, the
# The versions the install docs currently target, and the fix for the failure
# above. Both sides move together: vLLM owns the CLI and delegates `--omni`, and
# the two must share a major and a minor.
#
# This replaces an earlier pin to 0.28.0 on the reasoning that no stable 0.30
# existed. That is no longer true -- v0.30.0 shipped -- and the device-aware
# stage admission it adds is precisely what was missing.
VLLM_VERSION = "0.30.0"
VLLM_OMNI_REF = "v0.30.0"
VLLM_OMNI_REPO = "https://github.com/vllm-project/vllm-omni.git"

# The CUDA backend is named, not auto-detected, and the reason is the opposite
# of the modular stack's.
#
# vLLM-Omni's quickstart says `--torch-backend=auto`, which works when uv runs on
# the machine that will use the GPU. A Modal image build runs on a machine with
# no GPU at all, so uv's driver inspection finds nothing and falls back to the
# CPU-only index -- and the container then has a torch that cannot see CUDA.
# Naming the backend is the only way to get a GPU build out of a GPU-less build
# machine.
#
# CUDA 13.0, established by measurement rather than by reading.
#
# The first attempt named cu129, on the reasoning that vLLM's documentation says
# its binaries are compiled with CUDA 12.9 and that 0.28 predates the change
# which made the published wheel CUDA 13. That reasoning was wrong. With torch on
# cu129 the server died in seven seconds:
#
#     File "vllm/platforms/cuda.py", line 23, in <module>
#       import vllm._C_stable_libtorch
#     ImportError: libcudart.so.13: cannot open shared object file
#
# So the 0.28 wheel is built for CUDA 13.0 as well, and the documentation's prose
# about the default is stale for this version too. The error is what to trust.
VLLM_TORCH_BACKEND = "cu130"

# FlashInfer compiles its sampling kernels at first use with ninja and nvcc.
# This image carries the CUDA runtime libraries but no compiler, so the engine
# core for stage 0 died during warmup on the chain
#
#     flashinfer/sampling.py -> jit/cpp_ext.py get_cuda_path()
#       RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
#
# The modular stack hit this identically and was fixed the same way; the setting
# simply had to be carried across. It also removes a just-in-time compile from
# startup, which matters because startup here is already minutes rather than
# seconds.
OMNI_ENV_OVERRIDES = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    # The KV cache is allocated as one very large torch.zeros, which can fail
    # even when the device has room: the allocator cannot always find a
    # contiguous block that size. Measured on an H200, stage 0 reported
    # "Available KV cache memory: 125.82 GiB" and "GPU KV cache size: 1,374,288
    # tokens", then died in _allocate_kv_cache_tensors. The out-of-memory message
    # named this setting, and it was mistaken for a capacity problem first
    # because only the tail of that message was read.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

OMNI_PORT = 8091

# The stage layout is defined in deploy configs that vLLM-Omni loads
# automatically. Reading the installed directory is evidence of what the server
# actually had available, where a filename restated here would only be an
# assertion that happens to match.
OMNI_DEPLOY_DIR_RELATIVE = "deploy"

# Qwen3-Omni-30B-A3B is a 30B MoE. Qwen's own model card states a BF16 minimum
# of 78.85 GB for this checkpoint with a 15 second video input; audio-only is
# lighter, but the weights alone are around 60 GB and the talker adds roughly
# 10 GB more. That is why this phase runs on an 80 GB card and why the fit is
# worth reporting rather than assuming.
OMNI_MIN_VRAM_GB = 78.85

OMNI_FIXTURE = f"{DATA_ROOT}/fixtures/exp1_prompt.wav"
OMNI_PROMPT = "Answer in one short sentence."

# A cold start has to load roughly 60 GB of weights and bring up three stages.
OMNI_READY_TIMEOUT_SECONDS = 2400
OMNI_REQUEST_TIMEOUT_SECONDS = 600


def _noop() -> None:
    """Default commit. A plain filesystem needs nothing to make writes durable."""


def _json(report: dict) -> str:
    """Serialise a report as a string, so a driver needs no torch to read it."""
    import json

    return json.dumps(report, indent=2, sort_keys=False)


def _require(path: str, what: str) -> None:
    """Fail loudly when a cached asset is missing."""
    from pathlib import Path

    if not Path(path).exists():
        raise FileNotFoundError(
            f"{what} is not in the cache at {path}. "
            f"Run the cache operation first; it is the only one allowed to use the network."
        )


def _write_report(report: dict, filename: str) -> str:
    """Persist a report to the Volume path, returning where it landed."""
    from pathlib import Path

    path = Path(OMNI_DIR, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(report))
    return str(path)


def _bytes_on_disk(path: str) -> int:
    from pathlib import Path

    target = Path(path)
    if target.is_file():
        return target.stat().st_size
    return sum(f.stat().st_size for f in target.rglob("*") if f.is_file())


def _sha256_file(path: str) -> str:
    import hashlib
    from pathlib import Path

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _device() -> dict:
    """Device memory as the driver reports it."""
    import torch

    free, total = torch.cuda.mem_get_info()
    return {
        "visible_devices": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() else None,
        "total_bytes": int(total),
        "free_bytes": int(free),
        "used_bytes": int(total - free),
        "total_gb": round(total / 1e9, 3),
        "free_gb": round(free / 1e9, 3),
        "used_gb": round((total - free) / 1e9, 3),
    }


def _gpu_query() -> dict:
    """What the platform actually handed us, for the record."""
    import subprocess

    import torch

    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_devices": torch.cuda.device_count(),
        "device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
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


def _package_versions() -> dict:
    """Resolved versions of the libraries this phase actually runs."""
    from importlib.metadata import PackageNotFoundError, version

    names = ("vllm", "vllm-omni", "torch", "transformers", "huggingface-hub", "numpy")
    resolved = {}
    for name in names:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = None
    return resolved


def _deploy_config_text() -> dict:
    """The stage layout vLLM-Omni will actually load.

    Read from the installed package rather than restated here, because the
    deploy config is where stage count and placement are defined. It lists the
    whole deploy directory and returns any file naming this model family, so it
    works for whichever checkpoint is configured rather than only for the one
    this phase started with.
    """
    from pathlib import Path

    result = {"relative_dir": OMNI_DEPLOY_DIR_RELATIVE}
    try:
        import vllm_omni

        directory = Path(vllm_omni.__file__).parent / OMNI_DEPLOY_DIR_RELATIVE
        result["resolved_dir"] = str(directory)
        result["exists"] = directory.exists()
        if not directory.exists():
            return result
        names = sorted(p.name for p in directory.glob("*.yaml"))
        result["available"] = names
        chosen = [n for n in names if "qwen" in n.lower()]
        result["matching_qwen"] = chosen
        result["files"] = {
            name: (directory / name).read_text() for name in chosen[:3]
        }
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _served_model() -> str:
    """The model id the server actually registered, read rather than assumed.

    The request was rejected with "the model ... does not exist", which is the
    server saying our string does not match the id it registered. It advertises
    that id at /v1/models, so asking beats guessing -- and the alternative is a
    404 that a human would read as a route problem when it is a name problem.
    """
    import json
    import urllib.request

    with urllib.request.urlopen(
        f"http://127.0.0.1:{OMNI_PORT}/v1/models", timeout=30
    ) as response:
        payload = json.loads(response.read())
    for model in payload.get("data", []):
        return model.get("id")
    return None


def _routes() -> dict:
    """The route table the server publishes about itself.

    A hardcoded path cannot distinguish "wrong endpoint" from "endpoint
    disabled", and the run that answered /v1/models but returned 404 for
    /v1/chat/completions cost a cycle to exactly that ambiguity. So ask the
    server instead of assuming.
    """
    import json
    import urllib.request

    for path in ("/openapi.json", "/v1/openapi.json", "/docs/openapi.json"):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{OMNI_PORT}{path}", timeout=30
            ) as response:
                spec = json.loads(response.read())
            return {
                route: sorted(str(method).upper() for method in methods)
                for route, methods in (spec.get("paths") or {}).items()
            }
        except Exception:  # noqa: BLE001 - try the next location
            continue
    return {}


def _chat_endpoint(routes: dict):
    """Pick the chat route out of whatever the server advertises.

    The exact path first, then anything that looks like chat completions, then
    anything mentioning chat. Reported either way, so a wrong choice is visible
    in the artifact rather than inferred from a status code.
    """
    posts = [
        route
        for route, methods in routes.items()
        if "POST" in methods
    ]
    for candidate in ("/v1/chat/completions", "/chat/completions"):
        if candidate in posts:
            return candidate
    for route in posts:
        if "chat" in route and "completion" in route:
            return route
    for route in posts:
        if "chat" in route:
            return route
    return None


def _omni_command() -> list:
    """The documented serve command for Qwen3-Omni, plus one requirement.

    `--omni` is mandatory: vLLM owns the CLI and delegates it to vLLM-Omni.
    `--no-async-chunk` is not tuning. The documented `/v1/realtime` path is
    unsupported while async chunking is enabled, and async chunking is on by
    default, so the realtime path requires this flag.
    """
    return [
        "vllm",
        "serve",
        OMNI_MODEL_DIR,
        "--served-model-name",
        OMNI_SERVED_NAME,
        "--omni",
        "--port",
        str(OMNI_PORT),
        "--no-async-chunk",
        "--allowed-local-media-path",
        "/",
    ]


def _spawn(command: list):
    """Start a server and stream its output into a list."""
    import os
    import subprocess
    import threading

    logs = []
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, **OMNI_ENV_OVERRIDES},
    )

    def _drain() -> None:
        for line in process.stdout:
            logs.append(line.rstrip())

    threading.Thread(target=_drain, daemon=True).start()
    return process, logs


def _stop(process) -> dict:
    import subprocess

    if process is None:
        return {"exit_code": None, "stopped_by_us": None}
    stopped_by_us = process.poll() is None
    if stopped_by_us:
        process.terminate()
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=60)
    return {"exit_code": process.returncode, "stopped_by_us": stopped_by_us}


def _http_ready(url: str, timeout: float, process=None) -> tuple:
    """Poll until the server answers, the process dies, or the timeout expires.

    A server that exits will never answer, and waiting out a 40 minute timeout
    for it bills GPU time to learn nothing.
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
            time.sleep(2.0)
    return False, "timed out waiting for the server to answer"


def _stage_banners(log_lines: list) -> list:
    """Lines that mention a stage, as the server itself reports them.

    Marker-based, so the matched lines are returned rather than a verdict: a
    stage that is absent only because the wording changed looks identical to a
    stage that never started.
    """
    keys = ("stage", "thinker", "talker", "code2wav", "code_2_wav", "omni")
    lowered = [line.lower() for line in log_lines]
    return [
        line
        for line, low in zip(log_lines, lowered)
        if any(key in low for key in keys)
    ][:160]


def _extract_audio(payload: dict) -> dict:
    """Find the returned audio in a chat response, without guessing its shape.

    The response format for `modalities: ["text", "audio"]` is not pinned by
    anything we have read, so rather than assume a field name this searches the
    known candidates and, when it finds nothing, reports the response's own key
    structure. A wrong guess would look like a model that produced no audio.
    """
    import base64
    import json

    candidates = (
        ("choices.0.message.audio.data", lambda p: p["choices"][0]["message"]["audio"]["data"]),
        ("choices.0.message.audio", lambda p: p["choices"][0]["message"]["audio"]),
        ("audio.data", lambda p: p["audio"]["data"]),
        ("audio", lambda p: p["audio"]),
        ("choices.0.audio.data", lambda p: p["choices"][0]["audio"]["data"]),
    )
    for name, getter in candidates:
        try:
            value = getter(payload)
        except Exception:  # noqa: BLE001 - absence is the answer
            continue
        if isinstance(value, str) and len(value) > 64:
            try:
                return {"found_at": name, "bytes": base64.b64decode(value)}
            except Exception as exc:  # noqa: BLE001
                return {"found_at": name, "decode_error": f"{type(exc).__name__}: {exc}"}

    def shape(value, depth=0):
        if depth > 3:
            return "..."
        if isinstance(value, dict):
            return {key: shape(item, depth + 1) for key, item in list(value.items())[:12]}
        if isinstance(value, list):
            return [shape(value[0], depth + 1)] if value else []
        if isinstance(value, str):
            return f"str[{len(value)}]"
        return type(value).__name__

    return {"found_at": None, "response_shape": shape(payload), "response_keys": sorted(payload)}


def _text_from(payload: dict):
    """The text the model returned, wherever it put it."""
    try:
        return payload["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001 - absence is an answer
        return None


def cache_model(commit=_noop) -> str:
    """Download the pinned checkpoint. No GPU is required.

    Roughly 60 GB. Acquisition is kept separable from any GPU work for the same
    reason it was in the other two phases: moving bytes needs no accelerator,
    and billing an 80 GB card for a download would be absurd.
    """
    import time
    from pathlib import Path

    from huggingface_hub import snapshot_download

    Path(OMNI_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    existed = Path(OMNI_MODEL_DIR, "config.json").exists()

    started = time.perf_counter()
    snapshot_download(
        OMNI_MODEL_REPO,
        local_dir=OMNI_MODEL_DIR,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
    )
    elapsed = round(time.perf_counter() - started)

    report = {
        "operation": "cache_model",
        "repo": OMNI_MODEL_REPO,
        "path": OMNI_MODEL_DIR,
        "present_before": existed,
        "seconds": elapsed,
        "bytes": _bytes_on_disk(OMNI_MODEL_DIR),
        "gb": round(_bytes_on_disk(OMNI_MODEL_DIR) / 1e9, 2),
        "packages": _package_versions(),
    }
    report["files"] = sorted(p.name for p in Path(OMNI_MODEL_DIR).glob("*.json"))
    report["missing_config"] = not Path(OMNI_MODEL_DIR, "config.json").exists()

    print(
        f"[omni] cached {report['gb']} GB to {OMNI_MODEL_DIR}"
        f" (present before: {existed}, {elapsed}s)",
        flush=True,
    )
    commit()
    return _json(report)


def omni_turn(commit=_noop) -> str:
    """Serve Qwen3-Omni with vLLM-Omni and run one audio-to-audio turn.

    Minimal by design: one fixture in, one spoken response out, saved to disk.
    No browser, no concurrency, no duplex tuning. The point is to see the shape
    of the deployment -- which stages the server launches, where it puts them,
    and what that costs in memory and startup time -- and to get one real
    inference through the officially documented path.
    """
    import json
    import time
    import traceback
    import urllib.error
    import urllib.request
    from pathlib import Path

    Path(OMNI_DIR).mkdir(parents=True, exist_ok=True)

    clock = time.perf_counter
    started = clock()
    events = []

    def emit(name: str, **meta) -> None:
        events.append(
            {"event": name, "t_s": round(clock() - started, 3), "meta": meta}
        )

    report = {
        "operation": "omni_turn",
        "ok": True,
        "session_id": f"omni-{int(time.time())}",
        "clock": "perf_counter within one process, offset from server start",
        "question": (
            "an end-to-end speech model is served as a multi-stage pipeline; what does "
            "that deployment look like next to a single continuous model and an "
            "explicit three-model pipeline?"
        ),
        "environment": _gpu_query(),
        "packages": _package_versions(),
        "config": {
            "model_repo": OMNI_MODEL_REPO,
            "model_dir": OMNI_MODEL_DIR,
            "serve_command": _omni_command(),
            "vllm_version": VLLM_VERSION,
            "vllm_torch_backend": VLLM_TORCH_BACKEND,
            "port": OMNI_PORT,
            "fixture": OMNI_FIXTURE,
            "prompt": OMNI_PROMPT,
            "modalities": ["text", "audio"],
            "async_chunk": False,
            "env_overrides": OMNI_ENV_OVERRIDES,
            "documented_min_vram_gb": OMNI_MIN_VRAM_GB,
        },
        "notes": [
            "Audio output requires vLLM-Omni. Upstream vLLM's serve path for this "
            "checkpoint supports only the thinker and returns no speech.",
            "`--no-async-chunk` is a requirement of the realtime path, not tuning: "
            "/v1/realtime is unsupported while async chunking is enabled, and it is on "
            "by default.",
            "The fixture is passed as raw audio and the model's own processor handles "
            "it. The modular stack resampled explicitly because it had to; this "
            "architecture owns its own frontend, which is part of what is being "
            "compared.",
        ],
    }

    if not report["environment"]["cuda_available"]:
        # Never return quietly. A wrong torch build makes CUDA invisible, and the
        # first version of this returned a report that nothing printed and
        # nothing wrote, which is indistinguishable from the function never
        # having run -- and cost a diagnosis.
        report["ok"] = False
        report["error"] = "CUDA is unavailable; refusing to fall back to CPU."
        report["written_to"] = _write_report(report, "omni_turn.json")
        print(f"[omni] FAILED: {report['error']}", flush=True)
        print(
            f"[omni] torch={report['environment'].get('torch')}"
            f" cuda={report['environment'].get('torch_cuda')}"
            f" devices={report['environment'].get('visible_devices')}",
            flush=True,
        )
        commit()
        return _json(report)

    _require(OMNI_FIXTURE, "The Experiment 1 fixture")
    _require(OMNI_MODEL_DIR, "The Qwen3-Omni checkpoint")

    report["fixture"] = {
        "path": OMNI_FIXTURE,
        "bytes": Path(OMNI_FIXTURE).stat().st_size,
        "sha256": _sha256_file(OMNI_FIXTURE),
    }
    report["deploy_config"] = _deploy_config_text()
    copies = {}
    for name, text in (report["deploy_config"].get("files") or {}).items():
        target = Path(OMNI_DIR, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        copies[name] = str(target)
    report["deploy_config"].pop("files", None)
    report["deploy_config"]["copies"] = copies

    report["device_before"] = _device()

    process, logs = None, []
    try:
        emit("server_start")
        process, logs = _spawn(_omni_command())
        ready, reason = _http_ready(
            f"http://127.0.0.1:{OMNI_PORT}/v1/models",
            OMNI_READY_TIMEOUT_SECONDS,
            process,
        )
        emit("server_ready", ready=ready, outcome=reason)
        report["server"] = {
            "started": ready,
            "outcome": reason,
            "startup_s": round(clock() - started, 3),
        }
        if not ready:
            raise RuntimeError(f"the server did not start: {reason}")

        report["device_after_server"] = _device()

        # --- one turn ------------------------------------------------------
        served_model_name = _served_model()
        report["served_model_name"] = served_model_name
        if served_model_name != OMNI_SERVED_NAME:
            raise RuntimeError(
                f"--served-model-name did not take effect: the server reports "
                f"{served_model_name!r}, expected {OMNI_SERVED_NAME!r}. vLLM "
                f"defaults the API name to the --model argument, which was a local "
                f"path, so the flag exists to force the checkpoint's real id."
            )

        routes = _routes()
        report["routes"] = routes
        endpoint = _chat_endpoint(routes)
        report["request_endpoint"] = endpoint
        report["endpoint_candidates"] = sorted(
            route for route, methods in routes.items() if "POST" in methods
        )
        if endpoint is None:
            raise RuntimeError(
                f"the server advertises no chat route; POST routes were "
                f"{report['endpoint_candidates'][:40]}"
            )

        # One user message holding audio then text, which is the shape the
        # model's own documentation shows. The earlier attempt split them across
        # two messages, which was never exercised because the route 404'd first.
        body = json.dumps(
            {
                "model": OMNI_SERVED_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio_url", "audio_url": {"url": Path(OMNI_FIXTURE).as_uri()}},
                            {"type": "text", "text": OMNI_PROMPT},
                        ],
                    }
                ],
                "modalities": ["text", "audio"],
                "max_tokens": 256,
            }
        ).encode()

        emit("request_start", endpoint=endpoint)
        request_started = clock()
        request = urllib.request.Request(
            f"http://127.0.0.1:{OMNI_PORT}{endpoint}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=OMNI_REQUEST_TIMEOUT_SECONDS
            ) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            # Keep the body. A bare status code is what cost the last run: a 404
            # said nothing about whether the route was wrong or absent.
            detail = exc.read().decode(errors="replace")[:2000]
            report["request_error"] = {
                "status": exc.code,
                "endpoint": endpoint,
                "body": detail,
            }
            raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {detail[:400]}") from exc
        request_seconds = round(clock() - request_started, 3)
        emit("request_complete", seconds=request_seconds)

        report["request"] = {
            "seconds": request_seconds,
            "text": _text_from(payload),
            "usage": payload.get("usage"),
        }

        audio = _extract_audio(payload)
        if audio.get("bytes"):
            Path(OMNI_AUDIO).write_bytes(audio["bytes"])
            report["audio"] = {
                "found_at": audio["found_at"],
                "bytes": len(audio["bytes"]),
                "path": OMNI_AUDIO,
                "sha256": _sha256_file(OMNI_AUDIO),
            }
            emit("audio_written", bytes=len(audio["bytes"]))
        else:
            report["audio"] = audio
            emit("audio_missing")

        report["request_response_keys"] = sorted(payload)
        report["device_after_request"] = _device()
        emit("turn_done")
    except Exception:  # noqa: BLE001 - reported, not swallowed
        report["ok"] = False
        report["error"] = traceback.format_exc()
    finally:
        emit("server_stop")
        report["server_exit"] = _stop(process)
        if logs:
            Path(OMNI_SERVER_LOG).write_text("\n".join(logs) + "\n")
        report["server_log_lines"] = len(logs)
        report["server_log_path"] = OMNI_SERVER_LOG

    report["stage_banners"] = _stage_banners(logs)
    report["events"] = events
    report["written_to"] = _write_report(report, "omni_turn.json")

    print(f"[omni] ok={report['ok']} server_started={report.get('server', {}).get('started')}", flush=True)
    print(f"[omni] startup_s={report.get('server', {}).get('startup_s')}", flush=True)
    print(f"[omni] text={report.get('request', {}).get('text')!r}", flush=True)
    print(f"[omni] audio={ {k: v for k, v in (report.get('audio') or {}).items() if k != 'bytes'} }", flush=True)
    print(f"[omni] devices={report['environment'].get('device_names')} log_lines={len(logs)}", flush=True)
    commit()
    return _json(report)
