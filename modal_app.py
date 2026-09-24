"""voice-inference-lab -- Modal execution for voice model inference.

Two functions:

    modal run modal_app.py::inspect_gpu      # plumbing check
    modal run modal_app.py::cache_weights    # PHASE 1 -- CPU only, no GPU

Phase 1 exists for one reason: moving ~15.8 GB of weights must never happen on
a GPU. A CPU container pays to move the bytes once, into a persistent Volume.
"""

import modal

app = modal.App("voice-inference-lab")

# ---------------------------------------------------------------------------
# Pins. The whole point of this project is that none of these drift.
#
# Weights are pinned to an immutable revision SHA: a cached Volume can never go
# stale, and a later run can never be pulled toward a newer upstream commit.
#
# Code is pinned because it has to be. moshi 0.2.13 declares
# `torch<2.10,>=2.2.0` and its README says it is tested on PyTorch 2.2/2.4.
# Left to float -- as inspect_gpu deliberately does -- torch resolves to
# 2.14+cu130, which is outside that range. So the loader gets a torch version
# Moshi actually claims to support.
# ---------------------------------------------------------------------------
MOSHI_REPO = "kyutai/moshiko-pytorch-bf16"
MOSHI_REVISION = "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"  # 7.69B params, ~15.8 GB

# The Moshi language model, the Mimi codec, and the text tokenizer all ship
# inside MOSHI_REPO. Later phases read those filenames from moshi's own
# constants, so the pairing cannot drift.
#
# Caution: `kyutai/mimi` holds the Hugging Face `transformers` build of Mimi.
# That build uses different `state_dict` keys, and the native moshi loader
# rejects it. Do not point this file at that repo.
MOSHI_PACKAGE = "moshi==0.2.13"
TORCH_PACKAGE = "torch==2.4.1"
PYTHON_VERSION = "3.12"

# The Hugging Face cache lives on the Volume, never in the container, so it
# survives the container and is reused by the next one.
VOLUME_PATH = "/cache"
HF_HOME = f"{VOLUME_PATH}/hf"

volume = modal.Volume.from_name("voice-inference-lab-hf-cache", create_if_missing=True)

# PHASE 1 image. The downloader needs no torch and no CUDA, and leaving them
# out keeps the image small and the cold start short.
cache_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .uv_pip_install("huggingface_hub")
    .env({"HF_HOME": HF_HOME, "HF_XET_HIGH_PERFORMANCE": "1"})
)

# The plumbing check deliberately floats torch so it reports whatever the
# platform hands out by default. That contrast is the point of keeping it.
check_image = modal.Image.debian_slim().uv_pip_install("torch", "numpy")


def _snapshot(repo: str, revision: str, offline: bool) -> str:
    """Return the local path of a pinned snapshot.

    With ``offline=True`` huggingface_hub resolves purely from the local cache.
    If anything is missing it raises instead of quietly reaching for the
    network, which is what makes the cache guarantee real rather than assumed.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo, revision=revision, local_files_only=offline)


def _json(report: dict) -> str:
    """Serialise a report as a string.

    Remote functions return text rather than Python objects so the laptop never
    needs torch installed just to read a result.
    """
    import json

    text = json.dumps(report, indent=2)
    print(text)
    return text


@app.function(image=cache_image, volumes={VOLUME_PATH: volume}, timeout=3600, min_containers=0)
def cache_weights() -> str:
    """PHASE 1 -- CPU only. No GPU is attached to this container.

    Downloads the pinned snapshot into the Volume and commits it, so the
    ~15.8 GB transfer is billed as CPU time plus storage, never as A10 time.
    Reports whether the checkpoint was already present.
    """
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

    report = {
        "repo": MOSHI_REPO,
        "status": "already present" if already_present else "downloaded",
        "revision": MOSHI_REVISION,
        "path": path,
        "size_gb": round(size / 1e9, 2),
    }

    # Nothing written above is durable until this returns.
    volume.commit()
    report["volume_committed"] = True
    return _json(report)


@app.function(image=check_image, gpu="A10G", min_containers=0)
def inspect_gpu() -> str:
    """Plumbing check: proves CUDA is reachable and that a real tensor op runs.

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


@app.local_entrypoint()
def main() -> None:
    print(inspect_gpu.remote())
