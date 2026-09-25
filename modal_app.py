"""voice-inference-lab -- the Modal provider shim.

The experiment logic lives in `moshi_experiments`, which imports no provider SDK
and runs unchanged on any Linux host with an NVIDIA GPU. This file owns only
what is Modal's: the app handle, the container images, the persistent Volume,
and the GPU request. Nothing here decides what an experiment measures.

    modal run modal_app.py::inspect_gpu      # plumbing check
    modal run modal_app.py::cache_weights    # PHASE 1 -- CPU only, no GPU
    modal run modal_app.py::load_model       # PHASE 2 -- runs on the A10
    modal run modal_app.py::stream_session   # PHASE 3 -- runs on the A10

Modal chooses the GPU with a request string. AWS chooses it with an instance
type. That difference is the point of keeping two runners.
"""

import modal

import moshi_experiments as exp

app = modal.App("voice-inference-lab")

volume = modal.Volume.from_name("voice-inference-lab-hf-cache", create_if_missing=True)

# The experiment module is mounted into every container and lands on the
# container's PYTHONPATH, so `moshi_experiments` imports normally inside.
SOURCE = "moshi_experiments"

# PHASE 1 image. The downloader needs no torch and no CUDA, and leaving them
# out keeps the image small and the cold start short.
cache_image = (
    modal.Image.debian_slim(python_version=exp.PYTHON_VERSION)
    .uv_pip_install("huggingface_hub")
    .env({"HF_HOME": exp.HF_HOME, "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source(SOURCE)
)

# The pinned runtime with nothing local mounted yet. Modal requires `add_local_*`
# to be the last steps of an image build, so the base is kept separate and each
# consumer attaches its own sources at the end.
base_image = (
    modal.Image.debian_slim(python_version=exp.PYTHON_VERSION)
    .uv_pip_install(exp.TORCH_PACKAGE, exp.MOSHI_PACKAGE)
    .env({"HF_HOME": exp.HF_HOME, "HF_HUB_OFFLINE": "1"})
)

# PHASE 2 and PHASE 3 image. Needs the pinned torch and moshi, and is forced
# offline so it can only ever read the Volume -- a missing file becomes a loud
# error rather than a silent re-download.
load_image = base_image.add_local_python_source(SOURCE)

# The plumbing probe deliberately floats torch so it reports whatever the
# platform hands out by default. That contrast is the point of keeping it.
check_image = (
    modal.Image.debian_slim().uv_pip_install("torch", "numpy").add_local_python_source(SOURCE)
)

# The live browser session. Same pinned runtime as the offline phases, plus an
# ASGI framework so Modal can serve a WebSocket, plus the page itself.
# `load_image` is reused as the base rather than restated.
#
# The page lands at a fixed path. `moshi_browser.PAGE_PATH` must match it; a
# mismatch fails loudly on the first request rather than quietly serving
# nothing.
PAGE_REMOTE = "/root/browser/index.html"

live_image = (
    base_image.uv_pip_install("fastapi")
    .add_local_python_source(SOURCE, "moshi_browser")
    .add_local_file("browser/index.html", PAGE_REMOTE)
)


@app.function(image=cache_image, volumes={exp.DATA_ROOT: volume}, timeout=3600, min_containers=0)
def cache_weights() -> str:
    """PHASE 1 -- CPU only. No GPU is attached to this container.

    The Volume is committed only once the experiment has finished its writes.
    """
    return exp.cache_weights(commit=volume.commit)


@app.function(
    image=load_image,
    gpu="A10G",
    volumes={exp.DATA_ROOT: volume},
    timeout=1800,
    min_containers=0,
)
def load_model() -> str:
    """PHASE 2 -- loads the weights from the Volume. Never commits."""
    return exp.load_model()


@app.function(
    image=load_image,
    gpu="A10G",
    volumes={exp.DATA_ROOT: volume},
    timeout=1800,
    min_containers=0,
)
def stream_session() -> str:
    """PHASE 3 -- streams the fixture through Moshi and commits the audio."""
    return exp.stream_session(commit=volume.commit)


@app.function(image=check_image, gpu="A10G", min_containers=0)
def inspect_gpu() -> str:
    """Probe the GPU this platform hands out."""
    return exp.inspect_gpu()


@app.function(
    image=live_image,
    gpu="A10G",
    volumes={exp.DATA_ROOT: volume},
    timeout=3600,
    min_containers=0,
)
@modal.asgi_app()
def browser():
    """Live browser session: microphone in, model audio out, over one socket.

    An open connection keeps this container alive and billing, silence
    included. That is accepted and measured for this milestone rather than
    papered over, because the cost of an idle session is one of the questions
    the article asks.
    """
    import moshi_browser

    return moshi_browser.create_app(commit=volume.commit)
