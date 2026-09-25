"""voice-inference-lab -- the Modal runner for the modular stack.

The experiment logic lives in `modular_experiments`, which imports no provider
SDK. This file owns only what is Modal's: the app handle, the container images,
the persistent Volume, and the GPU request. Nothing here decides what an
experiment measures.

    modal run modal_modular.py::cache_models   # CPU only, no GPU. Downloads.
    modal run modal_modular.py::residency      # one A10. Loads all three.

Both operations are one container on one GPU on purpose. The three models stay
resident together so the stage timings share a clock and the stage boundaries
share a container, because what is being measured is pipeline scheduling rather
than distributed-service architecture.

`modal_app.py` remains the runner for the Moshi experiments. The two are not
merged: they need different images and different models, and the difference
between a continuous duplex stream and a three-model pipeline is the subject of
the comparison, not an accident to be smoothed over.
"""

import modal

import modular_experiments as exp

app = modal.App("voice-inference-lab-modular")

# The same Volume as the Moshi phases, under its own subdirectory. One place to
# look, and no chance of one stack overwriting the other's weights.
volume = modal.Volume.from_name("voice-inference-lab-hf-cache", create_if_missing=True)

SOURCE = "modular_experiments"

# Downloading needs no torch and no CUDA. faster-whisper is installed here so
# the CT2 allow-list comes from the library that will later read the files,
# rather than being restated and left to drift.
cache_image = (
    modal.Image.debian_slim(python_version=exp.PYTHON_VERSION)
    .uv_pip_install("huggingface_hub", exp.FASTER_WHISPER_PACKAGE)
    .env({"HF_HOME": exp.HF_HOME, "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source(SOURCE)
)

# The runtime image for all three models.
#
# - espeak-ng is Kokoro's out-of-dictionary fallback and degrades quietly
#   without it, so it is installed explicitly.
# - vLLM comes from its published CUDA 12.9 wheel rather than from PyPI. The
#   PyPI wheel is built for CUDA 13.0, and mixing it with a CUDA 12 torch fails
#   at import; mixing a CUDA 13 torch with CTranslate2's CUDA 12 cuDNN fails
#   later and worse, on Kokoro's first convolution. One CUDA major across the
#   stack is a requirement, and 12.9 is the one CTranslate2 forces.
# - cuBLAS and cuDNN 9 are installed because CTranslate2 requires them and does
#   not bundle them, and LD_LIBRARY_PATH is set here rather than from Python
#   because the dynamic loader reads it once, at process start.
# - HF_HUB_OFFLINE=1 keeps the runtime off the Hugging Face Hub: a cache miss
#   raises instead of quietly fetching, and the vLLM child process inherits the
#   rule. It is not a network sandbox and does not claim to be one. misaki was
#   observed pip-installing a spaCy model from GitHub on the first Kokoro load,
#   which is why that model is pinned in here instead of being resolved at run
#   time.
modular_image = (
    modal.Image.debian_slim(python_version=exp.PYTHON_VERSION)
    .apt_install("espeak-ng")
    .uv_pip_install(
        exp.VLLM_WHEEL_URL,
        exp.FASTER_WHISPER_PACKAGE,
        exp.KOKORO_PACKAGE,
        exp.SPACY_MODEL_PACKAGE,
        "numpy",
        "soundfile",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12==9.*",
        extra_options=f"--torch-backend={exp.VLLM_TORCH_BACKEND}",
    )
    .env(
        {
            "HF_HOME": exp.HF_HOME,
            "HF_HUB_OFFLINE": "1",
            "LD_LIBRARY_PATH": exp.NVIDIA_LIB_PATH,
        }
    )
    .add_local_python_source(SOURCE)
)


@app.function(
    image=cache_image, volumes={exp.DATA_ROOT: volume}, timeout=3600, min_containers=0
)
def cache_models() -> str:
    """Download all three model sets at pinned revisions. No GPU is attached."""
    return exp.cache_models(commit=volume.commit)


@app.function(
    image=modular_image,
    gpu="A10G",
    volumes={exp.DATA_ROOT: volume},
    timeout=3600,
    min_containers=0,
)
def residency() -> str:
    """Load all three models onto one A10 and measure the memory floor."""
    return exp.residency(commit=volume.commit)


@app.function(
    image=modular_image,
    gpu="A10G",
    volumes={exp.DATA_ROOT: volume},
    timeout=1800,
    min_containers=0,
)
def vllm_probe() -> str:
    """One vLLM start at a fixed fraction, with the complete log kept.

    No other model is loaded and no sweep runs, so a startup failure can be
    diagnosed without paying for the full three-model residency run each time.
    Same image and same environment as `residency`, deliberately.
    """
    return exp.vllm_startup_probe(commit=volume.commit)


@app.function(
    image=modular_image,
    volumes={exp.DATA_ROOT: volume},
    timeout=1800,
    min_containers=0,
)
def generate_fixture() -> str:
    """Synthesise the Experiment 1 spoken prompt once. No GPU attached.

    Fixture preparation, deliberately separate from measurement. It runs here
    because the pinned Kokoro weights already live in this image and cache, and
    it runs without a GPU because generating a few seconds of speech is not
    worth billing an A10 for.
    """
    return exp.generate_fixture(commit=volume.commit)


@app.function(
    image=modular_image,
    gpu="A10G",
    volumes={exp.DATA_ROOT: volume},
    timeout=3600,
    min_containers=0,
)
def sequential_turn() -> str:
    """Experiment 1 -- the fully sequential baseline, on one A10."""
    return exp.sequential_turn(commit=volume.commit)


@app.function(
    image=modular_image,
    gpu="A10G",
    volumes={exp.DATA_ROOT: volume},
    timeout=3600,
    min_containers=0,
)
def overlap_turn() -> str:
    """Experiment 2 -- the same turn with synthesis allowed to start earlier.

    Same image, same GPU, same container shape as Experiment 1. The only
    difference is the scheduling policy inside the experiment module.
    """
    return exp.overlap_turn(commit=volume.commit)


@app.function(
    image=modular_image,
    gpu="A10G",
    volumes={exp.DATA_ROOT: volume},
    timeout=3600,
    min_containers=0,
)
def paired_turns() -> str:
    """Both policies, back to back, in one container.

    The comparison the first two experiments could not make: same container,
    models loaded once, both first-call costs paid before any measured turn.
    """
    return exp.paired_turns(commit=volume.commit)
