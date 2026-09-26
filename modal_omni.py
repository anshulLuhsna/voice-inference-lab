"""voice-inference-lab -- the Modal runner for the Qwen Omni phase.

The experiment logic lives in `omni_experiments`, which imports no provider SDK.
This file owns only what is Modal's: the app handle, the container image, the
persistent Volume, and the GPU request.

    modal run modal_omni.py::cache_model    # CPU only, no GPU. ~60 GB.
    modal run modal_omni.py::omni_turn      # one A100 80GB. One turn.

This phase runs on an 80 GB card rather than the A10 the other two phases used,
because Qwen3-Omni-30B-A3B is a 30B mixture-of-experts model: Qwen's own model
card states a BF16 minimum of 78.85 GB for this checkpoint, and that figure is
for a 15 second video input. Audio-only is lighter, but the weights alone are
around 60 GB. Whether it actually fits is part of what this phase reports.

The modular runner stays separate. The images share nothing: this one carries
vLLM 0.28.0 and vLLM-Omni, the other carries a CUDA 12.9 vLLM wheel beside
CTranslate2 and Kokoro. Merging them would force one version of everything and
lose the point of the comparison.
"""

import modal

import omni_experiments as exp

app = modal.App("voice-inference-lab-omni")

# The same Volume as the other phases, under its own subdirectory.
volume = modal.Volume.from_name("voice-inference-lab-hf-cache", create_if_missing=True)

SOURCE = "omni_experiments"

# Downloading needs no torch and no CUDA, and moving 60 GB is not something to
# bill an 80 GB card for.
cache_image = (
    modal.Image.debian_slim(python_version=exp.PYTHON_VERSION)
    .uv_pip_install("huggingface_hub")
    .env({"HF_HOME": exp.HF_HOME, "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source(SOURCE)
)

# The runtime image, built the way vLLM-Omni's own quickstart describes:
# install vLLM first, then install vLLM-Omni from its repository.
#
# - `--torch-backend=auto` is the documented choice here and is safe in a way it
#   was not for the modular stack: this image has only one CUDA major in it.
# - vLLM-Omni is pinned to a release tag, not to `main`. The two projects must
#   share a major and a minor version, and the stable pair is 0.28.0 on both
#   sides.
# - ffmpeg is installed because audio input is decoded server-side.
# - HF_HUB_OFFLINE=1 keeps the runtime off the Hub. It is not a network sandbox
#   and does not claim to be one; it means a missing file raises rather than
#   quietly fetching 60 GB onto a billed GPU.
runtime_image = (
    modal.Image.debian_slim(python_version=exp.PYTHON_VERSION)
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        f"vllm=={exp.VLLM_VERSION}",
        extra_options=f"--torch-backend={exp.VLLM_TORCH_BACKEND}",
    )
    .uv_pip_install(
        f"vllm-omni @ git+{exp.VLLM_OMNI_REPO}@{exp.VLLM_OMNI_REF}",
        "numpy",
        "soundfile",
    )
    .env({"HF_HOME": exp.HF_HOME, "HF_HUB_OFFLINE": "1"})
    .add_local_python_source(SOURCE)
)


@app.function(
    image=cache_image, volumes={exp.DATA_ROOT: volume}, timeout=14400, min_containers=0
)
def cache_model() -> str:
    """Download the Qwen3-Omni checkpoint. No GPU is attached."""
    return exp.cache_model(commit=volume.commit)


@app.function(
    image=runtime_image,
    gpu="A10G:2",
    volumes={exp.DATA_ROOT: volume},
    timeout=7200,
    min_containers=0,
)
def omni_turn() -> str:
    """Serve Qwen3-Omni with vLLM-Omni and run one audio-to-audio turn.

    Two H100s, because the model's own deploy config is a two-GPU topology. From
    `vllm_omni/deploy/qwen3_omni_moe.yaml`, which the server loads automatically:

        # Qwen3-Omni-MoE production deploy, verified on 2x H100
        # (stage 0 on cuda:0, stages 1+2 on cuda:1).

        stage 0  devices "0"  gpu_memory_utilization 0.9   thinker
        stage 1  devices "1"  gpu_memory_utilization 0.6   talker
        stage 2  devices "1"  gpu_memory_utilization 0.1   code2wav

    Those fractions sum to 1.6 of a single device, so one card cannot run this
    configuration, and a bigger card does not help: a 141 GB H200 failed exactly
    as an 80 GB A100 did, with 139.04 GiB of 139.80 GiB already held by another
    process before stage 0 asked for its share. The constraint is the topology,
    not the memory.

    Two H100s rather than two H200s because the config states it was verified on
    2x H100, and because it is materially cheaper. No --stage-overrides is used:
    inventing a single-GPU split would measure a topology the project does not
    document or test.
    """
    return exp.omni_turn(commit=volume.commit)
