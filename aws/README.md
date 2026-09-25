# AWS EC2 reproduction

The same experiments as the Modal milestones, on a raw EC2 instance, for a
controlled comparison. The experiment logic is shared; only the launch and
lifecycle mechanisms differ, and they are supposed to differ.

## What is not here

Nothing in this directory provisions anything. No Terraform, no boto3, no IAM,
no subnets, no security groups, no quota management. The instance is created and
destroyed outside this repository, and shell access is assumed.

## Runtime target

The pinned runtime is `aws/requirements.txt`, taken from the Modal load image:
Python 3.12, `torch==2.4.1`, `moshi==0.2.13`, plus what those resolve.

The AMI exists to provide Linux, an NVIDIA driver, GPU device access, and a
CUDA-capable host. It is not the runtime. Do not use its PyTorch.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r aws/requirements.txt
```

## Run

Collect the machine first, before the runtime is installed, then again after.
Both are kept, because the difference between them is itself a provider fact.

```bash
.venv/bin/python aws/run_experiment.py collect-env --out artifacts --label pre-install
# install
.venv/bin/python aws/run_experiment.py collect-env --out artifacts --label post-install

# point the data root at the instance's local NVMe, read from the env artifact
.venv/bin/python aws/run_experiment.py suite --root /path/to/nvme/voice-lab --out artifacts
```

`VOICE_LAB_ROOT` is the single knob. It defaults to `/cache`, which is the Modal
mount point, so the same code serves both providers.

## What each step reproduces

| Step | Reproduces | Notes |
| --- | --- | --- |
| `inspect_gpu` | GPU and runtime probe | records raw bytes and raw nvidia-smi |
| `cache_weights` | model cache and download behaviour | the only step that needs the network |
| `load_model` | Moshi and Mimi load time and memory | |
| `stream_no_warmup` | unwarmed first-frame behaviour | 162 frames, warm-up 0 |
| `stream_warmup` | explicit warm-up | 162 frames, warm-up 5 |
| `stream_sustained` | session past the context window | 3400 frames, warm-up 5 |

Steps 4 to 6 override two module attributes before calling the experiment, so
they match the three Modal runs frame for frame.

## Phases are kept separate

Instance boot, download, model load, warm-up, first frame, and steady state are
separate numbers. None of them is reported as a single "cold start". Boot time in
particular cannot be seen from inside the instance and is not claimed.

## Spot instances

The machine is interruptible. Treat the local disk as ephemeral and never as the
only copy of a result. Artifacts are written as each step finishes, so a
completed step survives an interruption. If the instance is reclaimed, report it
as infrastructure behaviour rather than rerunning quietly. There is no
interruption recovery machinery, by choice.

## Copy results off before shutdown

```bash
# from your own machine
scp -r <user>@<host>:~/voice-inference/artifacts .
```

The important artifacts are the `*.json` reports and `series.csv` from the
sustained step. The `.wav` tracks are useful but large.
