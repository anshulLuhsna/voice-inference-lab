# Serving models on Modal: a guide from this project's mistakes

This is not a Modal tutorial. It is what building three voice architectures on
Modal — Moshi, a three-model STT→LLM→TTS pipeline, and Qwen3-Omni — actually
taught us, with the failure that taught it. Everything here was paid for.

Read it as one project's experience, not universal law. Where a claim is
environment-specific it says so.

---

## Part 1 — The shape of a Modal app

### Split the experiment from the provider

The single most useful structural decision in this repo: experiment logic lives
in a module with **no provider imports**, and a separate thin runner owns the app
handle, the image, the Volume, and the GPU request.

```
modular_experiments.py   # no modal import. Runs on any Linux box with a GPU.
modal_modular.py         # app, images, volume, gpu=...
```

The payoff is not portability theatre. It is that the experiment module can be
read, tested, and reasoned about without knowing anything about Modal, and that
`python3 -m py_compile` plus a handful of static checks can validate it locally
with no cloud account. Durability is the only operation a provider can define, so
it arrives as a callable:

```python
def _noop() -> None:
    """Default commit. A plain filesystem needs nothing to make writes durable."""

def stream_session(commit=_noop) -> str:
    ...
    commit()          # a no-op locally, volume.commit on Modal
```

### Operations return text

Return a JSON **string**, not a Python object. A driver process should not need
torch installed to read a result. It also sidesteps the deserialisation trap:
returning a torch tensor from a remote function makes the local client import
torch to unpickle it, which fails on a laptop.

### Images are cached per layer

Each method call on an `Image` is a cached layer. Break one and every later layer
rebuilds. Two consequences:

- **Put `add_local_*` last.** Modal requires it, and it fails loudly otherwise:
  `An image tried to run a build step after using image.add_local_*`.
- **`copy=False` (the default) mounts at container start, not build time.** So
  editing experiment code does not rebuild the image — which is why the inner
  loop here was fast. But it also means a running container keeps the source it
  started with; editing files never disturbs a live run.

```python
base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("espeak-ng")                       # system packages
    .uv_pip_install("kokoro==0.9.4", "numpy")       # python
    .env({"HF_HOME": "/cache/hf"})                  # container env vars
)
image = base.add_local_python_source("modular_experiments")   # always last
```

`uv_pip_install` accepts `extra_options`, which is the documented way to pass a
raw uv flag — confirmed in the installed source, where it is appended to the
argument list:

```python
.uv_pip_install("vllm==0.30.0", extra_options="--torch-backend=cu130")
```

### Volumes hold bytes; `commit()` makes them survive

```python
volume = modal.Volume.from_name("voice-inference-lab-hf-cache", create_if_missing=True)

@app.function(volumes={"/cache": volume})
def run() -> str:
    ...                      # write under /cache
    volume.commit()          # without this, the writes are gone when the container exits
```

Writes are **not durable until committed**. This bit us twice: a run that crashed
before the commit left nothing behind, and a report that was only *returned*
existed solely in a terminal's scrollback. Commit early and commit repeatedly —
this project now writes and commits after every measurement, so an interrupted
run still leaves what it measured.

### GPU requests

```python
@app.function(gpu="A10G")        # 24 GB
@app.function(gpu="A100-80GB")   # 80 GB
@app.function(gpu="H100:2")      # two cards; ":N" is the count
```

Modal chooses with a **request string**; AWS chooses with an instance type. That
difference is the reason this project keeps two separate runners rather than a
cloud abstraction.

Set `min_containers=0` unless you have measured a reason not to. It means every
run pays a cold start, which for a 70 GB checkpoint is minutes of billed GPU time
before any work happens.

### The CLI gotchas, all of which cost time here

```bash
modal run modal_app.py::cache_weights              # run one function
modal run -w result.json modal_modular.py::residency   # options BEFORE the ref
modal app logs ap-abc123                            # read logs after the fact
modal volume ls voice-inference-lab-hf-cache models
modal volume get <volume> <remote-path> . --force   # --force to overwrite
modal app stop ap-abc123                            # kill a runaway
```

Three traps:

- **`-w` goes before `FUNC_REF`.** After it, Modal interprets the flag as a
  parameter of the function, and you get `No such option '-w'`.
- **`modal run` does not print a function's return value.** Silence is not proof
  anything failed — or that anything ran. Use `-w file.json` if you want the
  result. This cost a full debugging cycle: three runs produced no output, and
  the real problem was that one of them had returned a valid report nobody saw.
- **Exit code 2 from a subprocess is argparse**, not a crash. A CLI rejecting an
  unknown flag exits 2 before loading anything. Distinguish it from a real
  failure (1) or an OOM.

And: a function that returns early without printing is indistinguishable from one
that never ran. **Never return silently.**

---

## Part 2 — Before you write a line: the checklist

For every model, write these down first and get them from the source, not from
memory. Every one of these was wrong at least once here.

| What | Why it matters |
| --- | --- |
| exact repo, exact revision | a floating ref makes runs incomparable |
| exact package versions | the resolved set drifts under you |
| exact GPU and count | the model may require more than one |
| exact load API | signature and defaults change between versions |
| exact streaming API | "streaming" often means per-segment, not continuous |
| exact CUDA major the wheel was built for | mismatches fail at import |
| memory required | from the vendor's table *and* measured |

Then verify from the source, not from documentation prose. vLLM's installation
guide states its binaries are compiled with CUDA 12.9; its wheel needed
`libcudart.so.13`. We believed the prose twice and were wrong twice. **The error
message is the authoritative source.**

### Pinning when the loader won't take a revision

Some loaders expose no way to request a revision. Then the pin is enforced
differently: fetch the default ref once, record the **resolved** commit, and set
`HF_HUB_OFFLINE=1` at runtime. Offline, resolution can only return what was
fetched, so a repointed upstream cannot leak in.

```python
# cache phase (network allowed), CPU container, no GPU
hf_hub_download(repo_id=REPO, filename="config.json")   # populates the HF cache
sha = _snapshot_sha(returned_path)                       # record it
assert sha == PINNED_REVISION                            # and check it

# runtime phase (HF_HUB_OFFLINE=1), GPU container
```

Be explicit that this is a weaker guarantee than a requested revision, and say
why: it is what the loader permits.

**Offline is scoped to `huggingface_hub`, not to the network.** A Kokoro load was
observed pip-installing a spaCy model from GitHub mid-run, because that library
shells out to pip. Anything a library installs at runtime belongs in the image.

---

## Part 3 — The traps, each with its failure

### 1. `--torch-backend=auto` installs CPU torch on a GPU-less build machine

uv inspects the driver to pick a PyTorch index. **Modal builds images on
machines with no GPU**, so the inspection finds nothing and it falls back to the
CPU index. The container then cannot see CUDA.

```
[omni] FAILED: CUDA is unavailable; refusing to fall back to CPU.
[omni] torch=2.13.0+cpu
```

Name the backend. `auto` is only correct when uv runs on the machine that will
use the GPU.

### 2. A vLLM wheel is built for exactly one CUDA major

```
File "vllm/platforms/cuda.py", line 23, in <module>
  import vllm._C_stable_libtorch
ImportError: libcudart.so.13: cannot open shared object file
```

Seen twice, at vLLM 0.30.0 **and** 0.28.0. Match the torch backend to the wheel's
CUDA major, not to what the docs claim and not to what other components want.

### 3. Two CUDA majors in one process do not coexist

CTranslate2 requires CUDA 12 and cuDNN 9. A CUDA 13 torch wants its own cuDNN.
cuDNN 9 is split into sublibraries that must all come from one build, so the
process ends up half-and-half:

```
RuntimeError: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed
  cudnn_status: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
```

Both models *loaded*. It failed on the first convolution. The process had
`libcublas.so.12` and `libcublas.so.13` mapped at once. If a dependency mandates
a CUDA major, the whole image shares it.

### 4. A compiled extension needs a compiler, not just runtime libraries

FlashInfer JIT-compiles its sampling kernels with ninja and nvcc on first use.
A runtime-only image has neither:

```
flashinfer/sampling.py -> jit/cpp_ext.py get_cuda_path()
  RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
```

`VLLM_USE_FLASHINFER_SAMPLER=0` is the documented workaround, and it also removes
a JIT compile from startup — worth having when startup is already minutes.

### 5. The first call through a model is not like the others

Measured here, on the same hardware:

| model | first call | later calls |
| --- | --- | --- |
| Moshi | 10,820 ms | 46 ms |
| Kokoro | 2,851 ms | 145 ms, 133 ms |

CUDA graph capture and lazy kernel init happen once. If your first measurement
includes it, you are measuring initialization. Push synthetic input through the
full path before the measured window, reset the state, and **report the warm-up
separately**. Warm every model you are about to time: Whisper and Kokoro were
warmed here and the LLM's first request still cost 68 ms against 25 ms.

### 6. `temperature: 0` does not make generation reproducible

Same prompt, same session, same parameters, two different completions — because
argmax over logits that differ in the last bits flips a near-tie. If you need
identical text, pin a seed and verify it; if you cannot, say the output varies.

### 7. Measure memory from the driver, not from parameter counts

`torch.cuda.mem_get_info()` is the only counter that sees every allocator.
CTranslate2 keeps no statistics at all, so PyTorch's counters miss it entirely.
Per-model footprint is the change in the driver's `used` across the load.

### 8. `gpu_memory_utilization` does not set a KV cache size

It is a fraction of **total** memory, not free memory, and it sets a total
budget. Two consecutive starts at the *same* setting, in the same container:

| | start 1 | start 2 |
| --- | --- | --- |
| KV tokens | 47,920 | 76,992 |
| reported weight load | 5.79 GB | 5.79 GB |

Identical weights, identical budget, and the split between cache and overhead
differed by 60%. So cache capacity is not a property of the setting, and a series
across different values also varies by start order.

### 9. An OOM message's advice is boilerplate; the useful part is in the middle

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.62 GiB.
GPU 0 has a total capacity of 139.80 GiB of which 761.62 MiB is free.
Process 1 has 139.04 GiB memory in use.
... If reserved but unallocated memory is large try setting
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
```

That last sentence is appended to every PyTorch OOM, and it is easy to act on it
when it is not your problem. It was tried here and changed nothing, and a run was
spent moving to a larger GPU on the strength of it.

**The diagnosis is in the middle of the message, not the end:** *Process 1 has
139.04 GiB memory in use*. Another process on the same device already held 139 of
the card's 139.8 GiB. That is a different problem from fragmentation, and the
next trap is what that process was.

The general rule: read the whole message, and read it before changing anything.

### 10. Multi-process stacks oversubscribe silently

vLLM-Omni runs each stage in its own process, and each budgets against the
**whole device**. The shipped config for Qwen3-Omni sets stage fractions of 0.9,
0.6 and 0.1 — summing to 1.6 of a single device, because it is written for two
GPUs:

```
# verified on 2x H100 (stage 0 on cuda:0, stages 1+2 on cuda:1)
```

That "Process 1" holding the whole card was a sibling stage. A bigger card does
not help: 80 GB, 141 GB and 2×80 GB all failed the same way, because the total
demanded scales with the card.

**Read the vendor's deploy config before choosing hardware.** It states the
intended device count, and the symptom is indistinguishable from "not enough
memory" — which is precisely why a larger GPU looks like the answer and isn't.

### 11. Read the whole error, and keep the whole log

Twice here, only the tail of a message was read and the wrong cause was
inferred — a fragmentation failure was called a capacity failure, and a cycle
was spent on a larger GPU that could not have helped.

Corollaries:
- Write logs to uniquely-named artifacts, not a fixed path that each run
  overwrites. One overwritten log destroyed the evidence that would have settled
  a question.
- To find a cause, take the **first** error in a log, not the last. Everything
  after a fatal error is the shutdown path unwinding. A traceback's causal line
  is its *last* line, which may sit well beyond a fixed context window.

### 12. A waited-on process can die

Polling an HTTP endpoint without watching the process means a server that exited
in 3 seconds still burns the full timeout. `_http_ready` here takes the process
and checks `poll()` each iteration, which turned a 15-minute stall into a
3-second failure. When it fails, record *why*: "process exited with code 1" is
not a diagnosis; the log tail is.

---

## Part 4 — Measuring honestly

The habits that mattered most here, all cheap:

- **Separate acquisition from computation.** Download on a CPU container with no
  GPU attached. Moving 70 GB does not need an accelerator, and billing one for a
  download is absurd.
- **Separate what is measured from what is not.** Model loading happens *before*
  `turn_start`, because folding a cold start into a turn describes a different
  experiment. Here it was the difference between 3 s and 108 s.
- **State the metric before comparing.** Time-to-first-audio and total completion
  are different questions; this project found overlap improved the first by 42%
  while the second moved by 1.6%. Neither policy is "faster" without naming one.
- **Compare in the same container.** Two runs in different containers differed by
  2.28× on transcription time — work the change under test could not affect. The
  paired, alternating, same-container design reduced that spread to 1.02×.
- **Report the spread of what should not have changed.** It is the cheapest
  validity check available, and it is what exposes an unreadable comparison.
- **One change at a time.** Every time two things changed at once here, the
  result was unattributable and had to be re-run.
- **When a result is negative, say so and say why.** A failed run that names its
  cause is worth more than a passing run that does not.

---

## The short version

1. Experiment logic separate from the provider shim; return JSON strings.
2. `add_local_*` last; `commit()` often; `min_containers=0` unless measured.
3. `-w` before the function ref, and remember nothing prints unless you print it.
4. Pin the repo, the revision, every package, and the CUDA major — and get them
   from the source and the error message, never from documentation prose.
5. Warm every model before you time anything, and report the warm-up separately.
6. Measure memory from the driver, and read the whole error, not its tail.
7. Read the vendor's deploy config before choosing hardware.
8. Change one thing at a time, and compare in one container.
