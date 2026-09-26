# Phase 3 — Qwen Omni: closed without an audio artifact

## Outcome

**No audio artifact was produced.** Eight runs across one, two and four GPUs and
two model sizes all failed before the server served a request. The architectural
question was answered anyway, and by stronger evidence than a working WAV would
have provided.

## The finding

> **Both Qwen Omni models ship as two-device, three-stage deployments, with
> memory fractions tuned for 80 GB-class cards.**

This is not inferred from logs. It is in the deploy config vLLM-Omni loads
automatically, and that config is committed alongside this note as
`qwen3_omni_moe.deploy.yaml`:

```
# Qwen3-Omni-MoE production deploy, verified on 2x H100
# (stage 0 on cuda:0, stages 1+2 on cuda:1).

stage 0  devices "0"  gpu_memory_utilization 0.9   Thinker
stage 1  devices "1"  gpu_memory_utilization 0.6   Talker
stage 2  devices "1"  gpu_memory_utilization 0.1   Code2Wav
```

Those fractions sum to **1.6 of a single device**, because the config is written
for two. The server's own log confirms the placement it applied:

```
[AsyncOmniEngine] Launching Orchestrator thread with 3 stages
[stage_init] Stage-0 set runtime devices: 0
[stage_init] Stage-1 set runtime devices: 1
```

The same two-device shape holds for **Qwen2.5-Omni-3B**, so this is a property of
the architecture family rather than of one large checkpoint.

## What was tried

| model | gpu | result |
| --- | --- | --- |
| Qwen3-Omni-30B-A3B | A100-80GB | OOM, stage 0 KV allocation |
| Qwen3-Omni-30B-A3B | H200 (141 GB) | OOM, same code path, 458 log lines |
| Qwen3-Omni-30B-A3B | H100 ×2 | placement correct, stage 0 short by 0.2 GiB |
| Qwen2.5-Omni-3B | A10G | OOM, stage 1 pinned to a device that does not exist |
| Qwen2.5-Omni-3B | A10G ×2 | placement correct, fractions fill a 22 GiB card |

Every failure is the same shape: stage budgets are fractions of the whole device,
sized for an 80 GB card. A bigger card does not help, because the demand scales
with it — which is why 80 GB, 141 GB and 2×80 GB all failed identically.

## What this means for the comparison

- **Moshi** — one model, one GPU, one continuous stream. The codec either side of
  the language model sits in the same process.
- **The modular trio** — three independent models in three containers. That
  layout was *chosen* here, for measurement reasons, and sentence-level overlap
  bought 42% on time-to-first-audio for +1.6% on total completion.
- **Qwen Omni** — staged *by the vendor*. The difference between a layout we
  chose and a layout we were handed is the sharper claim, and it holds at both
  model sizes.

## Runtime, verified

| | |
| --- | --- |
| model | `Qwen/Qwen3-Omni-30B-A3B-Instruct`, `Qwen/Qwen2.5-Omni-3B` |
| checkpoint size | 70.53 GB on disk (30B, measured) |
| vLLM | 0.28.0 |
| vLLM-Omni | v0.28.0 |
| CUDA | 13.0 (`--torch-backend=cu130`) |
| documented 30B minimum | 78.85 GB BF16, and that is for a 15 s video input |

There is **no stable v0.30.0 of vLLM-Omni** — only `v0.30.0rc1`. Stable releases
are even-numbered (0.18 → 0.28), and the two projects must share a major and a
minor, so 0.28.0 on both sides is the newest pair that is entirely released.

## Missing, and why it is recorded rather than pursued

1. One successful audio-in → audio-out inference.
2. First-audio timing.

Both need a server that starts. Getting there would have meant rewriting the
vendor's per-stage memory fractions to fit a 22 GB card — inventing a
configuration the project neither documents nor tests. That is the point at which
this stops being measurement of someone else's architecture and becomes debugging
their deployment, so it was stopped.

## Environment problems solved along the way

Recorded because they cost more than the model did, and because they generalise:

- `--torch-backend=auto` installs **CPU torch** on Modal, because images build on
  GPU-less machines and uv finds no driver to match.
- vLLM's published wheel needs **CUDA 13**, at 0.28.0 as well as 0.30.0, despite
  the documentation's prose about 12.9.
- FlashInfer JIT-compiles its sampling kernels with **nvcc**, which a
  runtime-only image does not have.
- A truncated error message was read twice and the wrong cause inferred twice.

See [`../../MODAL-GUIDE.md`](../../MODAL-GUIDE.md) for the general form of these.

## Follow-ups, not done here

- **Contention cause unresolved.** The modular overlap run showed the LLM's own
  completion 93 ms slower under overlap, and this harness cannot say whether that
  is GPU contention or the reader thread not being scheduled while synthesis
  runs. Needs the per-token timeline per turn, plus a read-and-discard arm.
- **LLM warm-up and seeding.** The first request of a cold session is 68 ms to
  first token against 25 ms afterwards, and produced a different completion at
  `temperature: 0`. Warming the LLM and using a seed would give six clean turns.
- **Qwen on 2× A10 with rewritten fractions**, if the audio artifact is ever
  wanted. Cheap, but a deviation from the shipped config and it should be
  labelled as one.
