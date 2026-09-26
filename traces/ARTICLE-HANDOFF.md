# voice-inference-lab — article handoff

An experimental lab that asks one question three ways: **what does it actually
cost to serve real-time voice AI?** Three architecture families, same measurement
discipline, honest about what was and was not established.

Everything below is drawn from committed traces under `traces/`. Where a claim
came from a vendor document rather than from our own measurement it says so. Two
of my own earlier claims are formally retracted at the end.

---

## The question, and why the answer is interesting

Everyone says "end-to-end" and "real-time". Neither word survives contact with a
deployment. The interesting variable is not model quality — it is **where the
staging lives**, and who decided it.

| | staging lives in | devices | who chose it |
| --- | --- | --- | --- |
| Moshi | one model, one process | 1 | the model's shape |
| Modular trio | three separate models | 1 (three resident) | **the engineer** |
| Qwen Omni | three stages in one model | 2 | **the vendor** |

That contrast is the article. Measured results follow.

---

## Phase 1 — Moshi

One continuous full-duplex model. Not an input→output API: two streams on one
clock, and it never decides a turn has ended.

**Measured** (`traces/`, `moshi_experiments.py`):

- Steady state **48.5 ms/frame** against an 80 ms budget.
- The first call through the model cost **10,820 ms**; after an explicit warm-up
  the first *real* frame cost **46 ms**. The warm-up must share one streaming
  context with the session, and `reset_streaming()` — not a new context — is what
  makes the reset work. `first_output_at_frame == 1` is the self-check.
- 3,400 frames (~272 s) crossing the 3,000-frame context window: **memory is
  preallocated, not merely bounded.** Allocated, reserved and free are identical
  at frames 0, 100, 2999, 3000, 3001 and 3399; growth across the session is
  exactly zero.
- Moshi's output lags its input by one frame. `mixed.wav`, `human.wav` and
  `moshi.wav` share one timeline, so a timestamp means the same moment in all
  three.

**The architectural point:** the codec and the language model are both inside the
same process, and the staging between them is the model's own. There is nothing
for a deployment engineer to place.

---

## Phase 2 — the modular trio

Whisper → Qwen → Kokoro, three independent models, deliberately **one container**
on one A10 with all three resident.

That layout was a choice, for measurement reasons: three containers would add two
cold starts and put the stage timings in three clock domains. The point was to
measure pipeline scheduling, not distributed-service architecture.

**Resident footprint** (`traces/modular/residency.json`):

| component | resident |
| --- | --- |
| Whisper (faster-whisper, CTranslate2 fp16) | 2.447 GB |
| Kokoro 82M | 0.688 GB |
| vLLM 0.40 budget, Qwen2.5-3B | ~9.1 GB, KV 47,920 tokens |

Device total 23.685 GB; **~11.4 GB free with all three resident.** Coexistence was
far cheaper than feared; the binding constraint was never memory.

**The experiment that matters** — sequential vs overlapped, paired, alternating,
same container, models loaded once, both first-call costs paid before any
measured turn:

| | sequential | overlapped |
| --- | --- | --- |
| time to first audio (median) | 830.6 ms | **478.6 ms** |
| total completion (median) | 831.2 ms | 844.6 ms |
| per-pair deltas | — | +367.9, +353.3, +321.2 ms |

**First audio 353 ms earlier — 42.4% — for +13.4 ms (+1.6%) on total
completion.** The overlap is visible in the trace: generation ran 184.6→782.4 ms,
synthesis ran 347.5→888.4 ms with no gaps, overlapping for 435 ms.

**Validity check that made the comparison readable:** transcription time spread
**1.02×** across the six turns. In the earlier cross-container attempt the same
metric spread **2.28×** — larger than the effect — and the comparison was
discarded.

**What did not get resolved.** The LLM's own observed completion was **93 ms
slower** under overlap, reproducibly. This harness cannot say whether that is GPU
contention or the reader thread not being scheduled while synthesis runs. It is
recorded as unresolved rather than attributed.

**What Kokoro forced.** It produced **one** chunk for a three-sentence reply,
because `KPipeline` splits its input on newlines before its internal
510-phoneme chunker. So clause-level overlap cannot come from the model; the
scheduler has to segment. That is a finding about the library, and it shaped the
experiment.

---

## Phase 3 — Qwen Omni

One model family, served by vLLM-Omni.

**The architecture, source-verified.** vLLM-Omni runs Qwen3-Omni as three stages:

| stage | role | config placement |
| --- | --- | --- |
| 0 | **Thinker** + API server | `devices: "0"` |
| 1 | **Talker** | `devices: "1"` |
| 2 | **Code2Wav** | `devices: "1"` |

The server's own log confirms it applied them:

```
[AsyncOmniEngine] Launching Orchestrator thread with 3 stages
[stage_init] Stage-0 set runtime devices: 0
[stage_init] Stage-1 set runtime devices: 1
```

The config file carries the vendor's own note: *"verified on 2x H100 (stage 0 on
cuda:0, stages 1+2 on cuda:1)"*. It is committed as
`traces/omni/qwen3_omni_moe.deploy.yaml`. **This is the article's Qwen payload**:
the staging is not inferred from logs, it is in the artifact you must load. And
the same two-device shape holds for Qwen2.5-Omni-3B, so it is the family, not the
checkpoint size.

**Where it got to, and where it stopped.** With vLLM-Omni **0.30.0** the server
**starts**: 572 s cold, three stages resident, **135.1 GB of 150.1 GB** on 2×
H200. The single remaining failure is a **404 on `/v1/chat/completions`** —
`/v1/models` answers, the chat route does not. That is a route-discovery step
(`GET /openapi.json`), not a hardware limit.

**No audio artifact exists.** The exit condition is 5 of 7 met: runtime config,
hardware footprint, stage layout, deploy config, limitations. Missing: one
audio-in → audio-out inference, and its first-audio timing.

**The real Phase 3 lesson, which cost the most to learn:** the failures were
**not** the deploy config. vLLM-Omni **0.28.0** used process-scoped NVML memory
estimation for LLM stages, so colocated stages could profile and allocate against
the whole device; 0.30.0 replaces it with budget summation and admission against
the physical GPU before launch, plus per-device locks. The tell was in our own
logs all along — `Available KV cache memory: 125.82 GiB (process-scoped)` on a
141 GB card whose other process already held 139.04 GiB.

---

## Claim ledger

**MEASURED (our runs):**
- Moshi 48.5 ms/frame steady state; memory constant across the context wrap.
- Moshi first call 10,820 ms → 46 ms after warm-up.
- Whisper 2.447 GB, Kokoro 0.688 GB resident; ~11.4 GB free with all three.
- Sequential vs overlapped: 830.6 → 478.6 ms first audio; 831.2 → 844.6 ms total.
- Overlap window 435 ms, from the trace.
- Policy-independent spread 1.02× (paired) vs 2.28× (cross-container).
- LLM completion +93 ms under overlap, **cause unresolved**.
- LLM first request 68 ms TTFT vs 25 ms afterwards, with a *different* completion
  at `temperature: 0`.
- Qwen3-Omni server start 572 s; 135.1 GB resident of 150.1 GB.
- Qwen request fails with HTTP 404 on `/v1/chat/completions`.

**SOURCE-VERIFIED (vendor documents):**
- Qwen3-Omni's three stages and their device placement; "verified on 2x H100".
- The same two-device shape for Qwen2.5-Omni.
- `gpu_memory_utilization` is a fraction of whole device, and stages **sharing** a
  device must sum to ≤ 1.0.
- vLLM-Omni 0.28's process-scoped NVML estimation and 0.30's device-aware
  admission; a reproduced 0.28 device-lock inversion with `--stage-overrides`.
- `--no-async-chunk` is required for `/v1/realtime`.
- vLLM's published wheels are CUDA 13.0 at both 0.28 and 0.30, despite the
  documentation's prose about 12.9.

**INFERRED (labelled as such):**
- In the single-GPU Qwen runs, stage 1 pinned to a nonexistent device `1`
  probably collapsed onto device 0. The placement lines are measured; the
  collapse is inference.
- That the 404 is a route-name difference rather than a disabled endpoint.

**RETRACTED — do not publish:**
1. *"Qwen3-Omni needs more than 80 GB."* **Withdrawn.** I inferred a capacity wall
   from the tail of a truncated OOM. A 141 GB H200 failed identically, which is
   what disproved it. The cause was memory accounting, not capacity.
2. *"The deploy config's fractions sum to 1.6 of a single device, so it cannot
   fit."* **Withdrawn.** The fractions are per *device*: 0.9 on GPU 0, and
   0.6 + 0.1 = 0.7 on GPU 1. Both under 1.0. They only overcommit if all three
   stages are forced onto one card, which the shipped config never does.

Both retractions matter: the second had already been drafted into a tweet.

---

## What video/audio the reader can actually listen to

| file | what it is |
| --- | --- |
| `mixed.wav` | Moshi duplex: human track summed with model track, one timeline |
| `artifacts/exp1/response.wav` | modular sequential: 10.375 s, three sentences |
| `artifacts/paired/turn-NN-*.wav` | modular paired turns, sequential and overlapped |
| — | Qwen: **none**. No audio was produced. |

---

## Reproducing any of it

```
modal run modal_app.py::cache_weights        # Moshi, CPU
modal run modal_app.py::stream_session       # Moshi, A10
modal run modal_modular.py::cache_models     # modular, CPU
modal run modal_modular.py::paired_turns     # the paired overlap comparison, A10
modal run modal_omni.py::cache_model         # Qwen, CPU, ~70 GB
modal run modal_omni.py::omni_turn           # Qwen, 2 GPUs
```

`MODAL-GUIDE.md` carries the deployment lessons as a standalone piece: twelve
traps with the failure that produced each, plus the checklist to run before
writing any deployment code.

---

## Follow-ups, deliberately not done

- **The overlap slowdown's cause.** Needs the per-token timeline per turn, plus a
  read-and-discard arm so the reader and synthesiser compete without GPU work.
- **LLM warm-up and seeding.** Would give six clean turns instead of five plus an
  outlier.
- **Qwen request route.** `GET /openapi.json` on the running server, then post to
  the right path. This is the one remaining step to an audio artifact, and it is
  a few minutes of work on hardware that has already been proven to start.
- **Contention/republish check** on the overlapping pair, if the article wants
  the mechanism rather than the observation.
- **AWS parity** — frozen, never run: no `g5.2xlarge` Spot capacity in any
  `ap-south-1` zone offering the type. Recorded in `infra/aws/README.md`.

---

## The one-sentence version

Moshi puts the staging inside a single continuous model, the modular trio puts it
under the engineer's control, and Qwen3-Omni puts it in the vendor's deploy
config — two devices, three stages — which is a fact about the artifact you must
load rather than an inference from logs.
