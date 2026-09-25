# Moshi v1 — handoff for the article

Self-hosted, full-duplex speech inference on Modal, measured end to end. This is
the evidence package, not prose for the article. Every number carries its source,
and section 11 separates what was measured from what was inferred.

Two harnesses drove the same model on the same GPU class:

- **offline**: a fixed 8 s audio fixture pushed through the loop as fast as the
  GPU accepts it
- **live**: a browser microphone over a WebSocket, driving the same loop

They exist so that the live path can be checked against a harness with no
network and no browser in it.

## 1. Final architecture

```
offline harness                          live harness
fixed 8 s fixture                        browser microphone
  repeated to fill the session             AudioWorklet, 24 kHz context
        |                                        |
        |                                   WebSocket, binary frames
        |                                   kind byte + int16 PCM
        |                                   1920 samples = 80 ms
        |                                        |
        +----------------> Modal ASGI app <-------+
                                |
              mimi.encode -> lm_gen.step -> mimi.decode
                     (one 80 ms frame per iteration)
                                |
        +-------------------+----+
        |                        |
  three WAV tracks          PCM back over the socket
  human/moshi/mixed               |
  via the Volume             AudioBufferSourceNode scheduling
                                  |
                            browser playback
```

The offline harness writes three full-length tracks on one shared timeline, so a
timestamp means the same moment in each. The live harness is deliberately not
Opus: raw int16 PCM at 48 KB/s keeps every browser frame a 1:1 match for one
`mimi.encode` call, which is what makes the live timings comparable to the
offline ones.

## 2. Configuration and pins

| | |
| --- | --- |
| GPU | A10G, requested as `gpu="A10G"`, driver-reported as `NVIDIA A10` |
| Compute capability | 8.6 |
| Device memory | `nvidia-smi` reports 23028 MiB; torch reports 23.7 GB decimal |
| Python | 3.12, pinned |
| PyTorch | `torch==2.4.1`, wheel `2.4.1+cu121` |
| CUDA runtime | 12.1 |
| moshi | `moshi==0.2.13` |
| Checkpoint | `kyutai/moshiko-pytorch-bf16` @ `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289` |
| Weight files | `model.safetensors`, `tokenizer-e351c8d8-checkpoint125.safetensors`, `tokenizer_spm_32k_3.model` |
| Parameters | Moshi 7,687,729,152; Mimi 79,308,609 |
| Storage | Modal Volume `voice-inference-lab-hf-cache` at `/cache` |
| Offline enforcement | `HF_HUB_OFFLINE=1` on the load and stream images |
| Lifecycle | `min_containers=0` on every function |
| Web framework | `fastapi` on the live image only |

The download phase deliberately has no torch and no CUDA, so moving 15.8 GB of
weights is never billed as GPU time.

The AMI-based AWS path exists in the repo but is frozen and produced no
measurements; see `infra/aws/README.md`.

## 3. Load, warm-up, first frame, steady state

Load, per component:

| Component | init | VRAM after |
| --- | --- | --- |
| Mimi | 680 ms | 0.391 GB |
| Moshi | 10,939 ms | 15.77 GB |

Warm-up, being the cost of pushing `EXPLICIT_WARMUP_FRAMES` (5) silent frames
through the full path. It is **not a constant**, and that is now measured across
twelve container initialisations:

| Condition | Warm-up |
| --- | --- |
| Offline, three solo runs | 12,283 / 12,452 / 13,225 ms |
| Live run 1, four initialisations | 12,421 / 12,864 / 17,357 / 19,843 ms |
| Live run 2, five initialisations | 15,091 / 17,788 / 18,943 / 21,368 / 24,960 ms |

Solo, the figure is tight, 12.28 to 13.23 s. Under concurrent container
initialisation it ranged **12.42 to 24.96 s**, and in run 2 no initialisation
came in under 15.09 s.

**Interpretation, and it is only an interpretation.** Five containers each
loading 15.8 GB of weights and capturing CUDA graphs while sharing one host is a
plausible cause and it fits the shape of the numbers. It has not been tested. If
"warm-up is about 12.4 s" is going in the article, it needs the qualifier "solo,
with one container initialising".

First real frame:

| Condition | First real frame |
| --- | --- |
| Without explicit warm-up | 10,820 ms |
| With explicit warm-up | 46.0 ms |

Steady state per frame:

| Run | mean | median | max |
| --- | --- | --- | --- |
| Offline, duplex run (post-warm-up) | 48.5 ms | 48.5 ms | 49.6 ms |
| Offline, sustained run | 48.4 ms | 48.4 ms | 49.4 ms |
| Offline, sustained, before the wrap | 48.9 ms | — | 53.8 ms |
| Offline, sustained, after the wrap | 49.4 ms | — | 51.1 ms |
| Live browser run 1 | 49.3 ms | — | — |
| Live browser run 2 | 53.4 ms | — | — |

**The live figure is one frame per run, not a steady-state average.** The live
path records the first post-warm-up frame only, so those two numbers are single
samples, and they differ by 8%. The offline columns are means over 156 to 3,394
frames and are the reliable steady-state measurement. No steady-state series
exists for the live path; producing one needs per-frame instrumentation on the
live side.

**Interpretation, not a universal claim.** The live microphone and WebSocket path
did not materially change the GPU compute cost per frame. Two independent
harnesses, one with a fixture and one with a browser and a network in the path,
produced per-frame numbers of the same order, and the offline steady-state range
covers the first live sample. This says nothing about browsers or WebSockets in
general, and the live sample is too small to claim equality.

## 4. VRAM

| Measure | Value |
| --- | --- |
| Moshi weights | 15.77 GB |
| Mimi | 0.39 GB |
| Peak allocated | 17.642 GB |
| Peak reserved | 17.922 GB |
| Minimum device free | 5.45 GB |
| Device total | 23.68 GB decimal, 22.07 GiB, same silicon |

Peak allocated, peak reserved and minimum free are **identical to three decimal
places across three runs** of very different lengths: 162 frames, 162 frames, and
3400 frames. The inference footprint is set at streaming-state initialisation and
does not vary with session length.

The raw byte count from the device is the correct cross-platform comparison.
`23.68 GB` decimal and `22.07 GiB` are the same number, and comparing the
rendered strings would wrongly suggest two different GPUs.

## 5. Full-duplex behaviour, and what an output frame is

Moshi is not an input/output API. It runs two continuous streams on one clock:
audio in, audio out, with a text stream alongside. It never decides a turn has
ended, which is why the offline session appends a fixed silence tail.

- One loop iteration consumes one 1920-sample frame and produces, once warm, one
  1920-sample frame of model audio.
- **The first iteration returns nothing.** Moshi's output streams are delayed
  against its inputs; the model's own `delays` reach a maximum of 1, and the
  step returns `None` until the offset exceeds that. Output therefore begins one
  frame, 80 ms, after the session starts. Measured: `first_output_at_frame = 1`
  on every run.
- An output frame is the model's own audio channel: `tokens[:, 1:]` — 8 codebooks
  — decoded by Mimi into 1920 samples. `tokens[:, 0]` is a separate text stream.
- In the offline harness this is preserved as three tracks on one timeline,
  `human.wav`, `moshi.wav` and `mixed.wav`, with the model track placed at
  `OUTPUT_OFFSET_FRAMES = 1` so each decoded frame lands where the model produced
  it.

**Verified by ear, not by number:** `mixed.wav` reads as a duplex conversation,
which is also what confirmed that the one-frame placement is correct.

## 6. Sustained session across the context boundary

One session of 3400 frames, 272 seconds, crossing the model's 3000-frame window:

| Frame | allocated | reserved | free |
| --- | --- | --- | --- |
| 0 | 17.395 GB | 17.738 GB | 5.58 GB |
| 100 | 17.395 GB | 17.738 GB | 5.58 GB |
| 2999 | 17.395 GB | 17.738 GB | 5.58 GB |
| 3000 | 17.395 GB | 17.738 GB | 5.58 GB |
| 3001 | 17.395 GB | 17.738 GB | 5.58 GB |
| 3399 | 17.395 GB | 17.738 GB | 5.58 GB |

Growth from frame 100 to the last frame: **exactly 0.0** on all three measures.
Latency crosses the boundary continuously: 48.9 ms mean before, 49.4 ms after,
with no step at 3000.

**Interpretation.** The memory is not merely bounded, it is preallocated. The
transformer's context cache is allocated at full context size when streaming
state is initialised, during warm-up, which is why the value at frame 0 already
equals the value after warm-up. Operationally it behaves as a ring: at 3000
frames it overwrites its oldest slot rather than growing.

## 7. Browser experiment

Session `8d77f5f0`, trace at `traces/session-8d77f5f0.jsonl`.

| Event | server_ms | Note |
| --- | --- | --- |
| `ws_connected` | 63.9 | socket accepted |
| `first_mic_audio` | 483.1 | 3840 bytes = 1920 int16 samples |
| `first_model_step` | 793.2 | first non-`None` from the step |
| `first_model_audio` | 793.6 | first decode |
| `first_chunk_sent` | 793.8 | this frame took **49.3 ms** |
| `client_playback_started` | 1120.6 | browser's own clock: 44891 ms |
| `disconnect` | 175093.8 | client-initiated, code 1005 |
| `session_end` | 175104.5 | |

Server-clock spans only:

- `first_mic_audio` to `first_chunk_sent`: **310.7 ms**. This window contains the
  deliberately discarded first frame plus two more frames of accumulation. It is
  a server-clock span, **not** audible-response latency.
- Socket open to first mic frame: 483.1 ms. That is the browser's audio graph
  starting, not our code.
- Socket usable: **175 s**, ended by the client. **No platform timeout was
  encountered.**

**Clock limitation, preserved deliberately.** `server_ms` and `browser_ms` are
unrelated clocks and the offset has never been measured. No browser end-to-end
latency has been computed by subtracting them, and none is claimed. The 326.8 ms
between first chunk sent and the server *hearing* the playback notification is a
server-clock interval measuring arrival of a message, not when sound left a
speaker.

The first-frame discard was **not audible** in the observed run. This is a
subjective report from the speaker, not an instrument reading.

### Second session, the verification run

Session `06271fc0`. The same shape, independently:

| Event | server_ms |
| --- | --- |
| `ws_connected` | 10.8 |
| `first_mic_audio` | 370.2 |
| `first_model_step` | 675.1 |
| `first_model_audio` | 676.3 |
| `first_chunk_sent` | 676.9, that frame 53.4 ms |
| `client_playback_started` | 983.4 |
| `disconnect` | 83540.1 |
| `session_end` | 83559.4 |

Server-clock spans: `first_mic_audio` to `first_chunk_sent` **306.7 ms**, and the
socket was usable **83.5 s**, again ended by the client with no platform timeout.
Two independent sessions put that span at 310.7 and 306.7 ms, which is the kind
of agreement that makes a number worth quoting.

The run also produced a **15 ms session**, `9d36ec40`: socket opened at 7.6 ms and
closed at 22.2 ms. A connection that arrives and leaves immediately, recorded
rather than filtered out.

## 8. Container and session lifecycle

Container lifecycle is distinguishable from session lifecycle only because
initialisation now writes its own durable record.

- `create_app()` runs **once per container**, not per connection. The model is
  loaded once per container and reused by connections to it.
- **Four initialisations** in the first browser session and **five** in the
  second, each session being one page load and one live socket.
  `load_and_warmup_ms` ran 20995 to 33901 in run 1 and 23545 to 38264 in run 2,
  and every initialisation carried a distinct `MODAL_TASK_ID`.
- The page request reported 33.2 s wall against 65.0 ms of handler execution, and
  three `/favicon.ico` requests reported 44.3 s, 24.6 s and 24.8 s against 378.8,
  187.1 and 188.3 ms.
- **Source-verified:** `@modal.concurrent(max_inputs=...)` is what allows one
  container to handle more than one input. It is not used, so a container serves
  one input at a time.
- The socket held the container: the container stayed alive for the session and
  scaled to zero on its own some minutes after disconnect, with no teardown from
  us. `min_containers=0` does not apply while a request is being served.

## 9. Limitations and unknowns

- **No browser end-to-end latency.** Clocks unrelated, offset unmeasured.
- **Capture at 24 kHz is not proven by the server trace.** The server only sees
  1920-sample frames, and a 48 kHz context would produce the same 3840-byte
  payloads. It is supported by client behaviour and a loud warning that never
  fired, nothing more.
- **The exact container linger time** was polled by hand, twice, not
  instrumented.
- **The multi-container cause is inferred, not tested.** See section 11.
- **The live per-frame figure is a single frame per run.** Two runs gave 49.3 ms
  and 53.4 ms. There is no live steady-state series, so no live steady-state
  average exists to compare against the offline one.
- **`container` is missing from the stdout init line.** The durable init record
  carries it; the printed line does not, because it is emitted before the field
  is set. The two artifacts therefore disagree, and the files are the authority.
- **Warm-up varies by a factor of two** across initialisations and the cause is
  unattributed.
- **Raw device bytes were never captured on Modal.** The probe was extended to
  report them, but that revision has not been run, so device memory for Moshi v1
  rests on `23028 MiB` and `23.7 GB` from earlier runs.
- **Model load time varied 8.0 s to 15.1 s** across identical runs and is
  unattributed. It is the largest single cost in a run.
- **Linear resampling** of the 44.1 kHz fixture to 24 kHz is crude and its effect
  on output quality is unmeasured.
- **One concurrent user, one session.** Nothing here speaks to multiple users,
  and concurrency was deliberately not added.
- **Sustained memory flatness is proven to 272 s**, not indefinitely.
- No quantisation, no SDPA/better attention, no optimisation of any kind was
  attempted.

## 10. Reproducible commands and artifact paths

```bash
modal run modal_app.py::inspect_gpu      # environment probe
modal run modal_app.py::cache_weights    # CPU only, downloads to the Volume
modal run modal_app.py::load_model       # load timing and memory
modal run modal_app.py::stream_session   # offline harness, 3400 frames
modal serve modal_app.py                 # live browser session
```

Retrieve artifacts:

```bash
modal volume get voice-inference-lab-hf-cache outputs/sustained/series.csv .
modal volume get voice-inference-lab-hf-cache outputs/browser/session-<id>.jsonl .
modal volume get voice-inference-lab-hf-cache outputs/browser/init/ .
```

In the repository:

| Path | What |
| --- | --- |
| `moshi_experiments.py` | the offline experiment logic, provider-free |
| `moshi_browser.py` | the live ASGI session |
| `browser/index.html` | capture and playback page |
| `modal_app.py` | the Modal shim: images, Volume, wrappers |
| `traces/session-8d77f5f0.jsonl`, `session-06271fc0.jsonl` | the two live session traces |
| `traces/session-9d36ec40.jsonl` | the 15 ms connection |
| `traces/init/` | one durable init record per container, five files from run 2 |
| `traces/README.md` | the first session's measurement summary |
| `session.wav`, `human.wav`, `moshi.wav`, `mixed.wav` | audio evidence |
| `infra/aws/`, `aws/` | the frozen AWS path, no measurements |

Commits, in order: `ff6518a` plumbing probe, `8df17be` pinned weight cache,
`08056a1` A10 load, `8c696da` streaming path, `a447421` duplex tracks,
`3d7518e` explicit warm-up, `c1caa8b` context boundary, `c208179` provider split,
`c7ccaa0` live session, `71b94a4` browser page, `83befe6` Modal serving,
`3b3b69e` first session evidence, `99bbcd4` durable init traces.

## 11. Claim ledger

**MEASURED** — our own instrumentation, our own runs.

- Steady per-frame compute: offline 48.4–49.4 ms, as means over 156 to 3,394
  frames; live 49.3 ms and 53.4 ms, each a single frame.
- Warm-up: 12,283 to 13,225 ms solo across three offline runs, and 12,421 to
  24,960 ms across nine live container initialisations.
- Model load: Mimi 680 ms, Moshi 10,939 ms.
- First real frame: 10,820 ms unwarmed, 46.0 ms warmed.
- Peak allocated 17.642 GB, peak reserved 17.922 GB, minimum free 5.45 GB.
- Sustained 3400 frames: memory growth exactly 0.0; latency 48.9 ms before the
  wrap, 49.4 ms after.
- Live sessions: 49.3 and 53.4 ms for the first post-warm-up frame; 310.7 and
  306.7 ms from first mic frame to first chunk sent, server clock; sockets usable
  175 s and 83.5 s; no platform timeout in either.
- Four container initialisations in one browser session, five in another, with
  their load and warm-up costs and their distinct container ids.
- Container scaled to zero after disconnect.

**SOURCE-VERIFIED** — read from pinned upstream source, not measured by us.

- Output delay is one frame: `delays` reach a maximum of 1, and the step returns
  `None` until the offset exceeds it.
- Model context is 3000 frames.
- The repository, weight filenames and tokenizer filename the loader expects.
- `moshi 0.2.13` declares `torch<2.10,>=2.2.0`.
- Streaming state is built when the streaming context is entered, and
  `reset_streaming` preserves the graph wrappers.
- `@modal.concurrent(max_inputs=...)` is what permits more than one input per
  container.
- Kyutai's own server discards the first inbound frame.

**INFERRED** — our reasoning over measurements and source, not directly tested.

- The context cache is preallocated and behaves as a ring, which is why memory is
  flat across the boundary.
- The four initialisations correspond one-to-one with the page request and the
  three favicon requests.
- The warm-up moves one-time work rather than removing it; total work is
  conserved, about 290 ms more across the session.
- The open socket, not the open page, is what holds the container.

**HYPOTHESIS** — not tested at all.

- That a `/favicon.ico` 404 cost a full GPU model load. This is the causal story
  that fits, and it is exactly the kind of claim that deserves its own experiment
  before it enters a comparison table.
- That concurrent container initialisation is what inflates warm-up from about
  12.4 s to as much as 25 s. Nine live initialisations fit the story, three solo
  offline ones are tight, and the cause is untested.
- That capture ran at 24 kHz.
- That the 0.5 ms latency difference across the wrap is drift rather than a wrap
  effect.
- That memory stays flat beyond 272 s.
- That any of this holds for more than one concurrent user.
