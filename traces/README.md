# Browser voice loop — first successful session

Session `8d77f5f0`, 2026-09-25. One person, one browser, one GPU, one socket.
Modal `serve`, A10G, the pinned runtime (torch 2.4.1+cu121, CUDA 12.1).

The raw trace is `session-8d77f5f0.jsonl` beside this file. Success criterion was
audible: the speaker spoke into the browser and heard Moshi answer from the
model we host on Modal.

## Measured, from the durable trace

| Event | server_ms | Note |
| --- | --- | --- |
| `ws_connected` | 63.9 | socket accepted, model already resident |
| `first_mic_audio` | 483.1 | 3840 bytes = 1920 int16 samples = one 80 ms frame |
| `first_model_step` | 793.2 | first non-`None` from `lm_gen.step` |
| `first_model_audio` | 793.6 | first `mimi.decode` output |
| `first_chunk_sent` | 793.8 | that frame took **49.3 ms** |
| `client_playback_started` | 1120.6 | browser reported its own clock: 44891 ms |
| `disconnect` | 175093.8 | client-initiated, code 1005 |
| `session_end` | 175104.5 | state reset, log flushed, commit |

Deltas, computed only within the server clock:

- first mic frame to first chunk sent: **310.7 ms**. That window contains the
  deliberately discarded first frame plus two more frames of accumulation.
- step to decoded audio: **0.6 ms**.
- socket lifetime: **175 s**, ended by the client. No platform timeout was hit.
- per-frame compute on the live path: **49.3 ms**, against **48.4–49.0 ms**
  measured offline. The browser path costs the same per frame as the fixture
  path, which is the point of using 1920-sample frames.

## Observed from the stdout log, not yet durable

`server_ready` is printed to stdout only, so these are not in the JSONL:

- It fired **four times** in one run: `load_and_warmup_ms` of 26309, 29324,
  20995, 33901 and `warmup_ms` of 12421, 17357, 12864, 19843.
- The first `GET /` reported `duration: 33.2 s, execution: 65.0 ms`. A 65 ms
  handler behind a 33 s wall clock: the first visitor waits the container cold
  start and the model load.
- Warm-up on the live path, 12.4 s, against 12.28 s and 12.45 s measured
  offline. Two independent paths agree within about 1%, so the warm-up is a
  property of the model and the GPU rather than of the harness.

## Not measured, and not claimed

- **Browser end-to-end latency.** `server_ms` and `browser_ms` are different
  clocks and the offset has never been measured. No cross-clock subtraction has
  been done and no end-to-end figure is stated.
- **Whether capture actually ran at 24 kHz.** The server only sees 1920-sample
  frames; a 48 kHz context would produce the same 3840-byte payloads. The page
  warns if the browser refuses 24 kHz, but the trace cannot prove which happened.
- **Whether the discarded first frame is audible.** Subjective; the speaker
  reported the interaction working.
- **How long the container lingers after disconnect.** A container was still
  listed active several minutes after `session_end`. It was not watched to zero,
  so no number is claimed.

## Findings to act on later, not now

- `server_ready` should also be written to the session log, so container
  initialisation becomes part of the durable artifact instead of stdout only.
- `runtime.commit()` is called from an async context. Modal warns
  `AsyncUsageWarning` and suggests `await runtime.commit.aio()`. Harmless here,
  worth correcting.
- **Four container initialisations in a single session** means the model load is
  paid per container, not per session. Each container holds ~17.9 GB reserved.
  This is the first real input to the idle-cost question, and it is larger than
  expected.

## Scope

One session, one concurrent user, no batching, no autoscaling, no idle policy.
This says nothing about multiple users or long-running use.
