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

## Observed by polling the platform

- **The container does scale to zero on its own.** One container was still
  listed active shortly after `session_end`, and it was gone a couple of minutes
  later, with no teardown from us. So the linger after disconnect is minutes,
  not zero and not indefinite. The exact duration was not instrumented, because
  this was polled by hand rather than measured.

## Container initialisation, investigated

One `server_ready` per container, and four of them in a single run. Separating
what is what:

**Measured.** Four initialisations in one `modal serve` session, with
`load_and_warmup_ms` of 26309, 29324, 20995 and 33901, and `warmup_ms` of 12421,
17357, 12864 and 19843. Three `/favicon.ico` requests reported wall durations of
44.3 s, 24.6 s and 24.8 s against handler executions of 378.8 ms, 187.1 ms and
188.3 ms. The page request reported 33.2 s against 65.0 ms. Four
initialisations, four HTTP requests, one WebSocket.

**Known from the platform.** `@modal.concurrent(max_inputs=...)` is what allows a
container to handle more than one input at a time. Without it a container serves
one input at a time, and we do not use it, deliberately, because concurrency is
out of scope. So while the WebSocket occupies a container, any other request
needs a different container, and a fresh container re-runs `create_app()` and
therefore reloads all 15.8 GB of weights.

**Inferred, not proven.** The four initialisations line up one-to-one with the
page load and the three favicon requests, and the favicon wall durations have the
shape of waiting on a container boot plus a model load. The mechanism above
explains the observation, but the counterfactual was never tested, so the
attribution is inference. A 404 on a favicon costing a GPU model load is the
kind of claim that deserves its own experiment before it goes in a comparison
table.

**Unknown.** Whether `modal serve` contributes anything of its own, being a dev
server that watches the working directory. Whether the ASGI app is rebuilt per
app revision separately from per container. And what the same run does under
`modal deploy`, which is the path the article would actually care about.

## Fixed in this milestone

- Container initialisation now writes its own durable record, one file per
  container, under `outputs/browser/init/`. The counting above can therefore be
  done from the Volume instead of read out of stdout, and a container that
  serves no session is still recorded, because it still cost.
- The session-end commit now uses `await runtime.commit.aio()` rather than
  blocking the event loop, which removes the `AsyncUsageWarning` Modal raised.

## Deliberately not done

Adding `@modal.concurrent` would collapse the extra containers, since a favicon
request would no longer need its own GPU container. It is out of scope for Moshi
v1, and it would change the very behaviour being measured.

## Scope

One session, one concurrent user, no batching, no autoscaling, no idle policy.
This says nothing about multiple users or long-running use.
