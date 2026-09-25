# voice-inference-lab

An experimental repository for understanding voice-model inference and serving
from first principles. We expect to implement more than one voice architecture
over time, starting with Kyutai Moshi.

We generalize only after two implementations exist and we can see what is
genuinely shared. Every abstraction here should exist because a real problem
forced it, not because we anticipated needing it.

## Where we are

1. **Plumbing check** (`inspect_gpu`) — done. Proves a Modal container can see
   an A10 and compute on it from Python.
2. **Weights, cached then loaded** (`cache_weights`, `load_model`) — done.
   Downloads on CPU, loads on GPU, from one persistent cache.
3. **One 45 second session** — done. Proved the streaming loop works, and
   showed that Moshi speaks while the input is still playing. Its output,
   `outputs/session.wav`, is kept.
4. **Short duplex fixture, synchronized tracks** — done. Streams a fixed 8
   second window and writes `human.wav`, `moshi.wav`, and `mixed.wav` on one
   shared timeline. Steady state is 48.5 ms per frame against an 80 ms budget,
   with memory flat across both tested session lengths.
5. **Explicit warm-up** — done. Moved the one-time initialization cost off the
   first real frame, from 10,820 ms to 46 ms, without keeping a GPU warm.
6. **Sustained session past the context window** (`stream_session`) — current
   milestone. Runs 3400 frames, about 272 seconds, to find out whether memory
   stays bounded when a session crosses Moshi's 3000 frame window.

## Why the phases are split

The pinned checkpoint is ~15.8 GB (7.69B parameters at BF16). Downloading that
on a GPU container would bill A10 time to move bytes that need no GPU at all.
So acquisition and computation are separate invocations against one shared
Modal Volume.

```
modal run modal_app.py::cache_weights    # CPU container, no GPU. Downloads.
modal run modal_app.py::load_model       # A10. Loads the weights only.
modal run modal_app.py::stream_session   # A10. Runs one streaming session.
modal run modal_app.py::inspect_gpu      # A10. Environment check.
```

Running any of these spends money on your Modal account. A GPU command also
requires a valid payment method on file.

## The streaming session

Moshi is not an input/output API. It runs two continuous streams at once:
human audio flows in, and model audio flows out, on the same clock. It never
decides that a turn has ended.

The session matches that shape. One loop drives one frame counter. Frame `i`
holds the input chunk submitted at that point, and the output frame produced at
that point. After the clip ends, silence is streamed for `TAIL_SECONDS`
(currently 5), so the model has room to speak. The tail is our construction,
not the model's behaviour.

Each frame follows the official runtime exactly:

1. `mimi.encode` turns 1920 input samples into 8 codebooks.
2. `lm_gen.step` advances the Moshi language model by one frame.
3. `mimi.decode` turns the model's 8 output codebooks back into audio.

Moshi emits no output on the first frame. Its output streams are delayed
against its inputs. The report records the frame index where output first
appears, which is 1.

### The three tracks

Every file covers the whole session and the same span of time, so a timestamp
means the same moment in all three.

| File | Contents |
| --- | --- |
| `human.wav` | the input clip, in place, silence elsewhere |
| `moshi.wav` | the decoded output, placed one frame later |
| `mixed.wav` | the two tracks summed at half scale |

The one frame placement of `moshi.wav` is `OUTPUT_OFFSET_FRAMES`. It puts each
decoded frame where the model produced it. Set it to 0 if the mix sounds 80 ms
early.

`mixed.wav` cannot clip, because each track is halved before the sum. The
report records the resulting peak.

The loop synchronises once per frame and is unpaced. It pushes frames as fast
as the GPU accepts them, so the timing numbers are the model's own speed, not
simulated wall-clock time.

### Explicit warm-up

The first call through the model is far more expensive than later calls: it
pays CUDA graph capture and lazy kernel setup. Measured, that cost fell entirely
on the first frame, which is the worst place for it.

So the session now warms the model before any real audio arrives. It pushes
`EXPLICIT_WARMUP_FRAMES` synthetic silent frames through the full path, then
resets the streaming state and runs the real fixture.

Two constraints make this work, and both are easy to get wrong:

- The warm-up must share one streaming context with the real session. The CUDA
  graph wrappers are built when the context is entered, so a second context
  would rebuild them and throw the warm-up away.
- The reset must be `reset_streaming()`, which clears the offsets and the step
  counter while leaving the graph wrappers in place.

The reset is self-verifying. `first_output_at_frame` must be `1`, because the
first frame after a reset always yields no output. A `0` means the reset never
happened.

`EXPLICIT_WARMUP_FRAMES = 0` restores the previous behaviour.

### Sustained session

`SUSTAINED_FRAMES` sets the session length, currently 3400 frames, which is
272 seconds. Moshi's context window is 3000 frames, or 240 seconds, so the run
goes about 32 seconds past the wrap. The report reads the window size from the
model rather than trusting a hardcoded number.

The fixed 8 second clip is repeated to fill the session. Content is irrelevant
to a memory lifetime test; only the frame clock matters. The silence tail is
kept at the end.

Memory and latency are sampled once per frame:

| Column | Meaning |
| --- | --- |
| `frame` | index within the session |
| `ms` | that frame's compute time |
| `allocated_gb` | `torch.cuda.memory_allocated()` after the frame |
| `reserved_gb` | `torch.cuda.memory_reserved()` after the frame |
| `free_gb` | device free memory from `torch.cuda.mem_get_info()` |

The three memory queries run after the frame timestamp is taken, and the clock
restarts after them, so sampling cannot inflate the reported compute time. The
sampling does add to the wall clock, which is disclosed rather than hidden.

3400 rows do not belong in a console, so the full series goes to a CSV on the
Volume. The report carries only the windows that answer the question: startup,
before the wrap, at the wrap, after the wrap, and the growth between frame 100
and the final frame.

### The fixture

`bria.mp3` from the metavoice repository, which is the sample Kyutai reference
in their own `sphn` README. Phase 1 downloads it to the Volume and records its
SHA-256.

The file is about 45 seconds of narration, so the session takes a fixed 8
second window from the start. The sustained session repeats that window to fill
the frames. Both numbers are constants in `modal_app.py`.

It is narration, not a question. Moshi will speak while it hears speech, but it
is not answering anything. Expect a duplex exchange, not a conversation.

The clip is decoded with `sphn` and resampled to 24 kHz only when its own rate
differs. That resample is linear interpolation and therefore crude.

No automatic speech detection is included. A reliable detector needs a real
model, and an energy threshold is a guess. Audible onset stays a judgement you
make, and the shared timeline makes that easy.

### Retrieving the results

The session commits the Volume, so the results outlive the container. The short
milestone's verified tracks stay under `outputs/`. The sustained session writes
under `outputs/sustained/`, so nothing already verified is overwritten.

```bash
modal volume get voice-inference-lab-hf-cache outputs/sustained/series.csv .
modal volume get voice-inference-lab-hf-cache outputs/sustained/mixed.wav .
```

Use `--force` on the second and later downloads, because a local file of the
same name already exists.

## What persists, and what doesn't

**Persists:** the Volume's *committed* bytes. Phase 1 commits the weights and
the fixture. Phase 3 commits the generated audio. Both survive the container
indefinitely and are reused by later runs.

**Does not persist:** the container filesystem, GPU memory, the loaded model
when the container exits, in-memory Hugging Face state, and stdout beyond
Modal's run logs.

Phase 2 deliberately never commits, so the cached weights stay untouched.

## Cost shape

- **Phase 1** — CPU container time plus download egress. It creates the
  persistent-storage cost: about 15.8 GB, billed for as long as the Volume
  exists. Deleting the Volume is the only way to stop that.
- **Phases 2 and 3** — GPU seconds only. Phase 3 adds roughly a megabyte of
  stored audio.
- Every function runs with `min_containers=0`, so nothing stays warm.

## How we know Hugging Face was not contacted again

Watching logs is the weak proof: `huggingface_hub` makes cheap revision checks
even with a warm cache, and a floating ref can invalidate resolution. So the
guarantee is structural instead.

1. **Offline enforcement.** The loader image sets `HF_HUB_OFFLINE=1` and
   resolves with `local_files_only=True`. Offline, the library raises rather
   than calling out: all files present → resolves with zero network; anything
   missing → a loud error, which is the intended behaviour.
2. **Immutable pins.** The checkpoint is pinned to a revision SHA, so nothing
   upstream can make the cache stale.
3. **Corroboration.** Each phase prints the resolved snapshot paths, so you can
   see it read local files.

## Pins

| What | Pinned to | Why |
| --- | --- | --- |
| `kyutai/moshiko-pytorch-bf16` | `2bfc9ae6…` | immutable; ~15.8 GB; holds Moshi, Mimi, and the tokenizer; CC-BY-4.0 |
| `moshi` | `==0.2.13` | loader API and weight compatibility |
| `torch` | `==2.4.1` | `moshi` requires `torch<2.10,>=2.2.0` |
| Python | `3.12` | inside `moshi`'s `>=3.10,<3.15` |

`moshi` `0.2.13` requires `torch<2.10`, so torch is pinned here rather than
left to float. The `inspect_gpu` function intentionally floats torch, because
its job is to report what the platform hands out by default.

## Deliberately absent

No WebSockets, browser code, serving, endpointing, concurrency, quantization,
or second model. No benchmarking. They arrive when a real problem requires
them.

## Next

Listen to `mixed.wav`, then let the next real problem drive the next layer of
structure.
