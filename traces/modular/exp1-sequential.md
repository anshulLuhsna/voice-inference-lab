# Experiment 1 — sequential modular baseline

Three models on one A10, in one container, with no overlap: Whisper transcribes
the whole clip, Qwen generates the whole reply, Kokoro synthesises the whole
reply, and only then is there audio. This is the baseline Experiment 2 changes.

Raw trace: `exp1-sequential.jsonl`.

## Fixture

A recorded prompt, synthesised once with the pinned Kokoro voice and then frozen.
Regenerating it per run would put synthesis inside the measurement window.

| | |
| --- | --- |
| `source_text` | Give me three practical tips for sleeping better, one short sentence each. |
| `model_repo` | `hexgrad/Kokoro-82M` |
| `model_revision` | `f3ff3571791e39611d31c381e3a41a3af07b4987` |
| `voice` | `af_heart` |
| `speed` | 1 |
| `generated_sample_rate` | 24000 |
| `duration_seconds` | 4.775 |
| `sha256` | `3550ae22b252aad67b4f5d5238ecc5a46d36ceed74c9afd6e4d54799cd55b706` |

The 16 kHz input Whisper needs is derived from that file rather than synthesised
again, so both rates are provably the same recording. 24000:16000 is exactly 3:2,
so no fractional rate is approximated. Measured: 114,600 samples at 24 kHz
(4.775 s) to 76,400 at 16 kHz (4.775 s).

## Result

**Transcript**, exact:

> Give me three practical tips for sleeping better, one short sentence each.

One segment, 0.000–4.240 s, language `en` at probability 1.0.

**Reply:**

> Establish a regular sleep schedule and stick to it. Avoid screens for at least
> an hour before bed. Create a relaxing bedtime routine to help you wind down.

**Spans**, server-side, one monotonic clock:

| span | ms |
| --- | --- |
| STT | 825.2 |
| LLM TTFT | 68.9 |
| LLM total | 542.8 |
| TTS first audio | 1689.6 |
| TTS total | 1704.0 |
| end to end | 3072.0 |

Output audio 10.375 s.

## What this run established for Experiment 2

Kokoro produced **one** chunk for the entire three-sentence reply, so first audio
and total synthesis are 14 ms apart: there is no incremental audio inside a
chunk. The reason is that `KPipeline` splits its input text on newlines
(`split_pattern=r'\n+'`) before its internal 510-phoneme chunker sees it, so a
single-paragraph reply is one chunk however many sentences it contains.

An overlapped policy therefore cannot be built by handing Kokoro the whole reply
and waiting. It has to feed one completed sentence at a time, with the sentence
boundaries found by the scheduler.

Model loading is excluded from the measured turn. `turn_start` sits 105,635 ms
into the process, so had loading been inside the turn the headline would have
read about 108 s instead of 3.07 s. Loading is reported separately.

## Pins

| | |
| --- | --- |
| Whisper weights | `dropbox-dash/faster-whisper-large-v3-turbo` @ `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` |
| `faster-whisper` | 1.2.1, `ctranslate2` 4.8.2, compute type `float16` |
| Qwen weights | `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| vLLM | 0.30.0+cu129, from the published CUDA 12.9 wheel |
| generation | `temperature` 0, `max_tokens` 160, fixed system prompt |
| Kokoro | 0.9.4, `hexgrad/Kokoro-82M` @ `f3ff3571…`, voice `af_heart` |
| torch | 2.13.0+cu129 |
| GPU | one A10, 23.685 GB, `gpu_memory_utilization` 0.40 |

## Artifacts

Generated files are gitignored; these paths are on the Volume and were copied to
`artifacts/exp1/` for listening.

- `artifacts/exp1/input.wav` — the fixture, copied in beside the output
- `artifacts/exp1/response.wav` — the synthesised reply, 24 kHz
- `artifacts/exp1/trace.jsonl` — the events
- `artifacts/exp1/sequential.json` — config, environment, derived spans
- `outputs/modular/exp1/vllm_startup.log` — the server's own log

## Limitations

- **Synthetic speech is cleaner than a real microphone.** This fixture is
  suitable for a controlled scheduling experiment and is **not** an ASR quality
  benchmark. The live browser run is the real-human-speech check.
- Kokoro produced the input, but that does not make Kokoro part of the measured
  input path. Fixture preparation happens once, offline, and its latency is
  excluded from every reported turn.
- Whisper transcribes digital silence as "Thank you." That is a property of the
  model, and it is why a real pipeline needs voice activity detection rather than
  an energy threshold alone.
- One run. No repetition, so no variance is available for these spans.
