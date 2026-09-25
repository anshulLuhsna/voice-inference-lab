# Experiment 2 — overlapped turn

One policy changed from Experiment 1: **synthesis begins at each completed
sentence instead of after the whole response.** Everything else is held fixed —
same fixture, same Whisper, same transcript, same prompt and generation
parameters, same Kokoro voice, same GPU, same container shape.

Raw trace: `exp2-overlap.jsonl`.

## Outcome: failed acceptance criterion 4

| # | Criterion | Result |
| --- | --- | --- |
| 1 | Same final transcript | **PASS** — exact match |
| 2 | Bounded three-sentence answer | **PASS** — three sentences, same text as Experiment 1 |
| 3 | Sentence 1 synthesised before `llm_done` | **PARTIAL** — synthesis *started* 337 ms before `llm_done`, finished 2.5 s after |
| 4 | First audio earlier than sequential | **FAIL** — 4,947.7 ms against 3,057.6 ms, i.e. 1.89 s later |
| 5 | Sentence audio concatenates intelligibly | **PASS** — 10.8 s, no clipping, round trip matches |
| 6 | Trace proves which operations overlapped | **PASS** |

**This run does not show that overlap is worse. It shows the comparison is not
attributable**, for two reasons below. The machinery works; the measurement is
confounded.

## What the overlap actually did

Turn-relative, all on one monotonic clock:

```
    0.0  turn_start
 1880.1  stt_final / llm_request_start
 1987.4  llm_ttft
 2108.0  llm_sentence_ready      clause 0
 2108.1  tts_clause_start        clause 0   <- 0.1 ms after it was ready
 2261.5  llm_sentence_ready      clause 1
 2427.7  llm_sentence_ready      clause 2
 2445.4  llm_done
 4947.7  tts_first_audio         clause 0   <- synthesis is still running
 4959.6  tts_clause_done         clause 0
 4959.6  first_audio_sent        clause 0
 ...
 5261.3  turn_done
```

The overlap is real and the trace proves it: clause 0's synthesis began 337 ms
before the last token arrived, and both later sentences were already queued
before `llm_done`.

Per clause:

| clause | text | ready | tts start | tts ms |
| --- | --- | --- | --- | --- |
| 0 | Establish a regular sleep schedule and stick to it. | 2108.0 | 2108.1 | **2851.5** |
| 1 | Avoid screens for at least an hour before bed. | 2261.5 | 4959.6 | **145.0** |
| 2 | Create a relaxing bedtime routine to help you wind down. | 2427.7 | 5104.6 | **133.3** |

## Cause 1: Kokoro's first call dominates time-to-first-audio

Clause 0 took 2,851 ms. Clauses 1 and 2, doing the same kind of work, took 145 ms
and 133 ms. That is a 20× difference for identical work.

Clause 0 is the first Kokoro synthesis in the container, so it pays CUDA graph
capture and lazy kernel initialisation — the effect the Moshi work already
measured, where the first call through the model cost 10,820 ms and later calls
cost 46 ms. Only 337 ms of clause 0's 2,851 ms overlapped with the LLM, so
contention cannot account for it.

**Time-to-first-audio therefore measures Kokoro's cold start, not the scheduling
policy.** Both experiments paid it.

## Cause 2: the two containers are not comparable

Policy-independent metrics, Experiment 2 relative to Experiment 1:

| metric | exp1 | exp2 | ratio |
| --- | --- | --- | --- |
| STT | 825.2 ms | 1880.1 ms | **2.28×** |
| LLM TTFT | 68.9 ms | 107.3 ms | **1.56×** |
| LLM total | 542.8 ms | 565.3 ms | 1.04× |
| Whisper load | 1378 ms | 1737 ms | 1.26× |
| Kokoro load | 2615 ms | 3833 ms | 1.47× |
| device peak | — | 12.389 GB | — |

The scheduling policy cannot touch STT or TTFT, yet both are substantially
slower. The Experiment 2 container was uniformly slower, so the cross-run delta
cannot be attributed to the policy. This was flagged as a caveat when the
comparison was written; the magnitude makes it disqualifying rather than a
footnote.

## No measurable contention on the LLM

The cost the overlap was expected to risk did not appear. Token arrival cadence
stayed at roughly 14–15 ms per chunk straight through the synthesis window, with
no stall when Kokoro was running, and `llm_total` moved from 542.8 to 565.3 ms
(+4%).

That is a weak result rather than a reassuring one: the overlap window was only
337 ms, because the LLM finishes in about 565 ms while a Kokoro call needs
1,700 ms or more. **The benefit of overlap on this workload is capped at roughly
the LLM's own duration**, which is small against a TTS-dominated turn. There is
little overlap available to measure.

## Concatenation check

- Clause durations 3.55 + 3.15 + 4.10 s sum to exactly `response_seconds` 10.8 s.
- Peak amplitude 0.518 — no clipping.
- The concatenated audio re-transcribed through Whisper returns the reply text
  exactly, so no word was dropped, duplicated or reordered at a clause boundary.

That last check is post-turn verification. It is not part of any measured span,
and it is not an ASR quality judgement.

## Pins

Identical to Experiment 1, which is the point. See
[`exp1-sequential.md`](./exp1-sequential.md). Sentence rule:
`([.?!])(\s|$)` — punctuation followed by whitespace, plus whatever remains when
the stream ends. No tokenizer was added.

## Artifacts

- `artifacts/exp2/input.wav`, `response.wav`, `clause-000.wav` … `clause-002.wav`
- `artifacts/exp2/trace.jsonl`, `artifacts/exp2/overlap.json`
- `outputs/modular/exp2/vllm_startup.log`

## Defects in this run's report

Found while reading the result and fixed for the next run:

- The reply text was never recorded, so the round-trip comparison was testing a
  good transcript against an empty string and always reported a mismatch.
- The clause timeline was expressed against process start while `timings_ms` used
  `turn_start`, so two sets of numbers describing the same events appeared
  unrelated.
- The trace header described its offsets as relative to `turn_start` when they
  are relative to the start of the operation.

## Limitations

- One run, and not a controlled A/B. The head-to-head comparison must be run in
  **one container with the models loaded once**, so that the only difference is
  the policy.
- Kokoro's first call must be paid before the measured turn, or reported
  separately, or it will keep dominating the first-audio metric.
- Synthetic input speech; not an ASR quality benchmark.
- The sentence rule is deliberately simple and has only been exercised on this
  fixture.
