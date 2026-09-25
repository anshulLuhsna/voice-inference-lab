# Experiment 3 — paired sequential vs overlapped

The comparison the first two experiments could not make. Both policies run
**alternating in one container**, models loaded once, both models' first-call
costs paid before any measured turn. The scheduling policy is then the only
thing that differs between the arms.

Three pairs, six turns. Raw trace: `paired-overlap-vs-sequential.jsonl`.

## Why this replaced the cross-run comparison

Experiments 1 and 2 ran in different containers, and the difference between them
showed up on work the policy cannot touch: transcription was **2.28×** slower in
the Experiment 2 container and time to first token **1.56×** slower. That is
larger than the effect being measured, so the cross-run delta could not be
attributed to anything.

Here the same metric spreads by **1.02×**. The comparison is now readable.

| | exp1/exp2 | paired |
| --- | --- | --- |
| STT spread | 2.28× | **1.02×** |

## Result

| turn | policy | STT | TTFT | LLM total | first audio | end to end | peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | sequential | 185.0 | 68.3 | 540.0 | 878.3 | 879.0 | 12.47 |
| 1 | overlapped | 184.5 | 25.8 | 597.8 | 510.4 | 890.3 | 12.47 |
| 2 | sequential | 188.1 | 24.7 | 496.9 | 830.6 | 831.2 | 12.49 |
| 3 | overlapped | 184.9 | 25.6 | 589.6 | 477.3 | 843.8 | 12.49 |
| 4 | sequential | 184.5 | 24.9 | 497.4 | 799.9 | 800.6 | 12.49 |
| 5 | overlapped | 186.9 | 25.0 | 590.2 | 478.6 | 844.6 | 12.49 |

**Time to first audio**, the metric the policy changes:

| | ms |
| --- | --- |
| sequential median | 830.6 |
| overlapped median | 478.6 |
| **median per-pair delta** | **+353.3 (42.4% earlier)** |
| per-pair deltas | +367.9, +353.3, +321.2 |

Positive means the overlapped policy produced audio earlier. All three pairs
agree in sign and are within 47 ms of each other.

**Total completion**, a different question:

| | ms |
| --- | --- |
| sequential median | 831.2 |
| overlapped median | 844.6 |
| delta | **+13.4 ms (+1.6%)** |

So: audio starts arriving about a third of a second sooner, and the turn as a
whole finishes about 13 ms later. Neither policy is "faster" without naming
which of those two is meant.

## Overlap is visible in the trace

From the overlapped arm's turn 1, turn-relative:

```
   184.6  llm_request_start
   347.5  llm_sentence_ready   clause 0
   347.5  tts_clause_start     clause 0
   510.4  tts_first_audio      clause 0   <- while generation is still running
   549.8  tts_clause_start     clause 1
   709.9  tts_first_audio      clause 1   <- still running
   764.2  tts_clause_start     clause 2
   782.4  llm_done                        <- generation ends here
   888.4  tts_clause_done      clause 2
```

Generation ran 184.6 → 782.4 ms. Synthesis ran 347.5 → 888.4 ms with no gaps.
The two overlap for **435 ms**, spanning all of clauses 0 and 1.

## The cost, and what this harness cannot say about it

`llm_total` is consistently higher under overlap: 589.6 and 590.2 ms against
496.9 and 497.4 ms, a **+93 ms (+18.7%)** effect, reproducible across all three
pairs.

**It is not attributable from this data.** The reader thread stamps `llm_done`
when it sees the stream end, so the inflation could be either:

- **GPU contention** — Kokoro kernels sharing the A10 slow vLLM's decoding, so
  the server genuinely finished later; or
- **client-side scheduling** — the main thread is running Kokoro forward passes
  and the reader thread does not get scheduled promptly, so the server finished
  on time and we observed it later.

Both are consistent with the numbers. Separating them needs the per-token arrival
timeline recorded per turn, which this run did not keep, and ideally a third arm
that reads the stream and discards the sentences, so the reader and synthesiser
compete without any GPU work. Until then this is a measured difference with an
unestablished cause.

Whatever it is, it does not reach the user: total completion moved by 13 ms.
Overlap hides the extra LLM time behind synthesis that had to happen anyway.

## Findings that affect future runs

**Warming Whisper and Kokoro is not enough.** The first measured turn shows a
time to first token of 68.3 ms against ~25 ms for every turn after it, and it
also produced a *different* third sentence. Turns 1–5 all produced the identical
reply; only turn 0 differs:

- turn 0: `...Create a relaxing bedtime routine to help you wind down.`
- turns 1–5: `...Create a comfortable, dark, and quiet sleep environment.`

So the first LLM request after a cold vLLM is both slower and different, and it
landed on a sequential arm, inflating pair 0's delta. Pairs 1 and 2, with both
arms warm, give +353.3 and +321.2 ms.

**`temperature: 0` is not reproducibility.** The same prompt at temperature 0
produced two different completions in the same session. Argmax over logits that
differ in the last bits across kernel paths will flip a near-tie. A `seed` was
not passed; passing one is the obvious next attempt, and if it does not fix it,
the honest conclusion is that the third clause is a near-tie and the text must be
pinned rather than the sampling.

## Consequences for the measurement

Clause 0's text is identical in every turn, so the primary metric — time to first
audio, which clause 0 drives — is a like-for-like comparison. The later clauses
are not, in turn 0 only.

## Fixture and pins

Unchanged from Experiment 1; see
[`exp1-sequential.md`](./exp1-sequential.md). Same fixture SHA, same models, same
prompt, same generation parameters, same voice, same GPU. Warm-up: Whisper
784 ms, Kokoro 1676 ms, outside every measured turn.

## Artifacts

- `artifacts/paired/turn-NN-<policy>.wav` — one per turn
- `artifacts/paired/trace.jsonl`, `artifacts/paired/paired.json`
- `outputs/modular/paired/vllm_startup.log`

## Limitations

- Three pairs, one session, one container. Enough to show a consistent sign and
  a range; not a distribution.
- The contention question above is unresolved.
- The first LLM request of a cold session is not representative and is included
  in the numbers.
- Synthetic input speech; not an ASR quality benchmark.
