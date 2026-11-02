# Design notes

Why serve-bench is built the way it is. Each section is a decision and the reason for it.

## Raw `/v1/completions`, not chat

All three servers expose the OpenAI completions endpoint. Using it instead of
`/v1/chat/completions` means:

- no chat template, so the prompt that reaches the model is exactly the text I built,
  and I control the prompt length;
- no "thinking" phase from Qwen3: the chat template would turn it on, the output would
  then split into `reasoning_content` and `content`, and the servers stream those two differently.

The cost: template rendering is not measured. On a 1.7B model that is noise next to prefill.

## Synthetic prompts from common words

Prompts are random sequences drawn from 142 common English words. Nearly every one is a single Qwen3 token
with its leading space, so N words come to about N tokens. The real count comes from the server's
`usage.prompt_tokens` and goes into every result: 512 words measured as 523 tokens on llama.cpp.
Prefill and decode cost do not depend on what the text means (no speculative decoding is enabled
anywhere), so natural text would add nothing but a dataset dependency.

Every run starts its prompts with a fresh random nonce. Without it a server would reuse KV cache
left over from the previous run with the same seed. That is a real bias: llama.cpp keeps slot
caches between requests, and mlx-lm and LM Studio have prompt caches too. In the prefix
experiment the shared prefix comes right after the nonce, so it is shared within a run and never
across runs.

## Fixed output length

Throughput and TPOT comparisons are only fair if every server generates the same number of
tokens. llama.cpp and LM Studio honour `ignore_eos`; mlx_lm.server does not. A random-word prompt
gives the model nothing to finish, so early EOS is rare. Each result stores
`mean_output_tokens`, so an early stop would show up there. TPOT always divides by the
server-reported `completion_tokens`, not by the number of SSE chunks: a server that packs several
tokens into one chunk would otherwise look slower per token.

## Metric definitions

They follow vLLM's `benchmark_serving`, so the numbers can be compared with published ones:

| metric | definition |
|---|---|
| TTFT | request sent → first non-empty text chunk |
| TPOT | (E2E − TTFT) / (output tokens − 1), per request, then percentiles over requests |
| ITL | gap between consecutive chunks, pooled over all requests |
| E2E | request sent → end of stream |
| throughput | output tokens of successful requests / (last end − first send) |
| goodput | requests meeting **both** TTFT ≤ 1 s and TPOT ≤ 50 ms, per second of the run (as in DistServe) |

Failed requests count against SLO attainment, but they are left out of the latency percentiles.

SLO choice: 1 s TTFT is the point where a chat UI starts to feel stuck. 50 ms TPOT is 20 tokens/s,
about 3× human reading speed. Both are CLI flags and are stored in every result.

## Closed loop and open loop

- **Closed loop** (`--pattern closed --concurrency C`): C users, each sends its next request
  when the previous one finishes. This is the right tool for a throughput-vs-batch-size curve.
  It hides queueing, though: when the server slows down, users slow down with it (coordinated
  omission).
- **Open loop** (`constant`, `poisson`, `burst`): requests are sent at precomputed times,
  whether or not earlier ones have finished. Only this shows what happens past capacity:
  the queue grows, TTFT explodes, goodput falls below the offered rate. All three patterns
  share the mean rate. `burst` sends B requests at once every B/rate seconds, so the
  patterns differ only in how bunched the arrivals are.

## Same batch budget for every server

Every server gets 8 concurrent sequences and a 32k-token KV budget:

| server | flags |
|---|---|
| llama.cpp | `-np 8 -c 32768`: 4096 tokens per slot |
| mlx_lm.server | `--decode-concurrency 8 --prompt-concurrency 8` |
| LM Studio | `lms load --parallel 8 -c 32768` |

The concurrency sweep runs up to 32 users, so it shows what happens past the batch limit:
requests queue, TTFT grows, and throughput levels off.

## Same model, two weight formats

Qwen3-1.7B: GGUF Q4_K_M (~4.9 bits/weight, 1.28 GB) for llama.cpp and LM Studio's llama.cpp
engine; MLX 4-bit, group 64 (~4.5 bits/weight, 0.97 GB) for mlx-lm and LM Studio's MLX engine.
The quantizations are not identical, so MLX-vs-llama.cpp numbers compare engine plus
quantization. LM Studio-vs-upstream comparisons use identical weight files, so they isolate
the wrapper and the bundled engine version.

A 1.7B model keeps the downloads small (the shared mac had ~34 GB of free disk) and leaves
memory for the other jobs running on the machine.

## A shared, busy machine

The benchmark ran on a shared M4 Max that was doing other work too. Countermeasures:

- `scripts/bench.sh` waits before every run until the 1-minute load average is below 6
  (`MAX_LOAD`);
- every result stores the load average, free memory, power source and the top five
  CPU-consuming process names, both before and after the run. Only process basenames are
  recorded, never paths or arguments;
- only one server runs at a time.

This reduces noise without removing it. If a number looks off, check the load columns in
`results/SUMMARY.md`.

## Reading stalls from client timestamps

`report.stall_stats` looks for prefill interference using only the chunk timestamps every result
file already has, so it needs no server changes and works the same on all four servers.

- **30 ms threshold.** It sits above every server's p90 gap at 3 req/s (9.8–25.5 ms). On the
  llama.cpp servers and `mlx_lm.server` the split barely moves between 20 and 50 ms. On LM Studio
  MLX it does move, because that engine sends tokens in bunches. The README says so.
- **Random placement as the baseline.** At high load stalls cover much of the run, so "a first
  token landed near a stall" happens by chance too. Moving the same stalls to random times gives
  the rate to beat. My first version compared with gaps that were not stalls and used a ±20 ms window, which
  made both MLX servers look unrelated: their first token comes 20–50 ms after the stall.
  Plotting the lag showed that, so the window now reaches 100 ms.
- **Stalls and first tokens only coincide.** This shows timing, not cause. A server-side trace, or
  runs with a different `-ub`, would be the direct test.

## Client

Python asyncio + httpx. Timestamps come from `time.perf_counter()` taken when a parsed SSE line
holds non-empty text. A Python client has a ceiling. `scripts/bench.sh fake ceiling` measures it against the fake
server at 1 ms/token, to check that it sits well above what these servers produce.

## Tests without a model

`servebench/fakeserver.py` is a stdlib OpenAI-compatible server with a known timing model: slot
count, prefill seconds per token, decode seconds per token. The tests check that the client
recovers those latencies, that queueing shows up in TTFT and not in TPOT, and that goodput and
the percentile maths are right. `make test` needs no model, no GPU and no network.
