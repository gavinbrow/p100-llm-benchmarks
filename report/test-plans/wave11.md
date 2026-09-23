# Wave 11 - finish the speed results

Scope set by the user on 2026-09-18: **speed only**. No quality, recall or tool-use tests.

Runner: `scripts/wave11.py` (detached, keeps the PC awake). Log: `logs/wave11.log`.
llama-bench rows go to `results/raw.jsonl` through `scripts/bench.ps1`; everything else to
`results/wave11.jsonl`. Order is by value, so stopping early still leaves the most useful parts.

## T - tensor split mode (new, highest value)

This build (b10970) has `-sm tensor`, which splits every layer across both cards and runs them
in parallel. The study so far used `-sm layer` throughout (`row` fails on these cards). A
one-off check on 2026-09-18, Qwen3.8-27B Q4_0, one repetition:

| split | pp512 | tg64 |
|---|---:|---:|
| layer | 140.3 | 14.55 |
| tensor | 168.6 | 21.18 |

| step | what | reps |
|---|---|---|
| T1 | every model file with a two-card result, `-sm layer,tensor` back to back, pp512 / pp2048 / tg128, empty cache | 3 |
| T2 | tensor at 32k for the dense and slow models; tensor at 128k for the fast long-context models | 2 |
| T3 | tensor at 256k for the six 256k models | 2 |

Layer-mode figures at depth already exist with the same settings (f16 KV, flash attention on).

## P - several agents at once (llama-batched-bench)

N sequences, each with its own prompt of `npp` tokens, then 128 generated tokens each. Reports
aggregate prefill and generation. Section 05 only did this with an empty cache.

| model | npp | parallel |
|---|---:|---|
| Nemotron-3.5-Lightning Q4_0 | 32,768 | 1, 2, 4, 8 |
| Nemotron-3.5-Lightning Q4_0 | 261,120 | 2, 4 |
| Qwen3.6-35B-A3B UD-Q4_K_M | 65,536 | 1, 2, 4 |
| Qwen3.5-9B Q8_0 | 65,536 | 1, 2, 4 |
| gpt-oss-20b | 32,768 | 1, 2, 4, 8 |
| Qwen3.8-27B Q4_0 (dense) | 16,384 | 1, 2, 4 |

## S - save and reload a long context (llama-server slots)

Fill a context, save the slot to disk, erase it, restore it, send a follow-up and check the
server re-reads only the new tokens. Then restart the server and restore again (a new
process; the file may still be in the OS cache, which cannot be cleared without admin).
Nemotron and Qwen3.6-35B-A3B at 128k and 256k, gpt-oss-20b at 126k.

## E - MoE experts in system RAM (`-ncmoe`)

The machine has 32 GB of DDR4-2133 and a 6-core i5-10600K. The test is whether the extra
room reaches contexts that ran out of memory:

- Qwen3-Coder-30B-A3B Q4_K_M with an **f16** cache at 128k and 256k. It reached 128k only
  with a q8_0 cache before, at 6.2 tok/s.
- Qwen3.6-35B-A3B UD-Q6_K at 256k (stopped at ~98k).
- Qwen3-Next-80B-A3B, Qwen3-Coder-Next and Qwen3.5-122B-A10B at 256k (stopped at 128k).

For each, the smallest number of expert layers on the CPU that loads, plus its empty-cache
speed so the offload cost is visible.

## B - backfill every missing prefill cell in the master table

These are 280 cells across 82 configurations: generation rows with no pp512 and/or pp2048
from the same invocation. Each is re-run with the row's exact settings and prompt
processing only (`-n 0`). Repetitions: 2 up to 32k, 1 beyond. The cheapest run first;
Nemotron at 512k and 1M run last (each fill takes about 1 h and 3 h).

These cells are measured separately from their generation figure, so the table marks them.
