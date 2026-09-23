# Wave 10 — holes to close before publishing

Started 2026-09-16 (overnight). Runner: `scripts/wave10.py`. Results: `results/wave10.jsonl`
(llama-server experiments) and `results/raw.jsonl` (llama-bench experiments, tags `wave10-*`).
Log: `logs/wave10.log`.

## Why another wave

A review of the study before publishing found six places where the report claims more than
the data shows, or where a reader would reasonably ask a question the data cannot answer.

| # | Hole | Where it bites | Experiment |
|---|---|---|---|
| 1 | **"One agent turn" is arithmetic, not a measurement.** It assumes the server reuses the cached context between turns. Hybrid models (SSM / linear attention) cannot roll their recurrent state back, so if the chat template re-renders history differently each turn (e.g. stripping earlier reasoning) the server may re-process the whole context: a 15-minute turn at 256k, not a 24-second one. | Section 12, every "usable as an agent" claim | **E2** real multi-turn sessions through `llama-server` |
| 2 | **Nothing checks the model can use a 256k context**, only that it runs. | Section 12 | **E2** plants one retrievable fact per turn (needle-in-a-haystack) |
| 3 | **Thermal state is uncontrolled.** Section 07 says peak 51 °C; that was the first day. Every run since 2026-09-15 peaked at 80–81 °C, two degrees under the 82 °C slowdown point, and the one-card deficit in section 12 is attributed to heat without a controlled test. The long-context numbers were all measured at 80 °C. | Sections 05, 07, 12 | **E1** sustained-load soak with throttle-reason telemetry |
| 4 | **Card 1 is on a PCIe 3.0 x4 link** (card 0 is x16), and every one-card figure is card 0 only. | Sections 05, 12 | **E0** cold canary on each card; **E1** soak on each card |
| 5 | **MTP's 110 tok/s was measured on an empty cache.** Agents never have one. | Findings 04, section 08, recommendations | **E3** MTP at 128k and 256k |
| 6 | **Tool results are read at the pp512 rate** in the turn estimate, and all depth fills used the default `-ub 512`. Larger micro-batches may read faster at depth, which is the number that dominates an agent turn and a cold fill. | Section 12 | **E4** pp2048 at 64k across `-ub` 512–4096 |
| 7 | **tok/s is not comparable across tokenizers.** A 256k-token context holds different amounts of text, and 39 tok/s means different words per second, depending on the vocabulary. | Every cross-family comparison | **E5** tokens per character on the same English and code samples |

## E0 — cold canary (≈5 min)

Qwen3.8-27B Q4_0, `llama-bench` quick suite (pp512, tg128), 3 reps, f16 KV, `-fa on`, on `CUDA0`
alone and then `CUDA1` alone, cards cold. Tags `wave10-card0-cold`, `wave10-card1-cold`.

## E2 — agent sessions at depth (≈4.5 h, highest priority)

`llama-server`, `-np 1`, f16 KV, `-fa on`, default everything else (including context
checkpoints and the 8 GB host prompt cache), OpenAI-style `/v1/chat/completions` so the
model's own chat template is applied, as it would be under any agent framework.

- **Document:** WikiText-2 train text, sized with the model's own tokenizer to `ctx − 24,000`
  tokens, with eight planted facts ("the access code for the Heron vault is 481516") at 5%,
  18%, 31%, 44%, 57%, 70%, 83% and 95% depth.
- **Turn 0:** system prompt + document + question about fact 1.
- **Turns 1–7:** the model's previous reply, then a ~2,000-token "tool result" (fresh WikiText
  text) and a question about the next fact. This is the shape of an agent loop: the context
  only grows, and every turn reads a tool result and writes a short answer.
- **Measured per turn:** tokens actually processed (`prompt_n`), tokens reused (`cache_n`),
  time to first token, generation rate, total context, and whether the answer contains the code.
- **Guard:** if any turn after the first re-processes more than half the context, the session
  records that and stops — it is the finding, and repeating it costs up to 25 minutes a turn.

Mode (a), reasoning off, full depth:

| Model | Context | Cards |
|---|---:|---|
| Nemotron-3.5-Lightning-30B-A3B Q4_0 | 262,144 | 2 |
| Qwen3.6-35B-A3B UD-Q4_K_M | 262,144 | 2 |
| Nex-N2.5-mini Q4_K_M | 262,144 | 2 |
| gemma-4-26B-A4B QAT Q4_0 | 262,144 | 2 |
| Qwen3.5-9B Q4_K_M | 262,144 | **1** |
| Qwen3-Coder-Next UD-Q2_K_XL | 131,072 | 2 |
| Qwen3-Next-80B-A3B-Instruct Q2_K | 131,072 | 2 |
| gpt-oss-20b MXFP4 (reasoning cannot be disabled) | 131,072 | 2 |

Mode (b), reasoning on, 65,536 context — the cache mechanism does not depend on depth, and a
full re-process at 64k costs minutes rather than half an hour: Nemotron, Qwen3.6-35B-A3B and
gemma-4-26B-A4B with reasoning on, and Nemotron and Qwen3.6-35B-A3B again with
`--reasoning-preserve`, which keeps earlier reasoning in the history.

## E1 — sustained load (≈2.3 h)

Qwen3.8-27B Q4_0 through `llama-server`, 512-token generations back to back for 25 minutes,
on card 0 alone, card 1 alone, and both; then Nemotron-3.5-Lightning on both for 20 minutes.
Each starts only once both cards are at or below 45 °C (15-minute cap). `nvidia-smi` at 1 Hz
logs temperature, power, graphics clock, P-state and the clock-event (throttle) reason bitmask.

## E3 — MTP at depth (≈1 h)

Nemotron-3.5-Lightning Q4_0, the same three workloads as the original MTP test (code, prose,
table), 256 tokens, greedy, raw `/completion` as before, each appended to a WikiText prefix of
`ctx − 4,096` tokens. With and without `--spec-type draft-mtp --spec-draft-n-max 2`, at 131,072
and 262,144. The prefix is filled once per server; the three workloads reuse it.

## E4 — micro-batch at depth (≈35 min)

`llama-bench -d 65536 -p 2048 -n 128 -b 4096 -ub {512,1024,2048,4096}`, 2 reps, one invocation
per `-ub`, Nemotron Q4_0 and Qwen3.6-35B-A3B UD-Q4_K_M. Tag `wave10-ubatch`.

## E5 — tokenizer efficiency (≈20 min)

`llama-tokenize --show-count` for one file per model family on 500,000 characters of
WikiText-2 test and on this repository's scripts (code). Tokens per 1,000 characters.

## E6 — every model at its full context window (≈16 h), added after launch

The biggest remaining hole. Every depth figure so far comes from `llama-bench -d N`, which does
fill the cache with N real tokens and then measures reading 512 more (pp512) and writing 128
(tg128) on top of them — so generation and marginal prefill *are* measured at depth. What it
never measures is the fill itself: how long it takes to read the whole prompt (time to first
token) and the average prefill speed across the whole window. And wave 9 only took models
trained to ≥262k out to 256k; models trained to 131k or less, and most small models, were never
taken to their own maximum.

For each of 29 models (one representative quant per model), `llama-server` is started at the
model's **largest context that allocates**, trying the trained window first and stepping down
(e.g. 262,144 → 196,608 → 131,072). The prompt is then grown through 4k, 16k, 32k, 64k, 128k,
256k, 512k and the full window minus 1,024 tokens. At each level: tokens in context, time to
first token for the whole prompt, average prefill speed, marginal prefill speed of the latest
segment, and generation speed (128 tokens) with that context. Peak VRAM per card per session.

**Method note.** Each level extends the previous prompt, so the server reuses the cached prefix
and reads only the new segment; TTFT is the sum of segment prefill times, which is the same work
a single cold prompt performs. E2's turn 0 is a true cold fill at `ctx − 24,000` for eight of
these models and serves as the cross-check. Prompts are cut at newlines so earlier tokens never
re-tokenize.

Order within E6: long-window models first (Nemotron to 1,048,576 first of all), then dense and
131k-trained models, then the two MLA models (GLM-4.7-Flash, Mistral-Small-4-119B) whose prefill
collapses with depth, capped at 131,072.

## Order

E0 → E2(a) → E2(b) → **E6** → E1 → E3 → E4 → E5. The first runner was launched with E6 absent;
`scripts/wave10-handoff.ps1` stops it the moment it finishes E2(b) (before E1 touches the GPUs)
and starts a second runner for the rest, logging to `logs/wave10b.log`.

## Scope

The needle test is the weakest meaningful retrieval check: exact recall of eight planted facts.
Passing it does not show a model reasons well over 256k tokens; failing it shows it cannot.
Speed and cache behaviour are the primary measurements. No new downloads.
