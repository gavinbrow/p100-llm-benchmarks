# Wave 9 — 256k context for agents, and one card vs two

Started 2026-09-16. Runner: `scripts/run-wave9.ps1`. Metadata source: `results/gguf-meta.jsonl`
(written by `scripts/ggufmeta.py`, a dependency-free GGUF header reader).

## Questions

1. **Which models are still usable as an agent with a 256k-token context?** An agent loop cares
   about two numbers at depth, not one: *generation* speed (tg128, how fast it answers) and
   *prompt processing* speed (pp512, how fast it ingests the next tool result on top of an
   already-full context). Both are measured at every depth.
2. **How much does NVIDIA-Nemotron-3.5-Lightning-30B-A3B keep at 256k**, and can it go further
   (its trained context is 1,048,576)?
3. **For models that fit on one 16 GB card, what does the second card actually buy?**

## Who can be tested at 256k

Two gates, both read from the GGUF header, not guessed:

- **Trained context ≥ 262,144.** Models trained to 131k (Llama-3.3-70B, DeepSeek-R1-Distill,
  Magistral, Mistral-Small-3.2, Nemotron-Super, gpt-oss-20b) would be running outside their
  training window; a speed number there says nothing about whether the model still works.
  Olmo-3.1 (65k), Qwen3-32B (41k), phi-4 (16k) likewise. GLM-4.7-Flash is 202k *and* already
  fell to 6.8 t/s at 128k in wave 8.
- **Weights + f16 KV cache at 256k ≤ ~31 GiB** (31.8 GiB total, minus compute buffers).

KV bytes per token = Σ over *attention-carrying* layers of 2 × n_kv_heads × head_dim × 2 bytes.
Hybrid models are cheap because most layers are SSM / linear-attention with a fixed-size state.

| Model (file) | Weights GiB | Attn layers / total | f16 KV @256k | Total | Verdict |
|---|---:|---:|---:|---:|---|
| Nemotron-3.5-Lightning-30B-A3B Q4_0 | 17.60 | 6 / 52 | 1.5 GiB | 19.1 | **test**; 512k (3.0) and 1M (6.0) also fit → stretch |
| Nemotron-3.5-Lightning-30B-A3B Q8_0 | 31.28 | 6 / 52 | 1.5 GiB | 32.8 | does not fit |
| Nex-N2.5-mini Q4_K_M | 20.79 | 10 / 40 | 5.0 GiB | 25.8 | **test** |
| Qwen3.6-35B-A3B UD-Q4_K_M | 20.61 | 10 / 40 | 5.0 GiB | 25.6 | **test** |
| Ornith-1.5-35B-A3B Q4_K_M | 20.36 | 10 / 41 | 5.0 GiB | 25.4 | **test** |
| gemma-4-26B-A4B QAT Q4_0 | 13.45 | 5 global / 30 | 5.0 GiB (+0.2 SWA) | 18.7 | **test** |
| Qwen3.5-9B Q8_0 | 8.87 | 8 / 32 | 8.0 GiB | 16.9 | **test** |
| Qwen3.8-27B UD-IQ4_XS | 13.27 | 16 / 65 | 16.0 GiB | 29.3 | **test** (dense — slow fill) |
| Qwen3.8-27B Q4_0 | 14.95 | 16 / 65 | 16.0 GiB | 31.0 | **try** — borderline |
| Qwen3-Coder-Next UD-Q2_K_XL | 24.92 | 12 / 48 | 6.0 GiB | 30.9 | **try** — borderline |
| Qwen3.5-122B-A10B i1-IQ1_M | 25.70 | 12 / 48 | 6.0 GiB | 31.7 | **try** — very likely fails |
| Qwen3-Next-80B-A3B-Instruct Q2_K | 26.23 | 12 / 48 | 6.0 GiB | 32.2 | **try** 256k (expected fail), 128k (3.0 → 29.2) |
| Qwen3-Coder-30B-A3B Q4_K_M | 17.28 | 48 / 48 | 24.0 GiB f16 / 12.75 q8_0 | 41 / 30 | **test with q8_0 KV** — only way in; labelled as such |
| Mistral-Small-4-119B i1-IQ1_M (MLA) | 24.85 | 36 / 36, compressed | small | — | **stretch** — pp was 37 t/s at 64k, so the fill alone is hours |
| gemma-4-31B QAT Q4_0 | 16.44 | 10 global / 60 | 20.0 GiB | 36.4 | does not fit |
| Seed-OSS-36B Q6_K (trained 512k) | 27.63 | 64 / 64 | 64 GiB | — | does not fit |

Duplicate quants of the same architecture (Qwen3.6-35B-A3B Q4_K_M / MXFP4 / Q6_K) are left out:
same KV geometry, and wave 7 already measured their quant differences.

## Method

- llama.cpp b10970 CUDA 12.4, `llama-bench`, **f16 KV, `-fa on`**, n_batch 2048 (default).
- "256k" = **261,120 tokens pre-filled** (`-d 261120`). pp512 then needs a 261,632-token context,
  which stays inside the 262,144 trained window. Measured: pp512 and tg128 at that depth.
- Each model: 131,072 first (skipped where wave 6–8 already has an f16/fa row), then 261,120.
  **One invocation per depth** — an allocation failure aborts a whole llama-bench run.
  If 131k fails, 256k is not attempted.
- Reps: 2 at depth (the fill is done once per test; reps repeat only the measurement).
- Wall time per invocation is logged, which is roughly the cold fill time — itself an agent
  number (how long before the first token when a session resumes with a full context).
- VRAM peak per card from the 1 Hz `nvidia-smi` sampler, as in every wave.

## Phase 2 — one card vs two

Every model ≤ ~15 GiB, measured back-to-back with identical settings, the only change being
`-dev CUDA0` vs both cards (layer split). Depth 0, `-p 512 -n 128`, f16 KV, `-fa on`, 5 reps.

LFM2.5-2.6B Q4_K_M / Q8_0 / F16, Qwen3.8-2B Q8_0, Qwen3.5-9B Q4_K_M / Q8_0, Llama-3.1-8B Q8_0,
gemma-4-12B QAT UD-Q4_K_XL, gpt-oss-20b MXFP4, Qwen3.8-27B UD-Q3_K_XL / UD-IQ4_XS / Q4_0,
Qwen3.6-27B Q4_0, gemma-4-26B-A4B QAT Q4_0, Mistral-Small-3.2-24B UD-Q4_K_XL, phi-4 Q8_0.
The last three are tight against 16 GiB and may not load on one card — a finding either way.

Then context on a single card, same depths on one and two cards, for the agent-relevant small
models where the KV arithmetic says it fits in 16 GiB:

| Model | Weights | KV at deepest | Depths |
|---|---:|---:|---|
| Qwen3.5-9B Q4_K_M | 5.29 | 8.0 GiB @256k | 32k, 131k, 261k |
| gpt-oss-20b MXFP4 | 11.28 | 3.0 GiB @131k (trained max) | 32k, 131k |
| gemma-4-26B-A4B QAT Q4_0 | 13.45 | 1.3 GiB @64k | 32k, 64k |
| Qwen3.8-27B UD-Q3_K_XL | 12.24 | 2.0 GiB @32k | 32k |

## Phase 3 — stretch (only after 1 and 2)

- Nemotron-3.5-Lightning Q4_0 at **524,288** and **1,047,552** tokens (trained 1,048,576).
- Mistral-Small-4-119B i1-IQ1_M at 131k, then 261k.

## Order

Phase 1 in priority order (Nemotron first, borderline/slow ones last) → Phase 2 → Phase 3.
Estimated ~15 h total, dominated by dense Qwen3.8-27B fills and the phase 3 stretches.

## Tags in `results/raw.jsonl`

`wave9-256k` (phase 1), `wave9-1gpu` / `wave9-2gpu` (phase 2), `wave9-stretch` (phase 3).

## Scope

Speed and memory only, same as every other wave. A model that runs at 256k is not shown to
*reason* well at 256k — long-context recall and agent task success are not measured here.
