# Wave 8 plan — 11 new models

Status: **preflight complete, downloads starting.**
Written 2026-09-15. Adds 11 models to the 18 already measured.

---

## 0. Preflight (done before committing 245 GB of bandwidth)

Three checks, because the two things that can waste a whole day here are a
filename that 404s after an hour of download, and an architecture the pinned
build cannot load.

**Filenames.** Every repo was queried through `https://huggingface.co/api/models/<repo>`
and the real `siblings.rfilename` list read, rather than trusting a filename from
the source table. All 11 exist and are single-file (no split GGUFs to concatenate).
Two names differ from what a guess would produce:

| Expected shape | Actual filename |
|---|---|
| `gemma-4-31B-it-qat-Q4_0.gguf` | `gemma-4-31B_q4_0-it.gguf` |
| `Mistral-Small-4-119B-2603-i1-IQ1_M.gguf` | `Mistral-Small-4-119B-2603.i1-IQ1_M.gguf` |

**Sizes.** Taken from HTTP `HEAD` (`x-linked-size`), not from the source table.
Total **245.02 GB** against **264.4 GB free** on C:. See §4.

**Architecture support.** The first 256 KB of each GGUF was range-fetched and the
header parsed to read `general.architecture`, then that ID was looked up in the
string table of the pinned `llama.dll`. This costs ~3 MB instead of 245 GB to
answer "will this even load".

| Model | arch | in build? |
|---|---|---|
| Qwen3-Coder-Next | `qwen3next` | yes |
| Qwen3-Next-80B-A3B-Instruct | `qwen3next` | yes |
| Nemotron-Super-49B v1.5 | `deci` | yes |
| Mistral-Small-4-119B | `mistral4` | yes |
| Qwen3.5-122B-A10B | `qwen35moe` | yes |
| Nex-N2.5-mini | `qwen35moe` | yes |
| gemma-4-31B-it-qat | `gemma4` | yes |
| GLM-4.7-Flash | `deepseek2` | yes |
| DeepSeek-R1-Distill-Qwen-32B | `qwen2` | yes |
| Olmo-3.1-32B-Think | `olmo2` | yes |
| Magistral-Small-2509 | `llama` | yes |

All 11 load-capable. Two corrections to the source table fall out of this:

- **GLM-4.7-Flash is not a new architecture.** bartowski built it on `deepseek2`
  (the DeepSeek-V2 MLA path). It is still a new *model family* worth measuring,
  but the "major missing family" framing overstates the architectural novelty.
- **Nex-N2.5-mini is `qwen35moe`** — the same arch as Qwen3.5-122B-A10B, so it is
  not the independent family the source suggests.
- **Magistral-Small-2509 is plain `llama` arch**, despite being built on
  Mistral Small 3.2.

---

## 1. What each model is actually testing

Not all 11 are equally informative. Grouping by the question they answer:

### A. Direct tests of the dequantisation-cost thesis

The central finding is that P100 throughput is set by how expensive the quant is
to decode, not by bandwidth, because Pascal has no `dp4a`. **I-quants are the most
`dp4a`-dependent format there is.** Three of these models are I-quants, and the
thesis makes a hard, falsifiable prediction: they should land at the *bottom* of
the efficiency table despite their small size.

| Model | Quant | Prediction |
|---|---|---|
| Nemotron-Super-49B v1.5 | `IQ4_XS` | dense, so `%peak` is computable — should be well under the 30.4% that `Q4_0` reached |
| Mistral-Small-4-119B | `i1-IQ1_M` | MoE, speed only |
| Qwen3.5-122B-A10B | `i1-IQ1_M` | MoE, and **10B active** — roughly 3× the active weights of every A3B model measured so far |

Qwen3.5-122B-A10B is the most interesting of the three: if active-parameter count
predicts MoE speed, it should land near a 10B dense model, not near the A3B MoEs.
That has not been tested at all yet — every MoE in the study so far is A3B/A4B.

### B. Controlled A/B against something already measured

These are worth more than their novelty suggests, because one variable moves:

| New model | Existing comparison | Variable isolated |
|---|---|---|
| Qwen3-Next-80B-A3B-**Instruct** Q2_K | Qwen3-Next-80B-A3B-**Thinking** Q2_K | post-training only — same arch, same quant, same size |
| Magistral-Small-2509 Q6_K | Mistral-Small-3.2-24B UD-Q4_K_XL | quant cost on a shared base model |
| gemma-4-31B-it-qat Q4_0 | gemma-4-26B-A4B-it-QAT Q4_0 | dense vs MoE within one family, same quant, same QAT recipe |
| Olmo-3.1-32B-Think Q5_K_M | Qwen3-32B Q6_K | two 32B dense models, K-quant, near the context ceiling |

The gemma pair is the cleanest MoE-vs-dense comparison available anywhere in this
study — same vendor, same quantisation-aware-training recipe, same `Q4_0` format.

### C. New families, speed characterisation only

GLM-4.7-Flash, DeepSeek-R1-Distill-Qwen-32B, Nex-N2.5-mini, Qwen3-Coder-Next.
No existing comparison point; these extend coverage.

---

## 2. Test matrix

Every model gets the same two passes. All runs use **f16 KV** with **`-fa on`**,
which wave 7 established as the only configuration worth running at depth (a
quantised cache costs 21–42% of throughput on this hardware to save under a GB).

**Pass 1 — headline.** `standard` suite at depth 0: `pp512`, `pp2048`, `tg128`,
5 repetitions. Gives the throughput number, the prompt-processing number, and
(for dense models) the achieved-bandwidth figure.

**Pass 2 — depth ladder.** `tg128` at increasing KV depth, one **separate
invocation per depth**. Reps: 3 at ≤32k, 2 above.

| Class | Models | Ladder |
|---|---|---|
| ≤22 GB | gemma-4-31B, Magistral, GLM-4.7-Flash, Nex-N2.5-mini, Olmo-3.1-32B, DeepSeek-R1-Distill | 0, 4k, 16k, 32k, 64k |
| ≤22 GB **MoE** | GLM-4.7-Flash, Nex-N2.5-mini | + 128k |
| ~25–26 GB | Qwen3-Coder-Next, Mistral-Small-4, Nemotron-Super-49B, Qwen3.5-122B, Qwen3-Next-80B-Instruct | 0, 4k, 16k, 32k |

Separate invocations per depth is not an efficiency mistake — it is deliberate.
`llama-bench` aborts the entire run when one depth fails to allocate, and failing
to allocate is the *expected* outcome at the top of these ladders, so batching
would throw away every shallower result that worked.

**Ladder extension rule:** if a ~25 GB model clears 32k, extend it to 64k. If a
≤22 GB model clears 64k, extend to 128k. Ceilings are a finding, not a failure —
a model that dies at 16k gets that recorded.

**VRAM context.** 31.8 GiB usable across both cards. A 25 GB model leaves ~6.8 GB
for KV and compute buffers. Prior evidence says a 25 GB **dense** model will not
allocate a 16k context at all (Seed-OSS-36B Q6_K and Llama-3.3-70B Q2_K_XL both
fail there, re-verified with `-fa on`). So **Nemotron-Super-49B IQ4_XS at 25.0 GB
dense is predicted to cap at 4k–8k.** If it clears 32k, that overturns a published
finding and is the most valuable single result in this wave.

---

## 3. Execution order

Ascending size. Smallest first means the first results land soonest, and the disk
stays comfortable longest. Downloads run sequentially in the background while the
previously-downloaded model is being benchmarked — these contend for disk but not
for GPU, and METHODOLOGY.md already records that download traffic does not affect
generation throughput.

| # | Model | GB | Group |
|---:|---|---:|---|
| 1 | gemma-4-31B-it-qat Q4_0 | 16.44 | B |
| 2 | Magistral-Small-2509 Q6_K | 18.02 | B |
| 3 | GLM-4.7-Flash Q5_K_M | 20.09 | B |
| 4 | Nex-N2.5-mini Q4_K_M | 20.79 | B |
| 5 | Olmo-3.1-32B-Think Q5_K_M | 21.29 | B |
| 6 | DeepSeek-R1-Distill-Qwen-32B Q5_K_M | 21.66 | B |
| 7 | Qwen3-Coder-Next UD-Q2_K_XL | 24.92 | A |
| 8 | Mistral-Small-4-119B i1-IQ1_M | 24.85 | A |
| 9 | Nemotron-Super-49B v1.5 IQ4_XS | 25.03 | A |
| 10 | Qwen3.5-122B-A10B i1-IQ1_M | 25.70 | A |
| 11 | Qwen3-Next-80B-A3B-Instruct Q2_K | 26.23 | A |

---

## 4. Disk — the one real constraint

**264.4 GB free, 245.02 GB required. That lands at ~19 GB free**, on a drive that
already holds 553.8 GB of GGUF weights.

19 GB is enough to finish but is not comfortable headroom for Windows. The default
here is to **keep everything** — they were asked for, and re-downloading costs
hours. The plan does not delete anything on its own.

Checkpoint: after model #8 the drive drops under ~70 GB free. If space is wanted
back, the cheapest candidates are the superseded quants of Qwen3.8-27B (7 quants
of one model, ~120 GB, and the quant sweep that needed them is complete and
published). That is a decision to raise, not to take.

---

## 5. Known post-processing work

`mastertable.ps1` will need three fixes before these rows are correct, for the
same class of reason that previously put gpt-oss-20b at an impossible 103% of
rated bandwidth:

1. **Quant splitter** — the regex expects `<base>-<QUANT>` with a hyphen and
   uppercase quant. It will fail on `gemma-4-31B_q4_0-it` (quant is mid-name,
   lowercase, underscore-delimited) and on the mradermacher `.i1-IQ1_M` files
   (dot before the `i1-` prefix).
2. **MoE detection** — four of these are MoE with no active-parameter marker in
   the filename: `Qwen3-Coder-Next`, `Mistral-Small-4-119B-2603`,
   `zai-org_GLM-4.7-Flash`, `nex-agi_Nex-N2.5-mini`. Left on the dense path they
   will report bandwidth figures that are meaningless and possibly above 100% of
   spec, which is the tell for exactly this bug.
3. **Params** — `params_b` comes from GGUF metadata, so total (not active) params
   are reported for MoE. Already true of existing rows; noted for consistency.

---

## 6. What this wave does *not* establish

No quality measurement. These 11 models add at least five more tokenizers to the
nine already in the study, so none of them can be perplexity-ranked against each
other or against existing rows. Every claim from this wave is a claim about
**speed and memory**. Where the source table asks to "compare answer quality" or
"assess quality", that is out of scope for this harness and is not being
answered — including for the two `IQ1_M` models, where aggressive quantisation
makes quality the *only* question that really matters.
