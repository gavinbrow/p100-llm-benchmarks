# Methodology

How the numbers in [the report](p100-report.html) and the [master table](MASTER-TABLE.md) were
taken, what they do not show, and where every model file came from.

## Test system

| Component | Detail |
|---|---|
| GPU | 2 × NVIDIA Tesla P100-PCIE-16GB (GP100, Pascal), compute capability 6.0 |
| VRAM | 16,287 MiB usable per card, 32,574 MiB total |
| Memory | HBM2, 732 GB/s per card (spec sheet) |
| Cooling | Passive datacentre heatsinks, in a desktop case |
| Interconnect | PCIe 3.0; card 0 at ×16, card 1 at ×4; no NVLink |
| Driver | 582.78, **TCC** mode (compute only, no display), ECC on |
| CPU | Intel Core i5-10600K (6 cores, 12 threads, 4.1 GHz) |
| System RAM | 32 GB, not used for weights or cache |
| OS | Windows 11 Pro 26200 |

## Software

- **llama.cpp b10970** (commit `bfdc32183`), the official Windows **CUDA 12.4** x64 release build.
- Tools from that build: `llama-bench`, `llama-server`, `llama-batched-bench`, `llama-perplexity`
  and `llama-tokenize`.

### Why CUDA 12.4 and not 13.x

The llama.cpp release page offers CUDA 12.4 and CUDA 13.x Windows builds. **CUDA 13 dropped
Pascal (`sm_60`)**, so the 13.x binaries will not run these cards. `nvidia-smi` reporting
"CUDA Version: 13.0" refers to the newest API the driver supports, not to what the card can
execute.

### Why llama.cpp and not vLLM, SGLang or ExLlamaV2

vLLM and SGLang require compute capability 7.0 (Volta) or later; the P100 is 6.0, so their
kernels do not exist for it. That leaves llama.cpp and its wrappers (Ollama, LM Studio). Plain
llama.cpp was used because its tools expose every flag this study varies: `-sm`, `-dev`,
`-ctk`/`-ctv`, `-d`, `-ub`, `-np` and the speculative-decoding options.

## Measurement

Every weight and the whole KV cache were in VRAM for every measurement. Both cards and layer
split (`-sm layer`) were used unless a row says otherwise.

**llama-bench** (Method: *bench* in the master table). A warmup pass, then N timed repetitions,
reported as mean ± standard deviation.

- `tg128` is writing: generating 128 tokens. `pp512` and `pp2048` are reading: processing a
  512- or 2,048-token prompt.
- `-d N` fills the KV cache to depth N first, untimed, then measures at that depth.
- 5 repetitions on an empty cache (3 for the split-mode comparison), 3 at depths up to 32k and 2
  beyond. Standard deviations at depth were 0.00–0.82 tok/s.
- `-sm layer,tensor` in one invocation for the split-mode comparisons, so both modes ran back to
  back on the same thermal state.

**One-fill server climb** (Method: *server*). llama-bench refills the cache from zero for every
test and every repetition, which would have meant several three-hour fills for a single
million-token cell. Instead one `llama-server` filled the context once and climbed: at each level
it read 512 new tokens, then 2,048, then wrote 128 (`-ctxcp 0 --cache-ram 0`, prompts sent as
token IDs). Back to back on warm cards at 32k it agreed with llama-bench within 3% for reading
512 tokens (303.5 against 311.5 ± 7.4 tok/s) and 6% for 2,048 (445.3 against 418.5). It is a
single measurement, and the server caps a conversation at the model's trained context, so its top
levels sit up to 4% below the llama-bench depths they stand in for (259,392 for 261,120; 128,320
for 130,048 and 131,072; 62,784 for 65,536).

**Markers in the tables.** A reading figure marked `*` was measured later in a separate llama-bench
run at the same settings; `†` comes from a one-fill server climb.

**Context fills.** One `llama-server` session per model at the largest context it could allocate
(its trained window, or the largest that fits in 32 GB, found by stepping down until one loaded).
The prompt grew through 4k, 16k, 32k, 64k, 128k, 256k, 512k and the full window, timing each step:
the cold time to first token, reading speed as the cache fills, and writing speed at each depth.

**Other tools.** `llama-server` for agent sessions, sustained load, speculative decoding and
context save/restore (`--slot-save-path`); `llama-batched-bench` for concurrent conversations;
`llama-perplexity` on WikiText-2 test (60 chunks at 512 context); `llama-tokenize` for the
tokenizer comparison. Speculative-decoding runs use greedy decoding (`temperature 0`,
`top_k 1`) so acceptance rates are comparable.

**GPU telemetry.** A parallel `nvidia-smi` process sampled VRAM, power, temperature, clocks and
clock-event (throttle) reasons once a second. For llama-bench runs the first 10 samples are
discarded so model loading does not count. Reported VRAM and power are peaks; power is the sum of
both cards.

### Effective bandwidth

A dense model reads every weight once per generated token, so `file size × tok/s` approximates
the memory bandwidth it achieves. Comparing that with the 732 GB/s spec shows how much of the card
is doing useful work. The arithmetic is invalid for mixture-of-experts models, which read only
their active experts, so MoE rows report speed only.

## Known limits

- **Quality is mostly unmeasured.** Perplexity is compared only within the Qwen family, whose
  token IDs were verified identical, and is a relative comparison on 60 chunks, not comparable
  with published absolute figures. Nothing ranks the other models on quality, including the two
  IQ1_M models at 119B and 122B. The agent test's recall check (eight six-digit facts) is a floor,
  not a quality ranking.
- **Heat.** Long runs are heat-soaked, steady-state numbers: right for agent work, pessimistic for
  one short prompt on idle cards. Long runs peaked at 80–81 °C, two degrees under the 82 °C
  slowdown point. Single-card runs used card 0 only; that the one-card deficit is thermal is
  inferred from telemetry and one cool-card rerun.
- **Fill times** add up the steps of a context fill. They run 6–12% slower than one cold read of
  the same length, probably from heat.
- **Tensor split** results come from one llama.cpp build, in which five models fail to load with
  it (one with an internal assertion). At depth, tensor figures from the last round are paired
  with layer figures from earlier rounds, so they include some drift in card temperature; the
  gains are 2–10× the gap between the two measurement methods.
- **Context ceilings** are for an f16 cache on both cards with layer split and one quant per
  model. A different quant, cache type or split mode moves them.
- **Speculative decoding** figures are one request per workload, with no error bars.
- **Concurrency** runs conversations in lockstep, so a real server would see somewhat less.
  Context reloads were measured with the saved file probably still in the OS file cache.
- **Tokenizer** figures come from one English and one code sample.
- **Missing cells.** Nemotron-3.5-Lightning reading 2,048 tokens at 1,047,552 is excluded: its
  re-run followed an unplanned restart that left the cooling degraded, and read 2–3× too slow.
  Four more runs failed, leaving 14 empty reading cells: DeepSeek-R1-Distill-Qwen-32B reading
  2,048 tokens at 0–32k (a CUDA error), and three Qwen3-Next-80B runs that exited at start. Each
  of those rows still has its writing speed, and all but two have a reading speed at the other
  prompt length.

## Models

Every file was downloaded unmodified from Hugging Face. The scripts expect LM Studio's layout,
`%USERPROFILE%\.lmstudio\models\<publisher>\<repo>\<file>.gguf`, so every model is also loadable
from LM Studio.

| Hugging Face repo | Files |
|---|---|
| [unsloth/Qwen3.8-2B-GGUF](https://huggingface.co/unsloth/Qwen3.8-2B-GGUF) | `Qwen3.8-2B-Q8_0.gguf` (also the draft model for Qwen3.6-35B-A3B) |
| [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) | `LFM2.5-2.6B-{F16,BF16,Q8_0,Q4_K_M,Q4_0}.gguf` |
| [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) | `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` |
| [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | `Qwen3.5-9B-{BF16,Q8_0,Q4_K_M,Q4_0}.gguf` |
| [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) | `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` |
| [unsloth/phi-4-GGUF](https://huggingface.co/unsloth/phi-4-GGUF) | `phi-4-Q8_0.gguf` |
| [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) | `gpt-oss-20b-MXFP4.gguf`, `eagle3-gpt-oss-20b-Q8_0.gguf` (draft head) |
| [bartowski/mistralai_Magistral-Small-2509-GGUF](https://huggingface.co/bartowski/mistralai_Magistral-Small-2509-GGUF) | `mistralai_Magistral-Small-2509-Q6_K.gguf` |
| [unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF) | `Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf` |
| [lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF](https://huggingface.co/lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF) | `gemma-4-26B-A4B-it-QAT-Q4_0.gguf` |
| [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) | `Qwen3.6-27B-Q4_0.gguf` |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | `Qwen3.8-27B-{Q4_0,Q8_0,UD-IQ4_XS,UD-Q3_K_XL,UD-Q6_K,UD-Q8_K_XL}.gguf` |
| [lmstudio-community/Qwen3.8-27B-GGUF](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF) | `Qwen3.8-27B-Q4_K_M.gguf` |
| [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) | `Qwen3.8-27B-DFlash2-Q8_0.gguf` (draft model) |
| [bartowski/zai-org_GLM-4.7-Flash-GGUF](https://huggingface.co/bartowski/zai-org_GLM-4.7-Flash-GGUF) | `zai-org_GLM-4.7-Flash-Q5_K_M.gguf` |
| [unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` |
| [google/gemma-4-31B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf) | `gemma-4-31B_q4_0-it.gguf` |
| [ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) | `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`, `mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` (MTP head); the Q8_0 is too large to load |
| [bartowski/allenai_Olmo-3.1-32B-Think-GGUF](https://huggingface.co/bartowski/allenai_Olmo-3.1-32B-Think-GGUF) | `allenai_Olmo-3.1-32B-Think-Q5_K_M.gguf` |
| [bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF) | `DeepSeek-R1-Distill-Qwen-32B-Q5_K_M.gguf` |
| [unsloth/Qwen3-32B-GGUF](https://huggingface.co/unsloth/Qwen3-32B-GGUF) | `Qwen3-32B-Q6_K.gguf` |
| [bartowski/nex-agi_Nex-N2.5-mini-GGUF](https://huggingface.co/bartowski/nex-agi_Nex-N2.5-mini-GGUF) | `nex-agi_Nex-N2.5-mini-Q4_K_M.gguf` |
| [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | `Qwen3.6-35B-A3B-{UD-Q6_K,UD-Q4_K_M,MXFP4_MOE}.gguf` |
| [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) | `Ornith-1.5-35B-Q4_K_M.gguf` |
| [lmstudio-community/Seed-OSS-36B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Seed-OSS-36B-Instruct-GGUF) | `Seed-OSS-36B-Instruct-Q6_K.gguf` |
| [bartowski/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-GGUF](https://huggingface.co/bartowski/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-GGUF) | `nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-IQ4_XS.gguf` (does not load) |
| [unsloth/Llama-3.3-70B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF) | `Llama-3.3-70B-Instruct-UD-Q2_K_XL.gguf` |
| [unsloth/Qwen3-Coder-Next-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) | `Qwen3-Coder-Next-UD-Q2_K_XL.gguf` |
| [bartowski/Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF) | `Qwen_Qwen3-Next-80B-A3B-Instruct-Q2_K.gguf` |
| [bartowski/Qwen_Qwen3-Next-80B-A3B-Thinking-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-Next-80B-A3B-Thinking-GGUF) | `Qwen_Qwen3-Next-80B-A3B-Thinking-Q2_K.gguf` |
| [mradermacher/Mistral-Small-4-119B-2603-i1-GGUF](https://huggingface.co/mradermacher/Mistral-Small-4-119B-2603-i1-GGUF) | `Mistral-Small-4-119B-2603.i1-IQ1_M.gguf` |
| [mradermacher/Qwen3.5-122B-A10B-i1-GGUF](https://huggingface.co/mradermacher/Qwen3.5-122B-A10B-i1-GGUF) | `Qwen3.5-122B-A10B.i1-IQ1_M.gguf` |
