# Pascal at the Limit

**How fast two ten-year-old Tesla P100s run local language models in 2026.**
545 speed measurements of 29 models from 2B to 122B parameters, with llama.cpp, timed from an
empty context to a million tokens. Every weight and the whole KV cache stay in VRAM; nothing is
offloaded to system RAM.

**[Read the full report](https://gavinbrow.github.io/p100-llm-benchmarks/)**: charts, every
table, and a filterable, sortable table of all 545 runs.

| | |
|---|---|
| GPUs | 2 × Tesla P100-PCIE-16GB (Pascal, compute capability 6.0), passively cooled in a desktop case |
| VRAM | 32,574 MiB usable, HBM2 at 732 GB/s per card |
| Link | PCIe 3.0, card 1 at ×4, no NVLink |
| Host | Intel Core i5-10600K, 32 GB RAM, Windows 11 |
| Software | llama.cpp b10970, official CUDA 12.4 build, driver 582.78 |

## Headline numbers

| | |
|---:|---|
| **110 tok/s** | Nemotron-3.5-Lightning-30B-A3B writing code with its built-in MTP draft head |
| **39 tok/s** | the same model with 256k tokens in context; 16 tok/s with its full million |
| **9×** | a 119B mixture-of-experts model (39 tok/s) against a 70B dense model (4.3 tok/s) on the same cards |
| **+21–44%** | dense models from 8B up with `-sm tensor`, which splits every layer across both cards |

## What we found

1. **Sparse beats dense, by a lot.** Mixture-of-experts models run 3–9× faster than dense models
   of comparable size. Qwen3.6-35B-A3B is both faster and lower in perplexity than every dense 27B
   setup tested.
2. **The card runs short of decode compute, not bandwidth.** Dense models use 16–65% of the
   732 GB/s peak, in order of how costly the quant format is to unpack. Pascal has no `dp4a`, so a
   Q3_K file 18% smaller than Q4_0 runs 15% slower.
3. **Split every layer across both cards.** `-sm tensor` makes dense models from 8B up 21–44%
   faster, and every model that loads with it 14–48% faster once the context is full. Five
   models, Nemotron-3.5-Lightning among them, do not load with it in this build.
4. **One card overheats; two share the heat.** Under 25 minutes of load a lone P100 throttles to
   906 MHz and loses 25%. Split across both cards, the same model loses 5.5%. Power was never the
   limit: at most 123 W per card of a 250 W allowance.
5. **Long context works.** Nemotron-3.5-Lightning writes at 50 tok/s with 128k tokens of context,
   39 at 256k and 16 at a million, because only 6 of its 52 layers keep a KV cache. With tensor
   split, five more models write at 28–31 tok/s at 256k.
6. **Judge a model at the context you will use.** GLM-4.7-Flash starts at 43 tok/s and falls to
   6.8 at 128k. Keep the KV cache at f16: q8_0 costs 21–42% at 72k and saves only 0.2–1.9 GB.
7. **Reading the context is the slow part.** Filling 256k takes 15 minutes on the fastest model,
   and a million takes almost three hours. `-ub 2048` cuts that by a quarter. About half the
   models run out of VRAM before their trained context.
8. **Agents are practical at 256k.** Each turn takes 12–14 s at 250k tokens, because the server
   re-reads only the ~2,000 new tokens. Four conversations at once give 1.8–1.9× the output of
   one, and a saved 128k context reloads in 1–9 s instead of 5–27 min.
9. **Short drafts win.** Speculative decoding pays with a draft length of 2. Nemotron's MTP head
   adds 59% on code, and a draft model adds up to 76% to a dense 27B on structured output. A
   separate draft model makes MoE models slower.
10. **Use the CUDA 12.4 build and F16 weights.** CUDA 13 dropped Pascal (`sm_60`), BF16 reads
    prompts 1.8× slower than F16, and `-sm row` cannot run on these cards (`VMM: no`).

## What to run

| For | Model | Settings | What you get |
|---|---|---|---|
| Fastest | Nemotron-3.5-Lightning-30B-A3B Q4_0 | MTP draft head, n-max 2 | 110 tok/s on code, 95 on prose |
| Agents, up to 1M tokens | Nemotron-3.5-Lightning-30B-A3B Q4_0 | f16 KV, `-fa on`, layer split | 39 tok/s at 256k, 16 at 1M; 12–14 s per turn at 250k |
| Small and fast | gpt-oss-20b MXFP4 | `-sm tensor` | 80 tok/s empty, 57 at 128k, in 12 GB |
| Long context, second choice | Nex-N2.5-mini or Ornith-1.5-35B, Q4_K_M | `-sm tensor`, f16 KV | 40 tok/s at 128k, 31 at 256k |
| Coding agent, up to 128k | Qwen3-Coder-Next UD-Q2_K_XL | `-sm tensor` | 33 tok/s empty, 29 at 128k |
| One card | Qwen3.5-9B Q4_K_M | single card | 18.6 tok/s at 256k in 14.1 GB |
| Most parameters | Mistral-Small-4-119B i1-IQ1_M | layer split | 39 tok/s, holds 128k |
| A dense 27B | Qwen3.8-27B Q4_0 | `-sm tensor`, or a DFlash2 draft at n-max 2 | 18.6 tok/s with tensor split; 20–24 with the draft |

Prefer Q4_0, Q8_0, MXFP4 or F16, the formats cheapest to unpack. Avoid Q3_K, ngram speculation,
a quantised KV cache, dense models over 25 GB (they leave almost no room for context), and
choosing a long-context model from an empty-cache benchmark.

## What is in this repository

| Path | Contents |
|---|---|
| [`docs/index.html`](docs/index.html) | The report, served by GitHub Pages ([`report/p100-report.html`](report/p100-report.html) is the same file) |
| [`report/MASTER-TABLE.md`](report/MASTER-TABLE.md) | All 545 writing-speed runs as one markdown table, each row fully qualified by its settings |
| [`report/METHODOLOGY.md`](report/METHODOLOGY.md) | Test system, measurement methods, known limits, and the source of every model file |
| [`report/test-plans/`](report/test-plans/) | The plan written before each of the later test rounds (8–11) |
| [`results/master.csv`](results/master.csv) | The master table as CSV |
| [`results/raw.jsonl`](results/raw.jsonl) | Every llama-bench result, one JSON line per test, with GPU telemetry |
| [`results/wave10.jsonl`](results/wave10.jsonl), [`wave11.jsonl`](results/wave11.jsonl) | llama-server experiments: context fills, agent sessions, sustained load, concurrency, save/restore |
| [`results/specdec.jsonl`](results/specdec.jsonl), [`perplexity.jsonl`](results/perplexity.jsonl), [`gguf-meta.jsonl`](results/gguf-meta.jsonl) | Speculative decoding, perplexity, and the GGUF header fields used for cache-size arithmetic |
| [`logs/`](logs/) | Per-run llama-bench JSON, stderr and once-a-second GPU telemetry, plus llama-server logs. 42 of the 434 JSON files are empty or cut off because that run failed to load or crashed; the matching `.stderr.txt` says why |
| [`scripts/`](scripts/) | Every benchmark, download and reporting script used |

## Reproducing

1. Download the llama.cpp **b10970** Windows release built for **CUDA 12.4** (not 13.x, which
   cannot run Pascal) and unpack it to `tools/llamacpp/`.
2. Download the GGUF files listed in [METHODOLOGY.md](report/METHODOLOGY.md#models). The scripts
   expect the LM Studio layout, `%USERPROFILE%\.lmstudio\models\<publisher>\<repo>\<file>.gguf`.
   For the perplexity and tokenizer tests, also unpack
   [wikitext-2-raw-v1.zip](https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip)
   (the copy llama.cpp's `scripts/get-wikitext-2.sh` uses) to `data/wikitext-2-raw/`.
3. Run a test round, for example `.\scripts\run-wave9.ps1` or `python scripts\wave11.py`.
4. Rebuild the tables and the data in the report with `.\scripts\mastertable.ps1`, then copy the
   report to `docs/` with `.\scripts\publish.ps1`.

`scripts/bench.ps1` is the harness the llama-bench rounds call. It runs `llama-bench`, samples
`nvidia-smi` once a second in parallel, discards the first 10 samples so model loading does not
count, and appends one JSON line per test to `results/raw.jsonl`. The scripts use absolute
paths (`C:\Projects\p100 testing` and the LM Studio models folder), set near the top of each
file; change them to match your machine.

## Scope and limits

This study measures **speed**. Quality is compared only within the Qwen family, whose tokenizers
produce identical token IDs, using perplexity on WikiText-2. Nothing here ranks other models on
quality, including the two IQ1_M models at 119B and 122B.

Long runs are heat-soaked, steady-state numbers, which is the right measure for agent work and a
pessimistic one for a single short prompt on idle cards. Fifteen reading-speed cells are empty
because their runs failed or were excluded after a cooling fault; the report lists them. The full
list of limits is in [METHODOLOGY.md](report/METHODOLOGY.md#known-limits).

## License

Scripts and report text: MIT (see [LICENSE](LICENSE)).
Measurement data in `results/` and `logs/`: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
`logs/wave10-sample-english.txt` is an excerpt of
WikiText-2 ([Merity et al., 2016](https://arxiv.org/abs/1609.07843)), used for the tokenizer
comparison, and stays under its original
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) license.
