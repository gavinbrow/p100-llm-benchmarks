# Wave 8 benchmarks - 11 new models. See report\test-plans\wave8.md for the design.
#
# Two passes per model:
#   1. standard suite at depth 0 (pp512, pp2048, tg128) - headline + bandwidth
#   2. a depth ladder, ONE INVOCATION PER DEPTH
#
# The per-depth invocation is deliberate, not wasteful. llama-bench aborts the
# whole run when a single depth fails to allocate, and failing to allocate is the
# expected outcome at the top of these ladders - several of these models are
# 25 GB against 31.8 GiB of total VRAM. Batching the depth list would throw away
# every shallower result that already succeeded.
#
# All depth runs pin f16 KV and -fa on. Wave 7 established that a quantised cache
# costs 21-42% of throughput on Pascal to save under a gigabyte, so the quantised
# ladder is a picture of a configuration nobody should run. Pinning -fa on rather
# than leaving it auto keeps the whole ladder comparable instead of letting the
# heuristic change a second variable partway up.
#
# Usage: .\run-wave8.ps1 [-Only <substring>]

param([string]$Only = "")

$ErrorActionPreference = "Continue"
$bench = "C:\Projects\p100 testing\scripts\bench.ps1"
$M     = "C:\Users\PC\.lmstudio\models"

# ladder: the planned depths. "extend" lists depths tried only if the last
# planned depth succeeded - a ceiling is a finding, but so is clearing one.
$models = @(
  @{n="gemma-4-31B-it-qat";     p="$M\google\gemma-4-31B-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf";
    ladder=@(0,4096,16384,32768,65536);  extend=@(131072)},
  @{n="Magistral-Small-2509";   p="$M\bartowski\mistralai_Magistral-Small-2509-GGUF\mistralai_Magistral-Small-2509-Q6_K.gguf";
    ladder=@(0,4096,16384,32768,65536);  extend=@()},
  @{n="GLM-4.7-Flash";          p="$M\bartowski\zai-org_GLM-4.7-Flash-GGUF\zai-org_GLM-4.7-Flash-Q5_K_M.gguf";
    ladder=@(0,4096,16384,32768,65536);  extend=@(131072)},
  @{n="Nex-N2.5-mini";          p="$M\bartowski\nex-agi_Nex-N2.5-mini-GGUF\nex-agi_Nex-N2.5-mini-Q4_K_M.gguf";
    ladder=@(0,4096,16384,32768,65536);  extend=@(131072)},
  @{n="Olmo-3.1-32B-Think";     p="$M\bartowski\allenai_Olmo-3.1-32B-Think-GGUF\allenai_Olmo-3.1-32B-Think-Q5_K_M.gguf";
    ladder=@(0,4096,16384,32768,65536);  extend=@()},
  @{n="DeepSeek-R1-Distill-32B";p="$M\bartowski\DeepSeek-R1-Distill-Qwen-32B-GGUF\DeepSeek-R1-Distill-Qwen-32B-Q5_K_M.gguf";
    ladder=@(0,4096,16384,32768,65536);  extend=@()},
  @{n="Qwen3-Coder-Next";       p="$M\unsloth\Qwen3-Coder-Next-GGUF\Qwen3-Coder-Next-UD-Q2_K_XL.gguf";
    ladder=@(0,4096,16384,32768);        extend=@(65536)},
  @{n="Mistral-Small-4-119B";   p="$M\mradermacher\Mistral-Small-4-119B-2603-i1-GGUF\Mistral-Small-4-119B-2603.i1-IQ1_M.gguf";
    ladder=@(0,4096,16384,32768);        extend=@(65536)},
  @{n="Nemotron-Super-49B";     p="$M\bartowski\nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-GGUF\nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-IQ4_XS.gguf";
    ladder=@(0,4096,16384,32768);        extend=@(65536)},
  @{n="Qwen3.5-122B-A10B";      p="$M\mradermacher\Qwen3.5-122B-A10B-i1-GGUF\Qwen3.5-122B-A10B.i1-IQ1_M.gguf";
    ladder=@(0,4096,16384,32768);        extend=@(65536)},
  @{n="Qwen3-Next-80B-Instruct";p="$M\bartowski\Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF\Qwen_Qwen3-Next-80B-A3B-Instruct-Q2_K.gguf";
    ladder=@(0,4096,16384,32768);        extend=@(65536)}
)

# Status comes back out-of-band in $script:DepthOk rather than as a return value.
# A function's return value in PowerShell is its ENTIRE output stream, so
# `return ($LASTEXITCODE -eq 0)` after calling bench.ps1 hands the caller an array
# of [every line bench printed..., $true] - which swallows the benchmark output
# and is always truthy, so the ceiling check below could never fire.
function Invoke-Depth($path, $depth) {
  $reps = if ($depth -ge 65536) { 2 } elseif ($depth -gt 0) { 3 } else { 5 }
  & $bench -Model $path -Suite quick -Tag "wave8-ladder" -Reps $reps `
           -Extra @("-d","$depth","-ctk","f16","-ctv","f16","-fa","on")
  $script:DepthOk = ($LASTEXITCODE -eq 0)
}

foreach ($m in $models) {
  if ($Only -and $m.n -notlike "*$Only*") { continue }
  if (-not (Test-Path $m.p)) { Write-Output "SKIP (not downloaded yet): $($m.n)"; continue }

  Write-Output ""
  Write-Output "################ $($m.n) ################"

  # Pass 1: headline. pp512/pp2048/tg128 at an empty cache, 5 reps.
  & $bench -Model $m.p -Suite standard -Tag "wave8" -Reps 5

  # Pass 2: the ladder.
  $script:DepthOk = $true
  foreach ($d in $m.ladder) {
    Write-Output "---- $($m.n)  d=$d ----"
    Invoke-Depth $m.p $d
    if (-not $script:DepthOk) { Write-Output "  CEILING: $($m.n) failed to allocate at d=$d - stopping ladder"; break }
  }

  # Only probe deeper if the model cleared its whole planned ladder. A model that
  # already hit its ceiling will fail every deeper depth too, at 10-20 min each.
  if ($script:DepthOk) {
    foreach ($d in $m.extend) {
      Write-Output "---- $($m.n)  d=$d (extension) ----"
      Invoke-Depth $m.p $d
      if (-not $script:DepthOk) { Write-Output "  CEILING: $($m.n) at d=$d"; break }
    }
  }
}

Write-Output ""
Write-Output "WAVE 8 COMPLETE"
