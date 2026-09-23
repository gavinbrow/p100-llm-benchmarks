# Wave 7: how deep can the KV cache actually go, and is the model still usable there?
#
# Every depth is a SEPARATE llama-bench invocation. A single invocation with a
# depth list is cheaper, but llama-bench aborts the whole run when one depth
# fails to allocate - and failing to allocate is the expected outcome at the top
# of this ladder, so batching would throw away the shallower results that worked.
#
# Phase A establishes which KV cache type fits at 72k. f16 is included knowing it
# will usually fail: "f16 cannot do this and q8_0 can" is the finding, not noise.
# Phase B then walks the depth ladder at whichever type Phase A showed to work.
#
# Reps drop to 2 past 64k. Each rep re-prefills the full cache, so a 3-rep run at
# 128k spends most of an hour on prefill that is never timed.

param(
  [switch]$SkipA,
  [switch]$SkipB,
  [string[]]$Only = @()
)

$ErrorActionPreference = "Continue"
$bench = "C:\Projects\p100 testing\scripts\bench.ps1"
$M     = "C:\Users\PC\.lmstudio\models"

# One representative per size class: the fastest model measured in that class at
# default settings, plus gpt-oss-20b as the wave-6 candidate for the same role.
$models = [ordered]@{
  "Qwen3.5-9B-Q4_0"           = "$M\unsloth\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q4_0.gguf"
  "gpt-oss-20b-MXFP4"         = "$M\ggml-org\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf"
  "Qwen3.8-27B-Q4_0"          = "$M\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_0.gguf"
  "Nemotron-3.5-30B-A3B-Q4_0" = "$M\ggml-org\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
}

$PROBE = 73728          # 72k: the shallowest depth that satisfies "70k or above"
$LADDER = @(0, 32768, 65536, 98304, 131072)
$KVTYPES = @("f16", "q8_0", "q4_0")

function Invoke-Depth($name, $path, $depth, $kv, $tag) {
  $reps = if ($depth -ge 65536) { 2 } else { 3 }
  # Quantised KV needs flash attention; pinning it on for every row here keeps
  # the f16 baseline comparable to the quantised ones rather than letting the
  # auto heuristic change two things at once.
  $extra = @("-d", "$depth", "-ctk", $kv, "-ctv", $kv, "-fa", "on")
  Write-Output "---- $name  d=$depth  kv=$kv ----"
  & $bench -Model $path -Suite quick -Tag $tag -Reps $reps -Extra $extra
}

foreach ($k in $models.Keys) {
  if ($Only.Count -gt 0 -and $Only -notcontains $k) { continue }
  $p = $models[$k]
  if (-not (Test-Path $p)) { Write-Output "SKIP (not downloaded): $k"; continue }

  Write-Output ""
  Write-Output "################ $k ################"

  $best = "q8_0"
  if (-not $SkipA) {
    foreach ($kv in $KVTYPES) { Invoke-Depth $k $p $PROBE $kv "kv-at-72k" }
  }
  if (-not $SkipB) {
    foreach ($d in $LADDER) { Invoke-Depth $k $p $d $best "longctx" }
  }
}

Write-Output ""
Write-Output "WAVE 7 COMPLETE"
