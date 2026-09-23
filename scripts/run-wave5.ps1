# Wave 5 benchmark pass: the big models.
#
# Run AFTER downloads are idle. Concurrent download traffic does not touch the
# GPU, but it did visibly inflate run-to-run variance once (13.88 +/- 0.59 vs
# 14.55 +/- 0.00 re-measured quiet), so headline numbers are taken quiet.

param([switch]$SkipStandard, [switch]$SkipContext)

$ErrorActionPreference = "Continue"
$bench = "C:\Projects\p100 testing\scripts\bench.ps1"
$M     = "C:\Users\PC\.lmstudio\models"

# name -> path. Ordered small-to-large so a VRAM failure late does not block the
# rest of the pass.
$models = [ordered]@{
  "Nemotron-3.5-30B-A3B-Q4_0" = "$M\ggml-org\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
  "Qwen3-32B-Q6_K"            = "$M\unsloth\Qwen3-32B-GGUF\Qwen3-32B-Q6_K.gguf"
  "Llama-3.3-70B-UD-Q2_K_XL"  = "$M\unsloth\Llama-3.3-70B-Instruct-GGUF\Llama-3.3-70B-Instruct-UD-Q2_K_XL.gguf"
  "Qwen3-Next-80B-A3B-Q2_K"   = "$M\bartowski\Qwen_Qwen3-Next-80B-A3B-Thinking-GGUF\Qwen_Qwen3-Next-80B-A3B-Thinking-Q2_K.gguf"
  "Seed-OSS-36B-Q6_K"         = "$M\lmstudio-community\Seed-OSS-36B-Instruct-GGUF\Seed-OSS-36B-Instruct-Q6_K.gguf"
  "Nemotron-3.5-30B-A3B-Q8_0" = "$M\ggml-org\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf"
}

foreach ($k in $models.Keys) {
  $p = $models[$k]
  if (-not (Test-Path $p)) { Write-Output "SKIP (not downloaded): $k"; continue }

  Write-Output ""
  Write-Output "################ $k ################"

  if (-not $SkipStandard) {
    # pp512/pp2048 + tg128 at defaults - the row that makes this model
    # comparable to every other model in the study.
    & $bench -Model $p -Suite standard -Tag "ladder" -Reps 5
  }
  if (-not $SkipContext) {
    # How generation holds up as the cache fills. 3 reps; these are slow.
    & $bench -Model $p -Suite context -Tag "ctx-scaling" -Reps 3
  }
}

Write-Output ""
Write-Output "WAVE 5 BENCH COMPLETE"
