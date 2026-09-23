# Wave 6 benchmark pass: the four common models, plus the gemma-4 MoE that was
# downloaded in an earlier wave and never measured.
#
# Run quiet - see run-wave5.ps1 for why downloads must be idle first.

param([switch]$SkipStandard, [switch]$SkipContext)

$ErrorActionPreference = "Continue"
$bench = "C:\Projects\p100 testing\scripts\bench.ps1"
$M     = "C:\Users\PC\.lmstudio\models"

$models = [ordered]@{
  "Llama-3.1-8B-Instruct-Q8_0" = "$M\bartowski\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
  "gpt-oss-20b-MXFP4"          = "$M\ggml-org\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf"
  "phi-4-Q8_0"                 = "$M\unsloth\phi-4-GGUF\phi-4-Q8_0.gguf"
  "gemma-4-26B-A4B-it-QAT-Q4_0"= "$M\lmstudio-community\gemma-4-26B-A4B-it-QAT-GGUF\gemma-4-26B-A4B-it-QAT-Q4_0.gguf"
  "Mistral-Small-3.2-24B-UD-Q4_K_XL" = "$M\unsloth\Mistral-Small-3.2-24B-Instruct-2506-GGUF\Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf"
}

foreach ($k in $models.Keys) {
  $p = $models[$k]
  if (-not (Test-Path $p)) { Write-Output "SKIP (not downloaded): $k"; continue }

  Write-Output ""
  Write-Output "################ $k ################"

  if (-not $SkipStandard) { & $bench -Model $p -Suite standard -Tag "ladder"      -Reps 5 }
  if (-not $SkipContext)  { & $bench -Model $p -Suite context  -Tag "ctx-scaling" -Reps 3 }
}

Write-Output ""
Write-Output "WAVE 6 BENCH COMPLETE"
