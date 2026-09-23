# Wave 7b: the depth ladder again, at f16 KV.
#
# Wave 7 walked the ladder at q8_0 because that is the type guaranteed to fit
# everywhere, which made the four models comparable. But its Phase A probe showed
# f16 beating q8_0 at 72k on every single model - by 27% on Qwen3.5-9B and 73% on
# gpt-oss-20b - so the q8_0 ladder is a picture of the configuration nobody
# should actually run. This fills in the one you should.
#
# 72k is already measured at f16 by wave 7 Phase A, so only the other four depths
# are repeated here. The three models are the ones fast enough for the result to
# matter; the dense 27B is left on its Phase A point because each of its deep
# runs costs 10-20 minutes and the conclusion there is already unambiguous.

$ErrorActionPreference = "Continue"
$bench = "C:\Projects\p100 testing\scripts\bench.ps1"
$M     = "C:\Users\PC\.lmstudio\models"

$models = [ordered]@{
  "Qwen3.5-9B-Q4_0"           = "$M\unsloth\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q4_0.gguf"
  "gpt-oss-20b-MXFP4"         = "$M\ggml-org\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf"
  "Nemotron-3.5-30B-A3B-Q4_0" = "$M\ggml-org\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
}

foreach ($k in $models.Keys) {
  $p = $models[$k]
  if (-not (Test-Path $p)) { Write-Output "SKIP (not downloaded): $k"; continue }
  Write-Output ""
  Write-Output "################ $k ################"
  foreach ($d in @(0, 32768, 65536, 98304, 131072)) {
    $reps = if ($d -ge 65536) { 2 } else { 3 }
    Write-Output "---- $k  d=$d  kv=f16 ----"
    & $bench -Model $p -Suite quick -Tag "longctx-f16" -Reps $reps `
             -Extra @("-d", "$d", "-ctk", "f16", "-ctv", "f16", "-fa", "on")
  }
}

Write-Output ""
Write-Output "WAVE 7B COMPLETE"
