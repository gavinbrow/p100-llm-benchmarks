# Wave 6: four models people actually run day to day.
#
# The library up to now was Qwen-heavy, which made the quant/architecture
# findings well-supported but said nothing about whether they generalise. These
# four are the most-downloaded members of four families not otherwise present -
# Mistral, OpenAI, Microsoft, Meta - so every result here is a check on whether
# the Pascal dequantisation story holds outside Qwen.
#
# Sizes are chosen to leave room for a large KV cache rather than to maximise
# quality: the long-context pass needs ~12-15 GB of the 32,574 MiB budget free.
#
# The eagle3 head is not a model under test. It is a draft for gpt-oss-20b and
# gives a second speculative-decoding mechanism to compare against Nemotron's MTP.

param([switch]$List)

$ErrorActionPreference = "Continue"
$dst = "C:\Users\PC\.lmstudio\models"

$manifest = @(
  @{r="unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF"; f="Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf"},
  @{r="ggml-org/gpt-oss-20b-GGUF";                        f="gpt-oss-20b-MXFP4.gguf"},
  @{r="ggml-org/gpt-oss-20b-GGUF";                        f="eagle3-gpt-oss-20b-Q8_0.gguf"},
  @{r="unsloth/phi-4-GGUF";                               f="phi-4-Q8_0.gguf"},
  @{r="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF";        f="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"}
)

if ($List) {
  foreach ($m in $manifest) { Join-Path $dst "$($m.r -replace '/','\')\$($m.f)" }
  return
}

foreach ($m in $manifest) {
  $dir = Join-Path $dst ($m.r -replace '/','\')
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  $out = Join-Path $dir $m.f
  $url = "https://huggingface.co/$($m.r)/resolve/main/$($m.f)"

  # A stub left behind by an interrupted run must not count as "already here",
  # so require a non-trivial size before skipping.
  if ((Test-Path $out) -and (Get-Item $out).Length -gt 1MB) {
    Write-Output ("SKIP  {0}  ({1:N1} GB already)" -f $m.f, ((Get-Item $out).Length/1GB))
    continue
  }
  if (Test-Path $out) { Remove-Item -LiteralPath $out -Force }

  Write-Output "GET   $($m.f)"
  curl.exe -L -C - --retry 5 --retry-delay 5 --retry-all-errors -s -S -o "$out.part" $url
  if ($LASTEXITCODE -ne 0) { Write-Output "  FAILED ($LASTEXITCODE)"; continue }
  Move-Item -LiteralPath "$out.part" -Destination $out -Force
  Write-Output ("  OK  {0:N1} GB" -f ((Get-Item $out).Length/1GB))
}

Write-Output "WAVE 6 COMPLETE"
