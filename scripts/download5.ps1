# Wave 5: the "bigger models" set. Downloads straight into the LM Studio tree.
#
# Selection rule: every file must be LARGER than the previous ceiling (29.3 GB /
# 35B params) in at least one dimension, must still fit 32,574 MiB of VRAM, and
# must answer a question the existing 109 runs do not.
#
#   Qwen3-Next-80B-A3B  Q2_K       26.2 GB  most params that fit (80B); new hybrid attn
#   Llama-3.3-70B       UD-Q2_K_XL 25.1 GB  largest DENSE - the bandwidth thesis at 70B
#   Seed-OSS-36B        Q6_K       27.6 GB  largest dense at a HIGH quant
#   Qwen3-32B           Q6_K       25.0 GB  fills the 27B->70B dense gap
#   Nemotron-3.5-30B    Q4_0       17.6 GB  new family + ships an MTP head
#   Nemotron-3.5-30B    mtp Q4_0    1.1 GB  the MTP test that 404'd in wave 4
#   Nemotron-3.5-30B    Q8_0       31.3 GB  deliberate VRAM-ceiling probe (may not fit)

param([switch]$List)

$ErrorActionPreference = "Continue"
$root = "C:\Users\PC\.lmstudio\models"

$manifest = @(
  @{r="bartowski/Qwen_Qwen3-Next-80B-A3B-Thinking-GGUF"; f="Qwen_Qwen3-Next-80B-A3B-Thinking-Q2_K.gguf"},
  @{r="unsloth/Llama-3.3-70B-Instruct-GGUF";             f="Llama-3.3-70B-Instruct-UD-Q2_K_XL.gguf"},
  @{r="lmstudio-community/Seed-OSS-36B-Instruct-GGUF";   f="Seed-OSS-36B-Instruct-Q6_K.gguf"},
  @{r="unsloth/Qwen3-32B-GGUF";                          f="Qwen3-32B-Q6_K.gguf"},
  @{r="ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF"; f="NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"},
  @{r="ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF"; f="mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"},
  @{r="ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF"; f="NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf"}
)

if ($List) { $manifest | ForEach-Object { Write-Output (Join-Path (Join-Path $root $_.r) $_.f) }; exit 0 }

foreach ($m in $manifest) {
  $dir = Join-Path $root $m.r
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $out = Join-Path $dir $m.f

  # wave 4 produced a 0-byte file when two downloads raced on the same .part,
  # and the plain -Test-Path skip then treated the stub as complete. Require
  # a non-trivial size before believing a file is already here.
  if ((Test-Path $out) -and (Get-Item $out).Length -gt 1MB) {
    Write-Output "SKIP (exists, $([math]::Round((Get-Item $out).Length/1GB,2)) GB): $($m.f)"
    continue
  }
  if (Test-Path $out) { Remove-Item -LiteralPath $out -Force }

  $url = "https://huggingface.co/$($m.r)/resolve/main/$($m.f)"
  Write-Output "GET $($m.r) :: $($m.f)"
  $t0 = Get-Date
  curl.exe -L -C - --retry 5 --retry-delay 5 --retry-all-errors -s -S -o "$out.part" $url
  if ($LASTEXITCODE -eq 0) {
    Move-Item -Force "$out.part" $out
    $gb = [math]::Round((Get-Item $out).Length / 1GB, 2)
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 0)
    $mbps = if ($secs -gt 0) { [math]::Round($gb * 1024 / $secs, 1) } else { 0 }
    Write-Output "  OK $gb GB in ${secs}s (${mbps} MB/s)"
  } else {
    Write-Output "  FAIL $($m.f): curl exit $LASTEXITCODE"
  }
}
Write-Output "WAVE 5 COMPLETE"
