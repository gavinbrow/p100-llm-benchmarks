# Downloads the benchmark model set into .\models using curl.exe (streaming + resumable).
# Usage: powershell -ExecutionPolicy Bypass -File download.ps1 -Wave 1
param([int]$Wave = 1)

$ErrorActionPreference = "Continue"
$root = "C:\Projects\p100 testing\models"

$manifest = @(
  # --- Wave 1: Qwen3.8-27B quant sweep (the headline model) ---
  # IQ4_XS / Q3_K_XL are chosen to fit on a SINGLE 16GB P100 (single- vs dual-GPU test)
  @{r="unsloth/Qwen3.8-27B-GGUF";  f="Qwen3.8-27B-Q4_0.gguf";          w=1},
  @{r="unsloth/Qwen3.8-27B-GGUF";  f="Qwen3.8-27B-UD-Q3_K_XL.gguf";    w=1},
  @{r="unsloth/Qwen3.8-27B-GGUF";  f="Qwen3.8-27B-UD-IQ4_XS.gguf";     w=1},
  @{r="unsloth/Qwen3.8-27B-GGUF";  f="Qwen3.8-27B-UD-Q6_K.gguf";       w=1},
  @{r="unsloth/Qwen3.8-27B-GGUF";  f="Qwen3.8-27B-Q8_0.gguf";          w=1},

  # --- Wave 2: the size ladder, 3B -> 9B ---
  @{r="LiquidAI/LFM2.5-2.6B-GGUF"; f="LFM2.5-2.6B-Q4_0.gguf";          w=2},
  @{r="LiquidAI/LFM2.5-2.6B-GGUF"; f="LFM2.5-2.6B-Q4_K_M.gguf";        w=2},
  @{r="LiquidAI/LFM2.5-2.6B-GGUF"; f="LFM2.5-2.6B-Q8_0.gguf";          w=2},
  @{r="LiquidAI/LFM2.5-2.6B-GGUF"; f="LFM2.5-2.6B-F16.gguf";           w=2},
  @{r="unsloth/Qwen3.5-9B-GGUF";   f="Qwen3.5-9B-Q4_0.gguf";           w=2},
  @{r="unsloth/Qwen3.5-9B-GGUF";   f="Qwen3.5-9B-Q4_K_M.gguf";         w=2},
  @{r="unsloth/Qwen3.5-9B-GGUF";   f="Qwen3.5-9B-Q8_0.gguf";           w=2},
  @{r="unsloth/Qwen3.5-9B-GGUF";   f="Qwen3.5-9B-BF16.gguf";           w=2},

  # --- Wave 3: MoE + max-size-that-fits-32GB ---
  @{r="unsloth/Qwen3.6-35B-A3B-GGUF"; f="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";  w=3},
  @{r="unsloth/Qwen3.6-35B-A3B-GGUF"; f="Qwen3.6-35B-A3B-MXFP4_MOE.gguf";  w=3},
  @{r="unsloth/Qwen3.6-35B-A3B-GGUF"; f="Qwen3.6-35B-A3B-UD-Q6_K.gguf";    w=3},
  @{r="unsloth/Qwen3.8-27B-GGUF";     f="Qwen3.8-27B-UD-Q8_K_XL.gguf";     w=3},

  # --- Wave 4: breadth. other families, other architectures, gen-over-gen ---
  # Qwen3.6-27B vs Qwen3.8-27B is a like-for-like generational comparison
  @{r="unsloth/Qwen3.6-27B-GGUF";  f="Qwen3.6-27B-Q4_0.gguf";                    w=4},
  # most-downloaded GGUF on HF right now; MoE coder
  @{r="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"; f="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"; w=4},
  # a non-Qwen MoE, to check the MoE advantage generalises across families
  @{r="ornith-ai/Ornith-1.5-35B-A3B-GGUF"; f="Ornith-1.5-35B-Q4_K_M.gguf";       w=4},
  # QAT (quantisation-aware trained) dense 12B - different quality/size curve
  @{r="unsloth/gemma-4-12B-it-qat-GGUF"; f="gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"; w=4},
  # gemma ships a tiny MTP head: a second speculative-decoding data point
  @{r="unsloth/gemma-4-12B-it-qat-GGUF"; f="mtp-gemma-4-12B-it-Q8_0.gguf";       w=4}
)

New-Item -ItemType Directory -Force -Path $root | Out-Null

foreach ($m in $manifest | Where-Object { $_.w -eq $Wave }) {
  $out = Join-Path $root $m.f
  if (Test-Path $out) {
    $gb = [math]::Round((Get-Item $out).Length / 1GB, 2)
    Write-Output "SKIP (exists, $gb GB): $($m.f)"
    continue
  }
  $url = "https://huggingface.co/$($m.r)/resolve/main/$($m.f)"
  Write-Output "GET $($m.r) :: $($m.f)"
  $t0 = Get-Date
  # -C - resume, -L follow redirects, --retry survive transient HF 5xx
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
Write-Output "WAVE $Wave COMPLETE"
