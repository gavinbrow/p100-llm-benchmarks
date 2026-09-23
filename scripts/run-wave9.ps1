# Wave 9 - 256k context for agents, then one card vs two. See report\test-plans\wave9.md.
#
# Same shape as run-wave8.ps1: one llama-bench invocation per depth, because an
# allocation failure aborts the whole run and throws away every depth that worked.
# A model's remaining depths are skipped after its first failure - a deeper depth
# needs strictly more memory, and each fill at this scale costs up to an hour.
#
# Usage: .\run-wave9.ps1 [-Phase 1|2|3|all] [-Only <substring>]

param([string]$Phase = "all", [string]$Only = "")

$ErrorActionPreference = "Continue"
$bench  = "C:\Projects\p100 testing\scripts\bench.ps1"
$MODELS = "C:\Users\PC\.lmstudio\models"   # not $M - see download8.ps1

# 256k = 261,120 tokens pre-filled, so pp512 on top (261,632) stays inside the
# 262,144-token trained window.
$D256 = 261120
$F16  = @("-ctk","f16","-ctv","f16","-fa","on")
$Q8   = @("-ctk","q8_0","-ctv","q8_0","-fa","on")

$nemo = "$MODELS\ggml-org\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"

# ---- phase 1: 256k. depths skip 131072 where waves 6-8 already have it. ----
$phase1 = @(
  @{n="Nemotron-3.5-Lightning Q4_0"; p=$nemo; depths=@($D256); kv=$F16},
  @{n="Nex-N2.5-mini Q4_K_M";  p="$MODELS\bartowski\nex-agi_Nex-N2.5-mini-GGUF\nex-agi_Nex-N2.5-mini-Q4_K_M.gguf"; depths=@($D256); kv=$F16},
  @{n="gemma-4-26B-A4B Q4_0";  p="$MODELS\lmstudio-community\gemma-4-26B-A4B-it-QAT-GGUF\gemma-4-26B-A4B-it-QAT-Q4_0.gguf"; depths=@($D256); kv=$F16},
  @{n="Qwen3.6-35B-A3B UD-Q4_K_M"; p="$MODELS\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"; depths=@(131072,$D256); kv=$F16},
  @{n="Ornith-1.5-35B Q4_K_M"; p="$MODELS\ornith-ai\Ornith-1.5-35B-A3B-GGUF\Ornith-1.5-35B-Q4_K_M.gguf"; depths=@(131072,$D256); kv=$F16},
  @{n="Qwen3.5-9B Q8_0";       p="$MODELS\unsloth\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q8_0.gguf"; depths=@(131072,$D256); kv=$F16},
  @{n="Qwen3-Coder-Next UD-Q2_K_XL"; p="$MODELS\unsloth\Qwen3-Coder-Next-GGUF\Qwen3-Coder-Next-UD-Q2_K_XL.gguf"; depths=@(131072,$D256); kv=$F16},
  @{n="Qwen3-Next-80B-Instruct Q2_K"; p="$MODELS\bartowski\Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF\Qwen_Qwen3-Next-80B-A3B-Instruct-Q2_K.gguf"; depths=@(131072,$D256); kv=$F16},
  @{n="Qwen3.5-122B-A10B i1-IQ1_M"; p="$MODELS\mradermacher\Qwen3.5-122B-A10B-i1-GGUF\Qwen3.5-122B-A10B.i1-IQ1_M.gguf"; depths=@(131072,$D256); kv=$F16},
  # q8_0 KV is the only way this all-attention model reaches 256k (24 GiB of f16 KV).
  @{n="Qwen3-Coder-30B-A3B Q4_K_M (q8_0 KV)"; p="$MODELS\unsloth\Qwen3-Coder-30B-A3B-Instruct-GGUF\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"; depths=@(131072,$D256); kv=$Q8},
  @{n="Qwen3.8-27B UD-IQ4_XS"; p="$MODELS\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-IQ4_XS.gguf"; depths=@(131072,$D256); kv=$F16},
  @{n="Qwen3.8-27B Q4_0";      p="$MODELS\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_0.gguf"; depths=@($D256); kv=$F16}
)

# ---- phase 2: one card vs two ----
$single = @(
  @{n="LFM2.5-2.6B Q4_K_M"; p="$MODELS\LiquidAI\LFM2.5-2.6B-GGUF\LFM2.5-2.6B-Q4_K_M.gguf"},
  @{n="LFM2.5-2.6B Q8_0";   p="$MODELS\LiquidAI\LFM2.5-2.6B-GGUF\LFM2.5-2.6B-Q8_0.gguf"},
  @{n="LFM2.5-2.6B F16";    p="$MODELS\LiquidAI\LFM2.5-2.6B-GGUF\LFM2.5-2.6B-F16.gguf"},
  @{n="Qwen3.8-2B Q8_0";    p="$MODELS\unsloth\Qwen3.8-2B-GGUF\Qwen3.8-2B-Q8_0.gguf"},
  @{n="Qwen3.5-9B Q4_K_M";  p="$MODELS\unsloth\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q4_K_M.gguf"},
  @{n="Qwen3.5-9B Q8_0";    p="$MODELS\unsloth\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q8_0.gguf"},
  @{n="Llama-3.1-8B Q8_0";  p="$MODELS\bartowski\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"},
  @{n="gemma-4-12B UD-Q4_K_XL"; p="$MODELS\unsloth\gemma-4-12B-it-qat-GGUF\gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"},
  @{n="gpt-oss-20b MXFP4";  p="$MODELS\ggml-org\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf"},
  @{n="Qwen3.8-27B UD-Q3_K_XL"; p="$MODELS\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q3_K_XL.gguf"},
  @{n="Qwen3.8-27B UD-IQ4_XS";  p="$MODELS\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-IQ4_XS.gguf"},
  @{n="gemma-4-26B-A4B Q4_0";   p="$MODELS\lmstudio-community\gemma-4-26B-A4B-it-QAT-GGUF\gemma-4-26B-A4B-it-QAT-Q4_0.gguf"},
  @{n="Mistral-Small-3.2-24B UD-Q4_K_XL"; p="$MODELS\unsloth\Mistral-Small-3.2-24B-Instruct-2506-GGUF\Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf"},
  @{n="phi-4 Q8_0";         p="$MODELS\unsloth\phi-4-GGUF\phi-4-Q8_0.gguf"},
  @{n="Qwen3.6-27B Q4_0";   p="$MODELS\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q4_0.gguf"},
  @{n="Qwen3.8-27B Q4_0";   p="$MODELS\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_0.gguf"}
)
$singleCtx = @(
  @{n="Qwen3.5-9B Q4_K_M"; p="$MODELS\unsloth\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q4_K_M.gguf"; depths=@(32768,131072,$D256)},
  @{n="gpt-oss-20b MXFP4"; p="$MODELS\ggml-org\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf"; depths=@(32768,130048)},
  @{n="gemma-4-26B-A4B Q4_0"; p="$MODELS\lmstudio-community\gemma-4-26B-A4B-it-QAT-GGUF\gemma-4-26B-A4B-it-QAT-Q4_0.gguf"; depths=@(32768,65536)},
  @{n="Qwen3.8-27B UD-Q3_K_XL"; p="$MODELS\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q3_K_XL.gguf"; depths=@(32768)}
)

# ---- phase 3: stretch ----
$phase3 = @(
  # 524288 completed 2026-09-16 15:52 (73.9 pp / 26.9 tg). The first run was stopped
  # ~65 min into the 1M fill; the resume starts at 1M.
  @{n="Nemotron-3.5-Lightning Q4_0 (1M trained)"; p=$nemo; depths=@(1047552); kv=$F16},
  @{n="Mistral-Small-4-119B i1-IQ1_M"; p="$MODELS\mradermacher\Mistral-Small-4-119B-2603-i1-GGUF\Mistral-Small-4-119B-2603.i1-IQ1_M.gguf"; depths=@(131072,$D256); kv=$F16}
)

# Status out-of-band: a PowerShell function returns its whole output stream.
function Invoke-One($path, $tag, $reps, [string[]]$extra) {
  $t0 = Get-Date
  & $bench -Model $path -Suite quick -Tag $tag -Reps $reps -Extra $extra
  $script:Ok = ($LASTEXITCODE -eq 0)
  Write-Output ("  [{0}] {1:N1} min" -f ($(if ($script:Ok) {"ok"} else {"FAIL"})), ((Get-Date) - $t0).TotalMinutes)
}

function Run-Ladder($list, $tag) {
  foreach ($m in $list) {
    if ($Only -and $m.n -notlike "*$Only*") { continue }
    if (-not (Test-Path $m.p)) { Write-Output "SKIP (missing file): $($m.n)"; continue }
    foreach ($d in $m.depths) {
      Write-Output ""; Write-Output "==== $(Get-Date -Format HH:mm)  $($m.n)  d=$d ===="
      Invoke-One $m.p $tag 2 (@("-d","$d") + $m.kv)
      if (-not $script:Ok) { Write-Output "  CEILING: $($m.n) at d=$d - skipping deeper"; break }
    }
  }
}

if ($Phase -in "1","all") { Write-Output "######## PHASE 1: 256k ########"; Run-Ladder $phase1 "wave9-256k" }

if ($Phase -in "2","all") {
  Write-Output ""; Write-Output "######## PHASE 2: one card vs two ########"
  foreach ($m in $single) {
    if ($Only -and $m.n -notlike "*$Only*") { continue }
    if (-not (Test-Path $m.p)) { Write-Output "SKIP (missing file): $($m.n)"; continue }
    Write-Output ""; Write-Output "==== $(Get-Date -Format HH:mm)  $($m.n)  d=0 ===="
    Invoke-One $m.p "wave9-1gpu" 5 (@("-dev","CUDA0") + $F16)
    Invoke-One $m.p "wave9-2gpu" 5 $F16
  }
  foreach ($m in $singleCtx) {
    if ($Only -and $m.n -notlike "*$Only*") { continue }
    $one = $true; $two = $true
    foreach ($d in $m.depths) {
      Write-Output ""; Write-Output "==== $(Get-Date -Format HH:mm)  $($m.n)  d=$d  1 vs 2 cards ===="
      if ($one) { Invoke-One $m.p "wave9-1gpu" 2 (@("-d","$d","-dev","CUDA0") + $F16); $one = $script:Ok }
      if ($two) { Invoke-One $m.p "wave9-2gpu" 2 (@("-d","$d") + $F16); $two = $script:Ok }
      if (-not ($one -or $two)) { break }
    }
  }
}

if ($Phase -in "3","all") { Write-Output ""; Write-Output "######## PHASE 3: stretch ########"; Run-Ladder $phase3 "wave9-stretch" }

Write-Output ""; Write-Output "WAVE 9 COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
