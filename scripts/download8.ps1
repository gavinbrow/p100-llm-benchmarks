# Wave 8 downloads - 11 models, 245 GB, into the LM Studio tree.
#
# Filenames here were NOT guessed. Every one was read from
# https://huggingface.co/api/models/<repo> (siblings.rfilename) during preflight,
# because two of them do not follow the convention the source table implied:
# google publishes "gemma-4-31B_q4_0-it.gguf" (quant mid-name, lowercase) and
# mradermacher uses a dot before the i1- prefix. See report\test-plans\wave8.md.
#
# Ordered ascending by size: first results land soonest and the disk stays
# comfortable longest. 264 GB free at start, ~19 GB at finish.

$ErrorActionPreference = "Continue"

# NOT $M. PowerShell variable names are case-insensitive, so $M and the $m loop
# variable below are the same variable - $M silently became the manifest hashtable
# on the first iteration and Join-Path stringified it, dumping 37 GB into a
# directory literally named "System.Collections.Hashtable".
$MODELS = "C:\Users\PC\.lmstudio\models"

$manifest = @(
  @{r="google/gemma-4-31B-it-qat-q4_0-gguf";                     f="gemma-4-31B_q4_0-it.gguf";                            gb=16.44},
  @{r="bartowski/mistralai_Magistral-Small-2509-GGUF";           f="mistralai_Magistral-Small-2509-Q6_K.gguf";             gb=18.02},
  @{r="bartowski/zai-org_GLM-4.7-Flash-GGUF";                    f="zai-org_GLM-4.7-Flash-Q5_K_M.gguf";                    gb=20.09},
  @{r="bartowski/nex-agi_Nex-N2.5-mini-GGUF";                    f="nex-agi_Nex-N2.5-mini-Q4_K_M.gguf";                    gb=20.79},
  @{r="bartowski/allenai_Olmo-3.1-32B-Think-GGUF";               f="allenai_Olmo-3.1-32B-Think-Q5_K_M.gguf";               gb=21.29},
  @{r="bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF";             f="DeepSeek-R1-Distill-Qwen-32B-Q5_K_M.gguf";             gb=21.66},
  @{r="unsloth/Qwen3-Coder-Next-GGUF";                           f="Qwen3-Coder-Next-UD-Q2_K_XL.gguf";                     gb=24.92},
  @{r="mradermacher/Mistral-Small-4-119B-2603-i1-GGUF";          f="Mistral-Small-4-119B-2603.i1-IQ1_M.gguf";              gb=24.85},
  @{r="bartowski/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-GGUF"; f="nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-IQ4_XS.gguf"; gb=25.03},
  @{r="mradermacher/Qwen3.5-122B-A10B-i1-GGUF";                  f="Qwen3.5-122B-A10B.i1-IQ1_M.gguf";                      gb=25.70},
  @{r="bartowski/Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF";         f="Qwen_Qwen3-Next-80B-A3B-Instruct-Q2_K.gguf";           gb=26.23}
)

foreach ($m in $manifest) {
  $dir = Join-Path $MODELS ($m.r -replace '/', '\')
  $out = Join-Path $dir $m.f
  New-Item -ItemType Directory -Force -Path $dir | Out-Null

  # A stub left behind by an interrupted run must not count as "already here",
  # so require a non-trivial size before skipping.
  if ((Test-Path $out) -and (Get-Item $out).Length -gt 1MB) {
    "SKIP (present): $($m.f)"
    continue
  }

  $free = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
  "==== $($m.f)  ($($m.gb) GB, ${free} GB free) ===="
  if ($free -lt ($m.gb + 5)) {
    "ABORT: only ${free} GB free, need $($m.gb) GB plus headroom. Stopping before $($m.f)."
    break
  }

  $url = "https://huggingface.co/$($m.r)/resolve/main/$($m.f)"
  $t0 = Get-Date
  # -C - resumes a partial file; downloading to .part means an interrupted run
  # can never leave a truncated file that looks complete to the skip check above.
  curl.exe -L -C - --retry 5 --retry-delay 5 --retry-all-errors -s -S -o "$out.part" $url
  if ($LASTEXITCODE -ne 0) { "  FAILED (curl $LASTEXITCODE): $($m.f)"; continue }

  Move-Item "$out.part" $out -Force
  $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
  $gb   = [math]::Round((Get-Item $out).Length / 1GB, 2)
  "  done: $gb GB in $mins min"
}

"WAVE 8 DOWNLOADS COMPLETE"
