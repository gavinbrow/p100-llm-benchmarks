# P100 benchmark runner.
# Wraps llama-bench, captures structured JSON + a VRAM/power/temp sample taken
# while the run is hot, and appends one row per test to results\raw.jsonl
#
# Usage:
#   .\bench.ps1 -Model <path> -Suite quick|standard|context|kv|devices -Tag "label"

param(
  [Parameter(Mandatory=$true)][string]$Model,
  [string]$Suite = "standard",
  [string]$Tag = "",
  [string[]]$Extra = @(),
  [int]$Reps = 5
)

$ErrorActionPreference = "Continue"
$LB   = "C:\Projects\p100 testing\tools\llamacpp\llama-bench.exe"
$OUT  = "C:\Projects\p100 testing\results\raw.jsonl"
$LOGD = "C:\Projects\p100 testing\logs"

if (-not (Test-Path $Model)) { Write-Output "MISSING MODEL: $Model"; exit 1 }
$name = [System.IO.Path]::GetFileNameWithoutExtension($Model)
if ($Tag -eq "") { $Tag = $Suite }

# ---- suite definitions -------------------------------------------------
# Each suite is an argument list appended to the base llama-bench invocation.
$suites = @{
  # sanity / fast signal
  "quick"    = @("-p","512","-n","128")

  # headline numbers: prompt processing + generation, both GPUs, fa auto
  "standard" = @("-p","512,2048","-n","128")

  # agent turn shape: read a ~2k-token tool result, then write (wave 10, used with -d)
  "agent"    = @("-p","2048","-n","128")

  # prompt processing only (wave 11 backfill of prefill cells, used with -d)
  "pp512"    = @("-p","512","-n","0")
  "pp2048"   = @("-p","2048","-n","0")

  # how does generation hold up as the KV cache fills?
  # -d N pre-fills the cache to depth N, then measures tg
  "context"  = @("-p","0","-n","128","-d","0,4096,16384,32768")

  # KV cache quantisation: does q8_0/q4_0 KV buy speed or only memory?
  "kv"       = @("-p","512","-n","128","-d","0,16384","-ctk","f16,q8_0,q4_0","-ctv","f16,q8_0,q4_0")

  # single GPU vs both (only valid for models that fit in 16GB)
  "devices"  = @("-p","512","-n","128","-dev","CUDA0","-dev","CUDA0,CUDA1")

  # flash attention on/off
  "fa"       = @("-p","512","-n","128","-fa","on,off")

  # batch/ubatch sweep - mostly affects prompt processing
  "batch"    = @("-p","2048","-n","0","-b","512,1024,2048","-ub","128,256,512")
}

if (-not $suites.ContainsKey($Suite)) { Write-Output "Unknown suite: $Suite"; exit 1 }

$stamp   = Get-Date -Format "yyyyMMdd-HHmmss"
$jsonOut = Join-Path $LOGD "$name.$Tag.$stamp.json"
$errOut  = Join-Path $LOGD "$name.$Tag.$stamp.stderr.txt"

$args = @("-m", $Model) + $suites[$Suite] + @("-r", "$Reps", "-o", "json") + $Extra

Write-Output "=== $name | suite=$Suite | reps=$Reps ==="
Write-Output "    $($args -replace '^-m$','')"

# Sample GPU state continuously while the benchmark runs. nvidia-smi writes
# straight to a CSV so we still get data if we have to kill the sampler.
$smiCsv = Join-Path $LOGD "$name.$Tag.$stamp.gpu.csv"
$sampler = Start-Process -FilePath "nvidia-smi" `
  -ArgumentList "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits","-l","1" `
  -RedirectStandardOutput $smiCsv -NoNewWindow -PassThru

$t0 = Get-Date
& $LB @args 1> $jsonOut 2> $errOut
$exit = $LASTEXITCODE
$elapsed = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

try { Stop-Process -Id $sampler.Id -Force -ErrorAction SilentlyContinue } catch {}
Start-Sleep -Milliseconds 500

# Aggregate peaks per GPU. Drop the first 5s of samples: that window is model
# load, not compute, so it would understate util and overstate nothing useful.
$gpu = $null
if (Test-Path $smiCsv) {
  $peak = @{}
  $lines = @(Get-Content $smiCsv -ErrorAction SilentlyContinue)
  if ($lines.Count -gt 10) { $lines = $lines[10..($lines.Count-1)] }
  foreach ($line in $lines) {
    $p = $line -split ',\s*'
    if ($p.Count -lt 5) { continue }
    $i = $p[0].Trim()
    if (-not $peak.ContainsKey($i)) { $peak[$i] = [ordered]@{mem_mib=0;util_pct=0;temp_c=0;power_w=0.0} }
    if ([int]$p[1]    -gt $peak[$i].mem_mib)  { $peak[$i].mem_mib  = [int]$p[1] }
    if ([int]$p[2]    -gt $peak[$i].util_pct) { $peak[$i].util_pct = [int]$p[2] }
    if ([int]$p[3]    -gt $peak[$i].temp_c)   { $peak[$i].temp_c   = [int]$p[3] }
    if ([double]$p[4] -gt $peak[$i].power_w)  { $peak[$i].power_w  = [double]$p[4] }
  }
  if ($peak.Count -gt 0) { $gpu = $peak }
}

if ($exit -ne 0) {
  $why = (Get-Content $errOut -Tail 6) -join ' | '
  Write-Output "  FAILED (exit $exit): $why"
  $rec = [ordered]@{ ok=$false; model=$name; suite=$Suite; tag=$Tag; ts=$stamp; error=$why } | ConvertTo-Json -Compress
  Add-Content -Path $OUT -Value $rec -Encoding utf8
  exit $exit
}

# llama-bench emits a JSON array, one object per test
$rows = Get-Content $jsonOut -Raw | ConvertFrom-Json
foreach ($r in $rows) {
  $rec = [ordered]@{
    ok            = $true
    model         = $name
    suite         = $Suite
    tag           = $Tag
    ts            = $stamp
    model_type    = $r.model_type
    model_size_gb = [math]::Round($r.model_size / 1GB, 3)
    params_b      = [math]::Round($r.model_n_params / 1e9, 2)
    n_prompt      = $r.n_prompt
    n_gen         = $r.n_gen
    n_depth       = $r.n_depth
    n_batch       = $r.n_batch
    n_ubatch      = $r.n_ubatch
    type_k        = $r.type_k
    type_v        = $r.type_v
    flash_attn    = $r.flash_attn
    split_mode    = $r.split_mode
    devices       = ($r.devices -join ',')
    n_gpu_layers  = $r.n_gpu_layers
    n_cpu_moe     = $r.n_cpu_moe
    tps_avg       = [math]::Round($r.avg_ts, 3)
    tps_stddev    = [math]::Round($r.stddev_ts, 3)
    ms_avg        = [math]::Round($r.avg_ns / 1e6, 2)
    reps          = $Reps
    build         = $r.build_commit
    gpu_peak      = $gpu
    wall_s        = $elapsed
  } | ConvertTo-Json -Depth 5 -Compress
  Add-Content -Path $OUT -Value $rec -Encoding utf8

  $label = if ($r.n_prompt -gt 0) { "pp$($r.n_prompt)" } else { "tg$($r.n_gen)" }
  if ($r.n_depth -gt 0) { $label += "@d$($r.n_depth)" }
  Write-Output ("  {0,-16} {1,8:N2} t/s  +/- {2,5:N2}   [k={3} v={4} fa={5} dev={6}]" -f `
    $label, $r.avg_ts, $r.stddev_ts, $r.type_k, $r.type_v, $r.flash_attn, ($r.devices -join '+'))
}
Write-Output "  done in ${elapsed}s -> $OUT"
