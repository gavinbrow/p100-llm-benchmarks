# Speculative decoding test. llama-bench can't do spec-dec, so this drives
# llama-server directly and reads the per-request timings it reports.
#
# Gains from speculative decoding are extremely workload-dependent - code and
# structured text draft well, freeform prose does not - so we run a fixed set of
# three prompt types and report each separately rather than a single average.

param(
  [Parameter(Mandatory=$true)][string]$Model,
  [string]$Draft = "",
  [string]$SpecType = "none",
  [int]$DraftMax = 3,
  [string]$Tag = "",
  [int]$NPredict = 256,
  [string[]]$Extra = @()
)

$ErrorActionPreference = "Continue"
$SRV  = "C:\Projects\p100 testing\tools\llamacpp\llama-server.exe"
$OUT  = "C:\Projects\p100 testing\results\specdec.jsonl"
$LOGD = "C:\Projects\p100 testing\logs"
$PORT = 8099

if ($Tag -eq "") { $Tag = $SpecType }
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$name  = [System.IO.Path]::GetFileNameWithoutExtension($Model)
$slog  = Join-Path $LOGD "server.$name.$Tag.$stamp.log"

# Three workloads with very different draft-predictability
$prompts = @(
  @{ kind="code";  text="Write a complete Python implementation of a red-black tree with insert, delete, and in-order traversal. Include type hints and docstrings." }
  @{ kind="prose"; text="Write a vivid, original short story about a lighthouse keeper who discovers something unexpected in the fog. Do not use cliches." }
  @{ kind="fact";  text="List the planets of the solar system in order from the sun. For each, give its diameter in km, orbital period, and number of moons, formatted as a markdown table." }
)

$args = @("-m", $Model, "--port", "$PORT", "--host", "127.0.0.1",
          "-ngl", "999", "-c", "8192", "--no-webui") + $Extra
if ($SpecType -ne "none") {
  $args += @("--spec-type", $SpecType, "--spec-draft-n-max", "$DraftMax")
  if ($Draft -ne "") { $args += @("-md", $Draft, "-ngld", "999") }
}

Write-Output "=== $name | spec=$SpecType | draft=$(if($Draft){Split-Path $Draft -Leaf}else{'none'}) | n-max=$DraftMax ==="

$srv = Start-Process -FilePath $SRV -ArgumentList $args -PassThru -NoNewWindow `
        -RedirectStandardOutput "$slog" -RedirectStandardError "$slog.err"

# wait for readiness (model load on these can take ~60s from cold cache)
$ready = $false
foreach ($i in 1..180) {
  Start-Sleep -Seconds 2
  if ($srv.HasExited) { break }
  try {
    $h = Invoke-RestMethod -Uri "http://127.0.0.1:$PORT/health" -TimeoutSec 3 -ErrorAction Stop
    if ($h.status -eq "ok") { $ready = $true; break }
  } catch {}
}

if (-not $ready) {
  $tail = (Get-Content "$slog.err" -Tail 8 -ErrorAction SilentlyContinue) -join ' | '
  Write-Output "  SERVER FAILED TO START: $tail"
  if (-not $srv.HasExited) { Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue }
  return
}

foreach ($p in $prompts) {
  # greedy decoding: makes acceptance rates comparable run to run
  $body = @{
    prompt      = $p.text
    n_predict   = $NPredict
    temperature = 0.0
    top_k       = 1
    cache_prompt = $false
  } | ConvertTo-Json -Depth 4

  try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$PORT/completion" -Method Post `
           -Body $body -ContentType "application/json" -TimeoutSec 900
  } catch {
    Write-Output "  $($p.kind): request failed - $($_.Exception.Message)"
    continue
  }

  $t = $r.timings
  $acc = $null
  if ($null -ne $t.draft_n -and $t.draft_n -gt 0) {
    $acc = [math]::Round($t.draft_n_accepted / $t.draft_n * 100, 1)
  }

  $rec = [ordered]@{
    model=$name; spec_type=$SpecType; tag=$Tag
    draft=$(if($Draft){Split-Path $Draft -Leaf}else{$null}); draft_n_max=$DraftMax
    kind=$p.kind; ts=$stamp
    predicted_n=$t.predicted_n
    predicted_ms=[math]::Round($t.predicted_ms,1)
    tps=[math]::Round($t.predicted_per_second,3)
    prompt_n=$t.prompt_n
    prompt_tps=[math]::Round($t.prompt_per_second,2)
    draft_n=$t.draft_n; draft_n_accepted=$t.draft_n_accepted; accept_pct=$acc
  } | ConvertTo-Json -Depth 4 -Compress
  Add-Content -Path $OUT -Value $rec -Encoding utf8

  $accs = if ($null -ne $acc) { "  accept=$acc%" } else { "" }
  Write-Output ("  {0,-6} {1,7:N2} t/s  ({2} tok){3}" -f $p.kind, $t.predicted_per_second, $t.predicted_n, $accs)
}

Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
