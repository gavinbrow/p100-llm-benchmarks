# Perplexity on wikitext-2 test, so the speed table can be paired with a
# quality column. Without this, "Q4_0 is fastest" is only half an argument -
# the question is always what you give up for it.
#
# --chunks is capped for wall-clock reasons: these are RELATIVE comparisons
# between quants of the same base model, all measured identically, not
# publication-grade absolute PPL figures. Chunk count is recorded per row.

param(
  [int]$Chunks = 40,
  [int]$Ctx = 512,
  [string[]]$Models = @()
)

$root = "C:\Projects\p100 testing"
Set-Location $root
# NB: PowerShell variable names are case-INSENSITIVE, so this must not be
# named $PPL - the parsed perplexity value below would overwrite the exe path.
$PPLEXE = "$root\tools\llamacpp\llama-perplexity.exe"
$DATA = "$root\data\wikitext-2-raw\wiki.test.raw"
$OUT  = "$root\results\perplexity.jsonl"

if ($Models.Count -eq 0) {
  $Models = @(
    "Qwen3.8-27B-UD-Q3_K_XL.gguf",
    "Qwen3.8-27B-UD-IQ4_XS.gguf",
    "Qwen3.8-27B-Q4_0.gguf",
    "Qwen3.8-27B-UD-Q6_K.gguf",
    "Qwen3.8-27B-Q8_0.gguf",
    "Qwen3.6-35B-A3B-MXFP4_MOE.gguf",
    "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
  )
}

foreach ($f in $Models) {
  $p = if (Test-Path $f) { $f } else { Join-Path "$root\models" $f }
  if (-not (Test-Path $p)) { Write-Output "SKIP missing: $f"; continue }
  $name = [System.IO.Path]::GetFileNameWithoutExtension($p)
  $log  = "$root\logs\ppl.$name.log"

  Write-Output "=== perplexity: $name (chunks=$Chunks ctx=$Ctx) ==="
  $t0 = Get-Date
  & $PPLEXE -m $p -f $DATA -c $Ctx --chunks $Chunks -ngl 999 2> $log | Out-Null
  $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 0)

  # llama-perplexity prints the running estimate to stderr; the last
  # "Final estimate: PPL = X +/- Y" line is what we want.
  $txt = Get-Content $log -Raw -ErrorAction SilentlyContinue
  $pplVal = $null; $pplErr = $null
  if ($txt -match 'Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)') {
    $pplVal = [double]$Matches[1]; $pplErr = [double]$Matches[2]
  } elseif ($txt -match '\[\d+\]([0-9.]+),\s*$') {
    $pplVal = [double]$Matches[1]
  }

  if ($null -eq $pplVal) {
    $tail = (Get-Content $log -Tail 4 -ErrorAction SilentlyContinue) -join ' | '
    Write-Output "  could not parse PPL: $tail"
  } else {
    Write-Output ("  PPL = {0} +/- {1}   ({2}s)" -f $pplVal, $pplErr, $secs)
  }

  $rec = [ordered]@{
    model=$name; ppl=$pplVal; ppl_stderr=$pplErr; chunks=$Chunks; ctx=$Ctx
    dataset="wikitext-2-raw/test"; wall_s=$secs
    ts=(Get-Date -Format "yyyyMMdd-HHmmss")
  } | ConvertTo-Json -Compress
  Add-Content -Path $OUT -Value $rec -Encoding utf8
}
Write-Output "PPL SWEEP COMPLETE"
