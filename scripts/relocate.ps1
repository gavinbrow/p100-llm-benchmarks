# Moves the benchmark weights out of .\models and into the LM Studio models tree
# so they show up in LM Studio's "My Models" pane alongside the user's own downloads.
#
# LM Studio indexes <downloadsFolder>\<publisher>\<repo>\<file>.gguf - a two-level
# layout. Anything flat or three-deep is ignored, so the HF repo each file came
# from is reproduced exactly.
#
# Same volume, so Move-Item is a rename: ~290 GB relocates instantly, no copy.

$ErrorActionPreference = "Stop"
$src = "C:\Projects\p100 testing\models"
$dst = "C:\Users\PC\.lmstudio\models"

# filename glob -> HF repo it was fetched from
$map = @(
  @{p="Qwen3.8-27B-DFlash2-*";        r="z-lab/Qwen3.8-27B-DFlash2-GGUF"},
  @{p="Qwen3.8-27B-*";                r="unsloth/Qwen3.8-27B-GGUF"},
  @{p="Qwen3.8-2B-*";                 r="unsloth/Qwen3.8-2B-GGUF"},
  @{p="LFM2.5-2.6B-*";                r="LiquidAI/LFM2.5-2.6B-GGUF"},
  @{p="Qwen3.5-9B-*";                 r="unsloth/Qwen3.5-9B-GGUF"},
  @{p="Qwen3.6-35B-A3B-*";            r="unsloth/Qwen3.6-35B-A3B-GGUF"},
  @{p="Qwen3.6-27B-*";                r="unsloth/Qwen3.6-27B-GGUF"},
  @{p="Qwen3-Coder-30B-A3B-*";        r="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"},
  @{p="Ornith-1.5-35B-*";             r="ornith-ai/Ornith-1.5-35B-A3B-GGUF"},
  @{p="*gemma-4-12B-it-qat-*";        r="unsloth/gemma-4-12B-it-qat-GGUF"}
)

if (-not (Test-Path $src)) { Write-Output "nothing to relocate"; exit 0 }

$moved = 0; $bytes = 0
foreach ($f in Get-ChildItem $src -Filter *.gguf) {
  # first matching pattern wins, so put the more specific globs earlier in $map
  $hit = $map | Where-Object { $f.Name -like $_.p } | Select-Object -First 1
  if (-not $hit) { Write-Output "NO MAPPING (left in place): $($f.Name)"; continue }

  $target = Join-Path $dst $hit.r
  New-Item -ItemType Directory -Force -Path $target | Out-Null
  $out = Join-Path $target $f.Name
  if (Test-Path $out) { Write-Output "EXISTS (skipped): $($hit.r)\$($f.Name)"; continue }

  Move-Item -LiteralPath $f.FullName -Destination $out
  $moved++; $bytes += $f.Length
  Write-Output ("MOVED {0,-45} -> {1}" -f $f.Name, $hit.r)
}
Write-Output ""
Write-Output "$moved files, $([math]::Round($bytes/1GB,1)) GB relocated to $dst"
