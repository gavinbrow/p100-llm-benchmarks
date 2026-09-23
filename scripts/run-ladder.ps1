# Size ladder: 2.6B -> 35B, holding quantisation as constant as the available
# files allow, so the numbers isolate model size rather than quant format.
# Q4_0 is used wherever it exists because the quant sweep showed it is the
# fastest format on Pascal; Q4_K_M is run alongside as the "what most people
# actually download" comparison.

$root = "C:\Projects\p100 testing"
Set-Location $root

$ladder = @(
  @{ f="LFM2.5-2.6B-Q4_0.gguf";           note="2.6B dense" },
  @{ f="LFM2.5-2.6B-Q4_K_M.gguf";         note="2.6B dense" },
  @{ f="LFM2.5-2.6B-Q8_0.gguf";           note="2.6B dense" },
  @{ f="LFM2.5-2.6B-F16.gguf";            note="2.6B dense, fp16 (P100 has 2:1 FP16)" },
  @{ f="Qwen3.5-9B-Q4_0.gguf";            note="9B dense" },
  @{ f="Qwen3.5-9B-Q4_K_M.gguf";          note="9B dense" },
  @{ f="Qwen3.5-9B-Q8_0.gguf";            note="9B dense" },
  @{ f="Qwen3.5-9B-BF16.gguf";            note="9B dense, bf16" },
  @{ f="Qwen3.8-27B-Q4_0.gguf";           note="27B dense" },
  @{ f="Qwen3.6-35B-A3B-MXFP4_MOE.gguf";  note="35B MoE, 3B active" },
  @{ f="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";  note="35B MoE, 3B active" }
)

# Largest things that still fit entirely in 32GB of VRAM
$biggest = @(
  @{ f="Qwen3.8-27B-UD-Q8_K_XL.gguf";     note="biggest dense that fits 32GB" },
  @{ f="Qwen3.6-35B-A3B-UD-Q6_K.gguf";    note="biggest MoE that fits 32GB" },
  @{ f="Qwen3.8-27B-Q8_0.gguf";           note="27B Q8_0" }
)

foreach ($m in ($ladder + $biggest)) {
  $p = Join-Path "$root\models" $m.f
  if (-not (Test-Path $p)) { Write-Output "SKIP missing: $($m.f)"; continue }
  Write-Output ""
  Write-Output "########## $($m.f)  --  $($m.note)"
  & "$root\scripts\bench.ps1" -Model $p -Suite standard -Tag "ladder" -Reps 5
}
Write-Output ""
Write-Output "LADDER COMPLETE"
