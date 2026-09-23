# Builds THE master table: every token-generation measurement in the study, one
# row each, fully qualified by the settings that produced it.
#
# Nothing is held constant off-screen, so any row can be reproduced from the row
# itself, and any two rows can be compared by reading across. Outputs
# results\master.csv, report\MASTER-TABLE.md and report\master-data.js, and injects
# the same data into report\p100-report.html.

$ErrorActionPreference = "Stop"
$root = "C:\Projects\p100 testing"
$raw  = Join-Path $root "results\raw.jsonl"
$csv  = Join-Path $root "results\master.csv"
$md   = Join-Path $root "report\MASTER-TABLE.md"
$js   = Join-Path $root "report\master-data.js"   # same-origin data file for the published report

$PEAK_BW_GBS = 732.2      # Tesla P100-PCIE-16GB HBM2, per card
$GIB_TO_GB   = 1.073741824

# Windows PowerShell 5.1's "-Encoding utf8" always writes a byte-order mark, which
# breaks json.loads, jq and Python's csv module on the published files. Every output
# below is written through .NET with this BOM-less encoding instead.
$UTF8 = New-Object System.Text.UTF8Encoding $false

# Splits "Qwen3.6-35B-A3B-UD-Q4_K_M" into base model and quant. The quant is the
# trailing type, optionally preceded by a publisher's prefix - unsloth's UD-
# (dynamic) or mradermacher's i1- (imatrix).
#
# Three shapes are not the trailing-hyphen convention:
#   - mradermacher delimits with a dot:  "Qwen3.5-122B-A10B.i1-IQ1_M"
#   - google puts the quant mid-name:    "gemma-4-31B_q4_0-it"
#   - bartowski prefixes the publisher:  "nex-agi_Nex-N2.5-mini-Q4_K_M"
# The first two need their own patterns; the third only needs stripping, and is
# stripped for every model so the table does not carry six different vendor
# prefixes in a column that is already wide.
function Split-Name([string]$n) {
  $n = $n -replace '^(bartowski|unsloth|mradermacher|ggml-org|google|nvidia|mistralai|allenai|nex-agi|zai-org|Qwen)_', ''

  # google QAT: "gemma-4-31B_q4_0-it" -> base "gemma-4-31B-it", quant "Q4_0"
  if ($n -match '^(?<a>.+?)_(?<q>[qQ]\d[^-]*)-(?<b>.+)$') {
    return @{ base = "$($Matches.a)-$($Matches.b)"; quant = $Matches.q.ToUpper() }
  }
  # trailing quant, after either a hyphen or mradermacher's dot
  if ($n -match '^(?<b>.+?)[-.](?<q>(UD-|i1-)?(IQ\d\S*|Q\d\S*|MXFP4(_MOE)?|BF16|F16))$') {
    return @{ base = $Matches.b; quant = $Matches.q }
  }
  return @{ base = $n; quant = "" }
}

# $all keeps the prompt-processing rows too - they are a different test of the same
# invocation, and the prefill columns below are built from them. $rows is generation only,
# one row per measurement, which is what the table lists.
$all  = Get-Content $raw -Encoding UTF8 | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.ok -and $_.tag -notlike 'harness-test*' }
$rows = $all | Where-Object { $_.n_gen -gt 0 }

# Prompt-processing rates measured in the SAME llama-bench invocation, at the same depth and
# settings, so every generation row can carry the prefill rate that goes with it. llama-bench
# emits pp and tg as separate tests of one run, which share a timestamp; the depth and the
# cache/device settings are part of the key because one invocation can sweep several.
$ppIndex = @{}
# Wave 11 runs layer and tensor split in one invocation, so the split mode and the expert
# offload are part of the key too (both are null on older rows, which keeps their keys stable).
foreach ($r in ($all | Where-Object { $_.n_prompt -gt 0 -and $_.tag -ne 'wave11-backfill' })) {
  $k = "$($r.ts)|$($r.n_depth)|$($r.type_k)|$($r.devices)|$($r.n_ubatch)|$($r.split_mode)|$($r.n_cpu_moe)"
  if (-not $ppIndex.ContainsKey($k)) { $ppIndex[$k] = @{} }
  $ppIndex[$k]["pp$($r.n_prompt)"] = $r.tps_avg
}

# Wave 11 backfill: prefill measured later, with prompt processing only, for generation rows
# whose own invocation skipped it. Keyed on the settings rather than the timestamp. Flash
# attention "auto" resolves to on for every model here, so -1 and 1 share a key.
$bfIndex = @{}
foreach ($r in ($all | Where-Object { $_.tag -eq 'wave11-backfill' -and $_.n_prompt -gt 0 })) {
  $fa = if ([int]$r.flash_attn -eq 0) { 0 } else { 1 }
  $k = "$($r.model)|$($r.n_depth)|$($r.type_k)|$($r.type_v)|$($r.devices)|$($r.n_ubatch)|$($r.n_batch)|$fa"
  if (-not $bfIndex.ContainsKey($k)) { $bfIndex[$k] = @{} }
  $bfIndex[$k]["pp$($r.n_prompt)"] = $r.tps_avg
}

# ---------- wave 11: llama-server one-fill sweeps (results\wave11.jsonl) ----------
# A second measurement method: one server fills the context once and climbs, reading 512 and
# then 2,048 new tokens and writing 128 at each level (see scripts\wave11b.py). Its rows fill
# prefill cells llama-bench left empty, and also appear as generation rows of their own, marked
# by the Method column. Rows named by an "exclusion" record (a throttled run) are dropped.
$w11 = Join-Path $root "results\wave11.jsonl"
$w11rows = @(Get-Content $w11 -Encoding UTF8 | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
$exFrom  = @($w11rows | Where-Object { $_.exp -eq 'exclusion' } | ForEach-Object { $_.match.ts_from })
$sweeps  = @($w11rows | Where-Object { $_.exp -eq 'depth_sweep' -and $_.method -eq 'server-climb-v2' -and
                                        $null -ne $_.pp512_tps -and $_.clean })
$sweeps  = @($sweeps | Where-Object { $t = $_.ts; -not ($exFrom | Where-Object { $t -ge $_ }) })
$swSummaries = @($w11rows | Where-Object { $_.exp -eq 'depth_sweep' -and $_.summary })
function Arg-Of($list, $flag, $default) {
  $a = @($list); $i = [array]::IndexOf($a, $flag)
  if ($i -ge 0) { $a[$i + 1] } else { $default }
}
function Sweep-Key($model, $ctk, $dev, $ub, $b, $split) { "$model|$ctk|$dev|$ub|$b|$split" }
$swIndex = @{}
foreach ($sw in $sweeps) {
  $k = Sweep-Key $sw.model (Arg-Of $sw.server_args '-ctk' 'f16') (Arg-Of $sw.server_args '-dev' 'auto') `
                 (Arg-Of $sw.server_args '-ub' '512') (Arg-Of $sw.server_args '-b' '2048') $sw.split
  if (-not $swIndex.ContainsKey($k)) { $swIndex[$k] = New-Object System.Collections.ArrayList }
  [void]$swIndex[$k].Add($sw)
}

function Is-MoE([string]$m) {
  ($m -match 'A\d+B|MoE|MXFP4_MOE') -or
  ($m -match '(^|_)(Ornith-1\.5-35B|gpt-oss-|Qwen3-Coder-Next|Mistral-Small-4-119B|zai-org_GLM-4\.7-Flash|GLM-4\.7-Flash|nex-agi_Nex-N2\.5-mini|Nex-N2\.5-mini)')
}

$flat = $rows | ForEach-Object {
  $row = $_
  $sp = Split-Name $_.model
  $sizeGB = [math]::Round($_.model_size_gb * $GIB_TO_GB, 3)

  # size x tok/s is achieved bandwidth only for DENSE models, which read every
  # weight once per token; a sparse MoE reads only its active experts.
  # Most MoE filenames advertise the active-parameter count ("-A3B-"), but some
  # name the family instead. gpt-oss-20b is 20.9B total / ~3.6B active and says
  # so nowhere in the filename; leaving it on the dense path put it at 103% of
  # rated bandwidth, which is the exact impossible figure that flags the mistake.
  # Wave 8 added four more MoE models whose filenames carry no active-param
  # marker: Qwen3-Coder-Next (80B/3B), Mistral-Small-4-119B (119B/6.5B),
  # GLM-4.7-Flash (30B/3B) and Nex-N2.5-mini (35B MoE). Matched by name because
  # the filename offers nothing else to key on.
  $isMoE = Is-MoE $_.model
  $effBW = if (-not $isMoE) { [math]::Round($sizeGB * $_.tps_avg, 1) } else { $null }

  $g = $_.gpu_peak; $memTot = 0; $pwrTot = 0.0
  if ($g) { foreach ($k in $g.PSObject.Properties.Name) {
    $memTot += [int]$g.$k.mem_mib; $pwrTot += [double]$g.$k.power_w } }

  # KV cache is f16/f16 unless the run overrode it; collapse to one cell since
  # every run in this study set K and V to the same type.
  $kv = if ($_.type_k -eq $_.type_v) { $_.type_k } else { "$($_.type_k)/$($_.type_v)" }

  # llama-bench writes devices="auto" when it took every visible GPU, and the
  # explicit device list only when -dev was passed. So "auto" here means both
  # cards; anything naming a single CUDA device means one.
  $gpus = if ($_.devices -eq 'auto') { 2 } elseif ($_.devices -match ',') { ($_.devices -split ',').Count } else { 1 }

  # flash_attn is tri-state in the JSON: -1 auto, 0 off, 1 on.
  $fa = switch ([int]$_.flash_attn) { 1 { "on" } 0 { "off" } default { "auto" } }

  $pp = $ppIndex["$($_.ts)|$($_.n_depth)|$($_.type_k)|$($_.devices)|$($_.n_ubatch)|$($_.split_mode)|$($_.n_cpu_moe)"]
  $faK = if ([int]$_.flash_attn -eq 0) { 0 } else { 1 }
  $bf = $bfIndex["$($_.model)|$($_.n_depth)|$($_.type_k)|$($_.type_v)|$($_.devices)|$($_.n_ubatch)|$($_.n_batch)|$faK"]
  $p5  = if ($pp -and $pp.ContainsKey('pp512'))  { $pp['pp512'] }  else { $null }
  $p20 = if ($pp -and $pp.ContainsKey('pp2048')) { $pp['pp2048'] } else { $null }
  $late = @()
  if ($null -eq $p5  -and $bf -and $bf.ContainsKey('pp512'))  { $p5  = $bf['pp512'];  $late += 'pp512:b' }
  if ($null -eq $p20 -and $bf -and $bf.ContainsKey('pp2048')) { $p20 = $bf['pp2048']; $late += 'pp2048:b' }
  # Still empty: take the server sweep at the same settings and depth. The server caps a
  # conversation at the model's trained context, so a sweep's top level can sit up to ~2%
  # below a llama-bench depth such as 261,120 (4% for Olmo, whose trained context is exactly the
  # 65,536 it was benchmarked at); within 5% counts as the same depth.
  if (($null -eq $p5 -or $null -eq $p20) -and $row.n_depth -gt 0) {
    $sm0  = if ($row.split_mode) { $row.split_mode } else { 'layer' }
    $cand = $swIndex[(Sweep-Key $row.model $row.type_k $row.devices $row.n_ubatch $row.n_batch $sm0)]
    if ($cand) {
      $d = [double]$row.n_depth; $best = $null; $bd = [double]::MaxValue
      foreach ($c in $cand) { $diff = [math]::Abs([double]$c.pp512_depth - $d); if ($diff -lt $bd) { $bd = $diff; $best = $c } }
      if ($best -and $bd -le 0.05 * $d) {
        if ($null -eq $p5)  { $p5  = $best.pp512_tps;  $late += 'pp512:s' }
        if ($null -eq $p20) { $p20 = $best.pp2048_tps; $late += 'pp2048:s' }
      }
    }
  }

  [PSCustomObject]@{
    model       = $sp.base
    quant       = $sp.quant
    arch        = if ($isMoE) { "MoE" } else { "dense" }
    params_b    = $_.params_b
    size_gb     = $sizeGB
    kv_depth    = $_.n_depth
    kv_type     = $kv
    flash_attn  = $fa
    gpus        = $gpus
    n_batch     = $_.n_batch
    tok_s       = $_.tps_avg
    stddev      = $_.tps_stddev
    pp512       = $p5
    pp2048      = $p20
    pp_later    = ($late -join ' ')
    split       = if ($_.split_mode) { $_.split_mode } else { 'layer' }
    cpu_moe     = if ($_.n_cpu_moe) { $_.n_cpu_moe } else { 0 }
    eff_bw_gbs  = $effBW
    pct_peak    = if ($effBW) { [math]::Round($effBW / $PEAK_BW_GBS * 100, 1) } else { $null }
    vram_mib    = $memTot
    power_w     = [math]::Round($pwrTot, 0)
    reps        = $_.reps
    tag         = $_.tag
    method      = 'bench'
  }
}

# The sweeps' own generation rows. Depth is the level: pp512 is read at it, pp2048 512 tokens
# later, and the 128 generated tokens start 2,560 tokens later.
$meta = @{}; foreach ($r in $rows) { if (-not $meta.ContainsKey($r.model)) { $meta[$r.model] = $r } }
$swFlat = foreach ($sw in $sweeps) {
  $m = $meta[$sw.model]
  if (-not $m) { continue }
  $sp = Split-Name $sw.model
  $sizeGB = [math]::Round($m.model_size_gb * $GIB_TO_GB, 3)
  $isMoE = Is-MoE $sw.model
  $effBW = if (-not $isMoE) { [math]::Round($sizeGB * $sw.tg_tps, 1) } else { $null }
  $sum = $swSummaries | Where-Object { $_.label -eq $sw.label -and $_.ts -ge $sw.ts } | Sort-Object ts | Select-Object -First 1
  $mem = 0; if ($sum -and $sum.gpu) { foreach ($k in $sum.gpu.PSObject.Properties.Name) { $mem += [int]$sum.gpu.$k.mem_max_mib } }
  $ctk = Arg-Of $sw.server_args '-ctk' 'f16'; $dev = Arg-Of $sw.server_args '-dev' 'auto'
  [PSCustomObject]@{
    model = $sp.base; quant = $sp.quant; arch = if ($isMoE) { "MoE" } else { "dense" }
    params_b = $m.params_b; size_gb = $sizeGB; kv_depth = $sw.pp512_depth; kv_type = $ctk
    flash_attn = Arg-Of $sw.server_args '-fa' 'on'; gpus = if ($dev -eq 'auto') { 2 } else { 1 }
    n_batch = [int](Arg-Of $sw.server_args '-b' '2048'); tok_s = $sw.tg_tps; stddev = $null
    pp512 = $sw.pp512_tps; pp2048 = $sw.pp2048_tps; pp_later = ''
    split = $sw.split; cpu_moe = 0
    eff_bw_gbs = $effBW; pct_peak = if ($effBW) { [math]::Round($effBW / $PEAK_BW_GBS * 100, 1) } else { $null }
    vram_mib = if ($mem) { $mem } else { $null }; power_w = $null; reps = 1; tag = 'wave11-sweep'; method = 'server'
  }
}
$flat = @($flat) + @($swFlat)

$flat = $flat | Sort-Object @{e='params_b'}, @{e='model'}, @{e='size_gb';Descending=$true}, @{e='kv_depth'}, @{e='kv_type'}, @{e='gpus'}, @{e='split'}, @{e='method'}
[System.IO.File]::WriteAllLines($csv, [string[]]($flat | ConvertTo-Csv -NoTypeInformation), $UTF8)
Write-Output "wrote $csv ($($flat.Count) rows)"

$sb = [System.Text.StringBuilder]::new()
[void]$sb.AppendLine("# Master table - every generation measurement")
[void]$sb.AppendLine()
[void]$sb.AppendLine("Generated $(Get-Date -Format 'yyyy-MM-dd HH:mm'). $($flat.Count) rows.")
[void]$sb.AppendLine("All figures are ``tg128`` (128 generated tokens), mean of N repetitions after a warmup pass.")
[void]$sb.AppendLine()
[void]$sb.AppendLine("**Columns.** *KV depth* is how full the cache was before generation started (0 = empty).")
[void]$sb.AppendLine("*KV type* is the cache quantisation. *GPUs* is how many cards the layers were split across.")
[void]$sb.AppendLine("*pp512* and *pp2048* are prompt processing at that same depth, from the same invocation:")
[void]$sb.AppendLine("how fast the model reads new input once the cache is that full. Blank where the run did not")
[void]$sb.AppendLine("measure that prompt length. A * marks a prefill figure measured in a later llama-bench run with the")
[void]$sb.AppendLine("same settings, in a separate invocation from the generation figure beside it; a † marks one taken")
[void]$sb.AppendLine("from a llama-server one-fill sweep at the same settings (up to 4% shallower at a model's trained limit).")
[void]$sb.AppendLine("*Method* is bench (llama-bench, tg128) or server (llama-server one-fill climb: a single")
[void]$sb.AppendLine("measurement, with the 128 generated tokens starting 2,560 tokens past the listed depth).")
[void]$sb.AppendLine("*Split* is how the model is divided across cards: layer (whole layers per card, the default)")
[void]$sb.AppendLine("or tensor (every layer divided across both cards). Every row ran entirely in VRAM: no weights")
[void]$sb.AppendLine("or cache were offloaded to system RAM.")
[void]$sb.AppendLine("*GB/s* is ``size_GB x tok/s``, the achieved memory bandwidth, and *%peak* compares it to the")
[void]$sb.AppendLine("P100's $PEAK_BW_GBS GB/s spec. Both are blank for MoE rows, where the arithmetic does not apply")
[void]$sb.AppendLine("(a sparse model reads only its active experts per token). *W* is the sum across both cards.")
[void]$sb.AppendLine()

$cols = @(
  @{h='Model';     e={$_.model};      a='l'},
  @{h='Quant';     e={$_.quant};      a='l'},
  @{h='Arch';      e={$_.arch};       a='l'},
  @{h='B';         e={$_.params_b};   a='r'},
  @{h='Size GB';   e={$_.size_gb};    a='r'},
  @{h='KV depth';  e={$_.kv_depth};   a='r'},
  @{h='KV type';   e={$_.kv_type};    a='l'},
  @{h='FA';        e={$_.flash_attn}; a='l'},
  @{h='GPUs';      e={$_.gpus};       a='r'},
  @{h='Split';     e={$_.split};      a='l'},
  @{h='tok/s';     e={'**' + $_.tok_s + '**'}; a='r'},
  @{h='+/-';       e={$_.stddev};     a='r'},
  @{h='pp512';     e={ if ($null -eq $_.pp512)  { '' } else { "$($_.pp512)"  + $(if ($_.pp_later -match 'pp512:b')  { '*' } elseif ($_.pp_later -match 'pp512:s')  { '†' } else { '' }) } }; a='r'},
  @{h='pp2048';    e={ if ($null -eq $_.pp2048) { '' } else { "$($_.pp2048)" + $(if ($_.pp_later -match 'pp2048:b') { '*' } elseif ($_.pp_later -match 'pp2048:s') { '†' } else { '' }) } }; a='r'},
  @{h='Method';    e={$_.method};     a='l'},
  @{h='GB/s';      e={$_.eff_bw_gbs}; a='r'},
  @{h='%peak';     e={$_.pct_peak};   a='r'},
  @{h='VRAM MiB';  e={$_.vram_mib};   a='r'},
  @{h='W';         e={$_.power_w};    a='r'}
)

[void]$sb.AppendLine("| " + (($cols | ForEach-Object { $_.h }) -join " | ") + " |")
[void]$sb.AppendLine("|" + (($cols | ForEach-Object { if ($_.a -eq 'r') { "---:" } else { ":---" } }) -join "|") + "|")
foreach ($r in $flat) {
  # Pipe the row INTO the column's scriptblock so $_ binds to the row. Calling
  # `& $c.e` directly would leave $_ bound to whatever the enclosing loop set.
  $vals = foreach ($c in $cols) {
    $v = $r | ForEach-Object $c.e
    if ($null -eq $v -or "$v" -eq "" -or "$v" -eq "****") { "" } else { "$v" }
  }
  [void]$sb.AppendLine("| " + ($vals -join " | ") + " |")
}

[System.IO.File]::WriteAllText($md, $sb.ToString() + [Environment]::NewLine, $UTF8)
Write-Output "wrote $md"

# ---------- data for the published report ----------
# Injected straight into the report's inline <script id="master-data"> block
# instead of shipped as a sibling file. It is only a few KB, and inlining means
# the page has no path or load-order dependency - it renders identically when
# opened as a local file, where a relative <script src> would not resolve.
$payload = [ordered]@{
  generated = (Get-Date -Format 'yyyy-MM-dd')
  peak_bw   = $PEAK_BW_GBS
  rows      = @($flat | ForEach-Object {
    [ordered]@{
      m = $_.model; q = $_.quant; a = $_.arch; b = $_.params_b; s = $_.size_gb
      d = $_.kv_depth; k = $_.kv_type; f = $_.flash_attn; g = $_.gpus
      t = $_.tok_s; sd = $_.stddev; p5 = $_.pp512; p20 = $_.pp2048
      sm = $_.split; me = $_.method
      m5  = if ($_.pp_later -match 'pp512:b')  { '*' } elseif ($_.pp_later -match 'pp512:s')  { '†' } else { '' }
      m20 = if ($_.pp_later -match 'pp2048:b') { '*' } elseif ($_.pp_later -match 'pp2048:s') { '†' } else { '' }
      bw = $_.eff_bw_gbs; pk = $_.pct_peak; v = $_.vram_mib
    }
  })
}
$json = $payload | ConvertTo-Json -Depth 5 -Compress
# "</script>" can never appear in these values, but guard anyway: a literal one
# inside an inline script terminates the block early and breaks the page.
$json = $json -replace '</script', '<\/script'
[System.IO.File]::WriteAllText($js, "window.MASTER=$json;", $UTF8)
Write-Output "wrote $js ($($flat.Count) rows)"

$page = Join-Path $root "report\p100-report.html"
if (Test-Path $page) {
  $html = Get-Content $page -Raw -Encoding UTF8
  $pat  = '(?s)/\*BEGIN-MASTER-DATA\*/.*?/\*END-MASTER-DATA\*/'
  if ($html -match $pat) {
    # Regex replacement, so $ and \ in the payload must not be read as group refs.
    $repl = '/*BEGIN-MASTER-DATA*/window.MASTER=' + $json + ';/*END-MASTER-DATA*/'
    $html = [regex]::Replace($html, $pat, [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $repl })
    [System.IO.File]::WriteAllText($page, $html, $UTF8)
    Write-Output "injected $($flat.Count) rows into $page"
  } else {
    Write-Output "WARN: marker block not found in $page - data NOT injected"
  }
}
