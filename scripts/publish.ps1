# Copies the report into docs/ for GitHub Pages.
#
# GitHub Pages will serve from the repo root or from /docs, and only those two.
# The report is generated as report\p100-report.html (mastertable.ps1 injects the
# data payload into that path), so docs\index.html is a copy, not the original -
# run this after any change to the report, or Pages serves a stale page.
#
# The .nojekyll marker stops Pages running the file through Jekyll, which would
# otherwise strip any path beginning with an underscore.

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$src  = Join-Path $root "report\p100-report.html"
$dst  = Join-Path $root "docs\index.html"

if (-not (Test-Path $src)) { throw "report not found: $src" }

New-Item -ItemType Directory -Force -Path (Split-Path $dst -Parent) | Out-Null
Copy-Item $src $dst -Force

$nojekyll = Join-Path $root "docs\.nojekyll"
if (-not (Test-Path $nojekyll)) { New-Item -ItemType File -Path $nojekyll | Out-Null }

$kb = [math]::Round((Get-Item $dst).Length / 1KB, 1)
Write-Output "published $dst ($kb KB)"
