# Reorders the running wave 10 without losing work.
#
# The first runner (PID passed in) was launched with e0 e2a e2b e1 e3 e4 e5. The full-window
# sweep (e6) was added afterwards and should run before the thermal, MTP, micro-batch and
# tokenizer experiments. This waits until that runner finishes e2b - detected by e1 logging
# its first line, which is followed by a cooldown wait of at least a few minutes, so nothing
# has touched the GPUs yet - stops it, and starts a second runner with e6 e1 e3 e4 e5.

param([Parameter(Mandatory=$true)][int]$RunnerPid)

$root = "C:\Projects\p100 testing"
$log  = "$root\logs\wave10.log"

while ($true) {
  if (Select-String -Path $log -Pattern 'E1 soak 27b-card0' -Quiet) { break }
  if (-not (Get-Process -Id $RunnerPid -ErrorAction SilentlyContinue)) { break }
  Start-Sleep -Seconds 15
}

Get-CimInstance Win32_Process -Filter "ParentProcessId=$RunnerPid" |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Stop-Process -Id $RunnerPid -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 5

$py = (Get-Command python).Source
Start-Process -FilePath $py -ArgumentList '-u', "`"$root\scripts\wave10.py`"", 'e6', 'e1', 'e3', 'e4', 'e5' `
  -WorkingDirectory "$root\scripts" -WindowStyle Hidden `
  -RedirectStandardOutput "$root\logs\wave10b.log" -RedirectStandardError "$root\logs\wave10b.err"
