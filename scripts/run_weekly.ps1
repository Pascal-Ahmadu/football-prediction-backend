<#
Runs the weekly pipeline unattended, for Windows Task Scheduler.

  1. Makes sure Docker Desktop and the fmep-pg container are up.
  2. Runs python -m app.pipeline.weekly with any arguments given to this script.
  3. Writes everything to data\logs\weekly-<date>-<time>.log, and deletes logs
     older than 60 days.

Run by hand exactly as the scheduler does:
    powershell -ExecutionPolicy Bypass -File scripts\run_weekly.ps1
    powershell -ExecutionPolicy Bypass -File scripts\run_weekly.ps1 --days-back 3 --days-ahead 4
#>

# Native programs (docker, python) write warnings to stderr; Windows PowerShell 5.1
# would treat those as errors. Failures are judged by exit codes instead.
$ErrorActionPreference = "Continue"
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
$logDir = Join-Path $project "data\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("weekly-{0:yyyyMMdd-HHmm}.log" -f (Get-Date))

function Write-Log([string]$message) {
    "{0:yyyy-MM-dd HH:mm:ss}  {1}" -f (Get-Date), $message | Out-File -Append -Encoding utf8 $log
}

function Test-Docker {
    docker info *> $null
    return $LASTEXITCODE -eq 0
}

Write-Log "starting: weekly pipeline $($args -join ' ')"

# Docker Desktop is not running after a reboot until someone opens it.
if (-not (Test-Docker)) {
    Write-Log "Docker is not running; starting Docker Desktop"
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    $deadline = (Get-Date).AddMinutes(5)
    while (-not (Test-Docker)) {
        if ((Get-Date) -gt $deadline) {
            Write-Log "FAILED: Docker did not start within 5 minutes"
            exit 1
        }
        Start-Sleep -Seconds 10
    }
}

docker start fmep-pg fmep-redis *> $null
Start-Sleep -Seconds 5  # give Postgres a moment to accept connections

Set-Location $project
$env:PYTHONIOENCODING = "utf-8"
# -u: unbuffered, so the log fills as the run progresses and a hung run shows where it stopped.
& $python -u -m app.pipeline.weekly @args *>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
$code = $LASTEXITCODE

if ($code -eq 0) {
    Write-Log "finished OK"
} else {
    Write-Log "FAILED with exit code $code"
}

Get-ChildItem $logDir -Filter "weekly-*.log" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-60) } |
    Remove-Item -Force

exit $code
