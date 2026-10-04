<#
Nightly database backup on this machine. Keeps 7 daily copies plus the first of
each month, so six seasons of work survive a disk failure.

    powershell -ExecutionPolicy Bypass -File scripts\backup_db.ps1
    powershell -ExecutionPolicy Bypass -File scripts\backup_db.ps1 -Restore data\backups\fmep-20261004.dump
#>

param([string]$Restore)

# Native tools write progress to stderr; judge them by exit code.
$ErrorActionPreference = "Continue"
$project = Split-Path -Parent $PSScriptRoot
$backups = Join-Path $project "data\backups"
New-Item -ItemType Directory -Force $backups | Out-Null
$log = Join-Path $project "data\logs\backup.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

function Write-Log([string]$message) {
    $line = "{0:yyyy-MM-dd HH:mm:ss}  {1}" -f (Get-Date), $message
    $line | Out-File -Append -Encoding utf8 $log
    Write-Output $line
}

docker start fmep-pg *> $null

if ($Restore) {
    if (-not (Test-Path $Restore)) { Write-Log "no such file: $Restore"; exit 1 }
    Write-Output "This REPLACES the current database with $Restore."
    $answer = Read-Host "Type restore to continue"
    if ($answer -ne "restore") { Write-Output "cancelled"; exit 1 }
    # cmd does the redirection: Windows PowerShell 5.1 would re-encode the bytes.
    cmd /c "docker exec -i fmep-pg pg_restore -U postgres -d fmep --clean --if-exists < ""$Restore"""
    Write-Log "restored from $Restore"
    exit $LASTEXITCODE
}

$file = Join-Path $backups ("fmep-{0:yyyyMMdd}.dump" -f (Get-Date))
# -Fc is the compressed custom format; pg_restore can rebuild from it selectively.
# The redirection is done by cmd, because Windows PowerShell 5.1 would treat the
# dump as text and corrupt it.
cmd /c "docker exec fmep-pg pg_dump -U postgres -d fmep -Fc > ""$file"""
$code = $LASTEXITCODE

if ($code -ne 0 -or -not (Test-Path $file) -or (Get-Item $file).Length -lt 1MB) {
    Write-Log "backup FAILED (exit $code)"
    if (Test-Path $file) { Remove-Item $file -Force }
    exit 1
}

Write-Log ("backup written: {0} ({1:N0} MB)" -f $file, ((Get-Item $file).Length / 1MB))

# Keep the last 7 days and the first backup of each month; drop the rest.
Get-ChildItem $backups -Filter "fmep-*.dump" |
    Where-Object {
        $_.LastWriteTime -lt (Get-Date).AddDays(-7) -and $_.Name -notmatch '^fmep-\d{6}01\.dump$'
    } |
    Remove-Item -Force

exit 0
