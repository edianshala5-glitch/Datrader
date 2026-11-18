# Updates this Datrader folder to the latest version on GitHub (Windows PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File D:\Datrader\update.ps1
#
# Only program files are replaced. Your settings (backend\.env), downloaded market data
# (backend\data) and installed packages (backend\.venv) are not in the download, so they
# are never touched.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$zip = Join-Path $env:TEMP "datrader-main.zip"
$tmp = Join-Path $env:TEMP "datrader-main"

Write-Host "Downloading the latest version of Datrader..."
Invoke-WebRequest "https://github.com/edianshala5-glitch/Datrader/archive/refs/heads/main.zip" -OutFile $zip

if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
Expand-Archive -Path $zip -DestinationPath $tmp
$src = (Get-ChildItem -Path $tmp -Directory | Select-Object -First 1).FullName

Write-Host "Updating files in $root ..."
# /E = include subfolders. No /MIR: files that exist only here (data, .env) are kept.
robocopy $src $root /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "Copy failed (robocopy exit code $LASTEXITCODE)" }

Remove-Item -Recurse -Force $tmp
Remove-Item -Force $zip
Write-Host "Done. Datrader is up to date." -ForegroundColor Green
