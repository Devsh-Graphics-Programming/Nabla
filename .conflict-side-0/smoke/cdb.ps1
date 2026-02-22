param(
    [Parameter(Mandatory=$true)][string]$Exe,
    [string]$cdb = "C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\cdb.exe"
)

$scriptDir = Split-Path -Parent $PSCommandPath

if (-not (Test-Path $Exe)) {
    Write-Host "Error: File $Exe does not exist!" -ForegroundColor Red
    exit -2
}
if (-not (Test-Path $cdb)) {
    Write-Host "Error: cdb not found at $cdb" -ForegroundColor Red
    exit -2
}

$p = Start-Process -FilePath $Exe -NoNewWindow -Wait -PassThru
$exitCode = $p.ExitCode

if ($exitCode -eq 0) {
    Write-Host "Application exited with code 0 (success)" -ForegroundColor Green
    exit 0
}

Write-Host "Application exited with code $exitCode (crash)" -ForegroundColor Red
Write-Host "Re-running with CDB attached!" -ForegroundColor Cyan

$dumpFile = "$env:TEMP\$name.dmp"

if (Test-Path $dumpFile) {
    Remove-Item $dumpFile -Force
}

& $cdb -lines -y srv* -c "sxn ld; g; !analyze -v; lm v; kp; .dump /ma $dumpFile; q" $Exe | Out-Host
Write-Host "Process execution completed" -ForegroundColor Yellow

$exeName = [IO.Path]::GetFileName($Exe)
$werPaths = @(
    "$env:PROGRAMDATA\Microsoft\Windows\WER",
    "$env:LOCALAPPDATA\Microsoft\Windows\WER"
)

$lastWer = Get-ChildItem $werPaths -Recurse -Filter Report.wer -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Where-Object { Select-String -Path $_.FullName -SimpleMatch $exeName -Quiet } |
    Select-Object -First 1

if ($lastWer) {
    Write-Host "WER report for ${exeName}:" -ForegroundColor Cyan
    Get-Content $lastWer.FullName | Out-Host
} else {
    Write-Host "Could not find last WER report for $exeName" -ForegroundColor DarkYellow
}

exit -1