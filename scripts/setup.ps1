param(
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

$Repos = @(
    @{ Name = "qa4sm-preprocessing"; Url = "https://github.com/aWst-austria/qa4sm-preprocessing" },
    @{ Name = "qa4sm-reader"; Url = "https://github.com/aWst-austria/qa4sm-reader" }
)

foreach ($r in $Repos) {
    $Path = Join-Path $RepoRoot $r.Name
    if (Test-Path $Path) {
        Write-Host "[$($r.Name)] Pulling latest..." -ForegroundColor Cyan
        Push-Location $Path
        git pull --ff-only
        Pop-Location
    }
    else {
        Write-Host "[$($r.Name)] Cloning..." -ForegroundColor Cyan
        Push-Location $RepoRoot
        git clone $r.Url
        Pop-Location
    }
}

Write-Host "`nRunning uv sync..." -ForegroundColor Cyan
$syncArgs = @()
if ($ForceReinstall) { $syncArgs += "--reinstall" }
Push-Location $RepoRoot
uv sync @syncArgs
Pop-Location

$exitCode = $LASTEXITCODE
if ($exitCode -eq 0) {
    Write-Host "`nSetup complete. Import check:" -ForegroundColor Green
    uv run python -c "
import qa4sm_reader; print(f'  qa4sm_reader: {qa4sm_reader.__version__}')
import qa4sm_preprocessing; print(f'  qa4sm_preprocessing: {qa4sm_preprocessing.__version__}')
"
}
else {
    Write-Host "`nSetup failed (exit code: $exitCode)" -ForegroundColor Red
}
exit $exitCode
