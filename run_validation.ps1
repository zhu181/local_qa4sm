param(
    [string]$Config = "validator/validation_run.template.json",
    [switch]$DryRun,
    [ValidateSet("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")]
    [string]$LogLevel = "INFO",
    [string]$LogFile = "",
    [int]$MaxWorkers = 0
)

$python = "C:/Users/30922/Documents/source/local_qa4sm/.venv/Scripts/python.exe"
$args = @("-m", "validator.cli", $Config, "--log-level", $LogLevel)

if ($DryRun) {
    $args += "--dry-run"
}

if ($LogFile -ne "") {
    $args += @("--log-file", $LogFile)
}

if ($MaxWorkers -gt 0) {
    $args += @("--max-workers", $MaxWorkers)
}

& $python @args
exit $LASTEXITCODE
