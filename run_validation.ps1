param(
    [Parameter(Position = 0)]
    [string]$Preset = "",
    [string]$Config = "",
    [object]$BBox = "",
    [switch]$Gpu,
    [string]$MemoryLimit = "",
    [int]$Heartbeat = 0,
    [string[]]$EnvVar = @(),
    [switch]$DryRun,
    [ValidateSet("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")]
    [string]$LogLevel = "INFO",
    [string]$LogFile = "",
    [int]$MaxWorkers = 0,
    [switch]$List,
    [switch]$Interactive
)

$presetsDir = Join-Path $PSScriptRoot "presets"
$defaultConfig = Join-Path $PSScriptRoot "validator" "validation_run.template.json"

$presetFiles = @()
if (Test-Path -LiteralPath $presetsDir) {
    $presetFiles = Get-ChildItem -LiteralPath $presetsDir -Filter "validation_run.*.json" | Sort-Object Name
}

function Show-Usage {
    Write-Host ""
    Write-Host "Usage: .\run_validation.ps1 [<preset>] [options]"
    Write-Host ""
    Write-Host "Presets (short names work, e.g. 'ismn-spl3', 'spl3', 'ismn-nsmc'):"
    Write-Host ""
    if ($presetFiles.Count -gt 0) {
        foreach ($pf in $presetFiles) {
            $name = $pf.BaseName -replace "^validation_run[_.]?", ""
            Write-Host ("  {0,-32} {1}" -f $name, $pf.FullName)
        }
    } else {
        Write-Host "  (no presets found in $presetsDir)"
    }
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Preset <name>     Run a ready-made preset (or pass it as the first arg)"
    Write-Host "  -Config <path>     Run an explicit config file (overrides -Preset)"
    Write-Host "  -BBox <lat0,lon0,lat1,lon1>"
    Write-Host "                     Limit the run to a bounding box (min_lat,min_lon,max_lat,max_lon)"
    Write-Host "  -Gpu               Use the Dask-parallel GPU path (sets QA4SM_USE_GPU=1)"
    Write-Host "  -MemoryLimit <size>"
    Write-Host "                     Dask worker memory limit, e.g. '16GB' or '8GiB' (QA4SM_DASK_MEMORY_LIMIT; default 60% of RAM)"
    Write-Host "  -Heartbeat <sec>   Heartbeat interval in seconds (QA4SM_HEARTBEAT_INTERVAL_SECONDS; default 60)"
    Write-Host "  -EnvVar <NAME=value>"
    Write-Host "                     Set any extra env var for the run (repeatable)"
    Write-Host "  -DryRun            Parse and print the config without running it"
    Write-Host "  -LogLevel <level>  DEBUG|INFO|WARNING|ERROR|CRITICAL (default INFO)"
    Write-Host "  -LogFile <path>    Write logs to a file (default: auto-created in logs\)"
    Write-Host "  -MaxWorkers <n>    Number of parallel workers"
    Write-Host "  -List              Show this help"
    Write-Host "  -Interactive       Prompt for preset, bbox, GPU, etc. (GPU defaults to ON)"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run_validation.ps1 spl3smpe -DryRun"
    Write-Host "  .\run_validation.ps1 ismn-spl3 -BBox 33,110,38,115 -Gpu"
    Write-Host "  .\run_validation.ps1 ismn-nsmc -BBox 41,-114,44,-112 -MemoryLimit 16GB"
    Write-Host "  .\run_validation.ps1 spl3 -EnvVar QA4SM_LOG_LEVEL=DEBUG"
    Write-Host "  .\run_validation.ps1 -Interactive"
}

if ($List) {
    Show-Usage
    exit 0
}

# --- logging setup -----------------------------------------------------------
$script:activeLogFile = ""
$autoLog = $false
if ($LogFile -eq "") {
    $logsDir = Join-Path $PSScriptRoot "logs"
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $logsDir ("validation_run_{0}.log" -f $stamp)
    $n = 1
    while (Test-Path -LiteralPath $LogFile) {
        $LogFile = Join-Path $logsDir ("validation_run_{0}_{1}.log" -f $stamp, (++$n))
    }
    $autoLog = $true
} elseif ($LogFile -ne "" -and (Split-Path -Parent $LogFile)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $LogFile) -Force | Out-Null
}
$script:activeLogFile = $LogFile

function Write-Log {
    param([string]$Message)
    $line = "{0} | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    if ($script:activeLogFile) {
        Add-Content -LiteralPath $script:activeLogFile -Value $line -Encoding utf8
    }
    Write-Host $line
}

Write-Log "=== run_validation.ps1 start ==="
Write-Log ("  Preset     : {0}" -f $(if ($Preset) { $Preset } else { "(default)" }))
Write-Log ("  Config     : {0}" -f $(if ($Config) { $Config } else { "(none)" }))
Write-Log ("  BBox       : {0}" -f $(if ($BBox) { "$BBox" } else { "(none)" }))
Write-Log ("  Gpu        : {0}" -f $(if ($Gpu) { "ON" } else { "off" }))
Write-Log ("  MemoryLimit: {0}" -f $(if ($MemoryLimit) { $MemoryLimit } else { "(default 60% RAM)" }))
Write-Log ("  Heartbeat  : {0}" -f $(if ($Heartbeat -gt 0) { $Heartbeat } else { "(default 60)" }))
Write-Log ("  EnvVar     : {0}" -f $(if ($EnvVar.Count) { ($EnvVar -join "; ") } else { "(none)" }))
Write-Log ("  LogLevel   : {0}" -f $LogLevel)
Write-Log ("  LogFile    : {0}" -f $LogFile)
Write-Log ("  MaxWorkers : {0}" -f $(if ($MaxWorkers -gt 0) { $MaxWorkers } else { "(default)" }))
Write-Log ("  DryRun     : {0}" -f $(if ($DryRun) { "yes" } else { "no" }))
Write-Log ("  Interactive: {0}" -f $(if ($Interactive) { "yes" } else { "no" }))
$sw = [System.Diagnostics.Stopwatch]::StartNew()

# --- interactive mode -------------------------------------------------------
function Read-Input {
    param([string]$Prompt)
    if ([Console]::IsInputRedirected) {
        Write-Host -NoNewline "$Prompt"
        return [Console]::In.ReadLine()
    }
    return (Read-Host $Prompt)
}

function Confirm-YesNo {
    param([string]$Prompt, [bool]$Default)
    $suffix = if ($Default) { "[Y/n]" } else { "[y/N]" }
    $answer = Read-Input -Prompt "$Prompt $suffix "
    if ([string]::IsNullOrWhiteSpace($answer)) { return $Default }
    return ($answer.Trim().ToLowerInvariant() -in @("y", "yes"))
}

function Confirm-Preset {
    Write-Host ""
    Write-Host "Available presets:"
    for ($i = 0; $i -lt $presetFiles.Count; $i++) {
        $name = $presetFiles[$i].BaseName -replace "^validation_run[_.]?", ""
        Write-Host ("  [{0}] {1}" -f ($i + 1), $name)
    }
    Write-Host ""
    $answer = Read-Input -Prompt "Pick a preset [number or name, Enter=1] "
    if ([string]::IsNullOrWhiteSpace($answer)) { return $presetFiles[0].FullName }
    $num = 0
    if ([int]::TryParse($answer.Trim(), [ref]$num) -and $num -ge 1 -and $num -le $presetFiles.Count) {
        return $presetFiles[$num - 1].FullName
    }
    return $answer.Trim()
}

if ($Interactive) {
    do {
        $Preset = Confirm-Preset

        $BBox = ""
        if (Confirm-YesNo -Prompt "Limit to a bounding box?" -Default $false) {
            for ($i = 0; $i -lt 3; $i++) {
                $BBox = Read-Input -Prompt "  min_lat,min_lon,max_lat,max_lon (e.g. 41,-114,44,-112) "
                $parts = @($BBox.Split(",") | ForEach-Object { $_.Trim() })
                $ok = ($parts.Count -eq 4)
                foreach ($p in $parts) { if ($p -notmatch "^-?[\d.]+$") { $ok = $false } }
                if ($ok) { break }
                Write-Warning "  Invalid bbox, need 4 numbers."
                $BBox = ""
            }
        }

        $Gpu = Confirm-YesNo -Prompt "Use GPU/Dask path?" -Default $true

        $MemoryLimit = ""
        $mem = Read-Input -Prompt "Dask memory limit (Enter for default 60% RAM, e.g. 16GB) "
        if ($mem -ne "") { $MemoryLimit = $mem }

        $DryRun = Confirm-YesNo -Prompt "Dry-run (parse only)?" -Default $false

        Write-Host ""
        Write-Log "---- Summary ----"
        Write-Log ("  Preset : {0}" -f $Preset)
        Write-Log ("  BBox   : {0}" -f $(if ($BBox) { $BBox } else { "(none - full)" }))
        Write-Log ("  GPU    : {0}" -f $(if ($Gpu) { "ON" } else { "off" }))
        Write-Log ("  Memory : {0}" -f $(if ($MemoryLimit) { $MemoryLimit } else { "default (60% RAM)" }))
        Write-Log ("  DryRun : {0}" -f $(if ($DryRun) { "yes" } else { "no" }))
        Write-Host "-----------------"
        $choice = Read-Input -Prompt "Run / Change / Quit (r/c/q) "
        if ($choice -and $choice.Trim().ToLowerInvariant() -in @("q", "quit")) {
            Write-Log "Interactive: aborted by user."
            exit 0
        }
    } while ($choice -and $choice.Trim().ToLowerInvariant() -in @("c", "change"))
}

# --- resolve the config to run -------------------------------------------
$runConfig = ""
$usedBBox = $false
if ($Config -ne "") {
    $runConfig = $Config
    if ($Preset -ne "") {
        Write-Warning "-Config and -Preset both given; using -Config ($Config)"
    }
} elseif ($Preset -ne "") {
    # allow a direct path to a preset file
    if (Test-Path -LiteralPath $Preset) {
        $runConfig = $Preset
    } else {
        $query = $Preset.ToLowerInvariant().Replace("-", "_").Trim()
        if ($query.StartsWith("validation_run.")) { $query = $query.Substring("validation_run.".Length) }
        if ($query.EndsWith(".json")) { $query = $query.Substring(0, $query.Length - 5) }

        $exact = Join-Path $presetsDir "validation_run.$query.json"
        if (Test-Path -LiteralPath $exact) {
            $runConfig = $exact
        } else {
            $matches1 = @($presetFiles | Where-Object {
                ($_.BaseName -replace "^validation_run[_.]?", "").ToLowerInvariant() -contains $query
            })
            if ($matches1.Count -eq 0) {
                $matches1 = @($presetFiles | Where-Object {
                    $_.BaseName -match $query
                })
            }
            if ($matches1.Count -eq 1) {
                $runConfig = $matches1[0].FullName
            } elseif ($matches1.Count -gt 1) {
                Write-Log "ERROR: Preset '$Preset' is ambiguous. Matches: $($matches1.BaseName -join ', ')"
                Write-Error "Preset '$Preset' is ambiguous. Matches: $($matches1.BaseName -join ', ')"
                Show-Usage
                exit 1
            } else {
                Write-Log "ERROR: Preset '$Preset' not found."
                Write-Error "Preset '$Preset' not found."
                Show-Usage
                exit 1
            }
        }
    }
} else {
    $runConfig = $defaultConfig
}

Write-Log "Resolved config: $runConfig"

if (-not (Test-Path -LiteralPath $runConfig)) {
    Write-Log "ERROR: Config not found: $runConfig"
    Write-Error "Config not found: $runConfig"
    exit 1
}

# --- optional bbox overlay -------------------------------------------------
if ($null -ne $BBox -and $BBox -ne "") {
    if ($BBox -is [array]) { $BBox = ($BBox -join ",") }
    $bbox = @($BBox.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries) | ForEach-Object { $_.Trim() })
    if ($bbox.Count -ne 4) {
        Write-Log "ERROR: -BBox needs exactly 4 numbers: <min_lat,min_lon,max_lat,max_lon> (got: '$BBox')"
        Write-Error "-BBox needs exactly 4 numbers: <min_lat,min_lon,max_lat,max_lon> (got: '$BBox')"
        exit 1
    }
    $valid = $true
    foreach ($v in $bbox) { if (-not ($v -match "^-?[\d.]+$")) { $valid = $false } }
    if (-not $valid) {
        Write-Log "ERROR: -BBox needs 4 numbers, got: '$BBox'"
        Write-Error "-BBox needs 4 numbers, got: '$BBox'"
        exit 1
    }

    try {
        $cfg = Get-Content -Raw -LiteralPath $runConfig | ConvertFrom-Json -Depth 30
    } catch {
        Write-Log "ERROR: Failed to parse config: $runConfig ($($_.Exception.Message))"
        Write-Error "Failed to parse config: $runConfig ($($_.Exception.Message))"
        exit 1
    }

    $cfg.validation_run | Add-Member -NotePropertyName min_lat -NotePropertyValue ([double]$bbox[0]) -Force
    $cfg.validation_run | Add-Member -NotePropertyName min_lon -NotePropertyValue ([double]$bbox[1]) -Force
    $cfg.validation_run | Add-Member -NotePropertyName max_lat -NotePropertyValue ([double]$bbox[2]) -Force
    $cfg.validation_run | Add-Member -NotePropertyName max_lon -NotePropertyValue ([double]$bbox[3]) -Force

    $tmpDir = Join-Path $env:TEMP "qa4sm_preset_runs"
    New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $baseName = [IO.Path]::GetFileNameWithoutExtension($runConfig)
    $baseName = $baseName -replace "^validation_run[_.]?", ""
    $bboxConfig = Join-Path $tmpDir ("validation_run.{0}_bbox_{1}.json" -f $baseName, $stamp)
    $cfg | ConvertTo-Json -Depth 30 | Set-Content -Encoding utf8 -LiteralPath $bboxConfig
    $runConfig = $bboxConfig
    $usedBBox = $true
    Write-Log "BBox overlay applied: $($bbox -join ',') -> $bboxConfig"
}

# --- run --------------------------------------------------------------------
$python = Join-Path $PSScriptRoot ".venv" "Scripts" "python.exe"
$cliArgs = @("-m", "validator.cli", $runConfig, "--log-level", $LogLevel)

if ($DryRun) { $cliArgs += "--dry-run" }
if ($LogFile -ne "") { $cliArgs += @("--log-file", $LogFile) }
if ($MaxWorkers -gt 0) { $cliArgs += @("--max-workers", $MaxWorkers) }

# --- environment -------------------------------------------------------------
if ($Gpu) {
    $env:QA4SM_USE_GPU = "1"
    Write-Log "ENV QA4SM_USE_GPU=1"
} else {
    Remove-Item Env:QA4SM_USE_GPU -ErrorAction SilentlyContinue
}
if ($MemoryLimit -ne "") {
    $normalizedMem = $MemoryLimit
    if ($MemoryLimit -match '^\d+$') {
        $normalizedMem = "${MemoryLimit}GB"
        Write-Log "NOTE: -MemoryLimit '$MemoryLimit' is a bare number; assuming $normalizedMem"
    }
    $env:QA4SM_DASK_MEMORY_LIMIT = $normalizedMem
    Write-Log "ENV QA4SM_DASK_MEMORY_LIMIT=$normalizedMem"
} else {
    Remove-Item Env:QA4SM_DASK_MEMORY_LIMIT -ErrorAction SilentlyContinue
}
if ($Heartbeat -gt 0) {
    $env:QA4SM_HEARTBEAT_INTERVAL_SECONDS = "$Heartbeat"
    Write-Log "ENV QA4SM_HEARTBEAT_INTERVAL_SECONDS=$Heartbeat"
} else {
    Remove-Item Env:QA4SM_HEARTBEAT_INTERVAL_SECONDS -ErrorAction SilentlyContinue
}
foreach ($kv in $EnvVar) {
    $eq = $kv.IndexOf("=")
    if ($eq -le 0) {
        Write-Log "ERROR: -EnvVar needs NAME=value, got: '$kv'"
        Write-Error "-EnvVar needs NAME=value, got: '$kv'"
        exit 1
    }
    Set-Item -Path ("Env:" + $kv.Substring(0, $eq)) -Value $kv.Substring($eq + 1)
    Write-Log "ENV $($kv.Substring(0, $eq))=$($kv.Substring($eq + 1))"
}

Write-Log "Running: $runConfig"
if ($Gpu) { Write-Log "  GPU/Dask path enabled" }
Write-Log ("Executing: {0} {1}" -f $python, ($cliArgs -join " "))

& $python @cliArgs
$code = $LASTEXITCODE

if ($usedBBox -and -not $DryRun) {
    Remove-Item -LiteralPath $runConfig -ErrorAction SilentlyContinue
}

$sw.Stop()
Write-Log ("=== run_validation.ps1 finished: exit={0} elapsed={1} ===" -f $code, $sw.Elapsed.ToString("hh\:mm\:ss"))
Write-Log "Log file: $LogFile"
if ($autoLog) {
    Write-Host ""
    Write-Host "Log written to: $LogFile"
}

exit $code
