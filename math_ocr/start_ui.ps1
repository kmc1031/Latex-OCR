# Run from any working directory. Stop the local server with Ctrl+C.
$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$pythonExecutable = Join-Path $projectDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw 'Project Python environment is missing. See README.md.'
}
& $pythonExecutable (Join-Path $projectDirectory 'ui_server.py') @args
