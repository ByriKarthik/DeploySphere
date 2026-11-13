$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "DeploySphere's virtual-environment Python was not found at $python."
}

Push-Location $PSScriptRoot
try {
    & $python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --workers 1
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $exitCode
