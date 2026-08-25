$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($env:PYTHON) {
    $Python = $env:PYTHON
}
else {
    $Python = "python"
}

& $Python `
    (Join-Path $Root "run_generator.py") `
    @args

exit $LASTEXITCODE
