param(
    [string]$MainFile = 'main.tex'
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$thesisDir = Join-Path $repoRoot 'docs\thesis'
$latexmkCmd = Get-Command latexmk -ErrorAction SilentlyContinue

if (-not (Test-Path $thesisDir)) {
    Write-Error "Thesis directory not found at $thesisDir."
    exit 1
}

if (-not $latexmkCmd) {
    Write-Error 'latexmk is not available on PATH.'
    exit 1
}

Push-Location $thesisDir
try {
    & $latexmkCmd.Source -xelatex -interaction=nonstopmode -halt-on-error $MainFile
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
