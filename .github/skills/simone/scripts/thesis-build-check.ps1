param(
    [string]$MainTex = "main.tex"
)

$ErrorActionPreference = "Stop"

$repo = if ($env:SIMONE_REPO) {
    $env:SIMONE_REPO
} else {
    "C:\Users\simcl\Documents\GitHub\Semantic Inference Module for Ontology-driven Node Extraction (SIMONE)"
}
$thesisDir = Join-Path $repo "docs\thesis"

if (-not (Test-Path -LiteralPath $thesisDir)) {
    Write-Error "thesis-build-check: thesis directory not found: $thesisDir"
}

Push-Location $thesisDir
try {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($MainTex)
    $windowsTexBin = "C:\texlive\2025\bin\windows"
    $latexmk = Get-Command latexmk -ErrorAction SilentlyContinue

    if ($latexmk) {
        & latexmk -g -pdf -interaction=nonstopmode -file-line-error $MainTex
    } elseif (Test-Path -LiteralPath (Join-Path $windowsTexBin "latexmk.exe")) {
        & (Join-Path $windowsTexBin "latexmk.exe") -g -pdf -interaction=nonstopmode -file-line-error $MainTex
        if ($LASTEXITCODE -ne 0) {
            Write-Host "thesis-build-check: latexmk failed; falling back to xelatex/biber cycle." -ForegroundColor Yellow
            & (Join-Path $windowsTexBin "xelatex.exe") -interaction=nonstopmode -file-line-error $MainTex
            if (Test-Path "$base.bcf") { & (Join-Path $windowsTexBin "biber.exe") $base }
            & (Join-Path $windowsTexBin "xelatex.exe") -interaction=nonstopmode -file-line-error $MainTex
            & (Join-Path $windowsTexBin "xelatex.exe") -interaction=nonstopmode -file-line-error $MainTex
        }
    } elseif (Test-Path -LiteralPath (Join-Path $windowsTexBin "xelatex.exe")) {
        & (Join-Path $windowsTexBin "xelatex.exe") -interaction=nonstopmode -file-line-error $MainTex
        if (Test-Path "$base.bcf") { & (Join-Path $windowsTexBin "biber.exe") $base }
        & (Join-Path $windowsTexBin "xelatex.exe") -interaction=nonstopmode -file-line-error $MainTex
        & (Join-Path $windowsTexBin "xelatex.exe") -interaction=nonstopmode -file-line-error $MainTex
    } else {
        Write-Error "thesis-build-check: no latexmk/xelatex found. Expected TeX Live in $windowsTexBin or latexmk on PATH."
    }

    $pdf = "$base.pdf"
    $log = "$base.log"
    $blg = "$base.blg"
    if (-not (Test-Path -LiteralPath $pdf)) {
        Write-Error "thesis-build-check: expected PDF missing: $pdf"
    }

    Write-Host "`n== PDF =="
    Get-Item -LiteralPath $pdf | Select-Object FullName, Length, LastWriteTime | Format-List

    Write-Host "`n== Log warnings/errors =="
    $patterns = "undefined|Undefined|Citation|Warning|Error|Overfull|Underfull"
    foreach ($file in @($log, $blg)) {
        if (Test-Path -LiteralPath $file) {
            Select-String -Path $file -Pattern $patterns | Select-Object -First 120 | ForEach-Object {
                "{0}:{1}: {2}" -f $_.Path, $_.LineNumber, $_.Line.Trim()
            }
        }
    }

    Write-Host "`n== PDF text spot check =="
    $pdftotext = Get-Command pdftotext -ErrorAction SilentlyContinue
    if ($pdftotext) {
        $text = & pdftotext $pdf -
        $text | Select-String -Pattern "Catalytic Experiments|Semantic Web Technologies|FAIR Data|Ontologies|Bibliography|undefined|\?\?" | Select-Object -First 100
    } else {
        Write-Host "pdftotext not available; skipping text spot check."
    }
}
finally {
    Pop-Location
}
