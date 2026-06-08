$ErrorActionPreference = "Stop"
$NotebookArgs = @($args)

$defaultRepo = "C:\Users\simcl\Sciebo\Masterarbeit_Simon_Clemens\05_Thesis"
$repo = if ($env:NOTEBOOKLM_LOCAL_REPO) { $env:NOTEBOOKLM_LOCAL_REPO } else { $defaultRepo }
$cli = if ($env:NOTEBOOKLM_LOCAL_CLI) { $env:NOTEBOOKLM_LOCAL_CLI } else { Join-Path $repo "venv\Scripts\notebooklm.exe" }

if (-not (Test-Path -LiteralPath $cli)) {
    Write-Error "notebooklm-local: CLI not found: $cli. Set NOTEBOOKLM_LOCAL_REPO or NOTEBOOKLM_LOCAL_CLI if it moved."
}

if (-not $env:NOTEBOOKLM_HOME) {
    $env:NOTEBOOKLM_HOME = "C:\Users\simcl\.notebooklm"
}
if (-not $env:NOTEBOOKLM_PROFILE) {
    $env:NOTEBOOKLM_PROFILE = "default"
}

$mutating = @("create", "delete", "remove", "add", "rename", "login", "refresh", "share", "unshare")
$requestedMutations = @($NotebookArgs | Where-Object { $mutating -contains $_.ToLowerInvariant() })
if ($requestedMutations.Count -gt 0 -and $env:SIMONE_NOTEBOOKLM_ALLOW_MUTATION -ne "1") {
    Write-Error "notebooklm-local: refusing mutating command token '$($requestedMutations[0])'. Set SIMONE_NOTEBOOKLM_ALLOW_MUTATION=1 only after explicit user intent."
}

Push-Location $repo
try {
    if (
        $NotebookArgs.Count -ge 2 -and
        $NotebookArgs[0].ToLowerInvariant() -eq "auth" -and
        $NotebookArgs[1].ToLowerInvariant() -eq "check"
    ) {
        $output = & $cli @NotebookArgs 2>&1 | Out-String
        $exitCode = $LASTEXITCODE
        try {
            $payload = $output | ConvertFrom-Json
            if ($payload.details) {
                $payload.details.PSObject.Properties.Remove("cookies_found")
                $payload.details.PSObject.Properties.Remove("cookie_domains")
                $payload.details.PSObject.Properties.Remove("cookies_by_domain")
            }
            $payload | ConvertTo-Json -Depth 16
        }
        catch {
            $output `
                -replace '"cookies_found"\s*:\s*\[[\s\S]*?\]\s*,?', '' `
                -replace '"cookie_domains"\s*:\s*\[[\s\S]*?\]\s*,?', '' `
                -replace '"cookies_by_domain"\s*:\s*\{[\s\S]*?\}\s*,?', ''
        }
        exit $exitCode
    }
    else {
        & $cli @NotebookArgs
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
