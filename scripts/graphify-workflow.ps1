param(
    [Parameter(Position = 0)]
    [ValidateSet('query', 'path', 'explain', 'help')]
    [string]$Command = 'help',

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$graphPath = Join-Path $repoRoot 'graphify-out\graph.json'
$graphifyCmd = Get-Command graphify -ErrorAction SilentlyContinue

function Show-Usage {
    Write-Host 'Usage:'
    Write-Host '  .\scripts\graphify-workflow.ps1 query  "<question>"'
    Write-Host '  .\scripts\graphify-workflow.ps1 path   "<node A>" "<node B>"'
    Write-Host '  .\scripts\graphify-workflow.ps1 explain "<concept>"'
}

if ($Command -eq 'help') {
    Show-Usage
    exit 0
}

if (-not (Test-Path $graphPath)) {
    Write-Error "No graph found at $graphPath. Run `graphify update .` first."
    exit 1
}

if (-not $graphifyCmd) {
    Write-Error 'graphify is not available on PATH.'
    exit 1
}

switch ($Command) {
    'query' {
        if ($Rest.Count -lt 1) {
            Show-Usage
            exit 1
        }

        $question = ($Rest -join ' ').Trim()
        & $graphifyCmd.Source query $question
        exit $LASTEXITCODE
    }
    'path' {
        if ($Rest.Count -lt 2) {
            Show-Usage
            exit 1
        }

        $source = $Rest[0]
        $target = $Rest[1]
        & $graphifyCmd.Source path $source $target
        exit $LASTEXITCODE
    }
    'explain' {
        if ($Rest.Count -lt 1) {
            Show-Usage
            exit 1
        }

        $concept = ($Rest -join ' ').Trim()
        & $graphifyCmd.Source explain $concept
        exit $LASTEXITCODE
    }
    default {
        Show-Usage
        exit 1
    }
}
