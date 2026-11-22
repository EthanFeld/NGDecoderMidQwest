param(
    [Parameter(Mandatory)][string]$Route,
    [string]$OutputDirectory,
    [ValidateSet('27', '13.5')][string]$FabricMHz = '27',
    [string]$Repository = (Resolve-Path "$PSScriptRoot/../../../..").Path,
    [string]$CadRoot = 'C:/oss-cad-suite/oss-cad-suite'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath $Repository).Path
$surface = Join-Path $repo 'past_experiments/surface_fpga'
$python = Join-Path $repo 'decoder_testing/.venv/Scripts/python.exe'
$source = (Resolve-Path -LiteralPath $Route).Path
if (!$OutputDirectory) { $OutputDirectory = Split-Path -Parent $source }
[void][IO.Directory]::CreateDirectory($OutputDirectory)
$destination = (Resolve-Path -LiteralPath $OutputDirectory).Path
$env:PATH = "$(Join-Path $CadRoot 'bin');$(Join-Path $CadRoot 'lib');$env:PATH"

$clockLabel = $FabricMHz.Replace('.', 'p')
$newRoute = Join-Path $destination "route-clock$clockLabel.json"
& $python (Join-Path $PSScriptRoot 'retime27.py') $source $newRoute --fabric-mhz $FabricMHz
if ($LASTEXITCODE -ne 0) { throw 'Clock-only route transform failed.' }
$raw = Join-Path $destination "surface-d15z-tilt-growth-${clockLabel}mhz-debug.fs"
& gowin_pack -d GW2A-18C -o $raw $newRoute
if ($LASTEXITCODE -ne 0) { throw 'Gowin packing failed.' }

$tokens = $errors = $null
$programmerScript = Join-Path $repo 'verify-fpga.ps1'
$ast = [System.Management.Automation.Language.Parser]::ParseFile($programmerScript, [ref]$tokens, [ref]$errors)
if ($errors.Count -gt 0) { throw ($errors | Out-String) }
$metadata = $ast.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Add-GowinImageMetadata'
}, $false)
if (!$metadata) { throw 'Could not find the Gowin metadata staging function.' }
. ([scriptblock]::Create($metadata.Extent.Text))
$staged = Add-GowinImageMetadata $raw

$payloadCheck = @'
from pathlib import Path
import sys

def rows(path):
    return [line for line in Path(path).read_bytes().splitlines() if line[:1] in (b"0", b"1")]

original, staged = map(rows, sys.argv[1:])
if not original or original != staged:
    raise ValueError("Staging changed the bitstream payload.")
print(f"{len(staged)} identical binary rows after adding programmer metadata.")
'@
& $python -c $payloadCheck $raw $staged
if ($LASTEXITCODE -ne 0) { throw 'Staged image payload verification failed.' }
Get-Content -LiteralPath $staged -TotalCount 8 | Select-String 'UserCode'
Get-FileHash -LiteralPath $raw,$staged
