param(
    [string]$Repository = (Resolve-Path "$PSScriptRoot/../../../..").Path,
    [ValidatePattern('^build/[A-Za-z0-9_-]+$')]
    [string]$Output = 'build/d15-tilt-growth-full-ram',
    [string]$CadRoot = 'C:/oss-cad-suite/oss-cad-suite',
    [ValidateSet(27, 54)][int]$ClockMHz = 54,
    [int]$GrowthCycles = 4000,
    [double]$GrowthResolution = 2,
    [switch]$SynthOnly,
    [switch]$RenderOnly
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path $Repository).Path
$surface = Join-Path $repo 'past_experiments/surface_fpga'
$python = Join-Path $repo 'decoder_testing/.venv/Scripts/python.exe'
$prefix = "past_experiments/surface_fpga/$Output"
$target = Join-Path $surface $Output
$resolution = $GrowthResolution.ToString([Globalization.CultureInfo]::InvariantCulture)
$clockMult = $ClockMHz / 27
$env:PATH = "$(Join-Path $CadRoot 'bin');$(Join-Path $CadRoot 'lib');$env:PATH"

function Invoke-Native([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program exited with $LASTEXITCODE." }
}

if (Test-Path (Join-Path $target 'graph.json')) {
    throw "Choose a fresh output directory. An experiment exists at $target."
}
[void][IO.Directory]::CreateDirectory((Join-Path $target 'rtl'))
[void][IO.Directory]::CreateDirectory((Join-Path $target 'tools'))
Copy-Item -LiteralPath (Join-Path $surface 'rtl/growth.sv') -Destination (Join-Path $target 'rtl/growth.sv')
Copy-Item -LiteralPath (Join-Path $surface 'tools/growth_model.cpp') -Destination (Join-Path $target 'tools/growth_model.cpp')
foreach ($patch in @('growth.patch', 'growth-model.patch')) {
    $path = Join-Path $PSScriptRoot $patch
    Invoke-Native git @('-C', $repo, '-c', 'core.autocrlf=false', 'apply', '--check', "--directory=$prefix", $path)
    Invoke-Native git @('-C', $repo, '-c', 'core.autocrlf=false', 'apply', "--directory=$prefix", $path)
}

Push-Location $surface
try {
    Invoke-Native $python @('surface_model.py', '--distance', '15', '--basis', 'z', '--rounds', '15',
        '--p', '.001', '--frontend', 'cluster', '--growth-p', '.007', '--growth-resolution', $resolution,
        '--growth-engine', 'dual-port', '--growth-skip-empty', '--growth-cycle-cap', "$GrowthCycles",
        '--clock-mhz', "$ClockMHz", '--output', $Output)
    Invoke-Native $python @('stencils.py', $Output)

    # Use the one-site write schedule in both the RTL and its cycle model.
    $header = Join-Path $target 'sc_growth.vh'
    $source = [IO.File]::ReadAllText($header)
    $short = '' + [char]96 + 'define SC_GROW_SHORT_UPDATES '
    if (!$source.Contains($short + '1')) { throw 'Expected the original two-site write schedule.' }
    [IO.File]::WriteAllText($header, $source.Replace($short + '1', $short + '0'))
    $topology = Join-Path $target 'growth_graph.txt'
    $lines = [IO.File]::ReadAllLines($topology)
    $fields = $lines[0].Split(' ')
    if ($fields.Count -ne 8 -or $fields[6] -ne '1') { throw 'Unexpected growth topology header.' }
    $fields[6] = '0'
    $lines[0] = $fields -join ' '
    [IO.File]::WriteAllLines($topology, $lines)
    if ($ClockMHz -eq 54 -and $GrowthCycles -eq 4000 -and $GrowthResolution -eq 2 -and
        !(Select-String -LiteralPath (Join-Path $target 'sc_image.vh') -SimpleMatch "32'h592e6076")) {
        throw 'Unexpected UART profile tag.'
    }

    if ($RenderOnly) { return }
    $env:TEMP = $env:TMP = $env:TMPDIR = $target
    $sources = "rtl/surface.sv rtl/rescue.sv rtl/lookup.sv $Output/rtl/growth.sv rtl/cluster_linkage.sv rtl/uart.sv rtl/link.sv"
    $script = "read_verilog -sv -D SC_CLOCK_MULT=$clockMult -D SC_CLOCK_DIV=1 -I$Output $sources; synth_gowin -family gw2a -nowidelut -top tang_nano20k -json $Output/netlist.json"
    Invoke-Native yosys @('-Q', '-q', '-l', "$Output/synth.log", '-p', $script)
    if ($SynthOnly) { return }
    Invoke-Native nextpnr-himbaechel @('-q', '-l', "$Output/route.log", '--Werror',
        '--json', "$Output/netlist.json", '--write', "$Output/route.json", '--report', "$Output/timing.json",
        '--device', 'GW2AR-LV18QN88C8/I7', '--vopt', 'family=GW2A-18C',
        '--vopt', 'cst=tang_nano20k.cst', '--freq', "$ClockMHz", '--sdc', 'board_clock.sdc')
    $suffix = ''
    if ($GrowthCycles -ne 4000) { $suffix += "-cap$GrowthCycles" }
    if ($GrowthResolution -ne 2) { $suffix += "-res" + $resolution.Replace('.', 'p') }
    $bitstream = "$Output/surface-d15z-tilt-growth-${ClockMHz}mhz$suffix.fs"
    Invoke-Native gowin_pack @('-d', 'GW2A-18C', '-o', $bitstream, "$Output/route.json")

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
    $staged = Add-GowinImageMetadata (Resolve-Path $bitstream).Path
    Get-Content -LiteralPath $staged -TotalCount 8 | Select-String 'UserCode'
    Get-FileHash -LiteralPath $staged
}
finally {
    Pop-Location
}
