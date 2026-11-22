[CmdletBinding()]
param(
    [ValidatePattern('^COM[1-9][0-9]*$')]
    [string]$Port,

    [string]$Python,
    [string]$Programmer = 'C:/Gowin/Gowin_V1.9.11.03_Education_x64/Programmer/bin/programmer_cli.exe',
    [ValidateSet(1, 4)]
    [int]$CableIndex = 1
)

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$surface = Join-Path $repo 'past_experiments/surface_fpga'
$tang = Join-Path $repo 'past_experiments/tang_nano_20k_d15'
if (-not $Python) {
    $Python = Join-Path $repo 'decoder_testing/.venv/Scripts/python.exe'
}
$Python = (Resolve-Path -LiteralPath $Python).Path
$Programmer = (Resolve-Path -LiteralPath $Programmer).Path

function Invoke-Python([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

function Invoke-Programmer([string[]]$Arguments) {
    & $Programmer @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Gowin Programmer failed with exit code $LASTEXITCODE."
    }
}

function Write-Sram([string]$Bitstream) {
    Invoke-Programmer @('--device', 'GW2AR-18C', '--run', '2', '--fsFile',
        $Bitstream, '--cable-index', $CableIndex, '--channel', '0')
    Start-Sleep -Milliseconds 250
}

function Add-GowinImageMetadata([string]$Bitstream) {
    # Gowin Programmer uses the device and configuration comments in its .fs parser.
    # The low 16 bits of the 0x0A footer command already carry the checksum.
    $footer = @(Get-Content -LiteralPath $Bitstream -Tail 12 |
        Select-String -Pattern '^000010100{40}([01]{16})$')
    if ($footer.Count -ne 1) {
        throw "Cannot identify the Gowin checksum in $Bitstream."
    }
    $checksum = [Convert]::ToUInt16($footer[0].Matches[0].Groups[1].Value, 2)
    $header = @(
        '//File Title: Bitstream file'
        '//Device: GW2AR-18'
        '//Device Version: C'
        '//Part Number: GW2AR-LV18QN88C8/I7'
        '//Device-package: GW2AR-18C-QFN88'
        '//BackgroundProgramming: OFF'
        ('//CheckSum: 0x{0:X4}' -f $checksum)
        ('//UserCode: 0x{0:X8}' -f $checksum)
        '//LoadingRate: 2.500MHz'
        '//CRCCheck: ON'
        '//Compress: OFF'
        '//Encryption: OFF'
        '//SecureMode: OFF'
        '//JTAGAsRegularIO: OFF'
        '//MultiBootSPIAddr: 0x00000000'
    )
    $directory = Join-Path $surface 'build/gowin-images'
    [void][IO.Directory]::CreateDirectory($directory)
    $name = [IO.Path]::GetFileNameWithoutExtension($Bitstream)
    $staged = Join-Path $directory "$name-gowin.fs"
    $writer = [IO.StreamWriter]::new($staged, $false, [Text.Encoding]::ASCII)
    try {
        foreach ($line in $header) {
            $writer.WriteLine($line)
        }
        foreach ($line in [IO.File]::ReadLines($Bitstream)) {
            if (-not $line.StartsWith('//')) {
                $writer.WriteLine($line)
            }
        }
    } finally {
        $writer.Dispose()
    }
    return $staged
}

function Assert-SurfaceReply([string]$Syndrome, [int]$Cycles, [bool]$Deferred, [int]$Logical) {
    $reply = Invoke-Python @((Join-Path $surface 'host.py'), $graph, '--port',
        $Port, '--noise-p', '.007', '--syndrome', $Syndrome) | ConvertFrom-Json
    if ($null -ne $reply.error -or $null -eq $reply.link_us -or
        $reply.fpga_cycles -ne $Cycles -or $reply.deferred -ne $Deferred -or
        $reply.result -ne $Logical -or $reply.residual_weight -ne [int]$Deferred) {
        throw "Surface Telescope returned an unexpected hardware reply: $($reply | ConvertTo-Json -Compress)"
    }
    $reply | ConvertTo-Json -Compress
}

$profiles = Get-Content -Raw -LiteralPath (Join-Path $surface 'bitstreams/manifest.json') |
    ConvertFrom-Json
$surfaceProfile = $profiles | Where-Object profile -EQ 'd7z-dual-p007-cut0-90mhz'
$graph = Join-Path $surface $surfaceProfile.graph
$surfaceImage = Join-Path $surface $surfaceProfile.bitstream
if ((Get-FileHash -LiteralPath $graph).Hash -ine $surfaceProfile.graph_sha256 -or
    (Get-FileHash -LiteralPath $surfaceImage).Hash -ine $surfaceProfile.bitstream_sha256) {
    throw 'The Surface Telescope graph and bitstream must match their manifest hashes.'
}
$surfaceImage = Add-GowinImageMetadata $surfaceImage
$run = Join-Path $tang 'run.py'
Invoke-Python @($run, 'verify')
$tangImage = Add-GowinImageMetadata (Join-Path $tang 'bitstreams/nano20k_uart.fs')

# The two USB Serial Ports are different functions of the same device.
if (-not $Port) {
    $findPort = @'
import sys
try:
    from serial.tools.list_ports import comports
except ImportError:
    sys.exit('Install pyserial 3.5 in the selected Python environment.')
ports = [
    port.device for port in comports()
    if (port.vid, port.pid) == (0x0403, 0x6010)
    and (port.serial_number or '').endswith('B')
]
if len(ports) != 1:
    sys.exit('Expected one FTDI channel B UART port. Use -Port to select it explicitly.')
print(ports[0])
'@
    $Port = Invoke-Python @('-c', $findPort)
}
Write-Host "Using $Port for UART"

# Check UART access before changing the image on the board.
$portCheck = @'
import sys
try:
    import serial
except ImportError:
    sys.exit('Install pyserial 3.5 in the selected Python environment.')
try:
    with serial.Serial(sys.argv[1], 115200, timeout=.35):
        print('Opened ' + sys.argv[1])
except serial.SerialException as exc:
    sys.exit(str(exc))
'@
& $Python -c $portCheck $Port
if ($LASTEXITCODE -ne 0) {
    throw "Cannot open $Port. Run from a local Windows PowerShell window and close any serial monitor using this port."
}
Invoke-Programmer @('--scan', '--cable-index', $CableIndex, '--channel', '0')

Write-Host 'Testing Surface Telescope distance-7 Z'
Write-Sram $surfaceImage
Assert-SurfaceReply ('00' * 24) 31 $false 0
Assert-SurfaceReply '000400120000000400000002101002002200002202000000' 431 $true 1

Write-Host 'Testing Tang Nano 20K distance-15 Z'
Write-Sram $tangImage
Invoke-Python @($run, 'smoke', '--port', $Port)
$predictions = Join-Path $tang '.build/board-decoded.npz'
$decode = Invoke-Python @($run, 'decode', '--port', $Port, '--p', '.005',
    '--input', (Join-Path $tang 'examples/shots.npy'), '--output', $predictions) | ConvertFrom-Json
if ($null -ne $decode.link_error -or $decode.shots -ne 64 -or
    $decode.accepted_on_fpga -ne 29 -or $decode.decoded_by_cpu -ne 35) {
    throw "Tang distance-15 hardware results did not match the sample: $($decode | ConvertTo-Json -Compress)"
}
$decode | ConvertTo-Json -Compress
$score = Invoke-Python @($run, 'score', '--predictions', $predictions,
    '--observables', (Join-Path $tang 'examples/observables.npy')) | ConvertFrom-Json
if ($score.logical_errors -ne 0) {
    throw "The Tang distance-15 sample produced $($score.logical_errors) logical errors."
}
$score | ConvertTo-Json -Compress
Write-Host 'Both UART hardware checks passed. The distance-15 image is in SRAM.'
