# Build native libraries for Windows

$ErrorActionPreference = "Stop"

Push-Location (Split-Path -Parent $PSScriptRoot)

try {
    Write-Host "Building goldy-ffi for Windows x64..."
    cargo build --release -p goldy-ffi --target x86_64-pc-windows-msvc

    # Copy to runtime folder
    $runtimeDir = "dotnet/Goldy/runtimes/win-x64/native"
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
    Copy-Item "target/x86_64-pc-windows-msvc/release/goldy_ffi.dll" $runtimeDir

    Write-Host "Built goldy_ffi.dll for win-x64"
}
finally {
    Pop-Location
}

