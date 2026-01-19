# Build native libraries for Windows and copy Slang dependencies

$ErrorActionPreference = "Stop"

Push-Location (Split-Path -Parent $PSScriptRoot)

try {
    Write-Host "Building goldy-ffi for Windows x64..."
    cargo build --release -p goldy-ffi --target x86_64-pc-windows-msvc

    # Copy FFI library to runtime folder
    $runtimeDir = "dotnet/Goldy/runtimes/win-x64/native"
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
    Copy-Item "target/x86_64-pc-windows-msvc/release/goldy_ffi.dll" $runtimeDir

    Write-Host "Built goldy_ffi.dll for win-x64"

    # Copy Slang libraries
    Write-Host "Copying Slang libraries..."
    $slangDir = "slang/bin/windows-x86_64"
    
    # Read manifest to get file list
    $manifest = Get-Content "slang/manifest.json" | ConvertFrom-Json
    $slangFiles = $manifest.platforms."windows-x86_64".files

    $copied = 0
    foreach ($file in $slangFiles) {
        $src = Join-Path $slangDir $file
        if (Test-Path $src) {
            Copy-Item $src $runtimeDir
            $copied++
            Write-Host "  Copied $file"
        } else {
            Write-Host "  Warning: $file not found at $src"
        }
    }

    Write-Host "Copied $copied Slang libraries to $runtimeDir"
}
finally {
    Pop-Location
}
