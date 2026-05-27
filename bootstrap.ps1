param(
    [switch]$Build,
    [switch]$Release
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VcpkgRoot   = "$ProjectRoot\external\vcpkg"
$VcpkgExe    = "$VcpkgRoot\vcpkg.exe"

# ---- 1. Bootstrap vcpkg if needed ----
if (-not (Test-Path $VcpkgExe)) {
    Write-Host "[bootstrap] Initializing vcpkg..." -ForegroundColor Cyan
    Push-Location $VcpkgRoot
    .\bootstrap-vcpkg.bat
    Pop-Location
    if (-not (Test-Path $VcpkgExe)) {
        Write-Error "vcpkg bootstrap failed"
        exit 1
    }
}

# ---- 2. Ensure cmake + ninja artifacts are cached ----
Write-Host "[bootstrap] Ensuring cmake + ninja artifacts..." -ForegroundColor Cyan
& $VcpkgExe use "microsoft:tools/kitware/cmake" "microsoft:tools/ninja-build/ninja" 2>&1 | Out-Null

# ---- 3. Locate artifact bin directories ----
$ArtifactBase = Get-ChildItem "$VcpkgRoot\downloads\artifacts" -Directory -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $ArtifactBase) {
    Write-Error "No artifact cache found. Run 'vcpkg use cmake ninja' manually once."
    exit 1
}

$CmakeDir  = Get-ChildItem "$($ArtifactBase.FullName)\tools.kitware.cmake\*\bin" -Directory -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

$NinjaDir  = Get-ChildItem "$($ArtifactBase.FullName)\tools.ninja.build.ninja\*" -Directory -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $CmakeDir -or -not $NinjaDir) {
    Write-Error "Artifact cache incomplete. Run 'vcpkg use microsoft:tools/kitware/cmake microsoft:tools/ninja-build/ninja' manually."
    exit 1
}

# ---- 4. Activate tools in current session ----
$env:Path = "$($CmakeDir.FullName);$($NinjaDir.FullName);$env:Path"

Write-Host "[bootstrap] cmake : $($CmakeDir.FullName)" -ForegroundColor Green
Write-Host "[bootstrap] ninja : $($NinjaDir.FullName)" -ForegroundColor Green

# ---- 5. Verify MSVC environment ----
$hasCl = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $hasCl) {
    Write-Warning "MSVC compiler (cl.exe) not on PATH."
    Write-Warning "Run this script from a 'Developer PowerShell for VS' or call vcvarsall.bat first."
}

# ---- 6. Build (if requested) ----
if ($Build) {
    $Preset = if ($Release) { "release" } else { "default" }
    Write-Host "`n[build] Configuring with preset '$Preset'..." -ForegroundColor Cyan
    Push-Location $ProjectRoot
    try {
        cmake --preset $Preset
        if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
        cmake --build build
        if ($LASTEXITCODE -ne 0) { throw "CMake build failed" }
        Write-Host "[build] Done." -ForegroundColor Green
    } finally {
        Pop-Location
    }
} else {
    Write-Host "`nEnvironment ready. Run manually:" -ForegroundColor Cyan
    Write-Host "  cmake --preset default" -ForegroundColor White
    Write-Host "  cmake --build build" -ForegroundColor White
    Write-Host "`nOr use: .\bootstrap.ps1 -Build           (Debug)" -ForegroundColor White
    Write-Host "         .\bootstrap.ps1 -Build -Release   (Release)" -ForegroundColor White
}
