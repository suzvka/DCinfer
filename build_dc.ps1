$vsRoot = "C:\Program Files\Microsoft Visual Studio\18\Community"
$msvcVer = "14.50.35717"
$sdkVer = "10.0.26100.0"
$sdkRoot = "C:\Program Files (x86)\Windows Kits\10"

$env:Path = "$vsRoot\VC\Tools\MSVC\$msvcVer\bin\Hostx64\x64;$vsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;$sdkRoot\bin\$sdkVer\x64;" + $env:Path
$env:INCLUDE = "$vsRoot\VC\Tools\MSVC\$msvcVer\include;$sdkRoot\Include\$sdkVer\ucrt;$sdkRoot\Include\$sdkVer\shared;$sdkRoot\Include\$sdkVer\um"
$env:LIB = "$vsRoot\VC\Tools\MSVC\$msvcVer\lib\x64;$sdkRoot\Lib\$sdkVer\ucrt\x64;$sdkRoot\Lib\$sdkVer\um\x64"

Remove-Item "$PSScriptRoot\build\CMakeCache.txt" -ErrorAction SilentlyContinue
Remove-Item "$PSScriptRoot\build\CMakeFiles" -Recurse -ErrorAction SilentlyContinue

Push-Location "$PSScriptRoot\build"
try {
    cmake .. -G Ninja
    if ($LASTEXITCODE -eq 0) {
        cmake --build .
    }
} finally {
    Pop-Location
}
