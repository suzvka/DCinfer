@echo off
rem RunOrtTest.bat - ORT integration test launcher (Debug workaround)
rem
rem vcpkg onnxruntime:x64-windows Debug DLL has a broken embedded ONNX
rem schema registry and cannot load any model. vcpkg also redeploys the
rem broken DLL at the end of every build, overwriting manual fixes.
rem So right before the test runs, copy the Release ORT DLLs (C ABI
rem compatible) over the output directory, then launch the test.
rem
rem Usage: RunOrtTest.bat <release_bin_dir> <test_bin_dir> <test_exe_path>

setlocal

set RELEASE_DIR=%~1
set BIN_DIR=%~2
set TEST_EXE=%~3

if not exist "%RELEASE_DIR%\onnxruntime.dll" (
    echo [RunOrtTest] ERROR: Release ORT DLL not found in %RELEASE_DIR%
    echo [RunOrtTest] Build Release first: cmake --build build --config Release
    exit /b 1
)

rem Skip deployment when the test already runs from the Release dir
rem (copying over the DLL of the pending process would fail)
if /i not "%BIN_DIR%"=="%RELEASE_DIR%" (
    copy /y "%RELEASE_DIR%\*.dll" "%BIN_DIR%\" >nul
    if errorlevel 1 (
        echo [RunOrtTest] ERROR: failed to deploy Release ORT DLLs to %BIN_DIR%
        exit /b 1
    )
)

"%TEST_EXE%"
exit /b %ERRORLEVEL%
