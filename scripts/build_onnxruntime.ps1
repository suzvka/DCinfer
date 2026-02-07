param(
	[ValidateSet('Debug','Release')] [string]$Config = 'Release',
	[ValidateSet('x64','Win32','arm64')] [string]$Arch = 'x64',
	[string]$OrtRoot = 'external/onnxruntime',
	[string]$OutDir = 'external/onnxruntime-build',
	[switch]$Cuda,
	[string]$CudaHome,
	[string]$CudnnHome,
	[string]$TensorRtHome
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ortRootAbs = Resolve-Path (Join-Path $repoRoot $OrtRoot)
$outDirAbs = Join-Path $repoRoot $OutDir

$buildDir = Join-Path $outDirAbs "build-$Arch-$Config"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$cmakeArgs = @(
	"-S", $ortRootAbs.Path,
	"-B", $buildDir,
	"-Donnxruntime_BUILD_SHARED_LIB=ON",
	"-Donnxruntime_BUILD_UNIT_TESTS=OFF",
	"-Donnxruntime_RUN_ONNX_TESTS=OFF",
	"-Donnxruntime_ENABLE_PYTHON=OFF",
	"-Donnxruntime_BUILD_CSHARP=OFF",
	"-Donnxruntime_BUILD_JAVA=OFF",
	"-Donnxruntime_BUILD_NODEJS=OFF"
)

if ($Cuda) {
	$cmakeArgs += "-Donnxruntime_USE_CUDA=ON"
	if ($CudaHome) { $cmakeArgs += "-DCUDA_TOOLKIT_ROOT_DIR=$CudaHome" }
	if ($CudnnHome) { $cmakeArgs += "-Donnxruntime_CUDNN_HOME=$CudnnHome" }
	if ($TensorRtHome) {
		$cmakeArgs += "-Donnxruntime_USE_TENSORRT=ON"
		$cmakeArgs += "-DTENSORRT_HOME=$TensorRtHome"
	}
}

Write-Host "Configuring ORT: $buildDir"
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building ORT: $Config $Arch"
cmake --build $buildDir --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Try to locate a dll/lib pair
$candidates = @(
	Join-Path $buildDir "$Config\onnxruntime.dll",
	Join-Path $buildDir "onnxruntime.dll"
)

Write-Host "Done. Set environment:"
Write-Host "  `$env:DC_ORT_ROOT='$($ortRootAbs.Path)'"
Write-Host "  `$env:DC_ORT_BUILD='$($buildDir)'"
