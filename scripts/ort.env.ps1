param(
	[string]$OrtRoot = "external/onnxruntime",
	[string]$OrtBuild = "external/onnxruntime-build"
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$ortRootAbs = (Resolve-Path (Join-Path $repoRoot $OrtRoot) -ErrorAction SilentlyContinue)
$ortBuildAbs = (Resolve-Path (Join-Path $repoRoot $OrtBuild) -ErrorAction SilentlyContinue)

if (-not $ortRootAbs) {
	Write-Host "DC_ORT_ROOT path not found: $(Join-Path $repoRoot $OrtRoot)"
} else {
	$env:DC_ORT_ROOT = $ortRootAbs.Path
	Write-Host "DC_ORT_ROOT=$($env:DC_ORT_ROOT)"
}

if (-not $ortBuildAbs) {
	Write-Host "DC_ORT_BUILD path not found: $(Join-Path $repoRoot $OrtBuild)"
} else {
	$env:DC_ORT_BUILD = $ortBuildAbs.Path
	Write-Host "DC_ORT_BUILD=$($env:DC_ORT_BUILD)"
}
