# DConnx

## ONNX Runtime from source (submodule)

This repository can be configured to use a locally built ONNX Runtime (ORT) to avoid mismatched `onnxruntime.lib`/`onnxruntime.dll` versions.

### 1) Add ORT as a submodule (run locally)

From the repo root:

`git submodule add https://github.com/microsoft/onnxruntime external/onnxruntime`

`git submodule update --init --recursive`

If you previously had a broken submodule folder, remove it first:

- delete `external/onnxruntime`
- delete `.git/modules/external/onnxruntime`

### 2) Build ORT (start with CPU-only)

Run (PowerShell):

`powershell -ExecutionPolicy Bypass -File scripts/build_onnxruntime.ps1 -Config Release -Arch x64 -OutDir external/onnxruntime-build`

### 3) Point this project to the built ORT

Set environment variables before building this project:

- `DC_ORT_ROOT`: ORT source root (default: `external/onnxruntime`)
- `DC_ORT_BUILD`: ORT build output root (default: `external/onnxruntime-build`)

You can apply defaults via:

`powershell -ExecutionPolicy Bypass -File scripts/ort.env.ps1`

### 4) Runtime deployment

The project copies `onnxruntime.dll` from `DC_ORT_BUILD` into the output folder so the EXE loads the expected DLL.
