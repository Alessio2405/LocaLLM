# LocalLM

Local-first chat app for Windows desktop and Android with:

- real GGUF model downloads
- embedded `llama-cpp-2` inference
- streamed assistant replies
- manual model loading
- dark and light themes with system default

## Included models

- `lfm-2.5-vl-1.6b`
- `qwen-3.5-2b-q80`
- `qwen-3.5-4b-q4km`

v1 is text-chat only. `mmproj` metadata is kept for later multimodal support, but only the primary GGUF file is downloaded and used.

## What works now

- model list and runtime status from Rust
- per-model download with progress and cancel
- detection of already-downloaded models on relaunch
- one loaded model at a time
- real local generation via `llama-cpp-2`
- token streaming from Rust to React through Tauri events
- persisted theme preference: `system`, `light`, `dark`

## Requirements

- Node.js
- Rust toolchain
- LLVM / libclang on Windows for `llama-cpp-2`

This machine now has LLVM installed through `winget`, and builds use:

```powershell
$env:LIBCLANG_PATH='C:\Program Files\LLVM\bin'
```

## Development

Install frontend packages:

```bash
npm install
```

Build the frontend:

```bash
npm run build
```

Check the Rust backend:

```powershell
$env:LIBCLANG_PATH='C:\Program Files\LLVM\bin'
cargo check --manifest-path src-tauri/Cargo.toml
```

Run the desktop app:

```powershell
$env:LIBCLANG_PATH='C:\Program Files\LLVM\bin'
npm run tauri dev
```

## Notes

- First-time Rust builds are slow because `llama.cpp` is compiled locally.
- Large models need substantial RAM and storage.
- Android packaging is scaffolded, but I only verified compilation on Windows in this pass.
