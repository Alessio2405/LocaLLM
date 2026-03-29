# TODO

## Immediate

- [ ] Verify that the shared `modelId` and `systemPrompt` persist after app restart.
- [ ] Verify that changing the shared model updates all existing chats and new chats.
- [ ] Verify that changing the shared system prompt updates all existing chats and new chats.
- [ ] Check exported chats to confirm hidden placeholder/template tokens are not present in markdown exports.
- [ ] Confirm the assistant text cleanup works for streamed replies, regenerated replies, and previously saved chats.

## Runtime And UX

- [ ] Revisit the desktop default model choice so first-run behavior prefers a model that is actually available and reliable on typical Windows machines.
- [ ] Surface clearer runtime errors in the UI when model import/load fails.
- [ ] Decide whether assistant-output cleanup should also happen in the Rust backend before events are emitted.
- [ ] Add a visible loading/import state when Ollama is rebuilding a model from a GGUF file.
- [ ] Review whether `temperature`, `topP`, and `maxTokens` should stay per-chat or also become shared settings.

## Repo Hygiene

- [ ] Ensure build output like `dist/` is not tracked by Git anymore if it still exists in the index.
- [ ] Add basic regression coverage for message sanitization and shared-settings migration behavior.
- [ ] Document the new shared model/system-prompt behavior in `README.md`.

## Android

- [ ] Initialize Android in this workspace with `npm run tauri android init`.
- [ ] Run the first Android emulator or device test with `npm run tauri android dev`.
- [ ] Validate embedded inference, startup time, and memory behavior on Android.
- [ ] Decide whether to keep the Windows Ollama path and Android embedded path split or unify runtimes later.
