import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { MODELS } from "./models";
import type {
  ChatCompleteEvent,
  ChatErrorEvent,
  ChatMessage,
  ChatTokenEvent,
  DownloadProgressEvent,
  GenerationStart,
  ModelInfo,
  RuntimeStatus,
  ThemePreference,
} from "./types";

const isTauri = () =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

function previewState(modelId = MODELS[0].id): RuntimeStatus {
  return {
    selectedModelId: modelId,
    loadedModelId: modelId,
    platform: "web-preview",
    inferenceReady: true,
    isGenerating: false,
    modelStates: Object.fromEntries(
      MODELS.map((model) => [
        model.id,
        {
          status: model.id === modelId ? "ready" : "downloaded",
          progress: 100,
          downloadedBytes: model.sizeBytes,
          totalBytes: model.sizeBytes,
          error: null,
          localPath: null,
        },
      ]),
    ),
  };
}

export async function getRuntimeStatus(): Promise<RuntimeStatus> {
  if (!isTauri()) {
    return previewState();
  }

  return invoke<RuntimeStatus>("get_runtime_status");
}

export async function listModels(): Promise<ModelInfo[]> {
  if (!isTauri()) {
    return MODELS;
  }

  return invoke<ModelInfo[]>("list_models");
}

export async function selectModel(modelId: string): Promise<RuntimeStatus> {
  if (!isTauri()) {
    return previewState(modelId);
  }

  return invoke<RuntimeStatus>("select_model", { modelId });
}

export async function downloadModel(modelId: string): Promise<RuntimeStatus> {
  if (!isTauri()) {
    return previewState(modelId);
  }

  return invoke<RuntimeStatus>("download_model", { modelId });
}

export async function cancelModelDownload(modelId: string): Promise<RuntimeStatus> {
  if (!isTauri()) {
    return previewState(modelId);
  }

  return invoke<RuntimeStatus>("cancel_model_download", { modelId });
}

export async function sendChat(
  modelId: string,
  messages: ChatMessage[],
): Promise<GenerationStart> {
  if (!isTauri()) {
    return {
      requestId: crypto.randomUUID(),
      messageId: crypto.randomUUID(),
    };
  }

  return invoke<GenerationStart>("send_chat", { modelId, messages });
}

export async function cancelGeneration(): Promise<RuntimeStatus> {
  if (!isTauri()) {
    return previewState();
  }

  return invoke<RuntimeStatus>("cancel_generation");
}

export async function getThemePreference(): Promise<ThemePreference> {
  if (!isTauri()) {
    return "system";
  }

  return invoke<ThemePreference>("get_theme_preference");
}

export async function setThemePreference(
  theme: ThemePreference,
): Promise<ThemePreference> {
  if (!isTauri()) {
    return theme;
  }

  return invoke<ThemePreference>("set_theme_preference", { theme });
}

export async function onDownloadProgress(
  handler: (event: DownloadProgressEvent) => void,
) {
  if (!isTauri()) {
    return () => {};
  }

  return listen<DownloadProgressEvent>("model-download-progress", (event) =>
    handler(event.payload),
  );
}

export async function onDownloadComplete(
  handler: (event: DownloadProgressEvent) => void,
) {
  if (!isTauri()) {
    return () => {};
  }

  return listen<DownloadProgressEvent>("model-download-complete", (event) =>
    handler(event.payload),
  );
}

export async function onDownloadError(
  handler: (event: DownloadProgressEvent) => void,
) {
  if (!isTauri()) {
    return () => {};
  }

  return listen<DownloadProgressEvent>("model-download-error", (event) =>
    handler(event.payload),
  );
}

export async function onChatToken(handler: (event: ChatTokenEvent) => void) {
  if (!isTauri()) {
    return () => {};
  }

  return listen<ChatTokenEvent>("chat-token", (event) => handler(event.payload));
}

export async function onChatComplete(
  handler: (event: ChatCompleteEvent) => void,
) {
  if (!isTauri()) {
    return () => {};
  }

  return listen<ChatCompleteEvent>("chat-complete", (event) =>
    handler(event.payload),
  );
}

export async function onChatError(handler: (event: ChatErrorEvent) => void) {
  if (!isTauri()) {
    return () => {};
  }

  return listen<ChatErrorEvent>("chat-error", (event) => handler(event.payload));
}
