import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { MODELS } from "./models";
import type {
  AppSettings,
  ChatCompleteEvent,
  ChatErrorEvent,
  ChatMessage,
  ChatStore,
  ChatGenerationSettings,
  ChatTokenEvent,
  DownloadProgressEvent,
  GenerationStart,
  ModelInfo,
  RuntimeStatus,
  ThemePreference,
} from "./types";

const isTauri = () =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const DEFAULT_SYSTEM_PROMPT =
  "You are a helpful local assistant named Dobby made by Alessio Doria. Be concise, practical, and explicit when you are unsure.";

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
  options: ChatGenerationSettings,
): Promise<GenerationStart> {
  if (!isTauri()) {
    return {
      requestId: crypto.randomUUID(),
      messageId: crypto.randomUUID(),
    };
  }

  return invoke<GenerationStart>("send_chat", { modelId, messages, options });
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

export async function loadChatStore(): Promise<ChatStore | null> {
  if (!isTauri()) {
    return null;
  }

  return invoke<ChatStore | null>("load_chat_store");
}

export async function saveChatStore(store: ChatStore): Promise<void> {
  if (!isTauri()) {
    return;
  }

  await invoke("save_chat_store", { store });
}

export async function getAppSettings(): Promise<AppSettings> {
  if (!isTauri()) {
    return {
      themePreference: "system",
      selectedModelId: MODELS[0].id,
      defaultModelId: MODELS[0].id,
      systemPrompt: DEFAULT_SYSTEM_PROMPT,
      autoLoadLastModel: true,
    };
  }

  return invoke<AppSettings>("get_app_settings");
}

export async function updateAppSettings(
  settings: AppSettings,
): Promise<AppSettings> {
  if (!isTauri()) {
    return settings;
  }

  return invoke<AppSettings>("update_app_settings", { settings });
}

export async function deleteModel(modelId: string): Promise<RuntimeStatus> {
  if (!isTauri()) {
    return previewState();
  }

  return invoke<RuntimeStatus>("delete_model", { modelId });
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
