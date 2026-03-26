export type Role = "user" | "assistant" | "system";

export type ThemePreference = "system" | "light" | "dark";
export type EffectiveTheme = "light" | "dark";

export type ModelDownloadState =
  | "remote"
  | "downloading"
  | "downloaded"
  | "loading"
  | "ready"
  | "error";

export type ModelInfo = {
  id: string;
  name: string;
  url: string;
  mmprojUrl?: string;
  description: string;
  sizeBytes: number;
  mmprojSizeBytes?: number;
  sizeHuman: string;
};

export type ModelRuntimeState = {
  status: ModelDownloadState;
  progress: number;
  downloadedBytes: number;
  totalBytes: number;
  error?: string | null;
  localPath?: string | null;
};

export type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  createdAt: string;
  streaming?: boolean;
  error?: string | null;
};

export type RuntimeStatus = {
  selectedModelId: string;
  loadedModelId?: string | null;
  platform: string;
  inferenceReady: boolean;
  isGenerating: boolean;
  modelStates: Record<string, ModelRuntimeState>;
};

export type DownloadProgressEvent = {
  modelId: string;
  status: ModelDownloadState;
  progress: number;
  downloadedBytes: number;
  totalBytes: number;
  localPath?: string | null;
  error?: string | null;
};

export type ChatTokenEvent = {
  requestId: string;
  messageId: string;
  token: string;
};

export type ChatCompleteEvent = {
  requestId: string;
  messageId: string;
};

export type ChatErrorEvent = {
  requestId: string;
  messageId: string;
  error: string;
};

export type GenerationStart = {
  requestId: string;
  messageId: string;
};
