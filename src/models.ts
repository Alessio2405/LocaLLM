import type { ModelInfo } from "./types";

export const DEFAULT_MODEL: ModelInfo = {
  id: "lfm-2.5-vl-1.6b",
  name: "LFM 2.5 VL 1.6B (Q4_0)",
  url: "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/LFM2.5-VL-1.6B-Q4_0.gguf",
  mmprojUrl:
    "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf",
  description: "Liquid AI multimodal model (text-only on web)",
  sizeBytes: 695_752_160,
  mmprojSizeBytes: 583_109_888,
  sizeHuman: "~664 MB",
};

export const TAURI_DEFAULT_MODEL: ModelInfo = {
  id: "qwen-3.5-2b-q80",
  name: "Qwen 3.5 2B (Q8_0)",
  url: "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q8_0.gguf?download=true",
  mmprojUrl:
    "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-F16.gguf",
  description: "Qwen multimodal model",
  sizeBytes: 2_012_012_800,
  mmprojSizeBytes: 668_227_264,
  sizeHuman: "2.68 GB",
};

export const HIGH_RAM_MAC_MODEL: ModelInfo = {
  id: "qwen-3.5-4b-q4km",
  name: "Qwen 3.5 4B (Q4_K_M)",
  url: "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf?download=true",
  mmprojUrl:
    "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf",
  description: "Qwen multimodal model for higher-memory devices",
  sizeBytes: 2_740_937_888,
  mmprojSizeBytes: 672_423_616,
  sizeHuman: "3.63 GB",
};

export const MOBILE_DEFAULT_MODEL_ID = DEFAULT_MODEL.id;
export const DESKTOP_DEFAULT_MODEL_ID = HIGH_RAM_MAC_MODEL.id;
export const BALANCED_DEFAULT_MODEL_ID = TAURI_DEFAULT_MODEL.id;

export function getPreferredModelId(platform?: string) {
  return platform === "android" ? MOBILE_DEFAULT_MODEL_ID : DESKTOP_DEFAULT_MODEL_ID;
}

export function getModelRecommendation(modelId: string, platform?: string) {
  if (modelId === getPreferredModelId(platform)) {
    return platform === "android" ? "Recommended for mobile" : "Recommended";
  }
  if (modelId === BALANCED_DEFAULT_MODEL_ID) {
    return "Balanced";
  }
  if (modelId === MOBILE_DEFAULT_MODEL_ID) {
    return "Fastest";
  }
  return "Higher quality";
}

export const MODELS = [DEFAULT_MODEL, TAURI_DEFAULT_MODEL, HIGH_RAM_MAC_MODEL];
