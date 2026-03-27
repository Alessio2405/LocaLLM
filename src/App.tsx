import { ChangeEvent, FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { MessageContent, getAssistantMessagePreview } from "./messageContent";
import {
  MODELS,
  getModelRecommendation,
  getPreferredModelId,
} from "./models";
import {
  cancelGeneration,
  cancelModelDownload,
  deleteModel,
  downloadModel,
  getAppSettings,
  getRuntimeStatus,
  loadChatStore,
  listModels,
  onChatComplete,
  onChatError,
  onChatToken,
  onDownloadComplete,
  onDownloadError,
  onDownloadProgress,
  saveChatStore,
  selectModel,
  sendChat,
  updateAppSettings,
} from "./tauri";
import type {
  AppSettings,
  ChatAttachment,
  ChatGenerationSettings,
  ChatMessage,
  ChatSession,
  ChatStore,
  EffectiveTheme,
  ModelInfo,
  ModelRuntimeState,
  RuntimeStatus,
  ThemePreference,
} from "./types";

const CHAT_STORAGE_KEY = "locallm.chat-sessions";
const ACTIVE_CHAT_STORAGE_KEY = "locallm.active-chat";
const INITIAL_DEFAULT_MODEL_ID = getPreferredModelId();

const getSystemTheme = (): EffectiveTheme =>
  typeof window !== "undefined" &&
  window.matchMedia("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";

const defaultGenerationSettings = (modelId: string): ChatGenerationSettings => ({
  modelId,
  systemPrompt:
    "You are a helpful local assistant. Be concise, practical, and explicit when you are unsure.",
  temperature: 0.7,
  topP: 0.9,
  maxTokens: 1024,
});

const createWelcomeMessage = (): ChatMessage => ({
  id: crypto.randomUUID(),
  role: "assistant",
  content:
    "Your chats stay local. Open Settings to pick a model, tune generation, or attach files for extra context.",
  createdAt: new Date().toISOString(),
});

const createChatSession = (modelId: string): ChatSession => {
  const now = new Date().toISOString();
  return {
    id: crypto.randomUUID(),
    title: "New chat",
    createdAt: now,
    updatedAt: now,
    messages: [createWelcomeMessage()],
    settings: defaultGenerationSettings(modelId),
    pinned: false,
    tags: [],
    titleLocked: false,
  };
};

const createFallbackStore = (modelId: string): ChatStore => {
  const chat = createChatSession(modelId);
  return { chats: [chat], activeChatId: chat.id };
};

function loadBrowserStore(modelId: string): ChatStore {
  if (typeof window === "undefined") {
    return createFallbackStore(modelId);
  }

  try {
    const chats = JSON.parse(window.localStorage.getItem(CHAT_STORAGE_KEY) || "null");
    const activeChatId = window.localStorage.getItem(ACTIVE_CHAT_STORAGE_KEY);
    if (!Array.isArray(chats) || !chats.length) {
      return createFallbackStore(modelId);
    }
    const normalized = chats.map((chat: ChatSession) => ({
      ...chat,
      settings: { ...defaultGenerationSettings(modelId), ...chat.settings },
      pinned: Boolean(chat.pinned),
      tags: normalizeTags(chat.tags || []),
      titleLocked: Boolean(chat.titleLocked),
    }));
    return {
      chats: sortChats(normalized),
      activeChatId:
        activeChatId && normalized.some((chat) => chat.id === activeChatId)
          ? activeChatId
          : normalized[0].id,
    };
  } catch {
    return createFallbackStore(modelId);
  }
}

function deriveChatTitle(messages: ChatMessage[]) {
  const firstUser = messages.find((message) => message.role === "user");
  if (!firstUser) return "New chat";
  const firstLine = firstUser.content.split("\n")[0].trim();
  return firstLine.length > 42 ? `${firstLine.slice(0, 42)}...` : firstLine;
}

const updateChatMetadata = (chat: ChatSession): ChatSession => ({
  ...chat,
  title:
    chat.titleLocked || (chat.title !== "New chat" && chat.title.trim())
      ? chat.title
      : deriveChatTitle(chat.messages),
  tags: normalizeTags(chat.tags || []),
});

const normalizeTags = (tags: string[]) =>
  Array.from(
    new Set(
      tags
        .map((tag) => tag.trim())
        .filter(Boolean)
        .map((tag) => tag.toLowerCase()),
    ),
  ).slice(0, 8);

const sortChats = (chats: ChatSession[]) =>
  [...chats].sort((left, right) => {
    if (Boolean(left.pinned) !== Boolean(right.pinned)) {
      return left.pinned ? -1 : 1;
    }
    return Date.parse(right.updatedAt) - Date.parse(left.updatedAt);
  });

const defaultModelState = (totalBytes = 0): ModelRuntimeState => ({
  status: "remote",
  progress: 0,
  downloadedBytes: 0,
  totalBytes,
  error: null,
  localPath: null,
});

const formatBytes = (value: number) =>
  value >= 1_000_000_000
    ? `${(value / 1_000_000_000).toFixed(2)} GB`
    : value >= 1_000_000
      ? `${(value / 1_000_000).toFixed(1)} MB`
      : value >= 1_000
        ? `${(value / 1_000).toFixed(1)} KB`
        : `${value} B`;

const formatChatTime = (iso: string) =>
  new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

const getModelActionLabel = (status: ModelRuntimeState["status"], isSelected: boolean) => {
  if (status === "ready" && isSelected) return "Ready";
  if (status === "downloaded") return "Load";
  if (status === "ready") return "Use";
  return "Select";
};

const getModelStorageLabel = (state: ModelRuntimeState, fallbackBytes: number) => {
  if (!state.localPath && (state.status === "downloaded" || state.status === "ready")) {
    return "Imported in Ollama";
  }
  const bytes = state.downloadedBytes || state.totalBytes || fallbackBytes;
  return bytes > 0 ? formatBytes(bytes) : "Not downloaded";
};

const sanitizeFileName = (value: string) =>
  value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "chat";

const isTextPreviewable = (file: File) =>
  file.type.startsWith("text/") ||
  /\.(txt|md|json|csv|yaml|yml|xml|html|css|js|ts|jsx|tsx|py|rs|java|kt|swift)$/i.test(
    file.name,
  );

async function extractPdfText(file: File) {
  try {
    const [{ getDocument, GlobalWorkerOptions }, worker] = await Promise.all([
      import("pdfjs-dist"),
      import("pdfjs-dist/build/pdf.worker.mjs?url"),
    ]);
    GlobalWorkerOptions.workerSrc = worker.default;
    const pdf = await getDocument({ data: await file.arrayBuffer() }).promise;
    const limit = Math.min(pdf.numPages, 8);
    const pages: string[] = [];
    for (let pageNumber = 1; pageNumber <= limit; pageNumber += 1) {
      const page = await pdf.getPage(pageNumber);
      const content = await page.getTextContent();
      pages.push(
        content.items
          .map((item: { str?: string }) => ("str" in item ? item.str : ""))
          .join(" ")
          .replace(/\s+/g, " ")
          .trim(),
      );
    }
    return pages.join("\n").slice(0, 12000);
  } catch {
    return "PDF attached, but text extraction failed.";
  }
}

async function extractDocxText(file: File) {
  try {
    const mammoth = await import("mammoth");
    const result = await mammoth.extractRawText({
      arrayBuffer: await file.arrayBuffer(),
    });
    return result.value.replace(/\r\n/g, "\n").slice(0, 12000);
  } catch {
    return "DOCX attached, but text extraction failed.";
  }
}

function getImageMetadata(file: File) {
  return new Promise<{ width: number; height: number }>((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      resolve({ width: image.width, height: image.height });
      URL.revokeObjectURL(url);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to inspect image"));
    };
    image.src = url;
  });
}

function fileToDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Failed to read image preview"));
    reader.onload = () => resolve(String(reader.result));
    reader.readAsDataURL(file);
  });
}

async function toChatAttachment(file: File): Promise<ChatAttachment> {
  if (file.type.startsWith("image/")) {
    const dimensions = await getImageMetadata(file).catch(() => null);
    return {
      id: crypto.randomUUID(),
      name: file.name,
      size: file.size,
      mimeType: file.type || "image/*",
      previewUrl: file.size <= 1_500_000 ? await fileToDataUrl(file) : null,
      kind: "image",
      width: dimensions?.width ?? null,
      height: dimensions?.height ?? null,
      included: true,
    };
  }
  if (file.type === "application/pdf" || /\.pdf$/i.test(file.name)) {
    return {
      id: crypto.randomUUID(),
      name: file.name,
      size: file.size,
      mimeType: file.type || "application/pdf",
      textContent: await extractPdfText(file),
      kind: "pdf",
      included: true,
    };
  }
  if (
    file.type ===
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
    /\.docx$/i.test(file.name)
  ) {
    return {
      id: crypto.randomUUID(),
      name: file.name,
      size: file.size,
      mimeType: file.type || "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      textContent: await extractDocxText(file),
      kind: "docx",
      included: true,
    };
  }
  if (isTextPreviewable(file)) {
    return {
      id: crypto.randomUUID(),
      name: file.name,
      size: file.size,
      mimeType: file.type || "text/plain",
      textContent: (await file.text()).slice(0, 12000),
      kind: "text",
      included: true,
    };
  }
  return {
    id: crypto.randomUUID(),
    name: file.name,
    size: file.size,
    mimeType: file.type || "application/octet-stream",
    kind: "binary",
    included: true,
  };
}

function buildPromptContent(content: string, attachments: ChatAttachment[]) {
  const includedAttachments = attachments.filter((attachment) => attachment.included !== false);
  if (!includedAttachments.length) return content;
  return [
    content,
    "",
    "Attached file context:",
    includedAttachments
      .map((attachment, index) => {
        const header = `File ${index + 1}: ${attachment.name} (${attachment.kind}, ${formatBytes(attachment.size)})`;
        if (attachment.textContent?.trim()) {
          return `${header}\nExtract:\n${attachment.textContent}`;
        }
        if (attachment.kind === "image") {
          return [
            header,
            `Image metadata: ${attachment.width ?? "?"}x${attachment.height ?? "?"}, ${attachment.mimeType}`,
            "Note: current runtime is text-only, so the image itself is not visually analyzed yet.",
          ].join("\n");
        }
        return `${header}\nNote: metadata only.`;
      })
      .join("\n\n"),
  ].join("\n");
}

const STRUCTURED_MESSAGE_BLOCK =
  /^(\s*[-*]\s|\s*\d+\.\s|\s*>\s|\s*#{1,6}\s|\s*\|)/m;

function isStructuredMessageBlock(block: string) {
  return STRUCTURED_MESSAGE_BLOCK.test(block);
}

function normalizeMessageContent(content: string) {
  return content
    .replace(/\r\n/g, "\n")
    .replace(/\u00c2/g, "")
    .replace(/\u00a0/g, " ")
    .split(/\n{2,}/)
    .map((block) => {
      const lines = block
        .split("\n")
        .map((line) => line.replace(/[ \t]+/g, " ").trim())
        .filter(Boolean);

      if (!lines.length) {
        return "";
      }

      const joined = lines.join("\n");
      return isStructuredMessageBlock(joined) ? joined : lines.join(" ");
    })
    .filter(Boolean)
    .join("\n\n");
}

function getMessageBlocks(content: string) {
  return normalizeMessageContent(content)
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean)
    .map((block) => ({
      content: block,
      structured: isStructuredMessageBlock(block),
    }));
}

function getMessagePreview(content: string) {
  return normalizeMessageContent(content).replace(/\n+/g, " ").trim();
}

function normalizeUserMessageContent(content: string) {
  return content
    .replace(/\r\n/g, "\n")
    .replace(/\u00c2/g, "")
    .replace(/\u00a0/g, " ");
}

function getAttachmentMetaLabel(attachment: ChatAttachment) {
  const details = [attachment.kind ?? "file", formatBytes(attachment.size)];
  if (attachment.kind === "image" && attachment.width && attachment.height) {
    details.push(`${attachment.width}x${attachment.height}`);
  }
  if (attachment.included === false) {
    details.push("excluded");
  }
  return details.join(" - ");
}

function buildPromptMessages(messages: ChatMessage[], settings: ChatGenerationSettings) {
  const promptMessages: ChatMessage[] = [];
  if (settings.systemPrompt.trim()) {
    promptMessages.push({
      id: "system-prompt",
      role: "system",
      content: settings.systemPrompt.trim(),
      createdAt: new Date().toISOString(),
    });
  }
  for (const message of messages) {
    promptMessages.push(
      message.role === "user"
        ? { ...message, content: buildPromptContent(message.content, message.attachments || []) }
        : message,
    );
  }
  return promptMessages;
}

function exportChatAsMarkdown(chat: ChatSession) {
  const parts = [`# ${chat.title}`, ""];
  if (chat.tags?.length) {
    parts.push(`Tags: ${chat.tags.map((tag) => `#${tag}`).join(", ")}`);
  }
  parts.push(`Model: ${chat.settings.modelId}`, "");
  for (const message of chat.messages) {
    parts.push(`## ${message.role === "user" ? "You" : "Assistant"}`);
    parts.push(message.content || "_Empty message_");
    if (message.attachments?.length) {
      parts.push("");
      parts.push("Attachments:");
      for (const attachment of message.attachments) {
        parts.push(`- ${attachment.name} (${attachment.kind || "file"}, ${formatBytes(attachment.size)})`);
      }
    }
    parts.push("");
  }
  return parts.join("\n");
}

function App() {
  const [models, setModels] = useState<ModelInfo[]>(MODELS);
  const [runtime, setRuntime] = useState<RuntimeStatus>({
    selectedModelId: INITIAL_DEFAULT_MODEL_ID,
    loadedModelId: null,
    platform: "booting",
    inferenceReady: false,
    isGenerating: false,
    modelStates: {},
  });
  const [appSettings, setAppSettings] = useState<AppSettings>({
    themePreference: "system",
    selectedModelId: INITIAL_DEFAULT_MODEL_ID,
    defaultModelId: INITIAL_DEFAULT_MODEL_ID,
    autoLoadLastModel: true,
  });
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([createChatSession(INITIAL_DEFAULT_MODEL_ID)]);
  const [activeChatId, setActiveChatId] = useState(chatSessions[0].id);
  const [draft, setDraft] = useState("");
  const [pendingAttachments, setPendingAttachments] = useState<ChatAttachment[]>([]);
  const [chatSearch, setChatSearch] = useState("");
  const [systemTheme, setSystemTheme] = useState<EffectiveTheme>(getSystemTheme);
  const [isBooting, setIsBooting] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isRenamingChat, setIsRenamingChat] = useState(false);
  const [renameDraft, setRenameDraft] = useState("");
  const [tagDraft, setTagDraft] = useState("");
  const composerFormRef = useRef<HTMLFormElement | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const messageEndRef = useRef<HTMLDivElement | null>(null);

  const activeChat = chatSessions.find((chat) => chat.id === activeChatId) ?? chatSessions[0];
  const activeSettings = activeChat?.settings ?? defaultGenerationSettings(INITIAL_DEFAULT_MODEL_ID);
  const selectedModel = models.find((model) => model.id === activeSettings.modelId) ?? models[0];
  const selectedModelState =
    runtime.modelStates[activeSettings.modelId] ?? defaultModelState(selectedModel?.sizeBytes || 0);
  const chatModelReady =
    selectedModelState.status === "ready" ||
    (runtime.inferenceReady && runtime.loadedModelId === activeSettings.modelId);
  const effectiveTheme =
    appSettings.themePreference === "system" ? systemTheme : appSettings.themePreference;
  const filteredChats = useMemo(() => {
    const query = chatSearch.trim().toLowerCase();
    const orderedChats = sortChats(chatSessions);
    if (!query) return orderedChats;
    return orderedChats.filter((chat) =>
      [chat.title, ...(chat.tags || []), ...chat.messages.map((message) => message.content)]
        .join(" ")
        .toLowerCase()
        .includes(query),
    );
  }, [chatSearch, chatSessions]);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: light)");
    const updateTheme = () => setSystemTheme(media.matches ? "light" : "dark");
    updateTheme();
    media.addEventListener("change", updateTheme);
    return () => media.removeEventListener("change", updateTheme);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = effectiveTheme;
  }, [effectiveTheme]);

  useEffect(() => {
    const textarea = composerRef.current;
    if (!textarea) return;
    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 240)}px`;
  }, [draft]);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [activeChatId, activeChat?.messages.length, runtime.isGenerating]);

  useEffect(() => {
    let cancelled = false;
    const cleanups: Array<() => void> = [];

    async function bootstrap() {
      try {
        const [availableModels, runtimeStatus, persistedSettings, persistedStore] = await Promise.all([
          listModels(),
          getRuntimeStatus(),
          getAppSettings(),
          loadChatStore(),
        ]);
        if (cancelled) return;

        setModels(availableModels);
        setRuntime(runtimeStatus);
        setAppSettings(persistedSettings);

        const fallbackModelId =
          persistedSettings.selectedModelId ||
          persistedSettings.defaultModelId ||
          availableModels[0]?.id ||
          getPreferredModelId(runtimeStatus.platform);
        const store = persistedStore || loadBrowserStore(fallbackModelId);
        const chats = sortChats((store.chats.length ? store.chats : [createChatSession(fallbackModelId)]).map((chat) => ({
          ...chat,
          settings: { ...defaultGenerationSettings(fallbackModelId), ...chat.settings },
          pinned: Boolean(chat.pinned),
          tags: normalizeTags(chat.tags || []),
          titleLocked: Boolean(chat.titleLocked),
        })));
        setChatSessions(chats);
        setActiveChatId(chats.some((chat) => chat.id === store.activeChatId) ? store.activeChatId : chats[0].id);

        cleanups.push(
          await onDownloadProgress((event) =>
            setRuntime((current) => ({
              ...current,
              modelStates: {
                ...current.modelStates,
                [event.modelId]: {
                  ...(current.modelStates[event.modelId] ?? defaultModelState()),
                  status: event.status,
                  progress: event.progress,
                  downloadedBytes: event.downloadedBytes,
                  totalBytes: event.totalBytes,
                  error: event.error ?? null,
                  localPath: event.localPath ?? null,
                },
              },
            })),
          ),
        );
        cleanups.push(
          await onDownloadComplete(() => {
            void refreshRuntime();
          }),
        );
        cleanups.push(
          await onDownloadError((event) =>
            setRuntime((current) => ({
              ...current,
              modelStates: {
                ...current.modelStates,
                [event.modelId]: {
                  ...(current.modelStates[event.modelId] ?? defaultModelState()),
                  status: event.status,
                  progress: event.progress,
                  downloadedBytes: event.downloadedBytes,
                  totalBytes: event.totalBytes,
                  error: event.error ?? "Download failed",
                  localPath: event.localPath ?? null,
                },
              },
            })),
          ),
        );
        cleanups.push(
          await onChatToken(({ messageId, token }) =>
            setChatSessions((current) =>
              current.map((chat) => ({
                ...chat,
                messages: chat.messages.map((message) =>
                  message.id === messageId ? { ...message, content: message.content + token, streaming: true } : message,
                ),
              })),
            ),
          ),
        );
        cleanups.push(
          await onChatComplete(({ messageId }) => {
            setChatSessions((current) =>
              current.map((chat) =>
                chat.messages.some((message) => message.id === messageId)
                  ? updateChatMetadata({
                      ...chat,
                      updatedAt: new Date().toISOString(),
                      messages: chat.messages.map((message) =>
                        message.id === messageId ? { ...message, streaming: false } : message,
                      ),
                    })
                  : chat,
              ),
            );
            void refreshRuntime();
          }),
        );
        cleanups.push(
          await onChatError(({ messageId, error }) => {
            setChatSessions((current) =>
              current.map((chat) =>
                chat.messages.some((message) => message.id === messageId)
                  ? {
                      ...chat,
                      messages: chat.messages.map((message) =>
                        message.id === messageId
                          ? { ...message, streaming: false, error, content: message.content || "Generation failed." }
                          : message,
                      ),
                    }
                  : chat,
              ),
            );
            void refreshRuntime();
          }),
        );
      } catch (error) {
        console.error("bootstrap failed", error);
      } finally {
        if (!cancelled) {
          setIsBooting(false);
        }
      }
    }

    void bootstrap();
    return () => {
      cancelled = true;
      for (const cleanup of cleanups) cleanup();
    };
  }, []);

  useEffect(() => {
    if (isBooting || !chatSessions.length || !activeChatId) return;
    const store: ChatStore = { chats: sortChats(chatSessions), activeChatId };
    const timeout = window.setTimeout(() => {
      void saveChatStore(store);
      if (typeof window !== "undefined") {
        window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chatSessions));
        window.localStorage.setItem(ACTIVE_CHAT_STORAGE_KEY, activeChatId);
      }
    }, 250);
    return () => window.clearTimeout(timeout);
  }, [activeChatId, chatSessions, isBooting]);

  const refreshRuntime = async () => setRuntime(await getRuntimeStatus());
  const persistAppSettings = async (nextSettings: AppSettings) => setAppSettings(await updateAppSettings(nextSettings));
  const updateActiveChat = (updater: (chat: ChatSession) => ChatSession) =>
    activeChat &&
    setChatSessions((current) =>
      current.map((chat) => (chat.id === activeChat.id ? updateChatMetadata(updater(chat)) : chat)),
    );

  const appendAssistantError = (chatId: string, error: string) =>
    setChatSessions((current) =>
      current.map((chat) =>
        chat.id === chatId
          ? {
              ...chat,
              messages: [
                ...chat.messages,
                {
                  id: crypto.randomUUID(),
                  role: "assistant",
                  content: "Generation failed.",
                  createdAt: new Date().toISOString(),
                  error,
                },
              ],
            }
          : chat,
      ),
    );

  const toggleChatPinned = (chatId: string) =>
    setChatSessions((current) =>
      current.map((chat) =>
        chat.id === chatId
          ? { ...chat, pinned: !chat.pinned, updatedAt: new Date().toISOString() }
          : chat,
      ),
    );

  const addTagToActiveChat = () => {
    const nextTag = tagDraft.trim().toLowerCase();
    if (!activeChat || !nextTag) return;
    updateActiveChat((chat) => ({
      ...chat,
      updatedAt: new Date().toISOString(),
      tags: normalizeTags([...(chat.tags || []), nextTag]),
    }));
    setTagDraft("");
  };

  const removeTagFromActiveChat = (tagToRemove: string) => {
    if (!activeChat) return;
    updateActiveChat((chat) => ({
      ...chat,
      updatedAt: new Date().toISOString(),
      tags: (chat.tags || []).filter((tag) => tag !== tagToRemove),
    }));
  };

  async function handleSelectModel(modelId: string) {
    setRuntime(await selectModel(modelId));
    await persistAppSettings({ ...appSettings, selectedModelId: modelId });
  }

  const handleDownload = async (modelId: string) => setRuntime(await downloadModel(modelId));

  async function handleSetChatModel(modelId: string) {
    updateActiveChat((chat) => ({ ...chat, updatedAt: new Date().toISOString(), settings: { ...chat.settings, modelId } }));
    await handleSelectModel(modelId);
  }

  async function runGeneration(chatId: string, visibleMessages: ChatMessage[], settings: ChatGenerationSettings) {
    if (runtime.loadedModelId !== settings.modelId) {
      setRuntime(await selectModel(settings.modelId));
    }
    const start = await sendChat(settings.modelId, buildPromptMessages(visibleMessages, settings), settings);
    setChatSessions((current) =>
      current.map((chat) =>
        chat.id === chatId
          ? updateChatMetadata({
              ...chat,
              updatedAt: new Date().toISOString(),
              messages: [
                ...visibleMessages,
                { id: start.messageId, role: "assistant", content: "", createdAt: new Date().toISOString(), streaming: true },
              ],
            })
          : chat,
      ),
    );
    await refreshRuntime();
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const content = draft.trim();
    if ((!content && !pendingAttachments.length) || !activeChat) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: content || "Please use the attached files as context.",
      createdAt: new Date().toISOString(),
      attachments: pendingAttachments,
    };
    const visibleMessages = [...activeChat.messages, userMessage];
    setChatSessions((current) =>
      current.map((chat) =>
        chat.id === activeChat.id ? updateChatMetadata({ ...chat, updatedAt: new Date().toISOString(), messages: visibleMessages }) : chat,
      ),
    );
    setDraft("");
    setPendingAttachments([]);

    try {
      await runGeneration(activeChat.id, visibleMessages, activeChat.settings);
    } catch (error) {
      appendAssistantError(activeChat.id, error instanceof Error ? error.message : "Unable to start generation.");
      void refreshRuntime();
    }
  }

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey) {
      return;
    }
    event.preventDefault();
    composerFormRef.current?.requestSubmit();
  };

  const handleCreateChat = () => {
    const modelId =
      appSettings.selectedModelId ||
      appSettings.defaultModelId ||
      models[0]?.id ||
      getPreferredModelId(runtime.platform);
    const chat = createChatSession(modelId);
    setChatSessions((current) => [chat, ...current]);
    setActiveChatId(chat.id);
    setDraft("");
    setPendingAttachments([]);
    setTagDraft("");
  };

  const handleDeleteChat = (chatId: string) =>
    (setTagDraft(""),
    setChatSessions((current) => {
      if (current.length === 1) {
        const replacement = createChatSession(
          appSettings.defaultModelId || models[0]?.id || getPreferredModelId(runtime.platform),
        );
        setActiveChatId(replacement.id);
        return [replacement];
      }
      const next = current.filter((chat) => chat.id !== chatId);
      if (chatId === activeChatId) setActiveChatId(next[0].id);
      return next;
    }));

  const handleExportChat = () => {
    if (!activeChat) return;
    const blob = new Blob([exportChatAsMarkdown(activeChat)], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${sanitizeFileName(activeChat.title)}.md`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const handleEditAndResend = (messageId: string) => {
    if (!activeChat) return;
    const index = activeChat.messages.findIndex((message) => message.id === messageId);
    if (index < 0) return;
    const message = activeChat.messages[index];
    setDraft(message.content);
    setPendingAttachments(message.attachments || []);
    setChatSessions((current) =>
      current.map((chat) =>
        chat.id === activeChat.id
          ? updateChatMetadata({ ...chat, updatedAt: new Date().toISOString(), messages: activeChat.messages.slice(0, index) })
          : chat,
      ),
    );
  };

  const handleRegenerate = async () => {
    if (!activeChat || runtime.isGenerating) return;
    const lastAssistantIndex = [...activeChat.messages].map((message, index) => ({ message, index })).reverse().find((entry) => entry.message.role === "assistant")?.index;
    if (lastAssistantIndex == null) return;
    const trimmedMessages = activeChat.messages.filter((_, index) => index < lastAssistantIndex);
    setChatSessions((current) =>
      current.map((chat) =>
        chat.id === activeChat.id ? updateChatMetadata({ ...chat, updatedAt: new Date().toISOString(), messages: trimmedMessages }) : chat,
      ),
    );
    try {
      await runGeneration(activeChat.id, trimmedMessages, activeChat.settings);
    } catch (error) {
      appendAssistantError(activeChat.id, error instanceof Error ? error.message : "Unable to regenerate.");
      void refreshRuntime();
    }
  };

  const handleAttachFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    const nextAttachments = await Promise.all(files.map(toChatAttachment));
    setPendingAttachments((current) => [...current, ...nextAttachments]);
    event.target.value = "";
  };

  const togglePendingAttachment = (attachmentId: string) =>
    setPendingAttachments((current) =>
      current.map((attachment) =>
        attachment.id === attachmentId
          ? { ...attachment, included: attachment.included === false }
          : attachment,
      ),
    );

  const canType = Boolean(activeChat);
  const canSend =
    (draft.trim().length > 0 || pendingAttachments.length > 0) &&
    !runtime.isGenerating &&
    Boolean(activeChat);
  const recommendedModelId = getPreferredModelId(runtime.platform);
  const sendBlockedByModel =
    selectedModelState.status === "remote" ||
    selectedModelState.status === "downloading" ||
    selectedModelState.status === "loading";
  const totalFiles = activeChat ? activeChat.messages.reduce((count, message) => count + (message.attachments?.length || 0), 0) : 0;

  return (
    <div className={`shell${isSettingsOpen ? " settings-open" : ""}`}>
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark">L</div><div><h1>LocalLM</h1><p>Private AI workspace</p></div></div>
        <button className="new-chat-button" type="button" onClick={handleCreateChat}>Start new chat</button>
        <div className="search-wrap"><input className="chat-search" type="search" value={chatSearch} onChange={(event) => setChatSearch(event.target.value)} placeholder="Search chats" /></div>
        <section className="panel conversations-panel"><div className="panel-label">Conversations</div><div className="chat-list">{filteredChats.map((chat) => <article key={chat.id} className={`chat-card${chat.id === activeChatId ? " active" : ""}`}><button className="chat-card-main" type="button" onClick={() => { setActiveChatId(chat.id); setTagDraft(""); }}><div className="chat-card-top"><strong>{chat.pinned ? `Pinned - ${chat.title}` : chat.title}</strong><span>{formatChatTime(chat.updatedAt)}</span></div><p>{getAssistantMessagePreview(chat.messages[chat.messages.length - 1]?.content || "New conversation")}</p>{chat.tags?.length ? <div className="tag-row">{chat.tags.map((tag) => <span key={tag} className="tag-chip">#{tag}</span>)}</div> : null}<div className="chat-card-meta"><span>{chat.messages.length} messages</span><span>{chat.messages.reduce((count, message) => count + (message.attachments?.length || 0), 0)} files</span></div></button><button className="chat-card-delete" type="button" onClick={() => handleDeleteChat(chat.id)}>Delete</button></article>)}</div></section>
        <section className="panel compact info-panel"><div className="panel-label">Workspace</div><div className="runtime-row"><span>Theme</span><strong>{effectiveTheme}</strong></div><div className="runtime-row"><span>Loaded model</span><strong>{runtime.loadedModelId ?? "None"}</strong></div><div className="runtime-row"><span>Platform</span><strong>{runtime.platform}</strong></div></section>
      </aside>
      <main className="workspace">
        <header className="topbar"><div><div className="eyebrow">Conversation</div>{isRenamingChat ? <input className="rename-input" value={renameDraft} onChange={(event) => setRenameDraft(event.target.value)} onBlur={() => { if (activeChat && renameDraft.trim()) updateActiveChat((chat) => ({ ...chat, title: renameDraft.trim(), titleLocked: true, updatedAt: new Date().toISOString() })); setIsRenamingChat(false); }} onKeyDown={(event) => { if (event.key === "Enter" && activeChat && renameDraft.trim()) { updateActiveChat((chat) => ({ ...chat, title: renameDraft.trim(), titleLocked: true, updatedAt: new Date().toISOString() })); setIsRenamingChat(false); } }} autoFocus /> : <h2>{activeChat?.title ?? "New chat"}</h2>}<p className="topbar-subtitle">{chatModelReady ? `${selectedModel?.name ?? "Model"} is ready for this chat` : selectedModelState.status === "downloaded" ? "Model is downloaded. Send a message and it will load automatically." : "Open Settings to download or load the model chosen for this chat"}</p></div><div className="topbar-controls"><div className="topbar-badge-group"><span className={`status-pill ${selectedModelState.status}`}>{selectedModelState.status}</span><span className="topbar-badge">{totalFiles} files in chat</span></div><button className="settings-button" type="button" onClick={() => activeChat && toggleChatPinned(activeChat.id)}>{activeChat?.pinned ? "Unpin" : "Pin"}</button><button className="settings-button" type="button" onClick={() => { setRenameDraft(activeChat?.title ?? ""); setIsRenamingChat(true); }}>Rename</button><button className="settings-button" type="button" onClick={handleExportChat}>Export</button><button className="settings-button" type="button" onClick={() => void handleRegenerate()}>Regenerate</button><button className="settings-button" type="button" onClick={() => setIsSettingsOpen((current) => !current)}>{isSettingsOpen ? "Close settings" : "Open settings"}</button></div></header>
        <section className="messages">{(activeChat?.messages || []).map((message) => <article key={message.id} className={`message-row ${message.role === "user" ? "user" : "assistant"}`}><div className="avatar">{message.role === "user" ? "You" : "AI"}</div><div className="bubble">{message.attachments?.length ? <div className="message-attachments">{message.attachments.map((attachment) => <div key={attachment.id} className={`attachment-chip message-chip${attachment.included === false ? " muted" : ""}`}>{attachment.previewUrl && attachment.kind === "image" ? <img className="attachment-preview" src={attachment.previewUrl} alt={attachment.name} /> : null}<div><span>{attachment.name}</span><small>{getAttachmentMetaLabel(attachment)}</small></div></div>)}</div> : null}{message.content ? <MessageContent role={message.role} content={message.content} /> : <p className="typing-indicator">Thinking...</p>}{message.role === "user" ? <div className="message-tools"><button className="message-tool" type="button" onClick={() => handleEditAndResend(message.id)}>Edit & resend</button></div> : null}{message.streaming ? <span className="stream-caret" /> : null}{message.error ? <p className="message-error">{message.error}</p> : null}</div></article>)}<div ref={messageEndRef} /></section>
        <form ref={composerFormRef} className="composer-wrap" onSubmit={handleSubmit}><div className="composer-card">{pendingAttachments.length ? <div className="attachment-tray">{pendingAttachments.map((attachment) => <div key={attachment.id} className={`attachment-chip${attachment.included === false ? " muted" : ""}`}>{attachment.previewUrl && attachment.kind === "image" ? <img className="attachment-preview" src={attachment.previewUrl} alt={attachment.name} /> : null}<div><strong>{attachment.name}</strong><small>{getAttachmentMetaLabel(attachment)}</small></div><button className="chip-toggle" type="button" onClick={() => togglePendingAttachment(attachment.id)}>{attachment.included === false ? "Include" : "Exclude"}</button><button className="chip-remove" type="button" onClick={() => setPendingAttachments((current) => current.filter((item) => item.id !== attachment.id))}>Remove</button></div>)}</div> : null}<textarea ref={composerRef} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={handleComposerKeyDown} placeholder={chatModelReady ? "Ask something, summarize files, or continue the conversation" : selectedModelState.status === "downloaded" ? "Write your message. The model will load when you send." : "You can type now. Download or load the model before sending."} rows={1} disabled={!canType} /><div className="composer-footer"><span>{runtime.isGenerating ? "Generating locally..." : chatModelReady ? "Text, PDF, DOCX, and image metadata are added as local context." : selectedModelState.status === "downloaded" ? "Model downloaded. Sending will load it automatically." : "You can type already. Sending unlocks after the model is available."}</span><div className="composer-actions"><input ref={fileInputRef} className="hidden-file-input" type="file" multiple accept=".txt,.md,.json,.csv,.yaml,.yml,.xml,.html,.css,.js,.ts,.jsx,.tsx,.py,.rs,.java,.kt,.swift,.pdf,.docx,image/*" onChange={handleAttachFiles} /><button className="send-button secondary" type="button" onClick={() => fileInputRef.current?.click()} disabled={!canType}>Add files</button>{runtime.isGenerating ? <button className="send-button secondary" type="button" onClick={() => void cancelGeneration().then(setRuntime)}>Stop</button> : null}<button className="send-button" type="submit" disabled={!canSend || sendBlockedByModel}>Send</button></div></div></div></form>
      </main>
      <aside className={`settings-panel${isSettingsOpen ? " open" : ""}`}>
        <div className="settings-header"><div><div className="panel-label">Settings</div><h3>Configuration</h3></div><button className="icon-button" type="button" onClick={() => setIsSettingsOpen(false)}>Close</button></div>
        <section className="panel settings-section"><div className="settings-row"><div><strong>Appearance</strong><p>Switch between system, light, and dark mode.</p></div><select aria-label="Theme preference" className="theme-select" value={appSettings.themePreference} onChange={(event) => void persistAppSettings({ ...appSettings, themePreference: event.target.value as ThemePreference })}><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select></div><div className="toggle-row"><div><strong>Auto-load last model</strong><p>Best effort load at startup if already downloaded.</p></div><button className="toggle-button" type="button" onClick={() => void persistAppSettings({ ...appSettings, autoLoadLastModel: !appSettings.autoLoadLastModel })}>{appSettings.autoLoadLastModel ? "On" : "Off"}</button></div><div className="settings-stack"><label className="field"><span>Default model</span><select className="theme-select" value={appSettings.defaultModelId ?? models[0]?.id ?? ""} onChange={(event) => void persistAppSettings({ ...appSettings, defaultModelId: event.target.value })}>{models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</select></label></div></section>
        <section className="panel settings-section"><div className="panel-label">Current chat</div><div className="settings-stack"><label className="field"><span>Model</span><select className="theme-select" value={activeSettings.modelId} onChange={(event) => void handleSetChatModel(event.target.value)}>{models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</select></label><label className="field"><span>Tags</span><div className="tag-editor"><input className="number-input" value={tagDraft} placeholder="Add a tag and press Enter" onChange={(event) => setTagDraft(event.target.value)} onKeyDown={(event: KeyboardEvent<HTMLInputElement>) => { if (event.key === "Enter") { event.preventDefault(); addTagToActiveChat(); } }} /><button className="toggle-button" type="button" onClick={addTagToActiveChat}>Add</button></div>{activeChat?.tags?.length ? <div className="tag-row">{activeChat.tags.map((tag) => <button key={tag} className="tag-chip removable" type="button" onClick={() => removeTagFromActiveChat(tag)}>#{tag}</button>)}</div> : null}</label><label className="field"><span>System prompt</span><textarea className="settings-textarea" rows={4} value={activeSettings.systemPrompt} onChange={(event) => updateActiveChat((chat) => ({ ...chat, updatedAt: new Date().toISOString(), settings: { ...chat.settings, systemPrompt: event.target.value } }))} /></label><div className="field-grid"><label className="field"><span>Temperature</span><input type="range" min="0" max="2" step="0.1" value={activeSettings.temperature} onChange={(event) => updateActiveChat((chat) => ({ ...chat, settings: { ...chat.settings, temperature: Number(event.target.value) } }))} /><small>{activeSettings.temperature.toFixed(1)}</small></label><label className="field"><span>Top P</span><input type="range" min="0.1" max="1" step="0.05" value={activeSettings.topP} onChange={(event) => updateActiveChat((chat) => ({ ...chat, settings: { ...chat.settings, topP: Number(event.target.value) } }))} /><small>{activeSettings.topP.toFixed(2)}</small></label></div><label className="field"><span>Max tokens</span><input className="number-input" type="number" min="64" max="4096" step="64" value={activeSettings.maxTokens} onChange={(event) => updateActiveChat((chat) => ({ ...chat, settings: { ...chat.settings, maxTokens: Number(event.target.value) } }))} /></label></div></section>
        <section className="panel settings-section"><div className="panel-label">Runtime</div><div className="runtime-grid"><div className="runtime-stat"><span>Status</span><strong>{runtime.loadedModelId === activeSettings.modelId ? "Ready" : "Needs load"}</strong></div><div className="runtime-stat"><span>Loaded model</span><strong>{runtime.loadedModelId ?? "None"}</strong></div><div className="runtime-stat"><span>Current chat</span><strong>{activeChat?.messages.length ?? 0} messages</strong></div><div className="runtime-stat"><span>Platform</span><strong>{runtime.platform}</strong></div></div></section>
        <section className="panel settings-section"><div className="panel-label">Available models</div><div className="model-list">{models.map((model) => { const modelState = runtime.modelStates[model.id] ?? defaultModelState(model.sizeBytes); const isSelected = model.id === activeSettings.modelId; return <article key={model.id} className={`model-card${isSelected ? " selected" : ""}`}><button className="model-main" type="button" onClick={() => void handleSetChatModel(model.id)}><div className="model-topline"><span>{model.name}</span><span className={`status-pill ${modelState.status}`}>{modelState.status}</span></div><p>{model.description}</p><div className="model-label-row"><span className={`recommendation-chip${model.id === recommendedModelId ? " primary" : ""}`}>{getModelRecommendation(model.id, runtime.platform)}</span>{isSelected ? <span className="recommendation-chip">Current chat</span> : null}</div><div className="model-meta-row"><span>{getModelStorageLabel(modelState, model.sizeBytes)}</span><span>{Math.round(modelState.progress)}%</span></div><div className="model-meta-row"><span>{model.sizeHuman}</span><span>{modelState.localPath ? "Local GGUF" : modelState.status === "downloaded" || modelState.status === "ready" ? "Imported model" : "Remote only"}</span></div><div className="progress-track"><div className="progress-bar" style={{ width: `${modelState.progress}%` }} /></div>{modelState.error ? <div className="model-error">{modelState.error}</div> : null}</button><div className="model-actions">{(modelState.status === "remote" || modelState.status === "error") ? <button className="model-action-button" type="button" onClick={() => void handleDownload(model.id)}>Download</button> : null}{modelState.status === "downloading" ? <button className="model-action-button secondary" type="button" onClick={() => void cancelModelDownload(model.id).then(setRuntime)}>Cancel</button> : null}{(modelState.status === "downloaded" || modelState.status === "ready") ? <button className="model-action-button" type="button" onClick={() => void handleSelectModel(model.id)}>{getModelActionLabel(modelState.status, isSelected)}</button> : null}{(modelState.status === "downloaded" || modelState.status === "ready" || modelState.status === "error") ? <button className="model-action-button secondary" type="button" onClick={() => void deleteModel(model.id).then(setRuntime)}>Delete</button> : null}</div></article>; })}</div></section>
      </aside>
    </div>
  );
}

export default App;
