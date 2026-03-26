import { FormEvent, useEffect, useMemo, useState } from "react";
import { MODELS } from "./models";
import {
  cancelGeneration,
  cancelModelDownload,
  downloadModel,
  getRuntimeStatus,
  getThemePreference,
  listModels,
  onChatComplete,
  onChatError,
  onChatToken,
  onDownloadComplete,
  onDownloadError,
  onDownloadProgress,
  selectModel,
  sendChat,
  setThemePreference,
} from "./tauri";
import type {
  ChatMessage,
  EffectiveTheme,
  ModelInfo,
  ModelRuntimeState,
  RuntimeStatus,
  ThemePreference,
} from "./types";

const starterMessages: ChatMessage[] = [
  {
    id: "welcome-1",
    role: "assistant",
    content:
      "Welcome to LocalLM. Download a model, load it on-device, and chat with real local inference.",
    createdAt: new Date().toISOString(),
  },
];

function getSystemTheme(): EffectiveTheme {
  if (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-color-scheme: light)").matches
  ) {
    return "light";
  }
  return "dark";
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>(starterMessages);
  const [draft, setDraft] = useState("");
  const [models, setModels] = useState<ModelInfo[]>(MODELS);
  const [runtime, setRuntime] = useState<RuntimeStatus>({
    selectedModelId: MODELS[0].id,
    loadedModelId: null,
    platform: "booting",
    inferenceReady: false,
    isGenerating: false,
    modelStates: {},
  });
  const [themePreference, setThemePreferenceState] =
    useState<ThemePreference>("system");
  const [systemTheme, setSystemTheme] = useState<EffectiveTheme>(getSystemTheme);
  const [isBooting, setIsBooting] = useState(true);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: light)");
    const updateTheme = () => setSystemTheme(media.matches ? "light" : "dark");
    updateTheme();
    media.addEventListener("change", updateTheme);
    return () => media.removeEventListener("change", updateTheme);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const cleanups: Array<() => void> = [];

    async function bootstrap() {
      const [availableModels, runtimeStatus, storedTheme] = await Promise.all([
        listModels(),
        getRuntimeStatus(),
        getThemePreference(),
      ]);

      if (cancelled) {
        return;
      }

      setModels(availableModels);
      setRuntime(runtimeStatus);
      setThemePreferenceState(storedTheme);
      setIsBooting(false);

      cleanups.push(
        await onDownloadProgress((event) => {
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
          }));
        }),
      );

      cleanups.push(
        await onDownloadComplete((event) => {
          setRuntime((current) => ({
            ...current,
            modelStates: {
              ...current.modelStates,
              [event.modelId]: {
                ...(current.modelStates[event.modelId] ?? defaultModelState()),
                status: "downloaded",
                progress: 100,
                downloadedBytes: event.totalBytes,
                totalBytes: event.totalBytes,
                error: null,
                localPath: event.localPath ?? null,
              },
            },
          }));
        }),
      );

      cleanups.push(
        await onDownloadError((event) => {
          setRuntime((current) => ({
            ...current,
            modelStates: {
              ...current.modelStates,
              [event.modelId]: {
                ...(current.modelStates[event.modelId] ?? defaultModelState()),
                status: "error",
                progress: event.progress,
                downloadedBytes: event.downloadedBytes,
                totalBytes: event.totalBytes,
                error: event.error ?? "Download failed",
                localPath: event.localPath ?? null,
              },
            },
          }));
        }),
      );

      cleanups.push(
        await onChatToken(({ messageId, token }) => {
          setMessages((current) =>
            current.map((message) =>
              message.id === messageId
                ? {
                    ...message,
                    content: message.content + token,
                    streaming: true,
                  }
                : message,
            ),
          );
        }),
      );

      cleanups.push(
        await onChatComplete(({ messageId }) => {
          setMessages((current) =>
            current.map((message) =>
              message.id === messageId
                ? {
                    ...message,
                    streaming: false,
                  }
                : message,
            ),
          );
          void refreshRuntime();
        }),
      );

      cleanups.push(
        await onChatError(({ messageId, error }) => {
          setMessages((current) =>
            current.map((message) =>
              message.id === messageId
                ? {
                    ...message,
                    streaming: false,
                    error,
                    content: message.content || "Generation failed.",
                  }
                : message,
            ),
          );
          void refreshRuntime();
        }),
      );
    }

    async function refreshRuntime() {
      const nextRuntime = await getRuntimeStatus();
      if (!cancelled) {
        setRuntime(nextRuntime);
      }
    }

    void bootstrap();

    return () => {
      cancelled = true;
      for (const cleanup of cleanups) {
        cleanup();
      }
    };
  }, []);

  useEffect(() => {
    const effectiveTheme =
      themePreference === "system" ? systemTheme : themePreference;
    document.documentElement.dataset.theme = effectiveTheme;
  }, [systemTheme, themePreference]);

  const selectedModel = useMemo(
    () =>
      models.find((model) => model.id === runtime.selectedModelId) ?? models[0],
    [models, runtime.selectedModelId],
  );

  const selectedModelState =
    runtime.modelStates[runtime.selectedModelId] ?? defaultModelState();
  const canSend =
    draft.trim().length > 0 && runtime.inferenceReady && !runtime.isGenerating;
  const effectiveTheme =
    themePreference === "system" ? systemTheme : themePreference;

  async function handleThemeChange(nextTheme: ThemePreference) {
    const applied = await setThemePreference(nextTheme);
    setThemePreferenceState(applied);
  }

  async function handleSelectModel(modelId: string) {
    const nextRuntime = await selectModel(modelId);
    setRuntime(nextRuntime);
  }

  async function handleDownload(modelId: string) {
    const nextRuntime = await downloadModel(modelId);
    setRuntime(nextRuntime);
  }

  async function handleCancelDownload(modelId: string) {
    const nextRuntime = await cancelModelDownload(modelId);
    setRuntime(nextRuntime);
  }

  async function handleCancelGeneration() {
    const nextRuntime = await cancelGeneration();
    setRuntime(nextRuntime);
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const content = draft.trim();
    if (!content || !selectedModel) {
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content,
      createdAt: new Date().toISOString(),
    };

    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setDraft("");

    const start = await sendChat(selectedModel.id, nextMessages);

    setMessages((current) => [
      ...current,
      {
        id: start.messageId,
        role: "assistant",
        content: "",
        createdAt: new Date().toISOString(),
        streaming: true,
      },
    ]);

    const nextRuntime = await getRuntimeStatus();
    setRuntime(nextRuntime);
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">L</div>
          <div>
            <h1>LocalLM</h1>
            <p>Private AI on your device</p>
          </div>
        </div>

        <button
          className="new-chat-button"
          type="button"
          onClick={() => setMessages(starterMessages)}
        >
          New chat
        </button>

        <section className="panel">
          <div className="panel-label">Models</div>
          <div className="model-list">
            {models.map((model) => {
              const modelState =
                runtime.modelStates[model.id] ?? defaultModelState(model.sizeBytes);
              const isSelected = model.id === runtime.selectedModelId;
              const actionLabel = getModelActionLabel(modelState.status, isSelected);

              return (
                <article
                  key={model.id}
                  className={`model-card${isSelected ? " selected" : ""}`}
                >
                  <button
                    className="model-main"
                    type="button"
                    onClick={() => void handleSelectModel(model.id)}
                  >
                    <div className="model-topline">
                      <span>{model.name}</span>
                      <span className={`status-pill ${modelState.status}`}>
                        {modelState.status}
                      </span>
                    </div>
                    <p>{model.description}</p>
                    <div className="model-meta-row">
                      <span>{model.sizeHuman}</span>
                      <span>{Math.round(modelState.progress)}%</span>
                    </div>
                    <div className="progress-track">
                      <div
                        className="progress-bar"
                        style={{ width: `${modelState.progress}%` }}
                      />
                    </div>
                    {modelState.error ? (
                      <div className="model-error">{modelState.error}</div>
                    ) : null}
                  </button>

                  <div className="model-actions">
                    {modelState.status === "remote" || modelState.status === "error" ? (
                      <button
                        className="model-action-button"
                        type="button"
                        onClick={() => void handleDownload(model.id)}
                      >
                        Download
                      </button>
                    ) : null}
                    {modelState.status === "downloading" ? (
                      <button
                        className="model-action-button secondary"
                        type="button"
                        onClick={() => void handleCancelDownload(model.id)}
                      >
                        Cancel
                      </button>
                    ) : null}
                    {(modelState.status === "downloaded" ||
                      modelState.status === "ready") && (
                      <button
                        className="model-action-button"
                        type="button"
                        onClick={() => void handleSelectModel(model.id)}
                      >
                        {actionLabel}
                      </button>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        <section className="panel compact">
          <div className="panel-label">Runtime</div>
          <div className="runtime-row">
            <span>Platform</span>
            <strong>{runtime.platform}</strong>
          </div>
          <div className="runtime-row">
            <span>Loaded</span>
            <strong>{runtime.loadedModelId ?? "None"}</strong>
          </div>
          <div className="runtime-row">
            <span>Theme</span>
            <strong>{effectiveTheme}</strong>
          </div>
        </section>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <div className="eyebrow">Local model</div>
            <h2>{selectedModel?.name ?? "Choose a model"}</h2>
          </div>
          <div className="topbar-controls">
            <select
              aria-label="Theme preference"
              className="theme-select"
              value={themePreference}
              onChange={(event) =>
                void handleThemeChange(event.target.value as ThemePreference)
              }
            >
              <option value="system">System</option>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
            <div className="topbar-meta">
              <span>{selectedModel?.sizeHuman}</span>
              <span>
                {runtime.inferenceReady ? "On-device ready" : selectedModelState.status}
              </span>
            </div>
          </div>
        </header>

        <section className="messages">
          {messages.map((message) => (
            <article
              key={message.id}
              className={`message-row ${message.role === "user" ? "user" : "assistant"}`}
            >
              <div className="avatar">{message.role === "user" ? "You" : "AI"}</div>
              <div className="bubble">
                {message.content ? (
                  message.content.split("\n").map((line, index) => (
                    <p key={`${message.id}-${index}`}>{line}</p>
                  ))
                ) : (
                  <p className="typing-indicator">Thinking...</p>
                )}
                {message.streaming ? <span className="stream-caret" /> : null}
                {message.error ? <p className="message-error">{message.error}</p> : null}
              </div>
            </article>
          ))}
        </section>

        <form className="composer-wrap" onSubmit={handleSubmit}>
          <div className="composer-card">
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder={
                runtime.inferenceReady
                  ? "Message LocalLM"
                  : "Download and load a model to start chatting"
              }
              rows={1}
              disabled={!runtime.inferenceReady || isBooting}
            />
            <div className="composer-footer">
              <span>
                {runtime.isGenerating
                  ? "Generating on-device..."
                  : runtime.inferenceReady
                    ? "Replies stay on-device."
                    : "Select a downloaded model to load it before chatting."}
              </span>
              <div className="composer-actions">
                {runtime.isGenerating ? (
                  <button
                    className="send-button secondary"
                    type="button"
                    onClick={() => void handleCancelGeneration()}
                  >
                    Stop
                  </button>
                ) : null}
                <button className="send-button" type="submit" disabled={!canSend}>
                  Send
                </button>
              </div>
            </div>
          </div>
        </form>
      </main>
    </div>
  );
}

function defaultModelState(totalBytes = 0): ModelRuntimeState {
  return {
    status: "remote",
    progress: 0,
    downloadedBytes: 0,
    totalBytes,
    error: null,
    localPath: null,
  };
}

function getModelActionLabel(
  status: ModelRuntimeState["status"],
  isSelected: boolean,
) {
  if (status === "ready" && isSelected) {
    return "Ready";
  }
  if (status === "downloaded") {
    return "Load";
  }
  if (status === "ready") {
    return "Use";
  }
  return "Select";
}

export default App;
