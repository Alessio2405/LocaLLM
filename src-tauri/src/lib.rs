use anyhow::{anyhow, Context};
use chrono::Utc;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use tauri::{AppHandle, Emitter, Manager, State, Theme};

#[cfg(target_os = "android")]
use encoding_rs::UTF_8;
#[cfg(target_os = "android")]
use std::num::NonZeroU32;
#[cfg(target_os = "android")]
use llama_cpp_2::context::params::LlamaContextParams;
#[cfg(target_os = "android")]
use llama_cpp_2::llama_backend::LlamaBackend;
#[cfg(target_os = "android")]
use llama_cpp_2::llama_batch::LlamaBatch;
#[cfg(target_os = "android")]
use llama_cpp_2::model::{AddBos, LlamaModel, params::LlamaModelParams};
#[cfg(target_os = "android")]
use llama_cpp_2::sampling::LlamaSampler;

const MODEL_DOWNLOAD_PROGRESS: &str = "model-download-progress";
const MODEL_DOWNLOAD_COMPLETE: &str = "model-download-complete";
const MODEL_DOWNLOAD_ERROR: &str = "model-download-error";
const CHAT_TOKEN: &str = "chat-token";
const CHAT_COMPLETE: &str = "chat-complete";
const CHAT_ERROR: &str = "chat-error";
const OLLAMA_API_URL: &str = "http://127.0.0.1:11434/api/chat";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelInfo {
    id: String,
    name: String,
    url: String,
    mmproj_url: Option<String>,
    description: String,
    size_bytes: u64,
    mmproj_size_bytes: Option<u64>,
    size_human: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatMessage {
    id: String,
    role: String,
    content: String,
    created_at: String,
    streaming: Option<bool>,
    error: Option<String>,
    attachments: Option<Vec<ChatAttachment>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatAttachment {
    id: String,
    name: String,
    size: u64,
    mime_type: String,
    text_content: Option<String>,
    preview_url: Option<String>,
    kind: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    included: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatGenerationSettings {
    model_id: String,
    system_prompt: String,
    temperature: f32,
    top_p: f32,
    max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatSession {
    id: String,
    title: String,
    created_at: String,
    updated_at: String,
    messages: Vec<ChatMessage>,
    settings: ChatGenerationSettings,
    pinned: Option<bool>,
    tags: Option<Vec<String>>,
    title_locked: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatStore {
    chats: Vec<ChatSession>,
    active_chat_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum ModelDownloadState {
    Remote,
    Downloading,
    Downloaded,
    Loading,
    Ready,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelRuntimeState {
    status: ModelDownloadState,
    progress: f32,
    downloaded_bytes: u64,
    total_bytes: u64,
    error: Option<String>,
    local_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeStatus {
    selected_model_id: String,
    loaded_model_id: Option<String>,
    platform: String,
    inference_ready: bool,
    is_generating: bool,
    model_states: HashMap<String, ModelRuntimeState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DownloadProgressEvent {
    model_id: String,
    status: ModelDownloadState,
    progress: f32,
    downloaded_bytes: u64,
    total_bytes: u64,
    local_path: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatTokenEvent {
    request_id: String,
    message_id: String,
    token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatCompleteEvent {
    request_id: String,
    message_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatErrorEvent {
    request_id: String,
    message_id: String,
    error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerationStart {
    request_id: String,
    message_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ThemePreference {
    System,
    Light,
    Dark,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PersistedSettings {
    theme_preference: ThemePreference,
    selected_model_id: Option<String>,
    default_model_id: Option<String>,
    #[serde(default = "default_system_prompt")]
    system_prompt: String,
    auto_load_last_model: bool,
}

impl Default for PersistedSettings {
    fn default() -> Self {
        Self {
            theme_preference: ThemePreference::System,
            selected_model_id: None,
            default_model_id: Some(preferred_default_model_id().into()),
            system_prompt: default_system_prompt(),
            auto_load_last_model: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppSettingsPayload {
    theme_preference: ThemePreference,
    selected_model_id: Option<String>,
    default_model_id: Option<String>,
    system_prompt: String,
    auto_load_last_model: bool,
}

#[derive(Clone)]
struct LoadedModel {
    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    local_path: Option<PathBuf>,
    ollama_model_name: Option<String>,
}

struct AppStateInner {
    settings: PersistedSettings,
    selected_model_id: String,
    loaded_model_id: Option<String>,
    loaded_model: Option<LoadedModel>,
    is_generating: bool,
    model_states: HashMap<String, ModelRuntimeState>,
    download_cancels: HashMap<String, Arc<AtomicBool>>,
    generation_cancel: Option<Arc<AtomicBool>>,
}

struct AppState {
    inner: Arc<Mutex<AppStateInner>>,
}

#[derive(Debug, Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    stream: bool,
    messages: Vec<OllamaChatMessage<'a>>,
    options: OllamaOptions,
}

#[derive(Debug, Serialize)]
struct OllamaChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    temperature: f32,
    top_p: f32,
    num_predict: u32,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    message: Option<OllamaChunkMessage>,
    done: Option<bool>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaChunkMessage {
    content: String,
}

fn preferred_default_model_id() -> &'static str {
    if cfg!(target_os = "android") {
        "lfm-2.5-vl-1.6b"
    } else {
        "qwen-3.5-4b-q4km"
    }
}

fn default_system_prompt() -> String {
    "You are a helpful local assistant named Dobby made by Alessio Doria. Be concise, practical, and explicit when you are unsure."
        .to_string()
}

fn seed_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "lfm-2.5-vl-1.6b".into(),
            name: "LFM 2.5 VL 1.6B (Q4_0)".into(),
            url: "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/LFM2.5-VL-1.6B-Q4_0.gguf".into(),
            mmproj_url: Some("https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf".into()),
            description: "Liquid AI model imported into Ollama from GGUF".into(),
            size_bytes: 695_752_160,
            mmproj_size_bytes: Some(583_109_888),
            size_human: "~664 MB".into(),
        },
        ModelInfo {
            id: "qwen-3.5-2b-q80".into(),
            name: "Qwen 3.5 2B (Q8_0)".into(),
            url: "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q8_0.gguf?download=true".into(),
            mmproj_url: Some("https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-F16.gguf".into()),
            description: "Qwen model imported into Ollama from GGUF".into(),
            size_bytes: 2_012_012_800,
            mmproj_size_bytes: Some(668_227_264),
            size_human: "2.68 GB".into(),
        },
        ModelInfo {
            id: "qwen-3.5-4b-q4km".into(),
            name: "Qwen 3.5 4B (Q4_K_M)".into(),
            url: "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf?download=true".into(),
            mmproj_url: Some("https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf".into()),
            description: "Higher quality Qwen model imported into Ollama from GGUF".into(),
            size_bytes: 2_740_937_888,
            mmproj_size_bytes: Some(672_423_616),
            size_human: "3.63 GB".into(),
        },
    ]
}

fn model_index() -> HashMap<String, ModelInfo> {
    seed_models()
        .into_iter()
        .map(|model| (model.id.clone(), model))
        .collect()
}

fn default_model_runtime_state(model: &ModelInfo) -> ModelRuntimeState {
    ModelRuntimeState {
        status: ModelDownloadState::Remote,
        progress: 0.0,
        downloaded_bytes: 0,
        total_bytes: model.size_bytes,
        error: None,
        local_path: None,
    }
}

fn app_data_dir(app: &AppHandle) -> anyhow::Result<PathBuf> {
    app.path()
        .app_data_dir()
        .context("failed to resolve app data directory")
}

fn model_dir(app: &AppHandle) -> anyhow::Result<PathBuf> {
    Ok(app_data_dir(app)?.join("models"))
}

fn model_file_path(app: &AppHandle, model: &ModelInfo) -> anyhow::Result<PathBuf> {
    Ok(model_dir(app)?.join(format!("{}.gguf", model.id)))
}

fn settings_path(app: &AppHandle) -> anyhow::Result<PathBuf> {
    Ok(app_data_dir(app)?.join("settings.json"))
}

fn chats_path(app: &AppHandle) -> anyhow::Result<PathBuf> {
    Ok(app_data_dir(app)?.join("chats.json"))
}

fn ensure_directories(app: &AppHandle) -> anyhow::Result<()> {
    std::fs::create_dir_all(model_dir(app)?)?;
    Ok(())
}

fn load_chat_store_from_disk(app: &AppHandle) -> anyhow::Result<Option<ChatStore>> {
    let path = chats_path(app)?;
    match std::fs::read_to_string(path) {
        Ok(raw) => {
            let store = serde_json::from_str(&raw).context("failed to parse chats.json")?;
            Ok(Some(store))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err.into()),
    }
}

fn save_chat_store_to_disk(app: &AppHandle, store: &ChatStore) -> anyhow::Result<()> {
    ensure_directories(app)?;
    let path = chats_path(app)?;
    std::fs::write(path, serde_json::to_vec_pretty(store)?)?;
    Ok(())
}

fn load_settings(app: &AppHandle) -> PersistedSettings {
    let path = match settings_path(app) {
        Ok(path) => path,
        Err(_) => return PersistedSettings::default(),
    };

    match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
        Err(_) => PersistedSettings::default(),
    }
}

fn save_settings(app: &AppHandle, settings: &PersistedSettings) -> anyhow::Result<()> {
    ensure_directories(app)?;
    let path = settings_path(app)?;
    std::fs::write(path, serde_json::to_vec_pretty(settings)?)?;
    Ok(())
}

fn inspect_model_states(app: &AppHandle) -> HashMap<String, ModelRuntimeState> {
    let mut states = HashMap::new();
    for model in seed_models() {
        let path = match model_file_path(app, &model) {
            Ok(path) => path,
            Err(_) => {
                states.insert(model.id.clone(), default_model_runtime_state(&model));
                continue;
            }
        };

        let mut state = match std::fs::metadata(&path) {
            Ok(metadata) if metadata.len() > 0 => ModelRuntimeState {
                status: ModelDownloadState::Downloaded,
                progress: 100.0,
                downloaded_bytes: metadata.len(),
                total_bytes: metadata.len(),
                error: None,
                local_path: Some(path.to_string_lossy().to_string()),
            },
            Ok(metadata) => ModelRuntimeState {
                status: ModelDownloadState::Remote,
                progress: 0.0,
                downloaded_bytes: 0,
                total_bytes: metadata.len(),
                error: None,
                local_path: None,
            },
            Err(_) => default_model_runtime_state(&model),
        };

        if matches!(state.status, ModelDownloadState::Remote)
            && ollama_model_exists(&ollama_model_name(&model.id))
        {
            state = ModelRuntimeState {
                status: ModelDownloadState::Downloaded,
                progress: 100.0,
                downloaded_bytes: 0,
                total_bytes: 0,
                error: None,
                local_path: None,
            };
        }

        states.insert(model.id.clone(), state);
    }

    states
}

impl AppState {
    fn new(app: &AppHandle) -> Self {
        let settings = load_settings(app);
        let model_states = inspect_model_states(app);
        let selected_model_id = settings
            .selected_model_id
            .clone()
            .or_else(|| settings.default_model_id.clone())
            .unwrap_or_else(|| preferred_default_model_id().into());

        Self {
            inner: Arc::new(Mutex::new(AppStateInner {
                settings,
                selected_model_id,
                loaded_model_id: None,
                loaded_model: None,
                is_generating: false,
                model_states,
                download_cancels: HashMap::new(),
                generation_cancel: None,
            })),
        }
    }
}

fn runtime_status(inner: &AppStateInner) -> RuntimeStatus {
    RuntimeStatus {
        selected_model_id: inner.selected_model_id.clone(),
        loaded_model_id: inner.loaded_model_id.clone(),
        platform: std::env::consts::OS.into(),
        inference_ready: inner.loaded_model_id.is_some(),
        is_generating: inner.is_generating,
        model_states: inner.model_states.clone(),
    }
}

fn app_settings_payload(inner: &AppStateInner) -> AppSettingsPayload {
    AppSettingsPayload {
        theme_preference: inner.settings.theme_preference.clone(),
        selected_model_id: inner.settings.selected_model_id.clone(),
        default_model_id: inner.settings.default_model_id.clone(),
        system_prompt: inner.settings.system_prompt.clone(),
        auto_load_last_model: inner.settings.auto_load_last_model,
    }
}

fn emit_download_event(
    app: &AppHandle,
    event_name: &str,
    model_id: &str,
    state: &ModelRuntimeState,
) {
    let payload = DownloadProgressEvent {
        model_id: model_id.to_string(),
        status: state.status.clone(),
        progress: state.progress,
        downloaded_bytes: state.downloaded_bytes,
        total_bytes: state.total_bytes,
        local_path: state.local_path.clone(),
        error: state.error.clone(),
    };
    let _ = app.emit(event_name, payload);
}

fn set_model_state(
    inner: &mut AppStateInner,
    model_id: &str,
    state: ModelRuntimeState,
) -> ModelRuntimeState {
    inner.model_states.insert(model_id.to_string(), state.clone());
    state
}

fn apply_theme_to_window(app: &AppHandle, preference: &ThemePreference) {
    if let Some(window) = app.get_webview_window("main") {
        let theme = match preference {
            ThemePreference::System => None,
            ThemePreference::Light => Some(Theme::Light),
            ThemePreference::Dark => Some(Theme::Dark),
        };
        let _ = window.set_theme(theme);
    }
}

fn ollama_model_name(model_id: &str) -> String {
    format!("locallm-{}", model_id)
}

#[cfg(not(target_os = "android"))]
fn run_ollama_command(args: &[&str]) -> anyhow::Result<std::process::Output> {
    Command::new("ollama")
        .args(args)
        .env("NO_COLOR", "1")
        .env("TERM", "dumb")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("Failed to run 'ollama {}'", args.join(" ")))
}

#[cfg(target_os = "android")]
fn run_ollama_command(args: &[&str]) -> anyhow::Result<std::process::Output> {
    Err(anyhow!(format!(
        "Ollama is not available on Android for command '{}'",
        args.join(" ")
    )))
}

#[cfg(not(target_os = "android"))]
fn ollama_model_exists(model_name: &str) -> bool {
    run_ollama_command(&["show", model_name])
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "android")]
fn ollama_model_exists(_: &str) -> bool {
    false
}

fn import_model_into_ollama(model_id: &str, local_path: &Path, app: &AppHandle) -> anyhow::Result<String> {
    let model_name = ollama_model_name(model_id);
    let modelfile_path = app_data_dir(app)?.join(format!("Modelfile-{}.txt", model_id));
    let normalized_path = local_path.to_string_lossy().replace('\\', "/");
    let modelfile = format!(
        r#"FROM {normalized_path}

TEMPLATE """{{{{- if .System }}}}System: {{{{ .System }}}}

{{{{ end }}}}{{{{- range .Messages }}}}{{{{- if eq .Role "user" }}}}User: {{{{ .Content }}}}

{{{{ else if eq .Role "assistant" }}}}Assistant: {{{{ .Content }}}}

{{{{ end }}}}{{{{ end }}}}Assistant: """

PARAMETER stop "User:"
PARAMETER stop "System:"
"#,
    );
    let should_rebuild = std::fs::read_to_string(&modelfile_path)
        .map(|existing| existing != modelfile)
        .unwrap_or(true)
        || !ollama_model_exists(&model_name);
    std::fs::write(&modelfile_path, modelfile)?;

    if !should_rebuild {
        return Ok(model_name);
    }

    let modelfile_string = modelfile_path.to_string_lossy().to_string();
    let output = run_ollama_command(&["create", &model_name, "-f", &modelfile_string])?;
    if !output.status.success() || !ollama_model_exists(&model_name) {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let details = if !stderr.is_empty() {
            stderr
        } else if !stdout.is_empty() {
            stdout
        } else {
            "unknown Ollama error".to_string()
        };
        return Err(anyhow!(format!(
            "Ollama failed to import the GGUF model. {details}"
        )));
    }

    Ok(model_name)
}

async fn stream_ollama_response(
    app: AppHandle,
    ollama_model_name: String,
    messages: Vec<ChatMessage>,
    options: ChatGenerationSettings,
    request_id: String,
    message_id: String,
    cancel_flag: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let payload = OllamaChatRequest {
        model: &ollama_model_name,
        stream: true,
        messages: messages
            .iter()
            .filter(|message| {
                message.role == "user"
                    || message.role == "assistant"
                    || message.role == "system"
            })
            .map(|message| OllamaChatMessage {
                role: &message.role,
                content: &message.content,
            })
            .collect(),
        options: OllamaOptions {
            temperature: options.temperature,
            top_p: options.top_p,
            num_predict: options.max_tokens,
        },
    };

    let response = reqwest::Client::new()
        .post(OLLAMA_API_URL)
        .json(&payload)
        .send()
        .await
        .context("Failed to contact the local Ollama API")?
        .error_for_status()
        .context("Ollama rejected the chat request")?;

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        if cancel_flag.load(Ordering::SeqCst) {
            let _ = app.emit(
                CHAT_COMPLETE,
                ChatCompleteEvent {
                    request_id,
                    message_id,
                },
            );
            return Ok(());
        }

        let chunk = chunk.context("Failed while reading Ollama stream")?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(newline_index) = buffer.find('\n') {
            let line = buffer[..newline_index].trim().to_string();
            buffer = buffer[newline_index + 1..].to_string();
            if line.is_empty() {
                continue;
            }

            let event: OllamaStreamChunk =
                serde_json::from_str(&line).context("Failed to parse Ollama stream chunk")?;

            if let Some(error) = event.error {
                return Err(anyhow!(error));
            }

            if let Some(message) = event.message {
                if !message.content.is_empty() {
                    let _ = app.emit(
                        CHAT_TOKEN,
                        ChatTokenEvent {
                            request_id: request_id.clone(),
                            message_id: message_id.clone(),
                            token: message.content,
                        },
                    );
                }
            }

            if event.done.unwrap_or(false) {
                let _ = app.emit(
                    CHAT_COMPLETE,
                    ChatCompleteEvent {
                        request_id,
                        message_id,
                    },
                );
                return Ok(());
            }
        }
    }

    let _ = app.emit(
        CHAT_COMPLETE,
        ChatCompleteEvent {
            request_id,
            message_id,
        },
    );
    Ok(())
}

#[cfg(target_os = "android")]
fn build_embedded_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages.iter().filter(|message| {
        message.role == "system" || message.role == "user" || message.role == "assistant"
    }) {
        match message.role.as_str() {
            "system" => {
                prompt.push_str("System: ");
                prompt.push_str(&message.content);
                prompt.push_str("\n\n");
            }
            "user" => {
                prompt.push_str("User: ");
                prompt.push_str(&message.content);
                prompt.push_str("\n\n");
            }
            "assistant" => {
                prompt.push_str("Assistant: ");
                prompt.push_str(&message.content);
                prompt.push_str("\n\n");
            }
            _ => {}
        }
    }
    prompt.push_str("Assistant: ");
    prompt
}

#[cfg(target_os = "android")]
fn embedded_sampler(seed: u32, temperature: f32, top_p: f32) -> LlamaSampler {
    if temperature <= 0.0 {
        LlamaSampler::chain_simple([LlamaSampler::greedy()])
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(top_p.clamp(0.05, 1.0), 1),
            LlamaSampler::temp(temperature.max(0.1)),
            LlamaSampler::dist(seed),
        ])
    }
}

#[cfg(target_os = "android")]
fn stream_embedded_response(
    app: AppHandle,
    model_path: PathBuf,
    messages: Vec<ChatMessage>,
    options: ChatGenerationSettings,
    request_id: String,
    message_id: String,
    cancel_flag: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let backend = LlamaBackend::init().context("Failed to initialize llama.cpp backend")?;
    let model = LlamaModel::load_from_file(&backend, &model_path, &LlamaModelParams::default())
        .with_context(|| format!("Failed to load model from {}", model_path.display()))?;

    let prompt = build_embedded_prompt(&messages);
    let prompt_tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .context("Failed to tokenize prompt for embedded inference")?;

    let context_window = (prompt_tokens.len() + options.max_tokens as usize + 128).clamp(1024, 8192);
    let mut context_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(context_window as u32).unwrap()));

    let n_threads = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get() as u32)
        .unwrap_or(2);
    context_params = context_params
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads);

    let mut ctx = model
        .new_context(&backend, context_params)
        .context("Failed to create llama.cpp context")?;

    let mut batch = LlamaBatch::new(512, 1);
    let last_prompt_index = prompt_tokens.len().saturating_sub(1);
    for (index, token) in prompt_tokens.iter().enumerate() {
        let is_last = index == last_prompt_index;
        batch
            .add(*token, index as i32, &[0], is_last)
            .context("Failed to add prompt token to llama batch")?;
    }

    ctx.decode(&mut batch)
        .context("Failed to decode embedded prompt")?;

    let mut sampler = embedded_sampler(
        (Utc::now().timestamp_millis() & 0xffff_ffff) as u32,
        options.temperature,
        options.top_p,
    );
    let mut decoder = UTF_8.new_decoder();
    let mut current_token_index = prompt_tokens.len() as i32;
    let max_token_index = current_token_index + options.max_tokens as i32;

    while current_token_index < max_token_index {
        if cancel_flag.load(Ordering::SeqCst) {
            let _ = app.emit(
                CHAT_COMPLETE,
                ChatCompleteEvent {
                    request_id,
                    message_id,
                },
            );
            return Ok(());
        }

        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            let _ = app.emit(
                CHAT_COMPLETE,
                ChatCompleteEvent {
                    request_id,
                    message_id,
                },
            );
            return Ok(());
        }

        let piece = model
            .token_to_piece(token, false, &mut decoder)
            .context("Failed to decode embedded token piece")?;
        if !piece.is_empty() {
            let _ = app.emit(
                CHAT_TOKEN,
                ChatTokenEvent {
                    request_id: request_id.clone(),
                    message_id: message_id.clone(),
                    token: piece,
                },
            );
        }

        batch.clear();
        batch
            .add(token, current_token_index, &[0], true)
            .context("Failed to add generated token to llama batch")?;
        current_token_index += 1;
        ctx.decode(&mut batch)
            .context("Failed to decode embedded token batch")?;
    }

    let _ = app.emit(
        CHAT_COMPLETE,
        ChatCompleteEvent {
            request_id,
            message_id,
        },
    );
    Ok(())
}

#[tauri::command]
fn list_models() -> Vec<ModelInfo> {
    seed_models()
}

#[tauri::command]
fn get_runtime_status(state: State<'_, AppState>) -> RuntimeStatus {
    let inner = state.inner.lock().unwrap();
    runtime_status(&inner)
}

#[tauri::command]
async fn download_model(
    app: AppHandle,
    model_id: String,
    state: State<'_, AppState>,
) -> Result<RuntimeStatus, String> {
    let models = model_index();
    let model = models
        .get(&model_id)
        .cloned()
        .ok_or_else(|| "Unknown model".to_string())?;

    ensure_directories(&app).map_err(|err| err.to_string())?;
    let final_path = model_file_path(&app, &model).map_err(|err| err.to_string())?;

    if final_path.exists() {
        let actual_bytes = std::fs::metadata(&final_path)
            .map(|metadata| metadata.len())
            .unwrap_or(model.size_bytes);
        let mut inner = state.inner.lock().unwrap();
        let current = inner
            .model_states
            .get(&model_id)
            .cloned()
            .unwrap_or_else(|| default_model_runtime_state(&model));
        let status = if current.status == ModelDownloadState::Ready {
            ModelDownloadState::Ready
        } else {
            ModelDownloadState::Downloaded
        };
        set_model_state(
            &mut inner,
            &model_id,
            ModelRuntimeState {
                status,
                progress: 100.0,
                downloaded_bytes: actual_bytes,
                total_bytes: actual_bytes,
                error: None,
                local_path: Some(final_path.to_string_lossy().to_string()),
            },
        );
        return Ok(runtime_status(&inner));
    }

    let cancel_flag = Arc::new(AtomicBool::new(false));
    let runtime_snapshot = {
        let mut inner = state.inner.lock().unwrap();
        inner.download_cancels.insert(model_id.clone(), cancel_flag.clone());
        let state_value = set_model_state(
            &mut inner,
            &model_id,
            ModelRuntimeState {
                status: ModelDownloadState::Downloading,
                progress: 0.0,
                downloaded_bytes: 0,
                total_bytes: model.size_bytes,
                error: None,
                local_path: None,
            },
        );
        emit_download_event(&app, MODEL_DOWNLOAD_PROGRESS, &model_id, &state_value);
        runtime_status(&inner)
    };

    let app_handle = app.clone();
    let state_handle = state.inner.clone();
    tauri::async_runtime::spawn(async move {
        let temp_path = final_path.with_extension("gguf.part");
        let result: anyhow::Result<u64> = async {
            let client = reqwest::Client::new();
            let response = client
                .get(&model.url)
                .send()
                .await
                .context("Failed to start download")?
                .error_for_status()
                .context("Model download request failed")?;

            let total_bytes = response.content_length().unwrap_or(0);
            let mut downloaded_bytes = 0_u64;
            let mut file = tokio::fs::File::create(&temp_path)
                .await
                .context("Failed to create temporary model file")?;
            let mut stream = response.bytes_stream();

            while let Some(chunk) = stream.next().await {
                if cancel_flag.load(Ordering::SeqCst) {
                    let _ = tokio::fs::remove_file(&temp_path).await;
                    return Err(anyhow!("Download canceled"));
                }

                let chunk = chunk.context("Failed while downloading model data")?;
                tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
                    .await
                    .context("Failed to write model chunk")?;
                downloaded_bytes += chunk.len() as u64;

                let progress = if total_bytes == 0 {
                    0.0
                } else {
                    ((downloaded_bytes as f64 / total_bytes as f64) * 100.0) as f32
                };

                let progress_state = {
                    let mut inner = state_handle.lock().unwrap();
                    let next = set_model_state(
                        &mut inner,
                        &model.id,
                        ModelRuntimeState {
                            status: ModelDownloadState::Downloading,
                            progress,
                            downloaded_bytes,
                            total_bytes,
                            error: None,
                            local_path: None,
                        },
                    );
                    inner.download_cancels.insert(model.id.clone(), cancel_flag.clone());
                    next
                };
                emit_download_event(
                    &app_handle,
                    MODEL_DOWNLOAD_PROGRESS,
                    &model.id,
                    &progress_state,
                );
            }

            tokio::io::AsyncWriteExt::flush(&mut file)
                .await
                .context("Failed to flush downloaded model")?;
            drop(file);

            let metadata = tokio::fs::metadata(&temp_path)
                .await
                .context("Failed to inspect downloaded model")?;
            if metadata.len() == 0 {
                let _ = tokio::fs::remove_file(&temp_path).await;
                return Err(anyhow!("Downloaded model file is empty"));
            }

            tokio::fs::rename(&temp_path, &final_path)
                .await
                .context("Failed to finalize downloaded model")?;
            Ok(metadata.len())
        }
        .await;

        let mut inner = state_handle.lock().unwrap();
        inner.download_cancels.remove(&model.id);
        let next_state = match result {
            Ok(actual_bytes) => set_model_state(
                &mut inner,
                &model.id,
                ModelRuntimeState {
                    status: ModelDownloadState::Downloaded,
                    progress: 100.0,
                    downloaded_bytes: actual_bytes,
                    total_bytes: actual_bytes,
                    error: None,
                    local_path: Some(final_path.to_string_lossy().to_string()),
                },
            ),
            Err(err) => {
                let fallback_status = if cancel_flag.load(Ordering::SeqCst) {
                    ModelDownloadState::Remote
                } else {
                    ModelDownloadState::Error
                };
                set_model_state(
                    &mut inner,
                    &model.id,
                    ModelRuntimeState {
                        status: fallback_status,
                        progress: 0.0,
                        downloaded_bytes: 0,
                        total_bytes: 0,
                        error: Some(err.to_string()),
                        local_path: None,
                    },
                )
            }
        };

        let event_name = if matches!(next_state.status, ModelDownloadState::Downloaded) {
            MODEL_DOWNLOAD_COMPLETE
        } else {
            MODEL_DOWNLOAD_ERROR
        };
        emit_download_event(&app_handle, event_name, &model.id, &next_state);
    });

    Ok(runtime_snapshot)
}

#[tauri::command]
fn cancel_model_download(model_id: String, state: State<'_, AppState>) -> RuntimeStatus {
    let inner = state.inner.lock().unwrap();
    if let Some(cancel) = inner.download_cancels.get(&model_id) {
        cancel.store(true, Ordering::SeqCst);
    }
    runtime_status(&inner)
}

#[tauri::command]
fn delete_model(
    app: AppHandle,
    model_id: String,
    state: State<'_, AppState>,
) -> Result<RuntimeStatus, String> {
    let models = model_index();
    let model = models
        .get(&model_id)
        .cloned()
        .ok_or_else(|| "Unknown model".to_string())?;
    let local_path = model_file_path(&app, &model).map_err(|err| err.to_string())?;

    if local_path.exists() {
        std::fs::remove_file(&local_path).map_err(|err| err.to_string())?;
    }

    let _ = run_ollama_command(&["rm", &ollama_model_name(&model_id)]);

    let mut inner = state.inner.lock().unwrap();
    if inner.loaded_model_id.as_deref() == Some(&model_id) {
        inner.loaded_model_id = None;
        inner.loaded_model = None;
    }
    if inner.selected_model_id == model_id {
        inner.selected_model_id = inner
            .settings
            .default_model_id
            .clone()
            .unwrap_or_else(|| preferred_default_model_id().into());
        inner.settings.selected_model_id = Some(inner.selected_model_id.clone());
        let _ = save_settings(&app, &inner.settings);
    }
    set_model_state(&mut inner, &model_id, default_model_runtime_state(&model));
    Ok(runtime_status(&inner))
}

#[tauri::command]
fn select_model(
    app: AppHandle,
    model_id: String,
    state: State<'_, AppState>,
) -> Result<RuntimeStatus, String> {
    let models = model_index();
    let model = models
        .get(&model_id)
        .cloned()
        .ok_or_else(|| "Unknown model".to_string())?;
    let local_path = model_file_path(&app, &model).map_err(|err| err.to_string())?;
    let ollama_name = ollama_model_name(&model_id);
    let has_local_file = local_path.exists();
    let has_ollama_model = ollama_model_exists(&ollama_name);

    let mut inner = state.inner.lock().unwrap();
    inner.selected_model_id = model_id.clone();
    inner.settings.selected_model_id = Some(model_id.clone());
    let _ = save_settings(&app, &inner.settings);

    if !has_local_file && !has_ollama_model {
        inner.loaded_model_id = None;
        inner.loaded_model = None;
        let current = set_model_state(
            &mut inner,
            &model_id,
            ModelRuntimeState {
                status: ModelDownloadState::Remote,
                progress: 0.0,
                downloaded_bytes: 0,
                total_bytes: model.size_bytes,
                error: None,
                local_path: None,
            },
        );
        emit_download_event(&app, MODEL_DOWNLOAD_PROGRESS, &model_id, &current);
        return Ok(runtime_status(&inner));
    }

    if let Some(previous) = inner.loaded_model_id.clone() {
        if let Some(previous_state) = inner.model_states.get_mut(&previous) {
            if previous_state.status == ModelDownloadState::Ready {
                previous_state.status = ModelDownloadState::Downloaded;
            }
        }
    }

    let loading_state = set_model_state(
        &mut inner,
        &model_id,
        ModelRuntimeState {
            status: ModelDownloadState::Loading,
            progress: 100.0,
            downloaded_bytes: std::fs::metadata(&local_path)
                .map(|metadata| metadata.len())
                .unwrap_or(0),
            total_bytes: std::fs::metadata(&local_path)
                .map(|metadata| metadata.len())
                .unwrap_or(0),
            error: None,
            local_path: has_local_file.then(|| local_path.to_string_lossy().to_string()),
        },
    );
    emit_download_event(&app, MODEL_DOWNLOAD_PROGRESS, &model_id, &loading_state);
    drop(inner);

    #[cfg(target_os = "android")]
    let imported_name: Option<String> = None;
    #[cfg(target_os = "android")]
    let loaded_model_path = Some(local_path.clone());

    #[cfg(not(target_os = "android"))]
    let imported_name = if has_ollama_model {
        ollama_name
    } else {
        import_model_into_ollama(&model_id, &local_path, &app).map_err(|err| err.to_string())?
    };
    #[cfg(not(target_os = "android"))]
    let imported_name = Some(imported_name);
    #[cfg(not(target_os = "android"))]
    let loaded_model_path = has_local_file.then(|| local_path.clone());

    let mut inner = state.inner.lock().unwrap();
    inner.loaded_model_id = Some(model_id.clone());
    inner.loaded_model = Some(LoadedModel {
        local_path: loaded_model_path,
        ollama_model_name: imported_name,
    });
    let ready_state = set_model_state(
        &mut inner,
        &model_id,
        ModelRuntimeState {
            status: ModelDownloadState::Ready,
            progress: 100.0,
            downloaded_bytes: std::fs::metadata(&local_path)
                .map(|metadata| metadata.len())
                .unwrap_or(0),
            total_bytes: std::fs::metadata(&local_path)
                .map(|metadata| metadata.len())
                .unwrap_or(0),
            error: None,
            local_path: has_local_file.then(|| local_path.to_string_lossy().to_string()),
        },
    );
    emit_download_event(&app, MODEL_DOWNLOAD_COMPLETE, &model_id, &ready_state);
    Ok(runtime_status(&inner))
}

#[tauri::command]
fn send_chat(
    app: AppHandle,
    model_id: String,
    messages: Vec<ChatMessage>,
    options: ChatGenerationSettings,
    state: State<'_, AppState>,
) -> Result<GenerationStart, String> {
    let (request_id, message_id, cancel_flag, loaded_model, state_handle) = {
        let mut inner = state.inner.lock().unwrap();
        if inner.is_generating {
            return Err("A generation is already in progress".into());
        }
        if inner.loaded_model_id.as_deref() != Some(&model_id) {
            return Err("The selected model is not loaded".into());
        }
        let loaded = inner
            .loaded_model
            .as_ref()
            .ok_or_else(|| "The selected model is not loaded".to_string())?;
        let loaded_model = loaded.clone();
        let request_id = format!("req-{}", Utc::now().timestamp_millis());
        let message_id = format!("assistant-{}", Utc::now().timestamp_nanos_opt().unwrap_or(0));
        let cancel_flag = Arc::new(AtomicBool::new(false));
        inner.is_generating = true;
        inner.generation_cancel = Some(cancel_flag.clone());
        (
            request_id,
            message_id,
            cancel_flag,
            loaded_model,
            state.inner.clone(),
        )
    };

    let app_handle = app.clone();
    let request_id_for_task = request_id.clone();
    let message_id_for_task = message_id.clone();

    #[cfg(target_os = "android")]
    let local_path = loaded_model
        .local_path
        .clone()
        .ok_or_else(|| "The selected model has no local file available".to_string())?;
    #[cfg(not(target_os = "android"))]
    let ollama_name = loaded_model
        .ollama_model_name
        .clone()
        .ok_or_else(|| "The selected model is not imported into Ollama".to_string())?;

    #[cfg(target_os = "android")]
    tokio::task::spawn_blocking(move || {
        let result = stream_embedded_response(
            app_handle.clone(),
            local_path,
            messages,
            options,
            request_id_for_task.clone(),
            message_id_for_task.clone(),
            cancel_flag,
        );

        let mut inner = state_handle.lock().unwrap();
        inner.is_generating = false;
        inner.generation_cancel = None;
        if let Err(err) = result {
            let _ = app_handle.emit(
                CHAT_ERROR,
                ChatErrorEvent {
                    request_id: request_id_for_task,
                    message_id: message_id_for_task,
                    error: err.to_string(),
                },
            );
        }
    });

    #[cfg(not(target_os = "android"))]
    tauri::async_runtime::spawn(async move {
        let result = stream_ollama_response(
            app_handle.clone(),
            ollama_name,
            messages,
            options,
            request_id_for_task.clone(),
            message_id_for_task.clone(),
            cancel_flag,
        )
        .await;

        let mut inner = state_handle.lock().unwrap();
        inner.is_generating = false;
        inner.generation_cancel = None;
        if let Err(err) = result {
            let _ = app_handle.emit(
                CHAT_ERROR,
                ChatErrorEvent {
                    request_id: request_id_for_task,
                    message_id: message_id_for_task,
                    error: err.to_string(),
                },
            );
        }
    });

    Ok(GenerationStart {
        request_id,
        message_id,
    })
}

#[tauri::command]
fn cancel_generation(state: State<'_, AppState>) -> RuntimeStatus {
    let inner = state.inner.lock().unwrap();
    if let Some(cancel) = &inner.generation_cancel {
        cancel.store(true, Ordering::SeqCst);
    }
    runtime_status(&inner)
}

#[tauri::command]
fn load_chat_store(app: AppHandle) -> Result<Option<ChatStore>, String> {
    load_chat_store_from_disk(&app).map_err(|err| err.to_string())
}

#[tauri::command]
fn save_chat_store(app: AppHandle, store: ChatStore) -> Result<(), String> {
    save_chat_store_to_disk(&app, &store).map_err(|err| err.to_string())
}

#[tauri::command]
fn get_theme_preference(state: State<'_, AppState>) -> ThemePreference {
    let inner = state.inner.lock().unwrap();
    inner.settings.theme_preference.clone()
}

#[tauri::command]
fn set_theme_preference(
    app: AppHandle,
    theme: ThemePreference,
    state: State<'_, AppState>,
) -> Result<ThemePreference, String> {
    let preference = {
        let mut inner = state.inner.lock().unwrap();
        inner.settings.theme_preference = theme.clone();
        save_settings(&app, &inner.settings).map_err(|err| err.to_string())?;
        inner.settings.theme_preference.clone()
    };
    apply_theme_to_window(&app, &preference);
    Ok(preference)
}

#[tauri::command]
fn get_app_settings(state: State<'_, AppState>) -> AppSettingsPayload {
    let inner = state.inner.lock().unwrap();
    app_settings_payload(&inner)
}

#[tauri::command]
fn update_app_settings(
    app: AppHandle,
    settings: AppSettingsPayload,
    state: State<'_, AppState>,
) -> Result<AppSettingsPayload, String> {
    let payload = {
        let mut inner = state.inner.lock().unwrap();
        inner.settings.theme_preference = settings.theme_preference;
        inner.settings.selected_model_id = settings.selected_model_id;
        inner.settings.default_model_id = settings.default_model_id;
        inner.settings.system_prompt = settings.system_prompt;
        inner.settings.auto_load_last_model = settings.auto_load_last_model;

        if let Some(selected) = inner.settings.selected_model_id.clone() {
            inner.selected_model_id = selected;
        } else if let Some(default_model_id) = inner.settings.default_model_id.clone() {
            inner.selected_model_id = default_model_id;
        }

        save_settings(&app, &inner.settings).map_err(|err| err.to_string())?;
        app_settings_payload(&inner)
    };
    apply_theme_to_window(&app, &payload.theme_preference);
    Ok(payload)
}

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            ensure_directories(app.handle())
                .map_err(|err| std::io::Error::other(err.to_string()))?;
            let state = AppState::new(app.handle());
            if state.inner.lock().unwrap().settings.auto_load_last_model {
                let selected_model_id = state.inner.lock().unwrap().selected_model_id.clone();
                let models = model_index();
                if let Some(model) = models.get(&selected_model_id) {
                    if let Ok(local_path) = model_file_path(app.handle(), model) {
                        let has_local_file = local_path.exists();
                        #[cfg(target_os = "android")]
                        let loaded_model = has_local_file.then(|| LoadedModel {
                            local_path: Some(local_path.clone()),
                            ollama_model_name: None,
                        });

                        #[cfg(not(target_os = "android"))]
                        let loaded_model = {
                            let imported_name = if has_local_file {
                                import_model_into_ollama(
                                    &selected_model_id,
                                    &local_path,
                                    app.handle(),
                                )
                                .ok()
                            } else {
                                let ollama_name = ollama_model_name(&selected_model_id);
                                ollama_model_exists(&ollama_name).then_some(ollama_name)
                            };

                            imported_name.map(|imported_name| LoadedModel {
                                local_path: has_local_file.then(|| local_path.clone()),
                                ollama_model_name: Some(imported_name),
                            })
                        };

                        if let Some(loaded_model) = loaded_model {
                            let mut inner = state.inner.lock().unwrap();
                            inner.loaded_model_id = Some(selected_model_id.clone());
                            inner.loaded_model = Some(loaded_model);
                            set_model_state(
                                &mut inner,
                                &selected_model_id,
                                ModelRuntimeState {
                                    status: ModelDownloadState::Ready,
                                    progress: 100.0,
                                    downloaded_bytes: std::fs::metadata(&local_path)
                                        .map(|metadata| metadata.len())
                                        .unwrap_or(0),
                                    total_bytes: std::fs::metadata(&local_path)
                                        .map(|metadata| metadata.len())
                                        .unwrap_or(0),
                                    error: None,
                                    local_path: has_local_file
                                        .then(|| local_path.to_string_lossy().to_string()),
                                },
                            );
                        }
                    }
                }
            }
            apply_theme_to_window(
                app.handle(),
                &state.inner.lock().unwrap().settings.theme_preference,
            );
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_models,
            get_runtime_status,
            download_model,
            cancel_model_download,
            delete_model,
            select_model,
            send_chat,
            cancel_generation,
            load_chat_store,
            save_chat_store,
            get_theme_preference,
            set_theme_preference,
            get_app_settings,
            update_app_settings
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
