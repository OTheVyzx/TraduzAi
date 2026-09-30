use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, Notify};
use tokio::time::{sleep, timeout, Duration};

use super::protocol::{ProtocolRequest, PROTOCOL_VERSION};
use super::sandbox::{ProcessSandbox, SandboxLimits};

const BROKER_MAIN_CLASS: &str = "com.traduzai.source.runtime.broker.BrokerMainKt";
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
// Real sources often need a cold worker start plus one or more redirected HTTP
// requests. Keep the UI asynchronous, but do not kill a healthy worker at the
// old 15-20 second threshold (the official repository smoke allows 120 s).
const SOURCE_QUERY_DEADLINE_MS: u64 = 60_000;
const MAX_DOWNLOAD_STORAGE_BYTES: u64 = 50 * 1024 * 1024 * 1024;
const MAX_CHAPTER_DOWNLOAD_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuntimeLaunchSpec {
    pub(crate) program: PathBuf,
    pub(crate) classpath: PathBuf,
    pub(crate) data_dir: Option<PathBuf>,
}

impl RuntimeLaunchSpec {
    pub(crate) fn from_resource_root(root: &Path) -> Option<Self> {
        let runtime = root.join("source-runtime");
        let program =
            windows_java_compatible_path(runtime.join("jre").join("bin").join("java.exe"));
        let libs = windows_java_compatible_path(runtime.join("broker").join("lib"));
        if !program.is_file() || !libs.is_dir() {
            return None;
        }
        let classpath = packaged_runtime_classpath(&libs)?;
        Some(Self {
            program,
            classpath,
            data_dir: None,
        })
    }

    fn from_environment() -> Option<Self> {
        let program = windows_java_compatible_path(PathBuf::from(std::env::var_os(
            "TRADUZAI_SOURCE_RUNTIME_JAVA",
        )?));
        let classpath = windows_java_compatible_path(PathBuf::from(std::env::var_os(
            "TRADUZAI_SOURCE_RUNTIME_CLASSPATH",
        )?));
        (program.is_file() && classpath.parent().is_some_and(Path::is_dir)).then_some(Self {
            program,
            classpath,
            data_dir: std::env::var_os("TRADUZAI_SOURCE_RUNTIME_DATA").map(PathBuf::from),
        })
    }

    fn discover(app: &AppHandle) -> Option<Self> {
        let mut spec = Self::from_environment().or_else(|| {
            app.path()
                .resource_dir()
                .ok()
                .as_deref()
                .and_then(Self::from_resource_root)
        })?;
        if spec.data_dir.is_none() {
            spec.data_dir = app
                .path()
                .app_data_dir()
                .ok()
                .map(|path| path.join("source-runtime"));
        }
        Some(spec)
    }
}

const VERSIONED_RUNTIME_FAMILIES: &[&str] = &[
    "kotlin-stdlib-jdk7-",
    "kotlin-stdlib-jdk8-",
    "kotlin-stdlib-",
    "kotlin-reflect-",
    "kotlin-script-runtime-",
    "kotlinx-coroutines-core-jvm-",
    "kotlinx-coroutines-jdk8-",
    "kotlinx-coroutines-reactive-",
    "kotlinx-serialization-core-jvm-",
    "kotlinx-serialization-json-jvm-",
    "kotlinx-serialization-json-okio-jvm-",
    "kotlinx-serialization-protobuf-jvm-",
];

fn artifact_version(file_name: &str, family: &str) -> Vec<u64> {
    file_name
        .strip_prefix(family)
        .and_then(|value| value.strip_suffix(".jar"))
        .unwrap_or_default()
        .split(|character: char| !character.is_ascii_digit())
        .filter_map(|part| (!part.is_empty()).then(|| part.parse().ok()).flatten())
        .collect()
}

fn packaged_runtime_classpath(libs: &Path) -> Option<PathBuf> {
    let mut unversioned = Vec::new();
    let mut selected: BTreeMap<&'static str, (Vec<u64>, PathBuf)> = BTreeMap::new();
    let mut has_broker = false;

    for entry in std::fs::read_dir(libs).ok()?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("jar") {
            continue;
        }
        let file_name = entry.file_name().to_string_lossy().into_owned();
        has_broker |= file_name.starts_with("runtime-broker");
        if let Some(&family) = VERSIONED_RUNTIME_FAMILIES
            .iter()
            .find(|family| file_name.starts_with(**family))
        {
            let version = artifact_version(&file_name, family);
            let replace = selected
                .get(family)
                .is_none_or(|(current, _)| version > *current);
            if replace {
                selected.insert(family, (version, path));
            }
        } else {
            unversioned.push(path);
        }
    }
    if !has_broker {
        return None;
    }

    unversioned.extend(selected.into_values().map(|(_, path)| path));
    unversioned.sort_by_key(|path| {
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_ascii_lowercase();
        let priority = if name.starts_with("runtime-broker") {
            0
        } else if name.starts_with("kotlin") {
            1
        } else {
            2
        };
        (priority, name)
    });
    let paths = unversioned.into_iter().map(windows_java_compatible_path);
    std::env::join_paths(paths).ok().map(PathBuf::from)
}

pub(crate) fn windows_java_compatible_path(path: PathBuf) -> PathBuf {
    let Some(value) = path.to_str() else {
        return path;
    };
    if let Some(unc) = value.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{unc}"));
    }
    if let Some(drive) = value.strip_prefix(r"\\?\") {
        return PathBuf::from(drive);
    }
    path
}

struct BrokerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr: BufReader<ChildStderr>,
    _sandbox: ProcessSandbox,
}

#[derive(Default)]
pub(crate) struct SourceRuntimeManager {
    // A broker speaks a single ordered JSONL stream. Serializing the complete
    // start/write/read transaction prevents a queued request from observing a
    // process that another request has just invalidated after a transport error.
    request_gate: Mutex<()>,
    process: Mutex<Option<BrokerProcess>>,
    request_sequence: AtomicU64,
    download_controls: StdMutex<HashMap<String, Arc<DownloadJobControl>>>,
}

#[derive(Default)]
struct DownloadJobControl {
    pause_requested: AtomicBool,
    running: AtomicBool,
    changed: Notify,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DownloadAction {
    Pause,
    Resume,
}

fn download_action_allowed(status: &str, action: DownloadAction) -> bool {
    match action {
        DownloadAction::Pause => matches!(status, "queued" | "running" | "retrying"),
        DownloadAction::Resume => matches!(status, "paused" | "failed"),
    }
}

fn download_retry_delay(attempt: u32, retry_after_seconds: Option<u64>) -> Option<Duration> {
    if attempt >= 3 {
        return None;
    }
    let seconds = retry_after_seconds
        .unwrap_or_else(|| 1u64 << attempt.min(8))
        .clamp(1, 300);
    Some(Duration::from_secs(seconds))
}

impl SourceRuntimeManager {
    fn download_control(&self, job_id: &str) -> Arc<DownloadJobControl> {
        self.download_controls
            .lock()
            .expect("download controls poisoned")
            .entry(job_id.to_string())
            .or_default()
            .clone()
    }

    fn forget_download_control(&self, job_id: &str) {
        self.download_controls
            .lock()
            .expect("download controls poisoned")
            .remove(job_id);
    }

    async fn ensure_started(&self, spec: &RuntimeLaunchSpec) -> Result<(), String> {
        let mut guard = self.process.lock().await;
        if let Some(process) = guard.as_mut() {
            if process
                .child
                .try_wait()
                .map_err(|error| error.to_string())?
                .is_none()
            {
                return Ok(());
            }
            *guard = None;
        }

        let mut command = Command::new(&spec.program);
        command
            .args([
                "-Xms64m",
                "-Xmx512m",
                "-Dfile.encoding=UTF-8",
                "-Dkotlin-logging.logStartupMessage=false",
                "-cp",
            ])
            .arg(&spec.classpath)
            .arg(BROKER_MAIN_CLASS)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(data_dir) = &spec.data_dir {
            std::fs::create_dir_all(data_dir).map_err(|error| {
                format!("Não foi possível preparar os dados do runtime: {error}")
            })?;
            command.env("TRADUZAI_SOURCE_RUNTIME_DATA", data_dir);
        }
        #[cfg(windows)]
        command.creation_flags(0x08000000);

        let mut child = command
            .spawn()
            .map_err(|error| format!("Não foi possível iniciar o runtime de fontes: {error}"))?;
        let sandbox = match ProcessSandbox::attach(&child, SandboxLimits::default()) {
            Ok(sandbox) => sandbox,
            Err(error) => {
                let _ = child.kill().await;
                return Err(format!("Sandbox obrigatória indisponível: {error}"));
            }
        };
        let stdin = child.stdin.take().ok_or("Runtime sem stdin")?;
        let stdout = child.stdout.take().ok_or("Runtime sem stdout")?;
        let stderr = child.stderr.take().ok_or("Runtime sem stderr")?;
        *guard = Some(BrokerProcess {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            stderr: BufReader::new(stderr),
            _sandbox: sandbox,
        });
        Ok(())
    }

    pub(crate) async fn request(
        &self,
        spec: &RuntimeLaunchSpec,
        method: &str,
        params: Value,
        deadline_ms: u64,
    ) -> Result<Value, String> {
        let _request_guard = self.request_gate.lock().await;
        self.ensure_started(spec).await?;
        let id = format!(
            "studio-{}",
            self.request_sequence.fetch_add(1, Ordering::Relaxed) + 1
        );
        let request = ProtocolRequest {
            version: PROTOCOL_VERSION,
            id: id.clone(),
            method: method.to_string(),
            deadline_ms,
            params,
        };
        let mut encoded = serde_json::to_vec(&request).map_err(|error| error.to_string())?;
        encoded.push(b'\n');

        let mut guard = self.process.lock().await;
        let response =
            async {
                let process = guard.as_mut().ok_or("Runtime de fontes não está ativo")?;
                process
                    .stdin
                    .write_all(&encoded)
                    .await
                    .map_err(|error| format!("Falha ao enviar requisição ao runtime: {error}"))?;
                process.stdin.flush().await.map_err(|error| {
                    format!("Falha ao confirmar requisição ao runtime: {error}")
                })?;

                let line = timeout(
                    Duration::from_millis(deadline_ms.saturating_add(1_000)),
                    read_bounded_runtime_line(&mut process.stdout, MAX_RESPONSE_BYTES),
                )
                .await
                .map_err(|_| "SOURCE_TIMEOUT: o runtime excedeu o prazo".to_string())??;
                if line.is_empty() {
                    let exit_code = process
                        .child
                        .try_wait()
                        .ok()
                        .flatten()
                        .and_then(|status| status.code());
                    let mut stderr = Vec::new();
                    let _ = timeout(
                        Duration::from_millis(250),
                        (&mut process.stderr).take(4_096).read_to_end(&mut stderr),
                    )
                    .await;
                    return Err(runtime_exit_message(
                        exit_code,
                        &String::from_utf8_lossy(&stderr),
                    ));
                }
                parse_runtime_response(&id, &line)
            }
            .await;

        // A resposta `ok:false` é uma falha do domínio (índice inválido, fonte
        // indisponível etc.), não uma quebra do transporte. O broker continua
        // saudável e deve ser reutilizado pela próxima chamada.
        if response.is_err() {
            if let Some(mut process) = guard.take() {
                let _ = process.child.kill().await;
                let _ = process.child.wait().await;
            }
        }
        match response? {
            RuntimeReply::Success(value) => Ok(value),
            RuntimeReply::DomainError(error) => Err(error),
        }
    }

    pub(crate) async fn shutdown(&self) -> Result<bool, String> {
        let _request_guard = self.request_gate.lock().await;
        let mut process = self.process.lock().await.take();
        let Some(mut process) = process.take() else {
            return Ok(false);
        };
        process
            .child
            .kill()
            .await
            .map_err(|error| format!("Falha ao encerrar o runtime: {error}"))?;
        let _ = process.child.wait().await;
        Ok(true)
    }

    async fn is_running(&self) -> bool {
        let mut guard = self.process.lock().await;
        match guard.as_mut() {
            Some(process) => process.child.try_wait().ok().flatten().is_none(),
            None => false,
        }
    }
}

pub(crate) async fn read_bounded_runtime_line<R>(
    reader: &mut R,
    max_bytes: usize,
) -> Result<String, String>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    let mut output = Vec::with_capacity(max_bytes.min(64 * 1024));
    loop {
        let available = reader
            .fill_buf()
            .await
            .map_err(|error| format!("Falha ao ler resposta do runtime: {error}"))?;
        if available.is_empty() {
            break;
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let take = newline.map_or(available.len(), |position| position + 1);
        if output.len().saturating_add(take) > max_bytes {
            return Err("PAYLOAD_TOO_LARGE: resposta do runtime excedeu o limite".to_string());
        }
        output.extend_from_slice(&available[..take]);
        reader.consume(take);
        if newline.is_some() {
            break;
        }
    }
    if output.last() == Some(&b'\n') {
        output.pop();
    }
    if output.last() == Some(&b'\r') {
        output.pop();
    }
    String::from_utf8(output)
        .map_err(|_| "INVALID_RESPONSE: resposta do runtime não é UTF-8".to_string())
}

pub(crate) fn runtime_exit_message(exit_code: Option<i32>, stderr: &str) -> String {
    let detail: String = stderr.trim().chars().take(2_048).collect();
    let status = exit_code
        .map(|code| format!("exit={code}"))
        .unwrap_or_else(|| "exit=desconhecido".to_string());
    if detail.is_empty() {
        format!("SOURCE_RUNTIME_CRASHED: o processo encerrou sem resposta ({status})")
    } else {
        format!("SOURCE_RUNTIME_CRASHED: o processo encerrou sem resposta ({status}): {detail}")
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum RuntimeReply {
    Success(Value),
    DomainError(String),
}

pub(crate) fn parse_runtime_response(
    expected_id: &str,
    line: &str,
) -> Result<RuntimeReply, String> {
    let response: Value = serde_json::from_str(line)
        .map_err(|error| format!("Resposta JSON inválida do runtime: {error}"))?;
    let version = response.get("v").and_then(Value::as_u64);
    if version != Some(PROTOCOL_VERSION.into()) {
        return Err("PROTOCOL_INCOMPATIBLE: resposta sem versão compatível".to_string());
    }
    let actual_id = response
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if actual_id != expected_id {
        return Err(format!(
            "CORRELATION_MISMATCH: esperado {expected_id}, recebido {actual_id}"
        ));
    }
    if response.get("ok").and_then(Value::as_bool) == Some(false) {
        let code = response
            .pointer("/error/code")
            .and_then(Value::as_str)
            .unwrap_or("SOURCE_FAILURE");
        let message = response
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or("Falha no runtime de fontes");
        let retry_after = response
            .pointer("/error/retryAfterSeconds")
            .and_then(Value::as_u64)
            .map(|seconds| format!(" [retry-after={}s]", seconds.min(300)))
            .unwrap_or_default();
        return Ok(RuntimeReply::DomainError(format!(
            "{code}: {message}{retry_after}"
        )));
    }
    Ok(RuntimeReply::Success(response))
}

pub(crate) fn retry_after_seconds_from_error(error: &str) -> Option<u64> {
    let marker = " [retry-after=";
    let start = error.rfind(marker)? + marker.len();
    let value = error.get(start..)?.strip_suffix("s]")?;
    value.parse::<u64>().ok().map(|seconds| seconds.min(300))
}

#[cfg(test)]
pub(crate) fn match_response_id(expected_id: &str, line: &str) -> Result<Value, String> {
    match parse_runtime_response(expected_id, line)? {
        RuntimeReply::Success(value) => Ok(value),
        RuntimeReply::DomainError(error) => Err(error),
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RuntimeStatus {
    configured: bool,
    running: bool,
    protocol: u16,
    sandbox: &'static str,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SourceSearchConfig {
    extension_package: String,
    query: String,
    source_id: Option<String>,
    filters: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SourceFiltersConfig {
    extension_package: String,
    source_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SourceCatalogConfig {
    extension_package: String,
    source_id: String,
    page: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SourceMangaConfig {
    extension_package: String,
    source_id: String,
    manga: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SourceChapterConfig {
    extension_package: String,
    source_id: String,
    chapter: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ReaderDownloadConfig {
    manga_id: String,
    chapter_id: String,
    extension_package: String,
    source_id: String,
    chapter: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ReaderTranslateConfig {
    job_id: String,
    project_json_path: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RepositoryPreviewConfig {
    url: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RepositoryAddConfig {
    url: String,
    certificate_fingerprint: String,
    index_sha256: String,
    signing_key: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RepositoryRemoveConfig {
    id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExtensionInstallConfig {
    repository_id: String,
    package_name: String,
    version_code: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExtensionEnabledConfig {
    package_name: String,
    enabled: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExtensionPackageConfig {
    package_name: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ReaderMangaConfig {
    manga: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ReaderRecordConfig {
    record_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ReaderProgressConfig {
    manga_id: String,
    chapter_id: String,
    last_page: u32,
    page_count: Option<u32>,
    read: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ReaderCategoriesConfig {
    categories: Option<Vec<Value>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ReaderAutomationConfig {
    enabled: bool,
    interval_hours: u16,
}

fn announce_status(app: &AppHandle, status: &RuntimeStatus) {
    let _ = app.emit("studio://source-runtime/status", status);
}

#[tauri::command]
pub(crate) async fn studio_source_runtime_status(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
) -> Result<RuntimeStatus, String> {
    let status = RuntimeStatus {
        configured: RuntimeLaunchSpec::discover(&app).is_some(),
        running: state.is_running().await,
        protocol: PROTOCOL_VERSION,
        sandbox: "required",
    };
    announce_status(&app, &status);
    Ok(status)
}

#[tauri::command]
pub(crate) async fn studio_source_runtime_start(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    let response = state
        .request(&spec, "runtime.hello", json!({}), 5_000)
        .await?;
    let status = RuntimeStatus {
        configured: true,
        running: true,
        protocol: PROTOCOL_VERSION,
        sandbox: "required",
    };
    announce_status(&app, &status);
    Ok(response)
}

#[tauri::command]
pub(crate) async fn studio_source_runtime_shutdown(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
) -> Result<bool, String> {
    let stopped = state.shutdown().await?;
    let status = RuntimeStatus {
        configured: RuntimeLaunchSpec::discover(&app).is_some(),
        running: false,
        protocol: PROTOCOL_VERSION,
        sandbox: "required",
    };
    announce_status(&app, &status);
    Ok(stopped)
}

#[tauri::command]
pub(crate) async fn studio_source_search(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: SourceSearchConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            "source.search",
            json!({
                "packageName": config.extension_package,
                "query": config.query,
                "sourceId": config.source_id,
                "filters": config.filters,
            }),
            SOURCE_QUERY_DEADLINE_MS,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_source_filters(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: SourceFiltersConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            "source.filters",
            json!({
                "packageName": config.extension_package,
                "sourceId": config.source_id,
            }),
            SOURCE_QUERY_DEADLINE_MS,
        )
        .await
}

async fn source_catalog_request(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    method: &str,
    config: SourceCatalogConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            method,
            json!({
                "packageName": config.extension_package,
                "sourceId": config.source_id,
                "page": config.page.unwrap_or(1),
            }),
            SOURCE_QUERY_DEADLINE_MS,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_source_popular(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: SourceCatalogConfig,
) -> Result<Value, String> {
    source_catalog_request(app, state, "source.popular", config).await
}

#[tauri::command]
pub(crate) async fn studio_source_latest(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: SourceCatalogConfig,
) -> Result<Value, String> {
    source_catalog_request(app, state, "source.latest", config).await
}

async fn source_manga_request(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    method: &str,
    config: SourceMangaConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            method,
            json!({
                "packageName": config.extension_package,
                "sourceId": config.source_id,
                "manga": config.manga,
            }),
            SOURCE_QUERY_DEADLINE_MS,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_source_manga_details(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: SourceMangaConfig,
) -> Result<Value, String> {
    source_manga_request(app, state, "source.manga-details", config).await
}

#[tauri::command]
pub(crate) async fn studio_source_chapters(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: SourceMangaConfig,
) -> Result<Value, String> {
    source_manga_request(app, state, "source.chapters", config).await
}

#[tauri::command]
pub(crate) async fn studio_source_pages(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: SourceChapterConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            "source.pages",
            json!({
                "packageName": config.extension_package,
                "sourceId": config.source_id,
                "chapter": config.chapter,
            }),
            SOURCE_QUERY_DEADLINE_MS,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_source_repository_preview(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: RepositoryPreviewConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            "repository.preview",
            json!({ "url": config.url }),
            20_000,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_source_repository_add(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: RepositoryAddConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            "repository.add",
            json!({
                "url": config.url,
                "certificateFingerprint": config.certificate_fingerprint,
                "indexSha256": config.index_sha256,
                "signingKey": config.signing_key,
            }),
            20_000,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_source_repository_list(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(&spec, "repository.list", json!({}), 5_000)
        .await
}

#[tauri::command]
pub(crate) async fn studio_source_repository_refresh(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(&spec, "repository.refresh", json!({}), 60_000)
        .await
}

#[tauri::command]
pub(crate) async fn studio_source_repository_remove(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: RepositoryRemoveConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            "repository.remove",
            json!({ "id": config.id }),
            5_000,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_source_extension_list(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(&spec, "extension.list", json!({}), 5_000)
        .await
}

#[tauri::command]
pub(crate) async fn studio_source_extension_install(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ExtensionInstallConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            "extension.install",
            json!({
                "repositoryId": config.repository_id,
                "packageName": config.package_name,
                "versionCode": config.version_code,
            }),
            120_000,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_source_extension_set_enabled(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ExtensionEnabledConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            "extension.set-enabled",
            json!({
                "packageName": config.package_name,
                "enabled": config.enabled,
            }),
            5_000,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_source_extension_uninstall(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ExtensionPackageConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            "extension.uninstall",
            json!({
                "packageName": config.package_name,
            }),
            10_000,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_source_extension_rollback(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ExtensionPackageConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            "extension.rollback",
            json!({ "packageName": config.package_name }),
            30_000,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_reader_library_list(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(&spec, "reader.library.list", json!({}), 5_000)
        .await
}

async fn reader_library_save(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    method: &str,
    config: ReaderMangaConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(&spec, method, json!({ "manga": config.manga }), 10_000)
        .await
}

#[tauri::command]
pub(crate) async fn studio_reader_library_add(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ReaderMangaConfig,
) -> Result<Value, String> {
    reader_library_save(app, state, "reader.library.add", config).await
}

#[tauri::command]
pub(crate) async fn studio_reader_library_update(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ReaderMangaConfig,
) -> Result<Value, String> {
    reader_library_save(app, state, "reader.library.update", config).await
}

#[tauri::command]
pub(crate) async fn studio_reader_library_remove(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ReaderRecordConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            "reader.library.remove",
            json!({ "recordId": config.record_id }),
            5_000,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_reader_history(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(&spec, "reader.history", json!({}), 5_000)
        .await
}

#[tauri::command]
pub(crate) async fn studio_reader_progress(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ReaderProgressConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            "reader.progress",
            json!({
                "mangaId": config.manga_id,
                "chapterId": config.chapter_id,
                "lastPage": config.last_page,
                "pageCount": config.page_count,
                "read": config.read.unwrap_or(false),
            }),
            5_000,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_reader_categories(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: Option<ReaderCategoriesConfig>,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    let categories = config.and_then(|value| value.categories);
    state
        .request(
            &spec,
            "reader.categories",
            json!({ "categories": categories }),
            5_000,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_reader_automation_get(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(&spec, "reader.automation.get", json!({}), 5_000)
        .await
}

#[tauri::command]
pub(crate) async fn studio_reader_automation_set(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ReaderAutomationConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(
            &spec,
            "reader.automation.set",
            json!({
                "enabled": config.enabled,
                "intervalHours": config.interval_hours,
            }),
            5_000,
        )
        .await
}

#[tauri::command]
pub(crate) async fn studio_reader_automation_run_now(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    state
        .request(&spec, "reader.automation.run-now", json!({}), 120_000)
        .await
}

#[tauri::command]
pub(crate) async fn studio_reader_download_enqueue(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ReaderDownloadConfig,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    let data_dir = spec
        .data_dir
        .clone()
        .ok_or("Diretório de dados do runtime indisponível")?;
    let job_id = uuid::Uuid::new_v4().to_string();
    let staging_base = data_dir.join("staging").to_absolute_normalized()?;
    let staging = staging_base.join(&job_id).to_absolute_normalized()?;
    if !staging.starts_with(&staging_base) || staging == staging_base {
        return Err("PATH_ESCAPE: staging fora da raiz permitida".to_string());
    }
    std::fs::create_dir_all(&staging)
        .map_err(|error| format!("Não foi possível criar o staging do download: {error}"))?;
    let download = json!({
        "jobId": job_id,
        "mangaId": config.manga_id,
        "chapterId": config.chapter_id,
        "extensionPackage": config.extension_package,
        "sourceId": config.source_id,
        "chapter": config.chapter,
        "status": "queued",
        "pageCount": 0,
        "downloadedPages": 0,
        "totalBytes": 0,
        "attempt": 0,
        "pages": [],
    });
    if let Err(error) = state
        .request(
            &spec,
            "reader.download.record",
            json!({ "download": download.clone() }),
            5_000,
        )
        .await
    {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error);
    }
    spawn_download_job(app, job_id);
    Ok(json!({ "v": PROTOCOL_VERSION, "ok": true, "result": download }))
}

fn spawn_download_job(app: AppHandle, job_id: String) {
    let control = app
        .state::<SourceRuntimeManager>()
        .download_control(&job_id);
    if control.running.swap(true, Ordering::AcqRel) {
        return;
    }
    tauri::async_runtime::spawn(async move {
        if let Err(error) = run_download_job(&app, &job_id, &control).await {
            let _ = mark_download_failed(&app, &job_id, &error).await;
        }
        control.running.store(false, Ordering::Release);
        control.changed.notify_waiters();
    });
}

async fn run_download_job(
    app: &AppHandle,
    job_id: &str,
    control: &DownloadJobControl,
) -> Result<(), String> {
    let spec = RuntimeLaunchSpec::discover(app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    let state = app.state::<SourceRuntimeManager>();
    let mut record = load_download_record(&state, &spec, job_id).await?;
    if record.get("status").and_then(Value::as_str) == Some("completed") {
        return Ok(());
    }
    if control.pause_requested.load(Ordering::Acquire) {
        set_download_status(&mut record, "paused", None);
        save_download_record(&state, &spec, &record).await?;
        emit_download_progress(app, &record);
        return Ok(());
    }

    set_download_status(&mut record, "running", None);
    save_download_record(&state, &spec, &record).await?;
    emit_download_progress(app, &record);

    let source_pages = if let Some(pages) = record.get("sourcePages").and_then(Value::as_array) {
        pages.clone()
    } else {
        let response = state.request(&spec, "source.pages", json!({
            "packageName": required_record_string(&record, "extensionPackage")?,
            "sourceId": required_record_string(&record, "sourceId")?,
            "chapter": record.get("chapter").cloned().ok_or("DOWNLOAD_RECORD_INVALID: chapter ausente")?,
        }), SOURCE_QUERY_DEADLINE_MS).await?;
        let pages = response
            .pointer("/result/pages")
            .and_then(Value::as_array)
            .cloned()
            .ok_or("DOWNLOAD_PAGES_INVALID: lista de páginas ausente")?;
        if pages.is_empty() || pages.len() > 500 {
            return Err(
                "PAGE_COUNT_INVALID: o capítulo deve ter entre 1 e 500 páginas".to_string(),
            );
        }
        let object = record
            .as_object_mut()
            .ok_or("DOWNLOAD_RECORD_INVALID: payload inválido")?;
        object.insert("sourcePages".to_string(), Value::Array(pages.clone()));
        object.insert("pageCount".to_string(), json!(pages.len()));
        save_download_record(&state, &spec, &record).await?;
        pages
    };

    let data_dir = spec
        .data_dir
        .clone()
        .ok_or("Diretório de dados do runtime indisponível")?;
    let staging_base = data_dir.join("staging").to_absolute_normalized()?;
    let staging = staging_base.join(job_id).to_absolute_normalized()?;
    if !staging.starts_with(&staging_base) || staging == staging_base {
        return Err("PATH_ESCAPE: staging fora da raiz permitida".to_string());
    }
    std::fs::create_dir_all(&staging)
        .map_err(|error| format!("Não foi possível preparar o staging do download: {error}"))?;

    let mut downloaded = record
        .get("pages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut total_bytes = record
        .get("totalBytes")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| {
            downloaded
                .iter()
                .filter_map(|page| page.get("size").and_then(Value::as_u64))
                .sum()
        });
    for (index, page) in source_pages.iter().enumerate().skip(downloaded.len()) {
        if control.pause_requested.load(Ordering::Acquire) {
            set_download_status(&mut record, "paused", None);
            save_download_record(&state, &spec, &record).await?;
            emit_download_progress(app, &record);
            return Ok(());
        }
        if total_download_storage(&state, &spec).await? >= MAX_DOWNLOAD_STORAGE_BYTES {
            control.pause_requested.store(true, Ordering::Release);
            set_download_status(
                &mut record,
                "paused",
                Some("Limite de armazenamento atingido; downloads pausados sem apagar arquivos."),
            );
            save_download_record(&state, &spec, &record).await?;
            emit_download_progress(app, &record);
            return Ok(());
        }

        let mut attempt = 0u32;
        let response = loop {
            let result = state
                .request(
                    &spec,
                    "source.download-page",
                    json!({
                        "packageName": required_record_string(&record, "extensionPackage")?,
                        "sourceId": required_record_string(&record, "sourceId")?,
                        "page": page,
                        "number": index + 1,
                        "stagingRoot": staging,
                    }),
                    SOURCE_QUERY_DEADLINE_MS,
                )
                .await;
            match result {
                Ok(response) => break response,
                Err(error) if is_retryable_download_error(&error) => {
                    let Some(delay) =
                        download_retry_delay(attempt, retry_after_seconds_from_error(&error))
                    else {
                        return Err(error);
                    };
                    attempt += 1;
                    set_download_status(&mut record, "retrying", Some(&error));
                    if let Some(object) = record.as_object_mut() {
                        object.insert("attempt".to_string(), json!(attempt));
                    }
                    save_download_record(&state, &spec, &record).await?;
                    emit_download_progress(app, &record);
                    tokio::select! {
                        _ = sleep(delay) => {},
                        _ = control.changed.notified() => {},
                    }
                    if control.pause_requested.load(Ordering::Acquire) {
                        set_download_status(&mut record, "paused", None);
                        save_download_record(&state, &spec, &record).await?;
                        emit_download_progress(app, &record);
                        return Ok(());
                    }
                    set_download_status(&mut record, "running", None);
                }
                Err(error) => return Err(error),
            }
        };
        let mut validated = validate_download_manifest(&staging, &response)?;
        let page = validated
            .pop()
            .ok_or("DOWNLOAD_MANIFEST_INVALID: página ausente")?;
        let size = page.get("size").and_then(Value::as_u64).unwrap_or(0);
        total_bytes = total_bytes.saturating_add(size);
        if total_bytes > MAX_CHAPTER_DOWNLOAD_BYTES {
            return Err("DOWNLOAD_TOO_LARGE: capítulo excede 1 GiB".to_string());
        }
        downloaded.push(page);
        let object = record
            .as_object_mut()
            .ok_or("DOWNLOAD_RECORD_INVALID: payload inválido")?;
        object.insert("status".to_string(), json!("running"));
        object.insert("pages".to_string(), Value::Array(downloaded.clone()));
        object.insert("downloadedPages".to_string(), json!(downloaded.len()));
        object.insert("pageCount".to_string(), json!(source_pages.len()));
        object.insert("totalBytes".to_string(), json!(total_bytes));
        object.insert("attempt".to_string(), json!(0));
        object.remove("error");
        save_download_record(&state, &spec, &record).await?;
        emit_download_progress(app, &record);
    }

    set_download_status(&mut record, "completed", None);
    save_download_record(&state, &spec, &record).await?;
    emit_download_progress(app, &record);
    Ok(())
}

async fn load_download_record(
    state: &SourceRuntimeManager,
    spec: &RuntimeLaunchSpec,
    job_id: &str,
) -> Result<Value, String> {
    let response = state
        .request(spec, "reader.download.list", json!({}), 5_000)
        .await?;
    response
        .pointer("/result/downloads")
        .and_then(Value::as_array)
        .and_then(|items| {
            items
                .iter()
                .find(|item| item.get("jobId").and_then(Value::as_str) == Some(job_id))
        })
        .cloned()
        .ok_or_else(|| "DOWNLOAD_NOT_FOUND: download não registrado".to_string())
}

async fn save_download_record(
    state: &SourceRuntimeManager,
    spec: &RuntimeLaunchSpec,
    record: &Value,
) -> Result<(), String> {
    state
        .request(
            spec,
            "reader.download.record",
            json!({ "download": record }),
            5_000,
        )
        .await?;
    Ok(())
}

fn required_record_string<'a>(record: &'a Value, name: &str) -> Result<&'a str, String> {
    record
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("DOWNLOAD_RECORD_INVALID: {name} ausente"))
}

fn set_download_status(record: &mut Value, status: &str, error: Option<&str>) {
    if let Some(object) = record.as_object_mut() {
        object.insert("status".to_string(), json!(status));
        match error {
            Some(message) => {
                object.insert("error".to_string(), json!(message));
            }
            None => {
                object.remove("error");
            }
        }
    }
}

fn is_retryable_download_error(error: &str) -> bool {
    [
        "SOURCE_TIMEOUT",
        "SOURCE_FAILURE",
        "SOURCE_RUNTIME_CRASHED",
        "IMAGE_HTTP_ERROR",
    ]
    .iter()
    .any(|code| error.starts_with(code))
}

async fn total_download_storage(
    state: &SourceRuntimeManager,
    spec: &RuntimeLaunchSpec,
) -> Result<u64, String> {
    let response = state
        .request(spec, "reader.download.list", json!({}), 5_000)
        .await?;
    Ok(response
        .pointer("/result/downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| item.get("totalBytes").and_then(Value::as_u64))
        .sum())
}

fn emit_download_progress(app: &AppHandle, record: &Value) {
    let _ = app.emit("studio://source-runtime/job-progress", record);
}

async fn mark_download_failed(app: &AppHandle, job_id: &str, error: &str) -> Result<(), String> {
    let spec = RuntimeLaunchSpec::discover(app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    let state = app.state::<SourceRuntimeManager>();
    let mut record = load_download_record(&state, &spec, job_id).await?;
    if record.get("status").and_then(Value::as_str) != Some("paused") {
        set_download_status(&mut record, "failed", Some(error));
        save_download_record(&state, &spec, &record).await?;
        emit_download_progress(app, &record);
    }
    Ok(())
}

async fn update_download_action(
    app: &AppHandle,
    state: &SourceRuntimeManager,
    job_id: &str,
    action: DownloadAction,
) -> Result<Value, String> {
    uuid::Uuid::parse_str(job_id).map_err(|_| "ID de download inválido".to_string())?;
    let spec = RuntimeLaunchSpec::discover(app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    let mut record = load_download_record(state, &spec, job_id).await?;
    let status = required_record_string(&record, "status")?.to_string();
    if !download_action_allowed(&status, action) {
        return Err(format!(
            "DOWNLOAD_STATE_INVALID: ação indisponível para {status}"
        ));
    }
    let control = state.download_control(job_id);
    match action {
        DownloadAction::Pause => {
            control.pause_requested.store(true, Ordering::Release);
            control.changed.notify_waiters();
            set_download_status(&mut record, "paused", None);
        }
        DownloadAction::Resume => {
            control.pause_requested.store(false, Ordering::Release);
            set_download_status(&mut record, "queued", None);
        }
    }
    save_download_record(state, &spec, &record).await?;
    emit_download_progress(app, &record);
    if action == DownloadAction::Resume {
        if control.running.load(Ordering::Acquire) {
            let app = app.clone();
            let job_id = job_id.to_string();
            let control = control.clone();
            tauri::async_runtime::spawn(async move {
                while control.running.load(Ordering::Acquire) {
                    control.changed.notified().await;
                }
                if !control.pause_requested.load(Ordering::Acquire) {
                    spawn_download_job(app, job_id);
                }
            });
        } else {
            spawn_download_job(app.clone(), job_id.to_string());
        }
    }
    Ok(json!({ "v": PROTOCOL_VERSION, "ok": true, "result": record }))
}

#[tauri::command]
pub(crate) async fn studio_reader_download_pause(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ReaderRecordConfig,
) -> Result<Value, String> {
    update_download_action(&app, &state, &config.record_id, DownloadAction::Pause).await
}

#[tauri::command]
pub(crate) async fn studio_reader_download_resume(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ReaderRecordConfig,
) -> Result<Value, String> {
    update_download_action(&app, &state, &config.record_id, DownloadAction::Resume).await
}

#[tauri::command]
pub(crate) async fn studio_reader_download_retry(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ReaderRecordConfig,
) -> Result<Value, String> {
    update_download_action(&app, &state, &config.record_id, DownloadAction::Resume).await
}

#[tauri::command]
pub(crate) async fn studio_reader_download_remove(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ReaderRecordConfig,
) -> Result<bool, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    uuid::Uuid::parse_str(&config.record_id).map_err(|_| "ID de download inválido".to_string())?;
    let base = spec
        .data_dir
        .clone()
        .ok_or("Diretório de dados do runtime indisponível")?
        .join("staging")
        .to_absolute_normalized()?;
    let target = base.join(&config.record_id).to_absolute_normalized()?;
    let control = state.download_control(&config.record_id);
    if control.running.load(Ordering::Acquire) {
        return Err("DOWNLOAD_BUSY: pause o download antes de removê-lo".to_string());
    }
    if !target.starts_with(&base) || target == base {
        return Err("PATH_ESCAPE: download fora da raiz permitida".to_string());
    }
    if !target.exists() {
        return Ok(false);
    }
    std::fs::remove_dir_all(target)
        .map_err(|error| format!("Falha ao remover download: {error}"))?;
    state
        .request(
            &spec,
            "reader.download.remove",
            json!({ "jobId": config.record_id }),
            5_000,
        )
        .await?;
    state.forget_download_control(&config.record_id);
    Ok(true)
}

#[tauri::command]
pub(crate) async fn studio_reader_download_list(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
) -> Result<Value, String> {
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    let response = state
        .request(&spec, "reader.download.list", json!({}), 5_000)
        .await?;
    if let Some(downloads) = response
        .pointer("/result/downloads")
        .and_then(Value::as_array)
    {
        for download in downloads {
            let Some(job_id) = download.get("jobId").and_then(Value::as_str) else {
                continue;
            };
            if matches!(
                download.get("status").and_then(Value::as_str),
                Some("queued" | "running" | "retrying")
            ) {
                spawn_download_job(app.clone(), job_id.to_string());
            }
        }
    }
    Ok(response)
}

#[tauri::command]
pub(crate) async fn studio_reader_translate_chapter(
    app: AppHandle,
    state: State<'_, SourceRuntimeManager>,
    config: ReaderTranslateConfig,
) -> Result<Value, String> {
    uuid::Uuid::parse_str(&config.job_id).map_err(|_| "ID de download inválido".to_string())?;
    let spec = RuntimeLaunchSpec::discover(&app)
        .ok_or("Runtime de fontes não foi encontrado nos recursos do aplicativo")?;
    let downloads = state
        .request(&spec, "reader.download.list", json!({}), 5_000)
        .await?;
    let download = downloads
        .pointer("/result/downloads")
        .and_then(Value::as_array)
        .and_then(|items| {
            items.iter().find(|item| {
                item.get("jobId").and_then(Value::as_str) == Some(config.job_id.as_str())
            })
        });
    let Some(download) = download else {
        return Err("DOWNLOAD_NOT_FOUND: download não registrado".to_string());
    };
    if !download_ready_for_reading(download) {
        return Err("DOWNLOAD_NOT_READY: o capítulo ainda não terminou de baixar".to_string());
    }
    let base = spec
        .data_dir
        .ok_or("Diretório de dados do runtime indisponível")?
        .join("staging")
        .to_absolute_normalized()?;
    let staging = base.join(&config.job_id).to_absolute_normalized()?;
    if !staging.starts_with(&base) || staging == base {
        return Err("PATH_ESCAPE: download fora da raiz permitida".to_string());
    }
    let project_path = PathBuf::from(config.project_json_path);
    let pages = tauri::async_runtime::spawn_blocking(move || {
        crate::chapter_import::prepare_source_runtime_chapter(&staging, &project_path)
    })
    .await
    .map_err(|error| format!("Falha ao preparar capítulo remoto: {error}"))??;
    Ok(json!({ "preparedPages": pages }))
}

fn download_ready_for_reading(download: &Value) -> bool {
    let completed = download.get("status").and_then(Value::as_str) == Some("completed");
    let page_count = download
        .get("pageCount")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let pages = download
        .get("pages")
        .and_then(Value::as_array)
        .map_or(0, Vec::len) as u64;
    completed && page_count > 0 && pages == page_count
}

fn validate_download_manifest(staging: &Path, response: &Value) -> Result<Vec<Value>, String> {
    let files = response
        .pointer("/result/files")
        .and_then(Value::as_array)
        .ok_or("DOWNLOAD_MANIFEST_INVALID: lista de arquivos ausente")?;
    if files.is_empty() || files.len() > 500 {
        return Err("DOWNLOAD_MANIFEST_INVALID: quantidade de páginas fora do limite".to_string());
    }
    let canonical_root = std::fs::canonicalize(staging)
        .map_err(|error| format!("DOWNLOAD_STAGING_INVALID: {error}"))?;
    let mut total = 0u64;
    let mut pages = Vec::with_capacity(files.len());
    for (position, entry) in files.iter().enumerate() {
        let relative = entry
            .get("relativePath")
            .and_then(Value::as_str)
            .ok_or("DOWNLOAD_MANIFEST_INVALID: caminho relativo ausente")?;
        let relative_path = Path::new(relative);
        if relative_path.components().count() != 1 || relative_path.is_absolute() {
            return Err("PATH_ESCAPE: manifesto contém caminho inválido".to_string());
        }
        let target = canonical_root.join(relative_path);
        let symlink = std::fs::symlink_metadata(&target)
            .map_err(|error| format!("DOWNLOAD_FILE_MISSING: {error}"))?;
        if symlink.file_type().is_symlink() || !symlink.is_file() {
            return Err("DOWNLOAD_FILE_INVALID: página não é arquivo regular".to_string());
        }
        let canonical_target = std::fs::canonicalize(&target)
            .map_err(|error| format!("DOWNLOAD_FILE_INVALID: {error}"))?;
        if !canonical_target.starts_with(&canonical_root) {
            return Err("PATH_ESCAPE: arquivo saiu do staging".to_string());
        }
        let bytes = std::fs::read(&canonical_target)
            .map_err(|error| format!("DOWNLOAD_FILE_INVALID: {error}"))?;
        if bytes.is_empty() || bytes.len() > 25 * 1024 * 1024 {
            return Err("DOWNLOAD_FILE_INVALID: tamanho de página fora do limite".to_string());
        }
        total = total.saturating_add(bytes.len() as u64);
        if total > 1024 * 1024 * 1024 {
            return Err("DOWNLOAD_TOO_LARGE: capítulo excede 1 GiB".to_string());
        }
        let expected_size = entry
            .get("size")
            .and_then(Value::as_u64)
            .ok_or("DOWNLOAD_MANIFEST_INVALID: tamanho ausente")?;
        if expected_size != bytes.len() as u64 {
            return Err("DOWNLOAD_INTEGRITY_FAILED: tamanho divergente".to_string());
        }
        let actual_hash = format!("{:x}", Sha256::digest(&bytes));
        if entry.get("sha256").and_then(Value::as_str) != Some(actual_hash.as_str()) {
            return Err("DOWNLOAD_INTEGRITY_FAILED: SHA-256 divergente".to_string());
        }
        let (width, height) = image::ImageReader::open(&canonical_target)
            .map_err(|error| format!("DOWNLOAD_IMAGE_INVALID: {error}"))?
            .with_guessed_format()
            .map_err(|error| format!("DOWNLOAD_IMAGE_INVALID: {error}"))?
            .into_dimensions()
            .map_err(|error| format!("DOWNLOAD_IMAGE_INVALID: {error}"))?;
        if width == 0 || height == 0 || u64::from(width) * u64::from(height) > 250_000_000 {
            return Err("DOWNLOAD_IMAGE_INVALID: dimensões fora do limite".to_string());
        }
        pages.push(json!({
            "id": format!("{}:{}", staging.file_name().and_then(|value| value.to_str()).unwrap_or_default(), relative),
            "number": entry.get("number").and_then(Value::as_u64).unwrap_or((position + 1) as u64),
            "mime": entry.get("mime").and_then(Value::as_str).unwrap_or("application/octet-stream"),
            "size": expected_size,
            "sha256": actual_hash,
            "width": width,
            "height": height,
        }));
    }
    Ok(pages)
}

trait AbsoluteNormalizedPath {
    fn to_absolute_normalized(&self) -> Result<PathBuf, String>;
}

impl AbsoluteNormalizedPath for PathBuf {
    fn to_absolute_normalized(&self) -> Result<PathBuf, String> {
        if self.is_absolute() {
            Ok(self.components().collect())
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(self).components().collect())
                .map_err(|error| error.to_string())
        }
    }
}

#[cfg(test)]
mod download_manifest_tests {
    use super::*;

    #[test]
    fn validates_hash_dimensions_and_returns_only_opaque_page_ids() {
        let temp = tempfile::tempdir().unwrap();
        let staging = temp.path().join("11111111-1111-4111-8111-111111111111");
        std::fs::create_dir(&staging).unwrap();
        let path = staging.join("page-0001.png");
        image::DynamicImage::new_rgb8(4, 6).save(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let response = json!({ "result": { "files": [{
            "relativePath": "page-0001.png",
            "mime": "image/png",
            "size": bytes.len(),
            "sha256": hash,
            "number": 1
        }]}});

        let pages = validate_download_manifest(&staging, &response).unwrap();

        assert_eq!(pages[0]["width"], 4);
        assert_eq!(pages[0]["height"], 6);
        assert_eq!(
            pages[0]["id"],
            "11111111-1111-4111-8111-111111111111:page-0001.png"
        );
        assert!(pages[0].get("relativePath").is_none());
    }

    #[test]
    fn rejects_manifest_path_escape_before_reading_outside_staging() {
        let temp = tempfile::tempdir().unwrap();
        let staging = temp.path().join("11111111-1111-4111-8111-111111111111");
        std::fs::create_dir(&staging).unwrap();
        let response = json!({ "result": { "files": [{
            "relativePath": "../outside.png", "mime": "image/png", "size": 1, "sha256": "00"
        }]}});

        let error = validate_download_manifest(&staging, &response).unwrap_err();

        assert!(error.starts_with("PATH_ESCAPE"));
    }

    #[test]
    fn download_retry_uses_retry_after_then_bounded_exponential_backoff() {
        assert_eq!(
            download_retry_delay(0, Some(7)),
            Some(Duration::from_secs(7))
        );
        assert_eq!(download_retry_delay(0, None), Some(Duration::from_secs(1)));
        assert_eq!(download_retry_delay(1, None), Some(Duration::from_secs(2)));
        assert_eq!(download_retry_delay(2, None), Some(Duration::from_secs(4)));
        assert_eq!(download_retry_delay(3, None), None);
        assert_eq!(
            download_retry_delay(0, Some(9_999)),
            Some(Duration::from_secs(300))
        );
    }

    #[test]
    fn only_active_download_states_can_be_paused_and_only_stopped_states_resumed() {
        assert!(download_action_allowed("queued", DownloadAction::Pause));
        assert!(download_action_allowed("running", DownloadAction::Pause));
        assert!(download_action_allowed("retrying", DownloadAction::Pause));
        assert!(!download_action_allowed("completed", DownloadAction::Pause));
        assert!(download_action_allowed("paused", DownloadAction::Resume));
        assert!(download_action_allowed("failed", DownloadAction::Resume));
        assert!(!download_action_allowed("running", DownloadAction::Resume));
    }

    #[test]
    fn only_a_complete_manifest_is_ready_for_reading_or_translation() {
        assert!(download_ready_for_reading(&json!({
            "status": "completed", "pageCount": 2, "pages": [{}, {}]
        })));
        assert!(!download_ready_for_reading(&json!({
            "status": "running", "pageCount": 2, "pages": [{}, {}]
        })));
        assert!(!download_ready_for_reading(&json!({
            "status": "completed", "pageCount": 2, "pages": [{}]
        })));
    }
}
