use serde_json::{json, Value};
use std::{
    path::{Component, Path, PathBuf},
    process::Stdio,
};
use tauri::{AppHandle, Manager};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
};

const REQUESTS_DIR: &str = "requests";

#[derive(Debug)]
struct StudioLiteWorker {
    program: String,
    script: PathBuf,
    cwd: PathBuf,
}

#[derive(Debug)]
struct StudioLiteRequest {
    request_path: PathBuf,
    cache_dir: PathBuf,
    output_path: Option<PathBuf>,
}

#[tauri::command]
pub async fn studio_lite_model_status(app: AppHandle, config: Value) -> Result<Value, String> {
    if config.get("project_path").is_none() {
        return run_studio_lite_global_value(app, "model_status", config).await;
    }
    run_studio_lite_value(app, "model_status", config, None).await
}

#[tauri::command]
pub async fn studio_lite_build_mask(app: AppHandle, config: Value) -> Result<String, String> {
    run_studio_lite_output(app, "build_mask", config, "masks", "mask", "png").await
}

#[tauri::command]
pub async fn studio_lite_inpaint_region(app: AppHandle, config: Value) -> Result<String, String> {
    run_studio_lite_output(app, "inpaint_region", config, "inpaint", "inpaint", "png").await
}

#[tauri::command]
pub async fn studio_lite_detect_page(app: AppHandle, config: Value) -> Result<Value, String> {
    run_studio_lite_value(
        app,
        "detect_page",
        config,
        Some(("detections", "detect", "json")),
    )
    .await
}

async fn run_studio_lite_output(
    app: AppHandle,
    action: &str,
    config: Value,
    subdir: &str,
    prefix: &str,
    extension: &str,
) -> Result<String, String> {
    let project_dir = resolve_project_dir(&config)?;
    let request = prepare_request(
        config,
        action,
        &project_dir,
        Some((subdir, prefix, extension)),
    )?;
    let value = run_worker(app, &request.request_path).await?;
    let output = output_path_from_response(&value, &project_dir, &request.cache_dir)?
        .or_else(|| request.output_path.clone())
        .ok_or_else(|| "Worker Studio Lite nao retornou output_path".to_string())?;
    ensure_inside_existing_or_parent(&request.cache_dir, &output)?;
    Ok(project_relative_string(&project_dir, &output))
}

async fn run_studio_lite_value(
    app: AppHandle,
    action: &str,
    config: Value,
    output_hint: Option<(&str, &str, &str)>,
) -> Result<Value, String> {
    let project_dir = resolve_project_dir(&config)?;
    let request = prepare_request(config, action, &project_dir, output_hint)?;
    let value = run_worker(app, &request.request_path).await?;
    if let Some(output) = output_path_from_response(&value, &project_dir, &request.cache_dir)? {
        ensure_inside_existing_or_parent(&request.cache_dir, &output)?;
    }
    Ok(value)
}

async fn run_studio_lite_global_value(
    app: AppHandle,
    action: &str,
    mut config: Value,
) -> Result<Value, String> {
    let cache_dir = std::env::temp_dir().join("traduzai-studio-lite");
    std::fs::create_dir_all(cache_dir.join(REQUESTS_DIR))
        .map_err(|e| format!("Erro ao criar cache temporario Studio Lite: {e}"))?;
    if let Some(obj) = config.as_object_mut() {
        obj.insert(
            "cache_dir".to_string(),
            Value::String(cache_dir.to_string_lossy().replace('\\', "/")),
        );
    }
    let request_path = cache_dir
        .join(REQUESTS_DIR)
        .join(format!("request-{}.json", uuid::Uuid::new_v4()));
    let request_payload = json!({
        "action": action,
        "cache_dir": cache_dir.to_string_lossy().replace('\\', "/"),
        "config": config,
    });
    std::fs::write(
        &request_path,
        serde_json::to_vec_pretty(&request_payload).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("Erro ao gravar request Studio Lite: {e}"))?;
    run_worker(app, &request_path).await
}

fn prepare_request(
    mut config: Value,
    action: &str,
    project_dir: &Path,
    output_hint: Option<(&str, &str, &str)>,
) -> Result<StudioLiteRequest, String> {
    let cache_dir = studio_lite_cache_dir(project_dir);
    std::fs::create_dir_all(cache_dir.join(REQUESTS_DIR))
        .map_err(|e| format!("Erro ao criar cache Studio Lite: {e}"))?;

    let output_path = output_hint
        .map(|(subdir, prefix, extension)| {
            resolve_or_create_output_path(&mut config, &cache_dir, subdir, prefix, extension)
        })
        .transpose()?;

    let request_path = cache_dir
        .join(REQUESTS_DIR)
        .join(format!("request-{}.json", uuid::Uuid::new_v4()));
    let request_payload = json!({
        "action": action,
        "project_dir": project_dir.to_string_lossy().replace('\\', "/"),
        "cache_dir": cache_dir.to_string_lossy().replace('\\', "/"),
        "output_path": output_path
            .as_ref()
            .map(|path| path.to_string_lossy().replace('\\', "/")),
        "config": config,
    });
    std::fs::write(
        &request_path,
        serde_json::to_vec_pretty(&request_payload).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("Erro ao gravar request Studio Lite: {e}"))?;

    Ok(StudioLiteRequest {
        request_path,
        cache_dir,
        output_path,
    })
}

fn resolve_or_create_output_path(
    config: &mut Value,
    cache_dir: &Path,
    subdir: &str,
    prefix: &str,
    extension: &str,
) -> Result<PathBuf, String> {
    let output_dir = cache_dir.join(safe_component(subdir)?);
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("Erro ao criar saida Studio Lite: {e}"))?;
    let output = config
        .get("output_path")
        .and_then(Value::as_str)
        .map(|raw| resolve_cache_output_path(cache_dir, raw))
        .transpose()?
        .unwrap_or_else(|| {
            output_dir.join(format!(
                "{prefix}-{}.{}",
                uuid::Uuid::new_v4(),
                extension.trim_start_matches('.')
            ))
        });
    ensure_inside_existing_or_parent(cache_dir, &output)?;
    if let Some(obj) = config.as_object_mut() {
        obj.insert(
            "output_path".to_string(),
            Value::String(output.to_string_lossy().replace('\\', "/")),
        );
    }
    Ok(output)
}

async fn run_worker(app: AppHandle, request_path: &Path) -> Result<Value, String> {
    let worker = studio_lite_worker(&app)?;
    if !worker.script.exists() {
        return Err(format!(
            "Worker Studio Lite nao encontrado: {}",
            worker.script.to_string_lossy()
        ));
    }

    let mut cmd = Command::new(&worker.program);
    cmd.arg(&worker.script)
        .arg("--request")
        .arg(request_path)
        .current_dir(&worker.cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1");

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Erro ao iniciar Studio Lite worker: {e}"))?;
    let stdout = child.stdout.take().expect("stdout not captured");
    let stderr = child.stderr.take().expect("stderr not captured");

    let stderr_handle = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr).lines();
        let mut text = String::new();
        while let Ok(Some(line)) = reader.next_line().await {
            if text.len() < 4000 {
                text.push_str(&line);
                text.push('\n');
            }
        }
        text
    });

    let mut reader = BufReader::new(stdout).lines();
    let mut last_json: Option<Value> = None;
    while let Ok(Some(line)) = reader.next_line().await {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
            if value.get("type").and_then(Value::as_str) == Some("error") {
                return Err(value
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("Erro no Studio Lite worker")
                    .to_string());
            }
            last_json = Some(value);
        }
    }

    let status = child.wait().await.map_err(|e| e.to_string())?;
    let stderr_text = stderr_handle.await.unwrap_or_default();
    if !status.success() {
        return Err(format_worker_error(status.to_string(), &stderr_text));
    }

    let value = last_json.unwrap_or_else(|| json!({ "status": "ok" }));
    normalize_worker_response(value)
}

fn normalize_worker_response(value: Value) -> Result<Value, String> {
    if value.get("ok").and_then(Value::as_bool) == Some(false) {
        if let Some(message) = value
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
        {
            return Err(message.to_string());
        }
        return Err("Studio Lite worker retornou erro".to_string());
    }
    if let Some(result) = value.get("result") {
        let mut normalized = result.clone();
        if let Some(obj) = normalized.as_object_mut() {
            for key in ["output_path", "mask_path", "inpaint_path"] {
                if obj.get(key).is_none() {
                    if let Some(path) = value.get(key).cloned() {
                        obj.insert(key.to_string(), path);
                    }
                }
            }
        }
        return Ok(normalized);
    }
    Ok(value)
}

fn format_worker_error(status: String, stderr: &str) -> String {
    let detail = stderr.trim();
    if detail.is_empty() {
        format!("Studio Lite worker encerrou com status {status}")
    } else {
        let tail = if detail.len() > 2000 {
            &detail[detail.len() - 2000..]
        } else {
            detail
        };
        format!("Studio Lite worker encerrou com status {status}: {tail}")
    }
}

fn studio_lite_worker(app: &AppHandle) -> Result<StudioLiteWorker, String> {
    let root = project_root(app)?;
    let script = studio_lite_worker_script(app, &root)?;

    Ok(StudioLiteWorker {
        program: python_program(&root),
        script,
        cwd: root,
    })
}

fn studio_lite_worker_script(app: &AppHandle, root: &Path) -> Result<PathBuf, String> {
    let dev_script = root.join("pipeline").join("studio_lite").join("worker.py");
    if dev_script.exists() {
        return Ok(dev_script);
    }
    let resource_dir = app.path().resource_dir().map_err(|e| e.to_string())?;
    for candidate in [
        resource_dir.join("pipeline").join("studio_lite").join("worker.py"),
        resource_dir.join("studio_lite").join("worker.py"),
    ] {
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Ok(resource_dir.join("pipeline").join("studio_lite").join("worker.py"))
}

fn python_program(root: &Path) -> String {
    #[cfg(windows)]
    let candidate = root
        .join("pipeline")
        .join("venv")
        .join("Scripts")
        .join("python.exe");
    #[cfg(not(windows))]
    let candidate = root
        .join("pipeline")
        .join("venv")
        .join("bin")
        .join("python3");

    if candidate.exists() {
        candidate.to_string_lossy().to_string()
    } else if cfg!(windows) {
        "python".to_string()
    } else {
        "python3".to_string()
    }
}

fn project_root(_app: &AppHandle) -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    if cwd.join("pipeline").join("studio_lite").join("worker.py").exists() {
        return Ok(cwd);
    }
    if let Some(parent) = cwd.parent() {
        let parent = parent.to_path_buf();
        if parent
            .join("pipeline")
            .join("studio_lite")
            .join("worker.py")
            .exists()
        {
            return Ok(parent);
        }
    }
    if cwd.join("package.json").exists() && cwd.join("src-tauri").exists() {
        return Ok(cwd);
    }
    Ok(cwd)
}

fn resolve_project_dir(config: &Value) -> Result<PathBuf, String> {
    let raw = config
        .get("project_path")
        .and_then(Value::as_str)
        .ok_or_else(|| "project_path obrigatorio para Studio Lite".to_string())?;
    let path = normalize_input_path(raw);
    let project_dir = if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("project.json"))
    {
        path.parent()
            .ok_or_else(|| "project_path sem diretorio pai".to_string())?
            .to_path_buf()
    } else {
        path
    };
    if !project_dir.exists() {
        return Err(format!(
            "Diretorio do projeto Studio Lite nao encontrado: {}",
            project_dir.to_string_lossy()
        ));
    }
    let project_file = project_dir.join("project.json");
    if !project_file.exists() {
        return Err(format!(
            "project.json nao encontrado para Studio Lite: {}",
            project_file.to_string_lossy()
        ));
    }
    project_dir
        .canonicalize()
        .map_err(|e| format!("Erro ao resolver projeto Studio Lite: {e}"))
}

fn normalize_input_path(raw_path: &str) -> PathBuf {
    let trimmed = raw_path.trim();
    let without_file_uri = trimmed
        .strip_prefix("file:///")
        .or_else(|| trimmed.strip_prefix("file://"))
        .unwrap_or(trimmed);
    PathBuf::from(without_file_uri)
}

fn studio_lite_cache_dir(project_dir: &Path) -> PathBuf {
    project_dir.join("editor_cache").join("studio_lite")
}

fn resolve_cache_output_path(cache_dir: &Path, raw: &str) -> Result<PathBuf, String> {
    let raw_path = normalize_input_path(raw);
    let candidate = if raw_path.is_absolute() {
        raw_path
    } else {
        cache_dir.join(raw_path)
    };
    ensure_no_parent_components(&candidate)?;
    Ok(candidate)
}

fn ensure_no_parent_components(path: &Path) -> Result<(), String> {
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err("Caminho Studio Lite nao pode conter '..'".to_string());
    }
    Ok(())
}

fn ensure_inside_existing_or_parent(root: &Path, path: &Path) -> Result<(), String> {
    let canonical_root = root
        .canonicalize()
        .map_err(|e| format!("Cache Studio Lite invalido: {e}"))?;
    let comparable_path = if path.exists() {
        path.canonicalize()
            .map_err(|e| format!("Saida Studio Lite invalida: {e}"))?
    } else {
        let parent = path
            .parent()
            .ok_or_else(|| "Saida Studio Lite sem diretorio pai".to_string())?;
        let canonical_parent = parent
            .canonicalize()
            .map_err(|e| format!("Diretorio de saida Studio Lite invalido: {e}"))?;
        canonical_parent.join(
            path.file_name()
                .ok_or_else(|| "Saida Studio Lite sem nome de arquivo".to_string())?,
        )
    };
    if !path_is_inside(&canonical_root, &comparable_path) {
        return Err("Saida Studio Lite fora de editor_cache/studio_lite".to_string());
    }
    Ok(())
}

fn path_is_inside(root: &Path, path: &Path) -> bool {
    let root = root
        .to_string_lossy()
        .replace('\\', "/")
        .trim_end_matches('/')
        .to_ascii_lowercase();
    let path = path
        .to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase();
    path == root || path.starts_with(&format!("{root}/"))
}

fn project_relative_string(project_dir: &Path, path: &Path) -> String {
    path.strip_prefix(project_dir)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn output_path_from_response(
    value: &Value,
    project_dir: &Path,
    cache_dir: &Path,
) -> Result<Option<PathBuf>, String> {
    let Some(raw) = value
        .get("output_path")
        .or_else(|| value.get("mask_path"))
        .or_else(|| value.get("inpaint_path"))
        .or_else(|| value.get("path"))
        .or_else(|| value.get("result").and_then(|result| result.get("output_path")))
        .or_else(|| value.get("result").and_then(|result| result.get("mask_path")))
        .or_else(|| value.get("result").and_then(|result| result.get("inpaint_path")))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
    else {
        return Ok(None);
    };
    let path = normalize_input_path(raw);
    if path.is_absolute() {
        return Ok(Some(path));
    }
    ensure_no_parent_components(&path)?;
    let normalized = raw.replace('\\', "/");
    if normalized == "editor_cache" || normalized.starts_with("editor_cache/") {
        Ok(Some(project_dir.join(path)))
    } else {
        Ok(Some(cache_dir.join(path)))
    }
}

fn safe_component(value: &str) -> Result<&str, String> {
    if value.is_empty()
        || value
            .chars()
            .any(|c| c == '/' || c == '\\' || c == ':' || c == '\0')
    {
        return Err("Componente de cache Studio Lite invalido".to_string());
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_project() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("project.json"), "{}").expect("project");
        dir
    }

    #[test]
    fn studio_lite_cache_stays_under_project_editor_cache() {
        let project = make_project();
        let cache = studio_lite_cache_dir(project.path());
        assert!(cache.ends_with(Path::new("editor_cache").join("studio_lite")));
        assert!(path_is_inside(project.path(), &cache));
    }

    #[test]
    fn resolve_cache_output_rejects_parent_escape() {
        let project = make_project();
        let cache = studio_lite_cache_dir(project.path());
        std::fs::create_dir_all(&cache).expect("cache");
        let err = resolve_cache_output_path(&cache, "../escape.png").expect_err("escape rejected");
        assert!(err.contains(".."));
    }

    #[test]
    fn ensure_inside_allows_new_file_under_existing_cache_subdir() {
        let project = make_project();
        let cache = studio_lite_cache_dir(project.path());
        let out_dir = cache.join("masks");
        std::fs::create_dir_all(&out_dir).expect("out dir");
        let output = out_dir.join("mask.png");
        ensure_inside_existing_or_parent(&cache, &output).expect("inside cache");
    }

    #[test]
    fn ensure_inside_rejects_sibling_file() {
        let project = make_project();
        let cache = studio_lite_cache_dir(project.path());
        std::fs::create_dir_all(&cache).expect("cache");
        let sibling = project.path().join("editor_cache").join("mask.png");
        let err = ensure_inside_existing_or_parent(&cache, &sibling).expect_err("outside cache");
        assert!(err.contains("fora"));
    }

    #[test]
    fn response_output_accepts_project_relative_cache_path() {
        let project = make_project();
        let cache = studio_lite_cache_dir(project.path());
        let value = json!({ "output_path": "editor_cache/studio_lite/masks/mask.png" });
        let output = output_path_from_response(&value, project.path(), &cache)
            .expect("response path")
            .expect("output path");
        assert_eq!(output, cache.join("masks").join("mask.png"));
    }
}
