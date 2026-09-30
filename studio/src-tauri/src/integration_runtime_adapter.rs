use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;

use crate::project_schema;

#[derive(Debug, Deserialize)]
pub struct RenderPreviewConfig {
    pub project_path: String,
    pub page_index: u32,
    pub page: Value,
    pub fingerprint: String,
}

#[derive(Debug, Serialize)]
pub struct RenderPreviewResult {
    pub output_path: String,
    pub renderer_backend: String,
}

#[derive(Debug, Deserialize)]
pub struct ExportConfig {
    pub project_path: String,
    pub format: String,
    pub output_path: String,
    #[serde(default)]
    pub export_mode: Option<String>,
}

fn safe_key(raw: &str) -> String {
    let mut value = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    value.truncate(48);
    let value = value.trim_matches('_');
    if value.is_empty() {
        "preview".into()
    } else {
        value.into()
    }
}

fn safe_extension(raw: &str) -> String {
    let value = raw
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    if value.is_empty() {
        "png".into()
    } else {
        value
    }
}

fn preview_extension(page: &Value) -> String {
    page.pointer("/image_layers/base/path")
        .or_else(|| page.get("arquivo_original"))
        .or_else(|| page.get("arquivo_traduzido"))
        .and_then(Value::as_str)
        .and_then(|path| Path::new(path).extension())
        .and_then(|value| value.to_str())
        .map(safe_extension)
        .unwrap_or_else(|| "png".into())
}

fn is_consumer_fast_runtime_root(root: &Path) -> bool {
    root.join("pipeline/main.py").is_file()
        && root
            .join("pipeline/consumer_fast/chapter_runner.py")
            .is_file()
        && root
            .join("pipeline/consumer_fast/physical_executor.py")
            .is_file()
}

fn find_checkout_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|candidate| is_consumer_fast_runtime_root(candidate))
        .map(Path::to_path_buf)
}

fn consumer_fast_runtime_root(start: &Path, configured: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(root) = configured {
        let resolved = root
            .canonicalize()
            .map_err(|error| format!("runtime Consumer Fast configurado inacessível: {error}"))?;
        if !is_consumer_fast_runtime_root(&resolved) {
            return Err(format!(
                "runtime Consumer Fast incompleto em {}; main.py/chapter_runner.py/physical_executor.py são obrigatórios",
                resolved.display()
            ));
        }
        return Ok(resolved);
    }
    find_checkout_root(start).ok_or_else(|| {
        "runtime Consumer Fast canônico não encontrado; configure TRADUZAI_CONSUMER_FAST_RUNTIME_ROOT".to_string()
    })
}

fn find_python_runtime(start: &Path) -> Option<PathBuf> {
    start.ancestors().find_map(|candidate| {
        #[cfg(windows)]
        let python = candidate.join("pipeline/venv/Scripts/python.exe");
        #[cfg(not(windows))]
        let python = candidate.join("pipeline/venv/bin/python3");
        python.is_file().then_some(python)
    })
}

fn find_shared_models_dir(start: &Path) -> Option<PathBuf> {
    start.ancestors().find_map(|candidate| {
        let models = candidate.join("models");
        models
            .join("huggingface/models--mayocream--aot-inpainting")
            .is_dir()
            .then_some(models)
    })
}

pub fn physical_models_dir(app: &AppHandle) -> Result<PathBuf, String> {
    if cfg!(debug_assertions) {
        let cwd = std::env::current_dir().map_err(|error| error.to_string())?;
        if let Some(models) = find_shared_models_dir(&cwd) {
            return Ok(models);
        }
    }
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|error| format!("falha ao localizar dados locais: {error}"))?
        .join("models"))
}

fn sidecar(app: &AppHandle) -> Result<(String, Option<String>), String> {
    if cfg!(debug_assertions) {
        let cwd = std::env::current_dir().map_err(|error| error.to_string())?;
        let configured_root =
            std::env::var_os("TRADUZAI_CONSUMER_FAST_RUNTIME_ROOT").map(PathBuf::from);
        let root = consumer_fast_runtime_root(&cwd, configured_root.as_deref())?;
        let program = if let Some(python) = find_python_runtime(&root) {
            python.to_string_lossy().into_owned()
        } else if cfg!(windows) {
            "python".into()
        } else {
            "python3".into()
        };
        return Ok((
            program,
            Some(root.join("pipeline/main.py").to_string_lossy().into_owned()),
        ));
    }
    let mut binary = app
        .path()
        .resource_dir()
        .map_err(|error| error.to_string())?
        .join("binaries/traduzai-pipeline");
    #[cfg(windows)]
    {
        binary = binary.with_extension("exe");
    }
    Ok((binary.to_string_lossy().into_owned(), None))
}

fn consumer_fast_command_args(script: Option<&str>, config_path: &Path) -> Vec<OsString> {
    let mut args = Vec::with_capacity(usize::from(script.is_some()) + 2);
    if let Some(script) = script {
        args.push(script.into());
    }
    args.push("--consumer-fast-v1".into());
    args.push(config_path.as_os_str().to_owned());
    args
}

pub async fn run_pipeline_config_file(
    app: &AppHandle,
    config_path: &Path,
) -> Result<String, String> {
    if !config_path.is_file() {
        return Err(format!(
            "configuração física não encontrada: {}",
            config_path.display()
        ));
    }
    let (program, script) = sidecar(app)?;
    let mut command = Command::new(&program);
    command
        .args(consumer_fast_command_args(script.as_deref(), config_path))
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| format!("falha ao iniciar pipeline física: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "stdout da pipeline indisponível".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "stderr da pipeline indisponível".to_string())?;
    let stderr_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut body = String::new();
        let _ = reader.read_to_string(&mut body).await;
        body
    });
    let mut lines = BufReader::new(stdout).lines();
    let mut output_path = String::new();
    let mut reported_error = None;
    while let Some(line) = lines.next_line().await.map_err(|error| error.to_string())? {
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        app.emit("pipeline-progress", &message).ok();
        match message.get("type").and_then(Value::as_str) {
            Some("complete") => {
                output_path = message
                    .get("output_path")
                    .or_else(|| message.get("project_path"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .replace('\\', "/");
            }
            Some("error") => {
                reported_error = Some(
                    message
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("erro da pipeline física")
                        .to_string(),
                );
            }
            _ => {}
        }
    }
    let status = child.wait().await.map_err(|error| error.to_string())?;
    let stderr = stderr_task.await.unwrap_or_default();
    if !status.success() || reported_error.is_some() {
        let reason = reported_error.unwrap_or_else(|| format!("processo encerrou com {status}"));
        return Err(if stderr.trim().is_empty() {
            reason
        } else {
            format!("{reason}\n{}", stderr.trim())
        });
    }
    Ok(output_path)
}

pub async fn render_preview_page(
    app: AppHandle,
    config: RenderPreviewConfig,
) -> Result<RenderPreviewResult, String> {
    let project_file = project_schema::resolve_project_file(Path::new(&config.project_path));
    if !project_file.is_file() {
        return Err("project.json não encontrado".into());
    }
    let root = project_file
        .parent()
        .ok_or_else(|| "project.json sem diretório pai".to_string())?;
    let cache = root.join("render-cache").join("preview");
    std::fs::create_dir_all(&cache)
        .map_err(|error| format!("erro ao preparar cache de preview: {error}"))?;
    let stem = format!(
        "{:03}-{}",
        config.page_index + 1,
        safe_key(&config.fingerprint)
    );
    let override_path = cache.join(format!("{stem}.json"));
    let output_path = cache.join(format!("{stem}.{}", preview_extension(&config.page)));
    std::fs::write(
        &override_path,
        serde_json::to_vec_pretty(&json!({ "page": config.page }))
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("erro ao gravar página temporária do preview: {error}"))?;

    let (program, script) = sidecar(&app)?;
    let mut command = Command::new(&program);
    if let Some(script) = script {
        command.arg(script);
    }
    command
        .arg("--render-preview-page")
        .arg(project_file.to_string_lossy().to_string())
        .arg(config.page_index.to_string())
        .arg(override_path.to_string_lossy().to_string())
        .arg(output_path.to_string_lossy().to_string())
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| format!("falha ao iniciar renderer existente: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "stdout do renderer indisponível".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "stderr do renderer indisponível".to_string())?;
    let stderr_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut body = String::new();
        let _ = reader.read_to_string(&mut body).await;
        body
    });
    let mut lines = BufReader::new(stdout).lines();
    let mut result_path = output_path.to_string_lossy().replace('\\', "/");
    let mut backend = "python".to_string();
    let mut reported_error = None;
    while let Some(line) = lines.next_line().await.map_err(|error| error.to_string())? {
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        match message.get("type").and_then(Value::as_str) {
            Some("progress") => {
                app.emit("pipeline-progress", &message).ok();
            }
            Some("complete") => {
                if let Some(path) = message.get("output_path").and_then(Value::as_str) {
                    result_path = path.replace('\\', "/");
                }
                if let Some(value) = message.get("renderer_backend").and_then(Value::as_str) {
                    backend = value.to_string();
                }
            }
            Some("error") => {
                reported_error = Some(
                    message
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("erro do renderer")
                        .to_string(),
                )
            }
            _ => {}
        }
    }
    let status = child.wait().await.map_err(|error| error.to_string())?;
    let stderr = stderr_task.await.unwrap_or_default();
    if !status.success() || reported_error.is_some() {
        let reason = reported_error.unwrap_or_else(|| format!("processo encerrou com {status}"));
        return Err(if stderr.trim().is_empty() {
            reason
        } else {
            format!("{reason}\n{}", stderr.trim())
        });
    }
    let actual = PathBuf::from(&result_path);
    let actual = if actual.is_absolute() {
        actual
    } else {
        root.join(actual)
    };
    if !actual.is_file() {
        return Err("renderer terminou sem produzir o raster de preview".into());
    }
    Ok(RenderPreviewResult {
        output_path: actual.to_string_lossy().replace('\\', "/"),
        renderer_backend: backend,
    })
}

fn confined_file(root: &Path, raw: &str) -> Result<PathBuf, String> {
    let candidate = root.join(raw);
    let canonical_root = root.canonicalize().map_err(|error| error.to_string())?;
    let canonical = candidate
        .canonicalize()
        .map_err(|error| format!("artefato final ausente {raw}: {error}"))?;
    if !canonical.starts_with(&canonical_root) || !canonical.is_file() {
        return Err(format!("artefato final escapou do projeto: {raw}"));
    }
    Ok(canonical)
}

fn final_page_paths(project_file: &Path, project: &Value) -> Result<Vec<PathBuf>, String> {
    let root = project_file
        .parent()
        .ok_or_else(|| "project.json sem diretório pai".to_string())?;
    let pages = project
        .get("paginas")
        .and_then(Value::as_array)
        .ok_or_else(|| "projeto sem páginas".to_string())?;
    let direct = pages
        .iter()
        .map(|page| {
            page.get("arquivo_final")
                .or_else(|| page.get("arquivo_traduzido"))
                .or_else(|| page.pointer("/image_layers/rendered/path"))
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(|value| confined_file(root, value))
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    if direct.iter().all(Option::is_some) {
        return Ok(direct.into_iter().flatten().collect());
    }
    let translated = root.join("translated");
    let mut fallback = std::fs::read_dir(&translated)
        .map_err(|error| format!("páginas traduzidas finais indisponíveis: {error}"))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    fallback.sort_by_key(|path| path.file_name().map(|name| name.to_os_string()));
    if fallback.len() != pages.len() {
        return Err(format!(
            "exportação exige {} páginas finais, encontrou {}",
            pages.len(),
            fallback.len()
        ));
    }
    Ok(fallback)
}

pub async fn export_project(config: ExportConfig) -> Result<Value, String> {
    if config.format != "cbz" || config.export_mode.as_deref() != Some("final") {
        return Err("adapter integrado aceita somente CBZ final".into());
    }
    let project_file = project_schema::resolve_project_file(Path::new(&config.project_path));
    let project = project_schema::load_project_value(&project_file)?;
    let pages = final_page_paths(&project_file, &project)?;
    let destination = PathBuf::from(&config.output_path);
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let temporary = destination.with_extension(format!("cbz.{}.tmp", std::process::id()));
    let result = (|| -> Result<(), String> {
        let file = std::fs::File::create(&temporary).map_err(|error| error.to_string())?;
        let mut archive = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        let digits = pages.len().max(1).to_string().len().max(3);
        for (index, path) in pages.iter().enumerate() {
            let reader = image::ImageReader::open(path)
                .map_err(|error| format!("imagem final inválida: {error}"))?;
            let dimensions = reader
                .into_dimensions()
                .map_err(|error| format!("imagem final inválida: {error}"))?;
            if dimensions.0 == 0 || dimensions.1 == 0 {
                return Err("imagem final tem dimensões vazias".into());
            }
            let extension = safe_extension(
                path.extension()
                    .and_then(|value| value.to_str())
                    .unwrap_or("png"),
            );
            archive
                .start_file(format!("{:0digits$}.{extension}", index + 1), options)
                .map_err(|error| error.to_string())?;
            let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
            archive
                .write_all(&bytes)
                .map_err(|error| error.to_string())?;
        }
        archive.finish().map_err(|error| error.to_string())?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    if destination.exists() {
        std::fs::remove_file(&destination).map_err(|error| error.to_string())?;
    }
    std::fs::rename(&temporary, &destination).map_err(|error| error.to_string())?;
    Ok(json!({ "path": destination.to_string_lossy() }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};

    fn png(path: &Path, color: [u8; 4]) {
        ImageBuffer::from_pixel(2, 3, Rgba(color))
            .save(path)
            .unwrap();
    }

    #[test]
    fn debug_launch_selects_consumer_fast_before_config() {
        let script = r"C:\repo\pipeline\main.py";
        let config = Path::new(r"C:\data\pipeline_config.json");

        assert_eq!(
            consumer_fast_command_args(Some(script), config),
            vec![
                script.into(),
                "--consumer-fast-v1".into(),
                config.as_os_str().to_owned(),
            ]
        );
    }

    #[test]
    fn packaged_launch_selects_consumer_fast_before_config() {
        let config = Path::new(r"C:\data\pipeline_config.json");

        assert_eq!(
            consumer_fast_command_args(None, config),
            vec!["--consumer-fast-v1".into(), config.as_os_str().to_owned(),]
        );
    }

    #[test]
    fn finds_pipeline_from_a_nested_studio_working_directory() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("pipeline")).unwrap();
        std::fs::write(root.path().join("pipeline/main.py"), "# fixture\n").unwrap();
        std::fs::create_dir_all(root.path().join("pipeline/consumer_fast")).unwrap();
        std::fs::write(
            root.path().join("pipeline/consumer_fast/chapter_runner.py"),
            "# fixture\n",
        )
        .unwrap();
        std::fs::write(
            root.path()
                .join("pipeline/consumer_fast/physical_executor.py"),
            "# fixture\n",
        )
        .unwrap();
        let nested = root.path().join("studio/src-tauri");
        std::fs::create_dir_all(&nested).unwrap();

        assert_eq!(find_checkout_root(&nested), Some(root.path().to_path_buf()));
    }

    #[test]
    fn rejects_general_pipeline_root_without_physical_consumer_fast_executor() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("pipeline")).unwrap();
        std::fs::write(
            root.path().join("pipeline/main.py"),
            "# general pipeline only\n",
        )
        .unwrap();

        assert!(find_checkout_root(root.path()).is_none());
        assert!(consumer_fast_runtime_root(root.path(), Some(root.path())).is_err());
    }

    #[test]
    fn finds_shared_python_runtime_above_a_worktree_without_its_own_venv() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join(".worktrees/studio/studio/src-tauri");
        std::fs::create_dir_all(&nested).unwrap();
        #[cfg(windows)]
        let python = root.path().join("pipeline/venv/Scripts/python.exe");
        #[cfg(not(windows))]
        let python = root.path().join("pipeline/venv/bin/python3");
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::write(&python, b"fixture").unwrap();

        assert_eq!(find_python_runtime(&nested), Some(python));
    }

    #[test]
    fn finds_shared_aot_models_above_a_worktree() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join(".worktrees/studio/studio/src-tauri");
        std::fs::create_dir_all(&nested).unwrap();
        let models = root.path().join("models");
        std::fs::create_dir_all(models.join("huggingface/models--mayocream--aot-inpainting"))
            .unwrap();

        assert_eq!(find_shared_models_dir(&nested), Some(models));
    }

    #[tokio::test]
    async fn final_cbz_uses_project_page_order_and_valid_images() {
        let root = tempfile::tempdir().unwrap();
        let first = root.path().join("z-last-name.png");
        let second = root.path().join("a-first-name.png");
        png(&first, [1, 2, 3, 255]);
        png(&second, [4, 5, 6, 255]);
        let project_file = root.path().join("project.json");
        std::fs::write(
            &project_file,
            serde_json::to_vec(&json!({
                "paginas": [
                    {"arquivo_traduzido": "z-last-name.png"},
                    {"arquivo_traduzido": "a-first-name.png"}
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        let destination = root.path().join("final.cbz");

        export_project(ExportConfig {
            project_path: project_file.to_string_lossy().into_owned(),
            format: "cbz".into(),
            output_path: destination.to_string_lossy().into_owned(),
            export_mode: Some("final".into()),
        })
        .await
        .unwrap();

        let file = std::fs::File::open(destination).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 2);
        assert_eq!(archive.by_index(0).unwrap().name(), "001.png");
        assert_eq!(archive.by_index(1).unwrap().name(), "002.png");
    }

    #[test]
    fn final_page_path_rejects_project_escape() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        let project_file = root.path().join("project.json");
        let relative = pathdiff_for_test(root.path(), outside.path());
        let error = final_page_paths(
            &project_file,
            &json!({
                "paginas": [{"arquivo_traduzido": relative}]
            }),
        )
        .unwrap_err();
        assert!(error.contains("escapou"));
    }

    fn pathdiff_for_test(root: &Path, target: &Path) -> String {
        let mut value = PathBuf::from("..");
        if let Some(name) = target.file_name() {
            value.push(name);
        }
        let candidate = root.join(&value);
        if candidate.exists() {
            return value.to_string_lossy().into_owned();
        }
        target.to_string_lossy().into_owned()
    }
}
