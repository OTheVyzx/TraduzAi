use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri_plugin_dialog::DialogExt;

#[derive(Debug, Serialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub pages: usize,
    pub has_project_json: bool,
    pub error: Option<String>,
}

#[tauri::command]
pub async fn open_file_dialog(app: tauri::AppHandle) -> Result<Option<String>, String> {
    let file = app
        .dialog()
        .file()
        .add_filter("Mangá", &["zip", "cbz", "jpg", "jpeg", "png", "webp"])
        .blocking_pick_file();

    Ok(file.map(|f| f.to_string()))
}

#[tauri::command]
pub async fn open_source_dialog(app: tauri::AppHandle) -> Result<Option<String>, String> {
    if let Some(file) = app
        .dialog()
        .file()
        .add_filter("Mangá", &["zip", "cbz", "jpg", "jpeg", "png", "webp"])
        .blocking_pick_file()
    {
        return Ok(Some(file.to_string()));
    }

    let folder = app.dialog().file().blocking_pick_folder();
    Ok(folder.map(|f| f.to_string()))
}

#[tauri::command]
pub async fn open_multiple_sources_dialog(app: tauri::AppHandle) -> Result<Vec<String>, String> {
    let mut results = Vec::new();

    // Primeiro tenta arquivos múltiplos
    if let Some(files) = app
        .dialog()
        .file()
        .add_filter("Mangá", &["zip", "cbz", "jpg", "jpeg", "png", "webp"])
        .blocking_pick_files()
    {
        for f in files {
            results.push(f.to_string());
        }
    }

    // Se não pegou arquivos, tenta pastas (infelizmente o pick_folders não é tão comum, mas vamos tentar se o usuário quiser selecionar várias subpastas de uma vez)
    if results.is_empty() {
        // Nota: Tauri v2 dialog pick_folders() existe se o plugin suportar.
        if let Some(folders) = app.dialog().file().blocking_pick_folders() {
            for f in folders {
                results.push(f.to_string());
            }
        }
    }

    Ok(results)
}

#[tauri::command]
pub async fn open_project_dialog(app: tauri::AppHandle) -> Result<Option<String>, String> {
    let folder = app.dialog().file().blocking_pick_folder();
    Ok(folder.map(|f| f.to_string()))
}

#[tauri::command]
pub async fn save_file_dialog(
    app: tauri::AppHandle,
    format: Option<String>,
) -> Result<Option<String>, String> {
    let format = format.unwrap_or_else(|| "zip_full".to_string());
    let (filter_name, filter_exts, default_name): (&str, &[&str], &str) = match format.as_str() {
        "cbz" => ("CBZ", &["cbz"], "traduzido.cbz"),
        "jpg_only" => ("ZIP", &["zip"], "paginas-traduzidas.zip"),
        _ => ("ZIP", &["zip"], "traduzido.zip"),
    };

    let file = app
        .dialog()
        .file()
        .add_filter(filter_name, filter_exts)
        .set_file_name(default_name)
        .blocking_save_file();

    Ok(file.map(|f| f.to_string()))
}

#[tauri::command]
pub async fn validate_import(path: String) -> Result<ValidationResult, String> {
    let path = normalize_path(&path);

    if !path.exists() {
        return Ok(ValidationResult {
            valid: false,
            pages: 0,
            has_project_json: false,
            error: Some("Arquivo não encontrado".into()),
        });
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "zip" | "cbz" => validate_archive(&path),
        "jpg" | "jpeg" | "png" | "webp" => Ok(ValidationResult {
            valid: true,
            pages: 1,
            has_project_json: false,
            error: None,
        }),
        _ => {
            if path.is_dir() {
                validate_directory(&path)
            } else {
                Ok(ValidationResult {
                    valid: false,
                    pages: 0,
                    has_project_json: false,
                    error: Some("Formato não suportado. Use .zip, .cbz, imagem ou pasta.".into()),
                })
            }
        }
    }
}

pub fn normalize_path(raw_path: &str) -> PathBuf {
    let trimmed = raw_path.trim();

    #[cfg(windows)]
    let normalized = {
        if let Some(stripped) = trimmed.strip_prefix("file:///") {
            stripped.to_string()
        } else if let Some(stripped) = trimmed.strip_prefix("file://") {
            stripped.to_string()
        } else {
            trimmed.to_string()
        }
    };

    #[cfg(not(windows))]
    let normalized = {
        if let Some(stripped) = trimmed.strip_prefix("file://") {
            stripped.to_string()
        } else {
            trimmed.to_string()
        }
    };

    PathBuf::from(normalized)
}

fn validate_archive(path: &PathBuf) -> Result<ValidationResult, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| format!("ZIP inválido: {}", e))?;

    let mut page_count = 0;
    let mut has_project_json = false;
    let image_exts = ["jpg", "jpeg", "png", "webp"];

    for i in 0..archive.len() {
        if let Ok(entry) = archive.by_index(i) {
            let name = entry.name().to_lowercase();
            if name.ends_with("project.json") {
                has_project_json = true;
            }
            if image_exts.iter().any(|ext| name.ends_with(ext)) {
                page_count += 1;
            }
        }
    }

    if page_count == 0 {
        return Ok(ValidationResult {
            valid: false,
            pages: 0,
            has_project_json,
            error: Some("Nenhuma imagem encontrada no arquivo.".into()),
        });
    }

    Ok(ValidationResult {
        valid: true,
        pages: page_count,
        has_project_json,
        error: None,
    })
}

fn validate_directory(path: &PathBuf) -> Result<ValidationResult, String> {
    let image_exts = ["jpg", "jpeg", "png", "webp"];
    let mut page_count = 0;
    let has_project_json = path.join("project.json").exists();

    for entry in walkdir::WalkDir::new(path).max_depth(3) {
        if let Ok(entry) = entry {
            if entry.file_type().is_file() {
                let name = entry.file_name().to_string_lossy().to_lowercase();
                if image_exts.iter().any(|ext| name.ends_with(ext)) {
                    page_count += 1;
                }
            }
        }
    }

    Ok(ValidationResult {
        valid: page_count > 0,
        pages: page_count,
        has_project_json,
        error: if page_count == 0 {
            Some("Nenhuma imagem encontrada na pasta.".into())
        } else {
            None
        },
    })
}

#[tauri::command]
pub async fn load_project_json(path: String) -> Result<serde_json::Value, String> {
    let normalized = normalize_path(&path);
    let project_path = if normalized
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("project.json"))
    {
        normalized
    } else {
        normalized.join("project.json")
    };
    let content = std::fs::read_to_string(&project_path)
        .map_err(|e| format!("Erro ao ler project.json: {}", e))?;
    serde_json::from_str(&content).map_err(|e| format!("JSON inválido: {}", e))
}

#[derive(Debug, Deserialize)]
pub struct SaveProjectConfig {
    pub project_path: String,
    pub project_json: serde_json::Value,
}

#[tauri::command]
pub async fn save_project_json(config: SaveProjectConfig) -> Result<(), String> {
    let base_path = normalize_path(&config.project_path);
    let project_file = if base_path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("project.json"))
    {
        base_path
    } else {
        base_path.join("project.json")
    };

    let content = serde_json::to_string_pretty(&config.project_json)
        .map_err(|e| format!("Erro ao serializar JSON: {}", e))?;

    std::fs::write(&project_file, content)
        .map_err(|e| format!("Erro ao salvar project.json: {}", e))?;

    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct ExportConfig {
    pub project_path: String,
    pub format: String, // "zip_full", "jpg_only", "cbz"
    pub output_path: String,
}

#[tauri::command]
pub async fn export_project(config: ExportConfig) -> Result<serde_json::Value, String> {
    let project_dir = normalize_path(&config.project_path);
    let output_path = normalize_path(&config.output_path);

    let file =
        std::fs::File::create(&output_path).map_err(|e| format!("Erro ao criar arquivo: {}", e))?;
    let mut zip_writer = zip::ZipWriter::new(file);

    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    let translated_dir = project_dir.join("translated");
    let originals_dir = project_dir.join("originals");
    let legacy_images_dir = project_dir.join("images");
    let project_json = project_dir.join("project.json");

    match config.format.as_str() {
        "jpg_only" | "cbz" => {
            add_directory_to_zip(&mut zip_writer, &translated_dir, "", options)?;
        }
        _ => {
            add_directory_to_zip(&mut zip_writer, &translated_dir, "translated/", options)?;

            if originals_dir.exists() {
                add_directory_to_zip(&mut zip_writer, &originals_dir, "originals/", options)?;
            } else if legacy_images_dir.exists() {
                add_directory_to_zip(&mut zip_writer, &legacy_images_dir, "images/", options)?;
            }

            if project_json.exists() {
                zip_writer
                    .start_file("project.json", options)
                    .map_err(|e| e.to_string())?;
                let data = std::fs::read(&project_json).map_err(|e| e.to_string())?;
                use std::io::Write;
                zip_writer.write_all(&data).map_err(|e| e.to_string())?;
            }
        }
    }

    zip_writer.finish().map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "path": output_path.to_string_lossy()
    }))
}

fn add_directory_to_zip(
    zip_writer: &mut zip::ZipWriter<std::fs::File>,
    dir: &PathBuf,
    prefix: &str,
    options: zip::write::SimpleFileOptions,
) -> Result<(), String> {
    if !dir.exists() {
        return Ok(());
    }

    for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let name = format!("{prefix}{}", entry.file_name().to_string_lossy());
        zip_writer
            .start_file(&name, options)
            .map_err(|e| e.to_string())?;
        let data = std::fs::read(&path).map_err(|e| e.to_string())?;
        use std::io::Write;
        zip_writer.write_all(&data).map_err(|e| e.to_string())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn unique_temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("traduzai-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_file(path: &PathBuf, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn zip_entries(zip_path: &PathBuf) -> Vec<String> {
        let file = std::fs::File::open(zip_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut names = Vec::new();

        for i in 0..archive.len() {
            let entry = archive.by_index(i).unwrap();
            names.push(entry.name().to_string());
        }

        names
    }

    #[tokio::test]
    async fn export_zip_full_includes_originals_project_and_translated() {
        let root = unique_temp_dir();
        let project_dir = root.join("project");
        std::fs::create_dir_all(project_dir.join("translated")).unwrap();
        std::fs::create_dir_all(project_dir.join("originals")).unwrap();

        write_file(
            &project_dir.join("translated").join("001.jpg"),
            b"translated",
        );
        write_file(&project_dir.join("originals").join("001.jpg"), b"original");
        write_file(&project_dir.join("project.json"), br#"{"obra":"Teste"}"#);

        let output = root.join("out.zip");
        export_project(ExportConfig {
            project_path: project_dir.to_string_lossy().to_string(),
            format: "zip_full".to_string(),
            output_path: output.to_string_lossy().to_string(),
        })
        .await
        .unwrap();

        let path = output.to_string_lossy().to_string();
        assert_eq!(output.to_string_lossy(), output.to_string_lossy());
        let entries = zip_entries(&PathBuf::from(path));
        assert!(entries.contains(&"translated/001.jpg".to_string()));
        assert!(entries.contains(&"originals/001.jpg".to_string()));
        assert!(entries.contains(&"project.json".to_string()));

        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn export_cbz_flattens_translated_pages() {
        let root = unique_temp_dir();
        let project_dir = root.join("project");
        std::fs::create_dir_all(project_dir.join("translated")).unwrap();
        write_file(
            &project_dir.join("translated").join("001.jpg"),
            b"translated",
        );

        let output = root.join("out.cbz");
        export_project(ExportConfig {
            project_path: project_dir.to_string_lossy().to_string(),
            format: "cbz".to_string(),
            output_path: output.to_string_lossy().to_string(),
        })
        .await
        .unwrap();

        let entries = zip_entries(&output);
        assert_eq!(entries, vec!["001.jpg".to_string()]);

        let file = std::fs::File::open(&output).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut contents = String::new();
        archive
            .by_name("001.jpg")
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        assert_eq!(contents, "translated");

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    #[cfg(windows)]
    fn normalize_path_strips_file_uri_prefix_on_windows() {
        let path = normalize_path("file:///C:/traduzai/teste/project.json");
        assert_eq!(path, PathBuf::from("C:/traduzai/teste/project.json"));
    }
}
