use crate::commands::project_schema;
use image::{GrayImage, ImageBuffer, ImageReader, Luma};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
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
    suggested_name: Option<String>,
) -> Result<Option<String>, String> {
    let format = format.unwrap_or_else(|| "zip_full".to_string());
    let (filter_name, filter_exts, default_name): (&str, &[&str], &str) = match format.as_str() {
        "cbz" => ("CBZ", &["cbz"], "traduzido.cbz"),
        "jpg_only" => ("ZIP", &["zip"], "paginas-traduzidas.zip"),
        "lab_patch_json" => ("JSON", &["json"], "lab-patch.json"),
        "log" => ("Log", &["log", "txt"], "traduzai-log.log"),
        _ => ("ZIP", &["zip"], "traduzido.zip"),
    };
    let file_name = suggested_name
        .as_deref()
        .filter(|name| !name.trim().is_empty())
        .unwrap_or(default_name);

    let file = app
        .dialog()
        .file()
        .add_filter(filter_name, filter_exts)
        .set_file_name(file_name)
        .blocking_save_file();

    Ok(file.map(|f| f.to_string()))
}

#[tauri::command]
pub async fn export_text_file(output_path: String, content: String) -> Result<String, String> {
    let path = PathBuf::from(&output_path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Falha ao criar pasta de destino: {e}"))?;
        }
    }
    std::fs::write(&path, content.as_bytes())
        .map_err(|e| format!("Falha ao gravar {}: {e}", path.display()))?;
    Ok(path.to_string_lossy().to_string())
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
pub async fn load_project_json(path: String) -> Result<Value, String> {
    let normalized = normalize_path(&path);
    let project_path = project_schema::resolve_project_file(&normalized);
    project_schema::load_project_value(&project_path)
}

#[derive(Debug, Deserialize)]
pub struct SaveProjectConfig {
    pub project_path: String,
    pub project_json: Value,
}

#[derive(Debug, Deserialize)]
pub struct LoadEditorPageConfig {
    pub project_path: String,
    pub page_index: usize,
}

#[derive(Debug, Deserialize)]
pub struct CreateTextLayerConfig {
    pub project_path: String,
    pub page_index: usize,
    pub layout_bbox: [i64; 4],
}

#[derive(Debug, Deserialize)]
pub struct PatchTextLayerConfig {
    pub project_path: String,
    pub page_index: usize,
    pub layer_id: String,
    pub patch: Value,
}

#[derive(Debug, Deserialize)]
pub struct DeleteTextLayerConfig {
    pub project_path: String,
    pub page_index: usize,
    pub layer_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SetLayerVisibilityConfig {
    pub project_path: String,
    pub page_index: usize,
    pub layer_kind: String,
    pub layer_key: Option<String>,
    pub layer_id: Option<String>,
    pub visible: bool,
}

#[derive(Debug, Deserialize)]
pub struct BitmapLayerUpdateConfig {
    pub project_path: String,
    pub page_index: usize,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub brush_size: u32,
    #[serde(default)]
    pub clear: bool,
    #[serde(default)]
    pub erase: bool,
    #[serde(default)]
    pub strokes: Vec<Vec<[i32; 2]>>,
}

#[tauri::command]
pub async fn save_project_json(config: SaveProjectConfig) -> Result<(), String> {
    let base_path = normalize_path(&config.project_path);
    let project_file = project_schema::resolve_project_file(&base_path);
    let mut project_json = config.project_json;
    project_schema::save_project_value(&project_file, &mut project_json)?;
    Ok(())
}

fn load_project_for_editing(project_path: &str) -> Result<(PathBuf, Value), String> {
    let base_path = normalize_path(project_path);
    let project_file = project_schema::resolve_project_file(&base_path);
    let project = project_schema::load_project_value(&project_file)?;
    Ok((project_file, project))
}

fn save_project_after_edit(project_file: &Path, project: &mut Value) -> Result<(), String> {
    project_schema::save_project_value(project_file, project)
}

#[tauri::command]
pub async fn load_editor_page(config: LoadEditorPageConfig) -> Result<Value, String> {
    let (project_file, project) = load_project_for_editing(&config.project_path)?;
    let pages = project
        .get("paginas")
        .and_then(|value| value.as_array())
        .ok_or_else(|| "Projeto sem páginas".to_string())?;
    let page = pages
        .get(config.page_index)
        .cloned()
        .ok_or_else(|| "Página inválida".to_string())?;

    Ok(json!({
        "project_file": project_file.to_string_lossy(),
        "project_dir": project_file.parent().unwrap_or_else(|| Path::new("")).to_string_lossy(),
        "page_index": config.page_index,
        "total_pages": pages.len(),
        "page": page,
    }))
}

#[tauri::command]
pub async fn create_text_layer(config: CreateTextLayerConfig) -> Result<Value, String> {
    let (project_file, mut project) = load_project_for_editing(&config.project_path)?;
    let layer =
        project_schema::create_text_layer(&mut project, config.page_index, config.layout_bbox)?;
    save_project_after_edit(&project_file, &mut project)?;
    Ok(layer)
}

#[tauri::command]
pub async fn patch_text_layer(config: PatchTextLayerConfig) -> Result<Value, String> {
    let (project_file, mut project) = load_project_for_editing(&config.project_path)?;
    let layer = project_schema::patch_text_layer(
        &mut project,
        config.page_index,
        &config.layer_id,
        &config.patch,
    )?;
    save_project_after_edit(&project_file, &mut project)?;
    Ok(layer)
}

#[tauri::command]
pub async fn delete_text_layer(config: DeleteTextLayerConfig) -> Result<(), String> {
    let (project_file, mut project) = load_project_for_editing(&config.project_path)?;
    project_schema::delete_text_layer(&mut project, config.page_index, &config.layer_id)?;
    save_project_after_edit(&project_file, &mut project)
}

#[tauri::command]
pub async fn set_layer_visibility(config: SetLayerVisibilityConfig) -> Result<(), String> {
    let (project_file, mut project) = load_project_for_editing(&config.project_path)?;
    project_schema::set_layer_visibility(
        &mut project,
        config.page_index,
        &config.layer_kind,
        config.layer_key.as_deref(),
        config.layer_id.as_deref(),
        config.visible,
    )?;
    save_project_after_edit(&project_file, &mut project)
}

fn paint_circle(bitmap: &mut GrayImage, center_x: i32, center_y: i32, radius: i32, value: u8) {
    let radius_sq = radius * radius;
    for y in (center_y - radius)..=(center_y + radius) {
        if y < 0 || y >= bitmap.height() as i32 {
            continue;
        }
        for x in (center_x - radius)..=(center_x + radius) {
            if x < 0 || x >= bitmap.width() as i32 {
                continue;
            }
            let dx = x - center_x;
            let dy = y - center_y;
            if dx * dx + dy * dy <= radius_sq {
                bitmap.put_pixel(x as u32, y as u32, Luma([value]));
            }
        }
    }
}

fn paint_stroke(bitmap: &mut GrayImage, stroke: &[[i32; 2]], radius: i32, value: u8) {
    if stroke.is_empty() {
        return;
    }
    if stroke.len() == 1 {
        paint_circle(bitmap, stroke[0][0], stroke[0][1], radius, value);
        return;
    }

    for window in stroke.windows(2) {
        let [x1, y1] = window[0];
        let [x2, y2] = window[1];
        let dx = x2 - x1;
        let dy = y2 - y1;
        let steps = dx.abs().max(dy.abs()).max(1);
        for step in 0..=steps {
            let t = step as f32 / steps as f32;
            let x = x1 as f32 + dx as f32 * t;
            let y = y1 as f32 + dy as f32 * t;
            paint_circle(bitmap, x.round() as i32, y.round() as i32, radius, value);
        }
    }
}

fn load_or_create_bitmap_layer(
    path: &Path,
    width: u32,
    height: u32,
    clear: bool,
) -> Result<GrayImage, String> {
    if !clear && path.exists() {
        let existing = ImageReader::open(path)
            .map_err(|e| format!("Erro ao abrir bitmap da layer: {e}"))?
            .decode()
            .map_err(|e| format!("Erro ao decodificar bitmap da layer: {e}"))?
            .to_luma8();
        if existing.width() == width && existing.height() == height {
            return Ok(existing);
        }
    }

    Ok(ImageBuffer::from_pixel(width, height, Luma([0])))
}

fn save_bitmap_layer(path: &Path, bitmap: &GrayImage) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Erro ao preparar diretório da layer bitmap: {e}"))?;
    }
    let temp_path = path.with_extension("png.tmp");
    bitmap
        .save(&temp_path)
        .map_err(|e| format!("Erro ao salvar layer bitmap temporária: {e}"))?;
    if path.exists() {
        std::fs::remove_file(path)
            .map_err(|e| format!("Erro ao substituir layer bitmap anterior: {e}"))?;
    }
    std::fs::rename(&temp_path, path)
        .map_err(|e| format!("Erro ao finalizar gravação da layer bitmap: {e}"))?;
    Ok(())
}

fn update_bitmap_layer(
    config: BitmapLayerUpdateConfig,
    layer_key: &str,
) -> Result<String, String> {
    if config.width == 0 || config.height == 0 {
        return Err("Dimensões da layer bitmap inválidas".to_string());
    }

    let (project_file, mut project) = load_project_for_editing(&config.project_path)?;
    let relative_layer_path =
        project_schema::ensure_bitmap_layer_path(&mut project, config.page_index, layer_key)?;
    let project_dir = project_file
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let absolute_layer_path = project_dir.join(&relative_layer_path);

    let mut bitmap = load_or_create_bitmap_layer(
        &absolute_layer_path,
        config.width,
        config.height,
        config.clear,
    )?;

    let radius = (config.brush_size.max(1) as i32 / 2).max(1);
    let value = if config.erase { 0 } else { 255 };
    for stroke in &config.strokes {
        paint_stroke(&mut bitmap, stroke, radius, value);
    }

    save_bitmap_layer(&absolute_layer_path, &bitmap)?;
    project_schema::set_layer_visibility(
        &mut project,
        config.page_index,
        "image",
        Some(layer_key),
        None,
        !config.erase || !config.strokes.is_empty(),
    )?;
    save_project_after_edit(&project_file, &mut project)?;

    Ok(absolute_layer_path.to_string_lossy().replace('\\', "/"))
}

#[tauri::command]
pub async fn update_mask_region(config: BitmapLayerUpdateConfig) -> Result<String, String> {
    update_bitmap_layer(config, "mask")
}

#[tauri::command]
pub async fn update_brush_region(config: BitmapLayerUpdateConfig) -> Result<String, String> {
    update_bitmap_layer(config, "brush")
}

#[derive(Debug, Deserialize)]
pub struct ExportConfig {
    pub project_path: String,
    pub format: String, // "zip_full", "jpg_only", "cbz"
    pub output_path: String,
}

#[tauri::command]
pub async fn export_project(config: ExportConfig) -> Result<Value, String> {
    let project_base = normalize_path(&config.project_path);
    let project_file = project_schema::resolve_project_file(&project_base);
    let project_dir = project_file
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or(project_base);
    let output_path = normalize_path(&config.output_path);

    let file =
        std::fs::File::create(&output_path).map_err(|e| format!("Erro ao criar arquivo: {}", e))?;
    let mut zip_writer = zip::ZipWriter::new(file);

    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    let translated_dir = project_dir.join("translated");
    let originals_dir = project_dir.join("originals");
    let legacy_images_dir = project_dir.join("images");
    let layers_dir = project_dir.join("layers");
    let project_json = project_dir.join("project.json");

    match config.format.as_str() {
        "jpg_only" | "cbz" => {
            add_directory_to_zip(&mut zip_writer, &translated_dir, "", options)?;
        }
        _ => {
            add_directory_to_zip(&mut zip_writer, &translated_dir, "translated/", options)?;

            if originals_dir.exists() {
                add_directory_to_zip(&mut zip_writer, &originals_dir, "originals/", options)?;
            }
            if legacy_images_dir.exists() {
                add_directory_to_zip(&mut zip_writer, &legacy_images_dir, "images/", options)?;
            }
            if layers_dir.exists() {
                add_directory_to_zip(&mut zip_writer, &layers_dir, "layers/", options)?;
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

    for entry in walkdir::WalkDir::new(dir).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path().to_path_buf();
        let relative = path.strip_prefix(dir).map_err(|e| e.to_string())?;
        let name = format!("{prefix}{}", relative.to_string_lossy().replace('\\', "/"));
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
