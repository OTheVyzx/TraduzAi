use crate::commands::project_schema;
use crate::export::psd::engine_data::{TextEngineSpec, TextJustification, TextOrientation};
use crate::export::psd::{export_psd, PsdLayer};
use chrono::Utc;
use image::{open, GrayImage, ImageBuffer, ImageReader, Luma, RgbaImage};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Write;
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
    let file = app
        .dialog()
        .file()
        .add_filter("Mangá", &["zip", "cbz", "jpg", "jpeg", "png", "webp"])
        .blocking_pick_file();

    Ok(file.map(|f| f.to_string()))
}
#[tauri::command]
pub async fn open_multiple_sources_dialog(app: tauri::AppHandle) -> Result<Vec<String>, String> {
    let files = app
        .dialog()
        .file()
        .add_filter("Mangá", &["zip", "cbz", "jpg", "jpeg", "png", "webp"])
        .blocking_pick_files();

    Ok(files
        .unwrap_or_default()
        .into_iter()
        .map(|f| f.to_string())
        .collect())
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

fn update_bitmap_layer(config: BitmapLayerUpdateConfig, layer_key: &str) -> Result<String, String> {
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
    #[serde(default)]
    pub export_mode: Option<String>,
}

struct ExportQualityBundle {
    status: String,
    files: Vec<(String, Vec<u8>)>,
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

    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    let translated_dir = project_dir.join("translated");
    let originals_dir = project_dir.join("originals");
    let legacy_images_dir = project_dir.join("images");
    let layers_dir = project_dir.join("layers");
    let project_json = project_dir.join("project.json");
    let project_json_data = if project_json.exists() {
        Some(std::fs::read(&project_json).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let project_value = project_json_data
        .as_ref()
        .and_then(|data| serde_json::from_slice::<Value>(data).ok());
    let quality_bundle = if matches!(config.format.as_str(), "zip_full") {
        project_value
            .as_ref()
            .map(|project| build_export_quality_bundle(&project_dir, project, config.export_mode.as_deref()))
            .transpose()?
    } else {
        None
    };

    let file =
        std::fs::File::create(&output_path).map_err(|e| format!("Erro ao criar arquivo: {}", e))?;
    let mut zip_writer = zip::ZipWriter::new(file);

    match config.format.as_str() {
        "jpg_only" | "cbz" => {
            add_directory_to_zip(&mut zip_writer, &translated_dir, "", options)?;
        }
        _ => {
            let mut manifest_files = Vec::new();
            add_directory_to_zip_tracked(
                &mut zip_writer,
                &translated_dir,
                "translated/",
                options,
                Some(&mut manifest_files),
            )?;

            if originals_dir.exists() {
                add_directory_to_zip_tracked(
                    &mut zip_writer,
                    &originals_dir,
                    "originals/",
                    options,
                    Some(&mut manifest_files),
                )?;
            }
            if legacy_images_dir.exists() {
                add_directory_to_zip_tracked(
                    &mut zip_writer,
                    &legacy_images_dir,
                    "images/",
                    options,
                    Some(&mut manifest_files),
                )?;
            }
            if layers_dir.exists() {
                add_directory_to_zip_tracked(
                    &mut zip_writer,
                    &layers_dir,
                    "layers/",
                    options,
                    Some(&mut manifest_files),
                )?;
            }

            if let Some(data) = project_json_data.as_ref() {
                add_file_to_zip_tracked(
                    &mut zip_writer,
                    "project.json",
                    data,
                    options,
                    Some(&mut manifest_files),
                )?;
            }

            if let Some(bundle) = quality_bundle {
                for (path, data) in bundle.files {
                    add_file_to_zip_tracked(
                        &mut zip_writer,
                        &path,
                        &data,
                        options,
                        Some(&mut manifest_files),
                    )?;
                }
                let manifest = json!({
                    "run_id": uuid::Uuid::new_v4().to_string(),
                    "created_at": current_iso_timestamp(),
                    "status": bundle.status,
                    "files": manifest_files,
                });
                let manifest_data =
                    serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
                add_file_to_zip_tracked(
                    &mut zip_writer,
                    "export_manifest.json",
                    &manifest_data,
                    options,
                    None,
                )?;
            }
        }
    }

    zip_writer.finish().map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "path": output_path.to_string_lossy()
    }))
}

#[derive(Debug, Deserialize)]
pub struct ExportPagePsdConfig {
    pub project_path: String,
    pub page_index: usize,
    pub output_path: String,
}

#[tauri::command]
pub async fn export_page_psd(config: ExportPagePsdConfig) -> Result<String, String> {
    let project_base = normalize_path(&config.project_path);
    let project_file = project_schema::resolve_project_file(&project_base);
    let project_dir = project_file
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Caminho do projeto inválido".to_string())?;

    let project = project_schema::load_project_value(&project_file)?;
    let pages = project
        .get("paginas")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "Projeto sem páginas".to_string())?;
    let page = pages
        .get(config.page_index)
        .ok_or_else(|| "Página não encontrada".to_string())?;

    let original_rel = page
        .get("arquivo_original")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Caminho original não encontrado na página".to_string())?;

    // Tenta arquivo_final, se não houver tenta arquivo_traduzido (que costuma ser o final renderizado)
    let inpaint_rel = resolve_inpaint_rel(page);

    let original_path = project_dir.join(original_rel);
    let original_img = ImageReader::open(&original_path)
        .map_err(|e| format!("Falha ao abrir imagem original: {e}"))?
        .decode()
        .map_err(|e| format!("Falha ao decodificar imagem original: {e}"))?
        .to_rgba8();

    let width = original_img.width();
    let height = original_img.height();

    let mut psd_layers = Vec::new();

    // 3. Text Layers (Prep)
    if let Some(text_layers) = page.get("text_layers").and_then(|v| v.as_array()) {
        for (i, layer_val) in text_layers.iter().enumerate() {
            let texto = layer_val
                .get("translated")
                .and_then(|v| v.as_str())
                .or_else(|| layer_val.get("traduzido").and_then(|v| v.as_str()))
                .or_else(|| layer_val.get("texto").and_then(|v| v.as_str()))
                .unwrap_or("");

            let bbox = resolve_text_layer_bbox(layer_val);

            if texto.trim().is_empty() || bbox.is_none() {
                continue;
            }
            let [x1, y1, x2, y2] = bbox.unwrap();
            let w = (x2 - x1).max(1);
            let h = (y2 - y1).max(1);

            let x = x1;
            let y = y1;

            let style = layer_val.get("style").or_else(|| layer_val.get("estilo"));
            let font_size = style
                .and_then(|s| s.get("tamanho").and_then(|v| v.as_f64()))
                .unwrap_or(28.0);
            let color_hex = style
                .and_then(|s| s.get("cor").and_then(|v| v.as_str()))
                .unwrap_or("#FFFFFF");
            let negrito = style
                .and_then(|s| s.get("bold").and_then(|v| v.as_bool()))
                .unwrap_or(false);
            let italico = style
                .and_then(|s| s.get("italico").and_then(|v| v.as_bool()))
                .unwrap_or(false);
            let alinhamento = style
                .and_then(|s| s.get("alinhamento").and_then(|v| v.as_str()))
                .unwrap_or("center");

            let vertical = layer_val
                .get("vertical")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let color = parse_hex_color(color_hex);
            let orientation = if vertical {
                TextOrientation::Vertical
            } else {
                TextOrientation::Horizontal
            };
            let justification = match alinhamento {
                "left" => TextJustification::Left,
                "right" => TextJustification::Right,
                _ => TextJustification::Center,
            };

            let text_pixels = RgbaImage::new(w.max(1) as u32, h.max(1) as u32);

            psd_layers.push(PsdLayer {
                name: format!("Texto {}", i + 1),
                x,
                y,
                pixels: text_pixels,
                hidden: false,
                text_spec: Some(TextEngineSpec {
                    text: texto.to_string(),
                    font_name: "ArialMT".to_string(),
                    font_size,
                    color,
                    faux_bold: negrito,
                    faux_italic: italico,
                    orientation,
                    justification,
                    box_width: w as f64,
                    box_height: h as f64,
                }),
            });
        }
    }

    // 4. Assemble final layers Bottom to Top
    let mut final_psd_layers = Vec::new();

    // Bottom: Original
    final_psd_layers.push(PsdLayer {
        name: "Original".to_string(),
        x: 0,
        y: 0,
        pixels: original_img,
        hidden: false,
        text_spec: None,
    });

    // Layer: Inpaint
    if let Some(rel) = inpaint_rel {
        let path = project_dir.join(rel);
        if path.exists() {
            if let Ok(img) = open(&path) {
                final_psd_layers.push(PsdLayer {
                    name: "Limpeza (Inpaint)".to_string(),
                    x: 0,
                    y: 0,
                    pixels: img.to_rgba8(),
                    hidden: false,
                    text_spec: None,
                });
            }
        }
    }

    // Utility Layers (Hidden)
    if let Some(image_layers) = page.get("image_layers").and_then(|v| v.as_object()) {
        for key in &["mask", "brush"] {
            if let Some(layer_val) = image_layers.get(*key) {
                if let Some(rel_path) = layer_val.get("path").and_then(|v| v.as_str()) {
                    let path = project_dir.join(rel_path);
                    if path.exists() {
                        if let Ok(img) = open(&path) {
                            final_psd_layers.push(PsdLayer {
                                name: if *key == "mask" {
                                    "Máscara de Detecção".to_string()
                                } else {
                                    "Pincel de Edição".to_string()
                                },
                                x: 0,
                                y: 0,
                                pixels: img.to_rgba8(),
                                hidden: true,
                                text_spec: None,
                            });
                        }
                    }
                }
            }
        }
    }

    // Top: Texts
    for layer in psd_layers {
        final_psd_layers.push(layer);
    }

    let psd_bytes = export_psd(width, height, &final_psd_layers)?;

    let output_path = normalize_path(&config.output_path);
    std::fs::write(&output_path, psd_bytes)
        .map_err(|e| format!("Falha ao gravar arquivo PSD: {e}"))?;

    Ok(output_path.to_string_lossy().to_string())
}

fn resolve_inpaint_rel(page: &Value) -> Option<&str> {
    page.get("image_layers")
        .and_then(|v| v.as_object())
        .and_then(|layers| layers.get("inpaint"))
        .and_then(|v| v.get("path"))
        .and_then(|v| v.as_str())
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            page.get("arquivo_final")
                .and_then(|v| v.as_str())
                .filter(|value| !value.trim().is_empty())
        })
}

fn parse_layer_bbox_array(value: Option<&Value>) -> Option<[i32; 4]> {
    let bbox = value?.as_array()?;
    if bbox.len() != 4 {
        return None;
    }

    let mut coords = [0_i32; 4];
    for (index, coord) in bbox.iter().enumerate() {
        let parsed = coord
            .as_i64()
            .or_else(|| coord.as_u64().and_then(|raw| i64::try_from(raw).ok()))?;
        let parsed = i32::try_from(parsed).ok()?;
        coords[index] = parsed;
    }

    if coords[2] <= coords[0] || coords[3] <= coords[1] {
        return None;
    }

    Some(coords)
}

fn resolve_text_layer_bbox(layer_val: &Value) -> Option<[i32; 4]> {
    parse_layer_bbox_array(layer_val.get("render_bbox"))
        .or_else(|| parse_layer_bbox_array(layer_val.get("layout_bbox")))
        .or_else(|| parse_layer_bbox_array(layer_val.get("bbox")))
        .or_else(|| parse_layer_bbox_array(layer_val.get("balloon_bbox")))
}

fn parse_hex_color(hex: &str) -> [u8; 4] {
    let hex = hex.trim_start_matches('#');
    if hex.len() >= 6 {
        let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
        let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
        let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
        [r, g, b, 255]
    } else {
        [0, 0, 0, 255]
    }
}

fn current_iso_timestamp() -> String {
    Utc::now().to_rfc3339()
}

fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn export_flag_label(flag: &str) -> &'static str {
    match flag {
        "critical_error" => "erros criticos",
        "glossary_violation" | "forbidden_term" | "missing_protected_term" => {
            "glossario violado"
        }
        "visual_text_leak" | "page_not_processed" => "ingles restante",
        "ocr_gibberish" | "ocr_suspect" => "ocr suspeito",
        "typesetting_overflow" | "text_too_large" => "texto grande demais",
        "inpaint_suspicious" => "inpaint suspeito",
        "invalid_mask" | "missing_mask" => "mascara ausente",
        _ => "warning",
    }
}

fn export_flag_severity(flag: &str) -> &'static str {
    match flag {
        "critical_error" | "visual_text_leak" | "page_not_processed" => "critical",
        "glossary_violation" | "forbidden_term" | "missing_protected_term" | "invalid_mask"
        | "missing_mask" => "high",
        "ocr_gibberish" | "ocr_suspect" | "typesetting_overflow" | "text_too_large"
        | "inpaint_suspicious" => "medium",
        _ => "low",
    }
}

fn ignored_export_flag(layer: &Value, flag: &str) -> bool {
    layer
        .get("qa_actions")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
        .iter()
        .any(|action| {
            action.get("flag_id").and_then(Value::as_str) == Some(flag)
                && action.get("status").and_then(Value::as_str) == Some("ignored")
        })
}

fn collect_export_issues(project: &Value) -> Vec<Value> {
    let mut issues = Vec::new();
    for (page_index, page) in project
        .get("paginas")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
        .iter()
        .enumerate()
    {
        let page_number = page
            .get("numero")
            .and_then(Value::as_u64)
            .unwrap_or((page_index + 1) as u64);
        for layer in page
            .get("text_layers")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
        {
            let region_id = layer.get("id").and_then(Value::as_str).unwrap_or("region");
            for flag in layer
                .get("qa_flags")
                .and_then(Value::as_array)
                .unwrap_or(&Vec::new())
                .iter()
                .filter_map(Value::as_str)
            {
                if ignored_export_flag(layer, flag) {
                    continue;
                }
                issues.push(json!({
                    "id": format!("{page_index}:{region_id}:{flag}"),
                    "page": page_number,
                    "region_id": region_id,
                    "type": flag,
                    "label": export_flag_label(flag),
                    "severity": export_flag_severity(flag),
                    "source_text": layer.get("original").and_then(Value::as_str).unwrap_or(""),
                    "translated_text": layer
                        .get("traduzido")
                        .and_then(Value::as_str)
                        .or_else(|| layer.get("translated").and_then(Value::as_str))
                        .unwrap_or(""),
                }));
            }
        }
    }
    issues
}

fn collect_user_actions(project: &Value) -> Vec<Value> {
    let mut actions = Vec::new();
    for page in project
        .get("paginas")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        for layer in page
            .get("text_layers")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
        {
            for action in layer
                .get("qa_actions")
                .and_then(Value::as_array)
                .unwrap_or(&Vec::new())
            {
                actions.push(action.clone());
            }
        }
    }
    actions
}

fn collect_glossary_hits(project: &Value) -> Vec<Value> {
    let mut hits = Vec::new();
    for page in project
        .get("paginas")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        for layer in page
            .get("text_layers")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
        {
            if let Some(layer_hits) = layer.get("glossary_hits").and_then(Value::as_array) {
                hits.extend(layer_hits.iter().cloned());
            }
        }
    }
    hits
}

fn collect_ocr_corrections(project: &Value) -> Vec<Value> {
    let mut corrections = Vec::new();
    for page in project
        .get("paginas")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        for layer in page
            .get("text_layers")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
        {
            if layer.get("normalization").is_some()
                || layer.get("raw_ocr").is_some()
                || layer.get("normalized_ocr").is_some()
            {
                corrections.push(json!({
                    "region_id": layer.get("id").and_then(Value::as_str).unwrap_or(""),
                    "raw_text": layer.get("raw_ocr").and_then(Value::as_str).unwrap_or(""),
                    "normalized_text": layer
                        .get("normalized_ocr")
                        .and_then(Value::as_str)
                        .or_else(|| layer.get("original").and_then(Value::as_str))
                        .unwrap_or(""),
                    "normalization": layer.get("normalization").cloned().unwrap_or(Value::Null),
                }));
            }
        }
    }
    corrections
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn build_issues_csv(issues: &[Value]) -> String {
    let mut rows = vec!["id,page,region_id,type,severity,label".to_string()];
    for issue in issues {
        rows.push(format!(
            "{},{},{},{},{},{}",
            csv_escape(issue.get("id").and_then(Value::as_str).unwrap_or("")),
            issue.get("page").and_then(Value::as_u64).unwrap_or(0),
            csv_escape(issue.get("region_id").and_then(Value::as_str).unwrap_or("")),
            csv_escape(issue.get("type").and_then(Value::as_str).unwrap_or("")),
            csv_escape(issue.get("severity").and_then(Value::as_str).unwrap_or("")),
            csv_escape(issue.get("label").and_then(Value::as_str).unwrap_or(""))
        ));
    }
    rows.join("\n")
}

fn build_qa_markdown_report(project: &Value, status: &str, issues: &[Value], user_actions: &[Value]) -> String {
    let critical = issues
        .iter()
        .filter(|issue| issue.get("severity").and_then(Value::as_str) == Some("critical"))
        .count();
    let warnings = issues.len().saturating_sub(critical);
    let title = project.get("obra").and_then(Value::as_str).unwrap_or("Projeto");

    let mut lines = vec![
        format!("# Relatorio QA - {title}"),
        String::new(),
        "## resumo".to_string(),
        format!("- status: {status}"),
        format!("- erros criticos: {critical}"),
        format!("- warnings: {warnings}"),
        String::new(),
        "## erros criticos".to_string(),
    ];
    for issue in issues
        .iter()
        .filter(|issue| issue.get("severity").and_then(Value::as_str) == Some("critical"))
    {
        lines.push(format!(
            "- pagina {} / {}: {}",
            issue.get("page").and_then(Value::as_u64).unwrap_or(0),
            issue.get("region_id").and_then(Value::as_str).unwrap_or(""),
            issue.get("label").and_then(Value::as_str).unwrap_or("")
        ));
    }
    lines.extend([
        String::new(),
        "## warnings".to_string(),
        "## glossario usado".to_string(),
        "## violacoes".to_string(),
        "## correcoes OCR".to_string(),
        "## mascaras invalidas".to_string(),
        "## inpaint suspeito".to_string(),
        "## paginas bloqueadas".to_string(),
        "## acoes do usuario".to_string(),
    ]);
    for action in user_actions {
        lines.push(format!(
            "- {}: {}",
            action.get("flag_id").and_then(Value::as_str).unwrap_or("flag"),
            action
                .get("ignored_reason")
                .and_then(Value::as_str)
                .unwrap_or("sem motivo")
        ));
    }
    lines.push(String::new());
    lines.join("\n")
}

fn build_export_quality_bundle(
    project_dir: &Path,
    project: &Value,
    export_mode: Option<&str>,
) -> Result<ExportQualityBundle, String> {
    let mode = export_mode.unwrap_or("with_warnings");
    let issues = collect_export_issues(project);
    let critical_count = issues
        .iter()
        .filter(|issue| issue.get("severity").and_then(Value::as_str) == Some("critical"))
        .count();
    let high_count = issues
        .iter()
        .filter(|issue| issue.get("severity").and_then(Value::as_str) == Some("high"))
        .count();

    if mode == "clean" && critical_count + high_count > 0 {
        return Err("Export clean bloqueado: ha flags critical/high ativas.".to_string());
    }
    if mode != "debug" && critical_count > 0 {
        return Err("Export bloqueado: ha flags critical ativas. Use modo debug.".to_string());
    }

    let status = if critical_count > 0 {
        "blocked_debug_export"
    } else if issues.is_empty() {
        "clean"
    } else {
        "with_warnings"
    }
    .to_string();

    let user_actions = collect_user_actions(project);
    let qa_report_json = json!({
        "summary": {
            "status": status,
            "total": issues.len(),
            "critical": critical_count,
            "high": high_count,
        },
        "issues": issues,
        "user_actions": user_actions,
    });
    let glossary_used = collect_glossary_hits(project);
    let ocr_corrections = collect_ocr_corrections(project);
    let issues_ref = qa_report_json
        .get("issues")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let actions_ref = qa_report_json
        .get("user_actions")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let markdown = build_qa_markdown_report(project, &status, &issues_ref, &actions_ref);
    let structured_log = std::fs::read(project_dir.join("structured_log.jsonl"))
        .unwrap_or_else(|_| b"{\"event\":\"export\",\"status\":\"generated\"}\n".to_vec());

    Ok(ExportQualityBundle {
        status,
        files: vec![
            ("qa_report.md".to_string(), markdown.into_bytes()),
            (
                "qa_report.json".to_string(),
                serde_json::to_vec_pretty(&qa_report_json).map_err(|e| e.to_string())?,
            ),
            ("issues.csv".to_string(), build_issues_csv(&issues_ref).into_bytes()),
            (
                "glossary_used.json".to_string(),
                serde_json::to_vec_pretty(&glossary_used).map_err(|e| e.to_string())?,
            ),
            (
                "ocr_corrections.json".to_string(),
                serde_json::to_vec_pretty(&ocr_corrections).map_err(|e| e.to_string())?,
            ),
            ("structured_log.jsonl".to_string(), structured_log),
        ],
    })
}

fn add_file_to_zip_tracked(
    zip_writer: &mut zip::ZipWriter<std::fs::File>,
    name: &str,
    data: &[u8],
    options: zip::write::SimpleFileOptions,
    mut manifest: Option<&mut Vec<Value>>,
) -> Result<(), String> {
    zip_writer
        .start_file(name, options)
        .map_err(|e| e.to_string())?;
    zip_writer.write_all(data).map_err(|e| e.to_string())?;
    if let Some(entries) = manifest.as_deref_mut() {
        entries.push(json!({
            "path": name,
            "sha256": sha256_hex(data),
        }));
    }
    Ok(())
}

fn add_directory_to_zip(
    zip_writer: &mut zip::ZipWriter<std::fs::File>,
    dir: &PathBuf,
    prefix: &str,
    options: zip::write::SimpleFileOptions,
) -> Result<(), String> {
    add_directory_to_zip_tracked(zip_writer, dir, prefix, options, None)
}

fn add_directory_to_zip_tracked(
    zip_writer: &mut zip::ZipWriter<std::fs::File>,
    dir: &PathBuf,
    prefix: &str,
    options: zip::write::SimpleFileOptions,
    mut manifest: Option<&mut Vec<Value>>,
) -> Result<(), String> {
    if !dir.exists() {
        return Ok(());
    }

    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path().to_path_buf();
        let relative = path.strip_prefix(dir).map_err(|e| e.to_string())?;
        let name = format!("{prefix}{}", relative.to_string_lossy().replace('\\', "/"));
        let data = std::fs::read(&path).map_err(|e| e.to_string())?;
        add_file_to_zip_tracked(zip_writer, &name, &data, options, manifest.as_deref_mut())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    const WEBP_2X2_RED: &[u8] = &[
        82, 73, 70, 70, 60, 0, 0, 0, 87, 69, 66, 80, 86, 80, 56, 32, 48, 0, 0, 0, 208, 1, 0, 157,
        1, 42, 2, 0, 2, 0, 1, 64, 38, 37, 160, 2, 116, 186, 1, 248, 0, 3, 176, 0, 254, 242, 235,
        127, 252, 216, 21, 205, 115, 239, 247, 255, 210, 224, 253, 46, 15, 210, 224, 255, 210, 144,
        0, 0,
    ];

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

    fn zip_entry_string(zip_path: &PathBuf, name: &str) -> String {
        let file = std::fs::File::open(zip_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut contents = String::new();
        archive
            .by_name(name)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        contents
    }

    fn quality_project_json(flags: Vec<&str>) -> String {
        let layers: Vec<Value> = flags
            .iter()
            .map(|flag| {
                json!({
                    "id": format!("region-{flag}"),
                    "original": "YOUNG MASTER?!",
                    "traduzido": "Jovem mestre?!",
                    "qa_flags": [flag],
                    "qa_actions": []
                })
            })
            .collect();
        json!({
            "obra": "Teste",
            "capitulo": 1,
            "paginas": [{
                "numero": 1,
                "text_layers": layers,
                "textos": layers,
                "image_layers": {
                    "mask": {"path": "layers/masks/001.png"}
                }
            }],
            "qa": {"summary": {"total": flags.len()}},
            "log": {"summary": {"qa_flags": flags.len()}}
        })
        .to_string()
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
            export_mode: None,
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
            export_mode: None,
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

    #[tokio::test]
    async fn export_zip_full_includes_quality_reports_and_manifest() {
        let root = unique_temp_dir();
        let project_dir = root.join("project");
        std::fs::create_dir_all(project_dir.join("translated")).unwrap();
        std::fs::create_dir_all(project_dir.join("layers").join("masks")).unwrap();

        write_file(
            &project_dir.join("translated").join("001.jpg"),
            b"translated",
        );
        write_file(&project_dir.join("layers").join("masks").join("001.png"), b"mask");
        write_file(&project_dir.join("structured_log.jsonl"), br#"{"event":"done"}"#);
        write_file(
            &project_dir.join("project.json"),
            quality_project_json(vec!["ocr_gibberish"]).as_bytes(),
        );

        let output = root.join("quality.zip");
        export_project(ExportConfig {
            project_path: project_dir.to_string_lossy().to_string(),
            format: "zip_full".to_string(),
            output_path: output.to_string_lossy().to_string(),
            export_mode: Some("with_warnings".to_string()),
        })
        .await
        .unwrap();

        let entries = zip_entries(&output);
        for required in [
            "project.json",
            "qa_report.md",
            "qa_report.json",
            "issues.csv",
            "glossary_used.json",
            "ocr_corrections.json",
            "export_manifest.json",
            "structured_log.jsonl",
            "layers/masks/001.png",
        ] {
            assert!(entries.contains(&required.to_string()), "missing {required}");
        }

        let manifest: Value =
            serde_json::from_str(&zip_entry_string(&output, "export_manifest.json")).unwrap();
        assert_eq!(manifest["status"], "with_warnings");
        let project_sha = manifest["files"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["path"] == "project.json")
            .and_then(|entry| entry["sha256"].as_str())
            .unwrap();
        assert_eq!(project_sha.len(), 64);

        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn export_clean_blocks_critical_flags() {
        let root = unique_temp_dir();
        let project_dir = root.join("project");
        std::fs::create_dir_all(project_dir.join("translated")).unwrap();
        write_file(
            &project_dir.join("translated").join("001.jpg"),
            b"translated",
        );
        write_file(
            &project_dir.join("project.json"),
            quality_project_json(vec!["visual_text_leak"]).as_bytes(),
        );

        let output = root.join("blocked.zip");
        let err = export_project(ExportConfig {
            project_path: project_dir.to_string_lossy().to_string(),
            format: "zip_full".to_string(),
            output_path: output.to_string_lossy().to_string(),
            export_mode: Some("clean".to_string()),
        })
        .await
        .unwrap_err();

        assert!(err.contains("critical"));
        assert!(!output.exists());
        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn export_debug_allows_critical_with_manifest_marking() {
        let root = unique_temp_dir();
        let project_dir = root.join("project");
        std::fs::create_dir_all(project_dir.join("translated")).unwrap();
        write_file(
            &project_dir.join("translated").join("001.jpg"),
            b"translated",
        );
        write_file(
            &project_dir.join("project.json"),
            quality_project_json(vec!["visual_text_leak"]).as_bytes(),
        );

        let output = root.join("debug.zip");
        export_project(ExportConfig {
            project_path: project_dir.to_string_lossy().to_string(),
            format: "zip_full".to_string(),
            output_path: output.to_string_lossy().to_string(),
            export_mode: Some("debug".to_string()),
        })
        .await
        .unwrap();

        let manifest: Value =
            serde_json::from_str(&zip_entry_string(&output, "export_manifest.json")).unwrap();
        assert_eq!(manifest["status"], "blocked_debug_export");

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    #[cfg(windows)]
    fn normalize_path_strips_file_uri_prefix_on_windows() {
        let path = normalize_path("file:///C:/traduzai/teste/project.json");
        assert_eq!(path, PathBuf::from("C:/traduzai/teste/project.json"));
    }

    #[test]
    fn resolve_inpaint_rel_prefers_image_layer_over_rendered_output() {
        let page = serde_json::json!({
            "arquivo_traduzido": "translated/001.jpg",
            "arquivo_final": "translated/001.jpg",
            "image_layers": {
                "inpaint": {
                    "path": "images/001.jpg"
                }
            }
        });

        assert_eq!(resolve_inpaint_rel(&page), Some("images/001.jpg"));
    }

    #[test]
    fn resolve_inpaint_rel_falls_back_to_legacy_final_path() {
        let page = serde_json::json!({
            "arquivo_final": "images/001.jpg"
        });

        assert_eq!(resolve_inpaint_rel(&page), Some("images/001.jpg"));
    }

    #[test]
    fn resolve_text_layer_bbox_prefers_render_bbox_over_layout_bbox() {
        let layer = serde_json::json!({
            "render_bbox": [120, 240, 420, 520],
            "layout_bbox": [100, 200, 500, 700],
            "bbox": [90, 180, 530, 740],
            "balloon_bbox": [80, 170, 540, 760]
        });

        assert_eq!(resolve_text_layer_bbox(&layer), Some([120, 240, 420, 520]));
    }

    #[test]
    fn resolve_text_layer_bbox_falls_back_when_render_bbox_is_missing_or_invalid() {
        let invalid_render = serde_json::json!({
            "render_bbox": [300, 300, 300, 450],
            "layout_bbox": [100, 200, 500, 700],
            "bbox": [90, 180, 530, 740]
        });
        assert_eq!(
            resolve_text_layer_bbox(&invalid_render),
            Some([100, 200, 500, 700])
        );

        let without_render = serde_json::json!({
            "bbox": [90, 180, 530, 740],
            "balloon_bbox": [80, 170, 540, 760]
        });
        assert_eq!(
            resolve_text_layer_bbox(&without_render),
            Some([90, 180, 530, 740])
        );

        let only_balloon = serde_json::json!({
            "balloon_bbox": [80, 170, 540, 760]
        });
        assert_eq!(
            resolve_text_layer_bbox(&only_balloon),
            Some([80, 170, 540, 760])
        );
    }

    #[tokio::test]
    async fn export_page_psd_supports_webp_original() {
        let root = unique_temp_dir();
        let project_dir = root.join("project");
        std::fs::create_dir_all(project_dir.join("originals")).unwrap();

        write_file(
            &project_dir.join("originals").join("001.webp"),
            WEBP_2X2_RED,
        );
        write_file(
            &project_dir.join("project.json"),
            br#"{
                "obra":"Teste",
                "paginas":[
                    {
                        "numero":1,
                        "arquivo_original":"originals/001.webp"
                    }
                ]
            }"#,
        );

        let output = root.join("page.psd");
        let result = export_page_psd(ExportPagePsdConfig {
            project_path: project_dir.to_string_lossy().to_string(),
            page_index: 0,
            output_path: output.to_string_lossy().to_string(),
        })
        .await;

        assert!(result.is_ok(), "resultado inesperado: {result:?}");
        let bytes = std::fs::read(&output).unwrap();
        assert!(bytes.starts_with(b"8BPS"));

        std::fs::remove_dir_all(root).ok();
    }
}
