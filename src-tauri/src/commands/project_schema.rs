use serde_json::{json, Map, Value};
use std::fs;
use std::path::{Path, PathBuf};

pub const PROJECT_VERSION_V2: &str = "2.0";

fn default_text_style() -> Value {
    json!({
        "fonte": "CCDaveGibbonsLower W00 Regular.ttf",
        "tamanho": 28,
        "cor": "#FFFFFF",
        "cor_gradiente": [],
        "contorno": "#000000",
        "contorno_px": 2,
        "glow": false,
        "glow_cor": "",
        "glow_px": 0,
        "sombra": false,
        "sombra_cor": "",
        "sombra_offset": [0, 0],
        "bold": false,
        "italico": false,
        "rotacao": 0,
        "alinhamento": "center",
        "force_upper": false
    })
}

fn as_object_mut<'a>(value: &'a mut Value, context: &str) -> Result<&'a mut Map<String, Value>, String> {
    value
        .as_object_mut()
        .ok_or_else(|| format!("{context} precisa ser um objeto JSON"))
}

fn as_array_mut<'a>(value: &'a mut Value, context: &str) -> Result<&'a mut Vec<Value>, String> {
    value
        .as_array_mut()
        .ok_or_else(|| format!("{context} precisa ser uma lista JSON"))
}

fn get_bbox(value: Option<&Value>) -> Option<Vec<Value>> {
    value.and_then(|bbox| {
        bbox.as_array().map(|items| {
            items.iter()
                .take(4)
                .map(|item| Value::from(item.as_i64().unwrap_or_default()))
                .collect::<Vec<_>>()
        })
    })
}

fn page_number(page: &Map<String, Value>, page_index: usize) -> usize {
    page.get("numero")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .unwrap_or(page_index + 1)
}

fn infer_inpaint_path(page: &Map<String, Value>) -> Option<String> {
    if let Some(path) = page
        .get("image_layers")
        .and_then(|layers| layers.get("inpaint"))
        .and_then(|layer| layer.get("path"))
        .and_then(|path| path.as_str())
    {
        return Some(path.to_string());
    }

    let original = page.get("arquivo_original").and_then(|value| value.as_str())?;
    let filename = Path::new(original).file_name()?.to_string_lossy();
    Some(format!("images/{filename}"))
}

fn normalize_text_layer_object(
    layer: &mut Value,
    page_num: usize,
    layer_index: usize,
) -> Result<(), String> {
    let layer_obj = as_object_mut(layer, "text_layer")?;

    let legacy_bbox = get_bbox(layer_obj.get("bbox"));
    let source_bbox = get_bbox(layer_obj.get("source_bbox")).or_else(|| legacy_bbox.clone());
    let layout_bbox = get_bbox(layer_obj.get("layout_bbox"))
        .or_else(|| source_bbox.clone())
        .or_else(|| legacy_bbox.clone())
        .unwrap_or_else(|| vec![0.into(), 0.into(), 32.into(), 32.into()]);

    let default_id = format!("tl_{page_num:03}_{:03}", layer_index + 1);
    let original = layer_obj
        .get("original")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
        .or_else(|| layer_obj.get("text").and_then(|value| value.as_str()).map(ToOwned::to_owned))
        .unwrap_or_default();
    let translated = layer_obj
        .get("translated")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
        .or_else(|| {
            layer_obj
                .get("traduzido")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
        })
        .unwrap_or_default();
    let ocr_confidence = layer_obj
        .get("ocr_confidence")
        .and_then(|value| value.as_f64())
        .or_else(|| layer_obj.get("ocr_confidence").and_then(|value| value.as_i64()).map(|v| v as f64))
        .or_else(|| layer_obj.get("confianca_ocr").and_then(|value| value.as_f64()))
        .or_else(|| {
            layer_obj
                .get("confidence")
                .and_then(|value| value.as_f64())
        })
        .unwrap_or(0.0);
    let style = layer_obj
        .get("style")
        .cloned()
        .or_else(|| layer_obj.get("estilo").cloned())
        .unwrap_or_else(default_text_style);

    layer_obj.insert("id".into(), Value::from(layer_obj.get("id").and_then(|value| value.as_str()).unwrap_or(&default_id)));
    layer_obj.insert("kind".into(), Value::from("text"));
    layer_obj.insert("source_bbox".into(), Value::Array(source_bbox.unwrap_or_else(|| layout_bbox.clone())));
    layer_obj.insert("layout_bbox".into(), Value::Array(layout_bbox.clone()));
    if !layer_obj.contains_key("render_bbox") {
        layer_obj.insert("render_bbox".into(), Value::Null);
    }
    layer_obj.insert("original".into(), Value::from(original));
    layer_obj.insert("translated".into(), Value::from(translated));
    layer_obj.insert("ocr_confidence".into(), Value::from(ocr_confidence));
    layer_obj.insert("style".into(), style);
    layer_obj.insert(
        "visible".into(),
        Value::from(layer_obj.get("visible").and_then(|value| value.as_bool()).unwrap_or(true)),
    );
    layer_obj.insert(
        "locked".into(),
        Value::from(layer_obj.get("locked").and_then(|value| value.as_bool()).unwrap_or(false)),
    );
    layer_obj.insert(
        "order".into(),
        Value::from(layer_obj.get("order").and_then(|value| value.as_i64()).unwrap_or(layer_index as i64)),
    );
    layer_obj.insert(
        "tipo".into(),
        Value::from(layer_obj.get("tipo").and_then(|value| value.as_str()).unwrap_or("fala")),
    );
    if !layer_obj.contains_key("render_preview_path") {
        layer_obj.insert("render_preview_path".into(), Value::Null);
    }
    if !layer_obj.contains_key("detector") {
        layer_obj.insert("detector".into(), Value::Null);
    }
    if !layer_obj.contains_key("line_polygons") {
        layer_obj.insert("line_polygons".into(), Value::Null);
    }
    if !layer_obj.contains_key("source_direction") {
        layer_obj.insert("source_direction".into(), Value::Null);
    }
    if !layer_obj.contains_key("rendered_direction") {
        layer_obj.insert("rendered_direction".into(), Value::Null);
    }
    if !layer_obj.contains_key("source_language") {
        layer_obj.insert("source_language".into(), Value::Null);
    }
    if !layer_obj.contains_key("rotation_deg") {
        layer_obj.insert("rotation_deg".into(), Value::from(0));
    }
    if !layer_obj.contains_key("detected_font_size_px") {
        layer_obj.insert("detected_font_size_px".into(), Value::Null);
    }
    if !layer_obj.contains_key("balloon_bbox") {
        layer_obj.insert("balloon_bbox".into(), Value::Array(layout_bbox));
    }
    if !layer_obj.contains_key("balloon_subregions") {
        layer_obj.insert("balloon_subregions".into(), Value::Array(Vec::new()));
    }

    Ok(())
}

fn ensure_image_layer_entry(
    layers_obj: &mut Map<String, Value>,
    key: &str,
    path: Option<String>,
    visible: bool,
    locked: bool,
) {
    let existing = layers_obj.entry(key.to_string()).or_insert_with(|| json!({}));
    let entry = existing.as_object_mut().unwrap();
    entry.insert("key".into(), Value::from(key));
    if !entry.contains_key("path") {
        entry.insert(
            "path".into(),
            path.clone().map(Value::from).unwrap_or(Value::Null),
        );
    } else if entry.get("path").is_some_and(Value::is_null) && path.is_some() {
        entry.insert("path".into(), Value::from(path.unwrap()));
    }
    entry.insert(
        "visible".into(),
        Value::from(entry.get("visible").and_then(|value| value.as_bool()).unwrap_or(visible)),
    );
    entry.insert(
        "locked".into(),
        Value::from(entry.get("locked").and_then(|value| value.as_bool()).unwrap_or(locked)),
    );
}

fn sync_legacy_aliases(page: &mut Value) -> Result<(), String> {
    let page_obj = as_object_mut(page, "pagina")?;
    let image_layers = page_obj
        .entry("image_layers")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "image_layers precisa ser objeto".to_string())?;

    let base_path = image_layers
        .get("base")
        .and_then(|layer| layer.get("path"))
        .cloned()
        .unwrap_or(Value::Null);
    let rendered_path = image_layers
        .get("rendered")
        .and_then(|layer| layer.get("path"))
        .cloned()
        .unwrap_or(Value::Null);

    page_obj.insert("arquivo_original".into(), base_path);
    page_obj.insert("arquivo_traduzido".into(), rendered_path);

    let text_layers = page_obj
        .entry("text_layers")
        .or_insert_with(|| Value::Array(Vec::new()));
    let layers = as_array_mut(text_layers, "text_layers")?;

    layers.sort_by_key(|layer| {
        layer
            .get("order")
            .and_then(|value| value.as_i64())
            .unwrap_or_default()
    });

    for (index, layer) in layers.iter_mut().enumerate() {
        if let Some(layer_obj) = layer.as_object_mut() {
            layer_obj.insert("order".into(), Value::from(index as i64));
        }
    }

    let legacy_texts = layers
        .iter()
        .map(|layer| {
            let bbox = layer
                .get("layout_bbox")
                .cloned()
                .or_else(|| layer.get("source_bbox").cloned())
                .unwrap_or_else(|| json!([0, 0, 32, 32]));
            json!({
                "id": layer.get("id").cloned().unwrap_or(Value::Null),
                "bbox": bbox,
                "tipo": layer.get("tipo").cloned().unwrap_or_else(|| Value::from("fala")),
                "original": layer.get("original").cloned().unwrap_or_else(|| Value::from("")),
                "traduzido": layer.get("translated").cloned().unwrap_or_else(|| Value::from("")),
                "confianca_ocr": layer.get("ocr_confidence").cloned().unwrap_or_else(|| Value::from(0.0)),
                "estilo": layer.get("style").cloned().unwrap_or_else(default_text_style),
            })
        })
        .collect::<Vec<_>>();

    page_obj.insert("textos".into(), Value::Array(legacy_texts));
    Ok(())
}

fn migrate_page(page: &mut Value, page_index: usize) -> Result<(), String> {
    let page_obj = as_object_mut(page, "pagina")?;
    let number = page_number(page_obj, page_index);
    page_obj.insert(
        "numero".into(),
        Value::from(page_obj.get("numero").and_then(|value| value.as_u64()).unwrap_or(number as u64)),
    );

    let original_path = page_obj
        .get("arquivo_original")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
        .or_else(|| {
            page_obj
                .get("image_layers")
                .and_then(|layers| layers.get("base"))
                .and_then(|layer| layer.get("path"))
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
        });
    let rendered_path = page_obj
        .get("arquivo_traduzido")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
        .or_else(|| {
            page_obj
                .get("image_layers")
                .and_then(|layers| layers.get("rendered"))
                .and_then(|layer| layer.get("path"))
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
        });

    let inferred_inpaint_path = infer_inpaint_path(page_obj);
    let image_layers = page_obj
        .entry("image_layers")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "image_layers precisa ser objeto".to_string())?;

    ensure_image_layer_entry(image_layers, "base", original_path, true, true);
    ensure_image_layer_entry(image_layers, "mask", None, false, false);
    ensure_image_layer_entry(image_layers, "inpaint", inferred_inpaint_path, false, true);
    ensure_image_layer_entry(image_layers, "brush", None, false, false);
    ensure_image_layer_entry(image_layers, "rendered", rendered_path, true, true);

    let legacy_texts = page_obj
        .get("textos")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let text_layers = page_obj
        .entry("text_layers")
        .or_insert_with(|| Value::Array(Vec::new()));
    let layers = as_array_mut(text_layers, "text_layers")?;

    if layers.is_empty() && !legacy_texts.is_empty() {
        for (layer_index, legacy_text) in legacy_texts.iter().enumerate() {
            let bbox = get_bbox(legacy_text.get("bbox")).unwrap_or_else(|| vec![0.into(), 0.into(), 32.into(), 32.into()]);
            let style = legacy_text
                .get("estilo")
                .cloned()
                .unwrap_or_else(default_text_style);
            layers.push(json!({
                "id": legacy_text
                    .get("id")
                    .and_then(|value| value.as_str())
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| format!("tl_{number:03}_{:03}", layer_index + 1)),
                "kind": "text",
                "source_bbox": bbox,
                "layout_bbox": legacy_text.get("bbox").cloned().unwrap_or_else(|| json!([0, 0, 32, 32])),
                "render_bbox": Value::Null,
                "original": legacy_text.get("original").cloned().unwrap_or_else(|| Value::from("")),
                "translated": legacy_text.get("traduzido").cloned().unwrap_or_else(|| Value::from("")),
                "ocr_confidence": legacy_text.get("confianca_ocr").cloned().unwrap_or_else(|| Value::from(0.0)),
                "style": style,
                "visible": true,
                "locked": false,
                "order": layer_index,
                "tipo": legacy_text.get("tipo").cloned().unwrap_or_else(|| Value::from("fala")),
                "render_preview_path": Value::Null,
                "detector": Value::Null,
                "line_polygons": Value::Null,
                "source_direction": Value::Null,
                "rendered_direction": Value::Null,
                "source_language": Value::Null,
                "rotation_deg": 0,
                "detected_font_size_px": Value::Null,
                "balloon_bbox": legacy_text.get("bbox").cloned().unwrap_or_else(|| json!([0, 0, 32, 32])),
                "balloon_subregions": [],
            }));
        }
    }

    for (layer_index, layer) in layers.iter_mut().enumerate() {
        normalize_text_layer_object(layer, number, layer_index)?;
    }

    sync_legacy_aliases(page)?;
    Ok(())
}

pub fn migrate_project_value(project: &mut Value) -> Result<(), String> {
    let project_obj = as_object_mut(project, "project")?;
    project_obj.insert("versao".into(), Value::from(PROJECT_VERSION_V2));
    if !project_obj.contains_key("app") {
        project_obj.insert("app".into(), Value::from("traduzai"));
    }
    let pages = project_obj
        .entry("paginas")
        .or_insert_with(|| Value::Array(Vec::new()));
    let page_list = as_array_mut(pages, "paginas")?;
    for (page_index, page) in page_list.iter_mut().enumerate() {
        migrate_page(page, page_index)?;
    }

    let total_pages = page_list.len();
    let total_texts = page_list
        .iter()
        .map(|page| {
            page.get("text_layers")
                .and_then(|value| value.as_array())
                .map(Vec::len)
                .unwrap_or_default()
        })
        .sum::<usize>();

    let stats = project_obj
        .entry("estatisticas")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "estatisticas precisa ser objeto".to_string())?;
    stats.insert("total_paginas".into(), Value::from(total_pages as u64));
    stats.insert("total_textos".into(), Value::from(total_texts as u64));

    Ok(())
}

pub fn load_project_value(project_file: &Path) -> Result<Value, String> {
    let content =
        fs::read_to_string(project_file).map_err(|e| format!("Erro ao ler project.json: {e}"))?;
    let mut project: Value =
        serde_json::from_str(&content).map_err(|e| format!("JSON inválido: {e}"))?;
    migrate_project_value(&mut project)?;
    Ok(project)
}

pub fn save_project_value(project_file: &Path, project: &mut Value) -> Result<(), String> {
    migrate_project_value(project)?;
    let content = serde_json::to_string_pretty(project)
        .map_err(|e| format!("Erro ao serializar JSON: {e}"))?;
    if let Some(parent) = project_file.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Erro ao preparar diretório do projeto: {e}"))?;
    }
    let temp_file = project_file.with_extension("json.tmp");
    fs::write(&temp_file, content)
        .map_err(|e| format!("Erro ao salvar project.json temporário: {e}"))?;
    if project_file.exists() {
        fs::remove_file(project_file)
            .map_err(|e| format!("Erro ao substituir project.json anterior: {e}"))?;
    }
    fs::rename(&temp_file, project_file)
        .map_err(|e| format!("Erro ao finalizar gravação atômica do project.json: {e}"))?;
    Ok(())
}

pub fn default_bitmap_layer_path(page_number: usize, layer_key: &str) -> Option<String> {
    let page_name = format!("{page_number:03}.png");
    match layer_key {
        "mask" => Some(format!("layers/mask/{page_name}")),
        "brush" => Some(format!("layers/brush/{page_name}")),
        _ => None,
    }
}

pub fn ensure_bitmap_layer_path(
    project: &mut Value,
    page_index: usize,
    layer_key: &str,
) -> Result<String, String> {
    migrate_project_value(project)?;
    let project_obj = as_object_mut(project, "project")?;
    let pages = project_obj
        .get_mut("paginas")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| "Projeto sem páginas".to_string())?;
    let page = pages
        .get_mut(page_index)
        .ok_or_else(|| "Página inválida".to_string())?;
    let page_obj = as_object_mut(page, "pagina")?;
    let number = page_number(page_obj, page_index);
    let fallback = default_bitmap_layer_path(number, layer_key)
        .ok_or_else(|| format!("Layer bitmap não suportada: {layer_key}"))?;
    let image_layers = page_obj
        .get_mut("image_layers")
        .and_then(|value| value.as_object_mut())
        .ok_or_else(|| "image_layers inválido".to_string())?;
    let entry = image_layers
        .entry(layer_key.to_string())
        .or_insert_with(|| json!({}));
    let entry_obj = entry
        .as_object_mut()
        .ok_or_else(|| format!("image_layers.{layer_key} inválido"))?;
    let path = entry_obj
        .get("path")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
        .unwrap_or(fallback);
    entry_obj.insert("key".into(), Value::from(layer_key));
    entry_obj.insert("path".into(), Value::from(path.clone()));
    entry_obj.insert("visible".into(), Value::from(entry_obj.get("visible").and_then(|value| value.as_bool()).unwrap_or(false)));
    entry_obj.insert("locked".into(), Value::from(entry_obj.get("locked").and_then(|value| value.as_bool()).unwrap_or(false)));
    Ok(path)
}

pub fn create_text_layer(
    project: &mut Value,
    page_index: usize,
    layout_bbox: [i64; 4],
) -> Result<Value, String> {
    migrate_project_value(project)?;
    let project_obj = as_object_mut(project, "project")?;
    let pages = project_obj
        .get_mut("paginas")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| "Projeto sem páginas".to_string())?;
    let page = pages
        .get_mut(page_index)
        .ok_or_else(|| "Página inválida".to_string())?;
    let page_obj = as_object_mut(page, "pagina")?;
    let number = page_number(page_obj, page_index);
    let layers = page_obj
        .get_mut("text_layers")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| "text_layers inválido".to_string())?;

    let order = layers.len();
    let layer = json!({
        "id": format!("tl_{number:03}_{:03}", order + 1),
        "kind": "text",
        "source_bbox": layout_bbox,
        "layout_bbox": layout_bbox,
        "render_bbox": Value::Null,
        "original": "",
        "translated": "",
        "ocr_confidence": 1.0,
        "style": default_text_style(),
        "visible": true,
        "locked": false,
        "order": order,
        "tipo": "fala",
        "render_preview_path": Value::Null,
        "detector": Value::Null,
        "line_polygons": Value::Null,
        "source_direction": Value::Null,
        "rendered_direction": Value::Null,
        "source_language": Value::Null,
        "rotation_deg": 0,
        "detected_font_size_px": Value::Null,
        "balloon_bbox": layout_bbox,
        "balloon_subregions": [],
    });
    layers.push(layer.clone());
    sync_legacy_aliases(page)?;
    Ok(layer)
}

pub fn patch_text_layer(
    project: &mut Value,
    page_index: usize,
    layer_id: &str,
    patch: &Value,
) -> Result<Value, String> {
    migrate_project_value(project)?;
    let project_obj = as_object_mut(project, "project")?;
    let pages = project_obj
        .get_mut("paginas")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| "Projeto sem páginas".to_string())?;
    let page = pages
        .get_mut(page_index)
        .ok_or_else(|| "Página inválida".to_string())?;
    let page_obj = as_object_mut(page, "pagina")?;
    let number = page_number(page_obj, page_index);
    let layers = page_obj
        .get_mut("text_layers")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| "text_layers inválido".to_string())?;

    let patch_obj = patch
        .as_object()
        .ok_or_else(|| "Patch da text layer precisa ser objeto".to_string())?;

    let mut updated = None;
    for (layer_index, layer) in layers.iter_mut().enumerate() {
        if layer
            .get("id")
            .and_then(|value| value.as_str())
            .is_some_and(|id| id == layer_id)
        {
            let layer_obj = as_object_mut(layer, "text_layer")?;
            for (key, value) in patch_obj {
                if key == "style" && value.is_object() {
                    let style = layer_obj.entry("style").or_insert_with(default_text_style);
                    let style_obj = style
                        .as_object_mut()
                        .ok_or_else(|| "style inválido".to_string())?;
                    for (style_key, style_value) in value.as_object().unwrap() {
                        style_obj.insert(style_key.clone(), style_value.clone());
                    }
                } else {
                    layer_obj.insert(key.clone(), value.clone());
                }
            }
            normalize_text_layer_object(layer, number, layer_index)?;
            updated = Some(layer.clone());
            break;
        }
    }

    let updated = updated.ok_or_else(|| format!("Text layer não encontrada: {layer_id}"))?;
    sync_legacy_aliases(page)?;
    Ok(updated)
}

pub fn delete_text_layer(
    project: &mut Value,
    page_index: usize,
    layer_id: &str,
) -> Result<(), String> {
    migrate_project_value(project)?;
    let project_obj = as_object_mut(project, "project")?;
    let pages = project_obj
        .get_mut("paginas")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| "Projeto sem páginas".to_string())?;
    let page = pages
        .get_mut(page_index)
        .ok_or_else(|| "Página inválida".to_string())?;
    let page_obj = as_object_mut(page, "pagina")?;
    let layers = page_obj
        .get_mut("text_layers")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| "text_layers inválido".to_string())?;

    let before = layers.len();
    layers.retain(|layer| {
        layer
            .get("id")
            .and_then(|value| value.as_str())
            .is_none_or(|id| id != layer_id)
    });
    if layers.len() == before {
        return Err(format!("Text layer não encontrada: {layer_id}"));
    }

    sync_legacy_aliases(page)?;
    Ok(())
}

pub fn set_layer_visibility(
    project: &mut Value,
    page_index: usize,
    layer_kind: &str,
    layer_key: Option<&str>,
    layer_id: Option<&str>,
    visible: bool,
) -> Result<(), String> {
    migrate_project_value(project)?;
    let project_obj = as_object_mut(project, "project")?;
    let pages = project_obj
        .get_mut("paginas")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| "Projeto sem páginas".to_string())?;
    let page = pages
        .get_mut(page_index)
        .ok_or_else(|| "Página inválida".to_string())?;
    let page_obj = as_object_mut(page, "pagina")?;

    match layer_kind {
        "image" => {
            let key = layer_key.ok_or_else(|| "layer_key é obrigatório para image layer".to_string())?;
            let image_layers = page_obj
                .get_mut("image_layers")
                .and_then(|value| value.as_object_mut())
                .ok_or_else(|| "image_layers inválido".to_string())?;
            let entry = image_layers
                .get_mut(key)
                .and_then(|value| value.as_object_mut())
                .ok_or_else(|| format!("Image layer não encontrada: {key}"))?;
            entry.insert("visible".into(), Value::from(visible));
        }
        "text" => {
            let id = layer_id.ok_or_else(|| "layer_id é obrigatório para text layer".to_string())?;
            let layers = page_obj
                .get_mut("text_layers")
                .and_then(|value| value.as_array_mut())
                .ok_or_else(|| "text_layers inválido".to_string())?;
            let mut found = false;
            for layer in layers.iter_mut() {
                if layer
                    .get("id")
                    .and_then(|value| value.as_str())
                    .is_some_and(|current| current == id)
                {
                    let layer_obj = as_object_mut(layer, "text_layer")?;
                    layer_obj.insert("visible".into(), Value::from(visible));
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(format!("Text layer não encontrada: {id}"));
            }
        }
        other => return Err(format!("Tipo de layer não suportado: {other}")),
    }

    sync_legacy_aliases(page)?;
    Ok(())
}

pub fn resolve_project_file(base_path: &Path) -> PathBuf {
    if base_path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("project.json"))
    {
        base_path.to_path_buf()
    } else {
        base_path.join("project.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_project() -> Value {
        json!({
            "versao": "1.0",
            "app": "traduzai",
            "obra": "Teste",
            "capitulo": 7,
            "paginas": [
                {
                    "numero": 1,
                    "arquivo_original": "originals/001.jpg",
                    "arquivo_traduzido": "translated/001.jpg",
                    "textos": [
                        {
                            "id": "t1_1",
                            "bbox": [10, 12, 80, 90],
                            "tipo": "fala",
                            "original": "Hello",
                            "traduzido": "Oi",
                            "confianca_ocr": 0.91,
                            "estilo": {
                                "fonte": "ComicNeue-Bold.ttf",
                                "tamanho": 20,
                                "cor": "#FFFFFF",
                                "cor_gradiente": [],
                                "contorno": "#000000",
                                "contorno_px": 2,
                                "glow": false,
                                "glow_cor": "",
                                "glow_px": 0,
                                "sombra": false,
                                "sombra_cor": "",
                                "sombra_offset": [0, 0],
                                "bold": false,
                                "italico": false,
                                "rotacao": 0,
                                "alinhamento": "center"
                            }
                        }
                    ]
                }
            ]
        })
    }

    #[test]
    fn migrate_project_value_creates_v2_layers_and_aliases() {
        let mut project = legacy_project();
        migrate_project_value(&mut project).expect("migration should succeed");

        assert_eq!(project["versao"], "2.0");
        assert_eq!(project["paginas"][0]["image_layers"]["base"]["path"], "originals/001.jpg");
        assert_eq!(project["paginas"][0]["image_layers"]["rendered"]["path"], "translated/001.jpg");
        assert_eq!(project["paginas"][0]["image_layers"]["inpaint"]["path"], "images/001.jpg");
        assert_eq!(project["paginas"][0]["text_layers"][0]["layout_bbox"], json!([10, 12, 80, 90]));
        assert_eq!(project["paginas"][0]["text_layers"][0]["translated"], "Oi");
        assert_eq!(project["paginas"][0]["textos"][0]["traduzido"], "Oi");
        assert_eq!(project["estatisticas"]["total_textos"], 1);
    }

    #[test]
    fn create_patch_delete_text_layer_updates_legacy_aliases() {
        let mut project = legacy_project();
        let created = create_text_layer(&mut project, 0, [100, 110, 180, 220]).expect("layer should be created");
        let layer_id = created["id"].as_str().unwrap().to_string();
        assert_eq!(project["paginas"][0]["text_layers"].as_array().unwrap().len(), 2);

        let patched = patch_text_layer(
            &mut project,
            0,
            &layer_id,
            &json!({
                "translated": "Novo texto",
                "tipo": "narracao",
                "style": {
                    "tamanho": 36,
                    "glow": true
                }
            }),
        )
        .expect("layer should be patched");

        assert_eq!(patched["translated"], "Novo texto");
        assert_eq!(patched["tipo"], "narracao");
        assert_eq!(patched["style"]["tamanho"], 36);
        assert_eq!(project["paginas"][0]["textos"][1]["traduzido"], "Novo texto");

        delete_text_layer(&mut project, 0, &layer_id).expect("layer should be removed");
        assert_eq!(project["paginas"][0]["text_layers"].as_array().unwrap().len(), 1);
        assert_eq!(project["paginas"][0]["textos"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn ensure_bitmap_layer_path_assigns_default_paths() {
        let mut project = legacy_project();
        let path = ensure_bitmap_layer_path(&mut project, 0, "mask").expect("mask path");
        assert_eq!(path, "layers/mask/001.png");
        assert_eq!(project["paginas"][0]["image_layers"]["mask"]["path"], "layers/mask/001.png");
    }
}
