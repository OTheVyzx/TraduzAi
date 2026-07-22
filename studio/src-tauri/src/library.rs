use serde::Serialize;
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use tauri::Manager;
use uuid::Uuid;

const LIBRARY_FILE_NAME: &str = "studio-library.json";

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LibraryLoadResult {
    pub(crate) document: Value,
    pub(crate) recovered_from_backup: bool,
}

fn empty_library_document() -> Value {
    json!({
        "schemaVersion": 1,
        "selectedWorkId": Value::Null,
        "works": [],
        "preferences": {
            "chapterView": "grid",
            "thumbnailSize": 176
        }
    })
}

fn backup_path(path: &Path) -> PathBuf {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    path.with_extension(format!("{extension}.bak"))
}

fn parse_document(path: &Path) -> Result<Value, String> {
    let payload = fs::read_to_string(path)
        .map_err(|error| format!("Falha ao ler {}: {error}", path.display()))?;
    let document: Value = serde_json::from_str(payload.trim_start_matches('\u{feff}'))
        .map_err(|error| format!("Catalogo invalido em {}: {error}", path.display()))?;
    if !document.is_object() {
        return Err(format!(
            "Catalogo invalido em {}: raiz deve ser um objeto",
            path.display()
        ));
    }
    Ok(document)
}

pub(crate) fn load_library_from_path(path: &Path) -> Result<LibraryLoadResult, String> {
    if path.exists() {
        if let Ok(document) = parse_document(path) {
            return Ok(LibraryLoadResult {
                document,
                recovered_from_backup: false,
            });
        }

        let backup = backup_path(path);
        if backup.exists() {
            return parse_document(&backup).map(|document| LibraryLoadResult {
                document,
                recovered_from_backup: true,
            });
        }

        return parse_document(path).map(|document| LibraryLoadResult {
            document,
            recovered_from_backup: false,
        });
    }

    let backup = backup_path(path);
    if backup.exists() {
        return parse_document(&backup).map(|document| LibraryLoadResult {
            document,
            recovered_from_backup: true,
        });
    }

    Ok(LibraryLoadResult {
        document: empty_library_document(),
        recovered_from_backup: false,
    })
}

fn replace_file(path: &Path, payload: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("Caminho de catalogo invalido: {}", path.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("Falha ao criar pasta do catalogo: {error}"))?;

    let temp_path = parent.join(format!(".{}-{}.tmp", LIBRARY_FILE_NAME, Uuid::new_v4()));
    let write_result = (|| -> Result<(), String> {
        let mut file = File::create(&temp_path)
            .map_err(|error| format!("Falha ao criar arquivo temporario do catalogo: {error}"))?;
        file.write_all(payload)
            .map_err(|error| format!("Falha ao gravar catalogo temporario: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("Falha ao sincronizar catalogo temporario: {error}"))?;
        drop(file);

        if path.exists() {
            fs::remove_file(path)
                .map_err(|error| format!("Falha ao substituir catalogo existente: {error}"))?;
        }
        fs::rename(&temp_path, path)
            .map_err(|error| format!("Falha ao promover catalogo temporario: {error}"))?;
        Ok(())
    })();

    if write_result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    write_result
}

pub(crate) fn save_library_to_path(path: &Path, document: &Value) -> Result<(), String> {
    if !document.is_object() {
        return Err("Catalogo invalido: raiz deve ser um objeto".to_string());
    }
    let payload = serde_json::to_vec_pretty(document)
        .map_err(|error| format!("Falha ao serializar catalogo: {error}"))?;

    if path.exists() {
        if parse_document(path).is_ok() {
            let current = fs::read(path)
                .map_err(|error| format!("Falha ao preservar catalogo atual: {error}"))?;
            replace_file(&backup_path(path), &current)?;
        }
    }

    replace_file(path, &payload)
}

fn library_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_data_dir()
        .map(|directory| directory.join(LIBRARY_FILE_NAME))
        .map_err(|error| format!("Falha ao localizar dados do Studio: {error}"))
}

#[tauri::command]
pub(crate) fn studio_load_library(app: tauri::AppHandle) -> Result<LibraryLoadResult, String> {
    load_library_from_path(&library_path(&app)?)
}

#[tauri::command]
pub(crate) fn studio_save_library(app: tauri::AppHandle, document: Value) -> Result<(), String> {
    save_library_to_path(&library_path(&app)?, &document)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn missing_library_returns_an_empty_document() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("studio-library.json");

        let loaded = load_library_from_path(&path).unwrap();

        assert!(!loaded.recovered_from_backup);
        assert_eq!(loaded.document["schemaVersion"], 1);
        assert_eq!(loaded.document["works"], json!([]));
    }

    #[test]
    fn saves_and_loads_a_library_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("studio-library.json");
        let document = json!({"schemaVersion": 1, "works": [{"id": "work-1"}]});

        save_library_to_path(&path, &document).unwrap();
        let loaded = load_library_from_path(&path).unwrap();

        assert_eq!(loaded.document, document);
        assert!(!loaded.recovered_from_backup);
    }

    #[test]
    fn loads_backup_when_primary_is_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("studio-library.json");
        std::fs::write(&path, "{").unwrap();
        std::fs::write(
            path.with_extension("json.bak"),
            r#"{"schemaVersion":1,"works":[]}"#,
        )
        .unwrap();

        let loaded = load_library_from_path(&path).unwrap();

        assert!(loaded.recovered_from_backup);
        assert_eq!(loaded.document["works"], json!([]));
    }

    #[test]
    fn preserves_the_previous_valid_document_as_backup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("studio-library.json");
        let first = json!({"schemaVersion": 1, "works": [{"id": "first"}]});
        let second = json!({"schemaVersion": 1, "works": [{"id": "second"}]});

        save_library_to_path(&path, &first).unwrap();
        save_library_to_path(&path, &second).unwrap();
        std::fs::write(&path, "invalid").unwrap();

        let loaded = load_library_from_path(&path).unwrap();
        assert_eq!(loaded.document, first);
        assert!(loaded.recovered_from_backup);
    }
}
