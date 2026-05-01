use crate::local_memory::{
    memory_db_path, LocalMemoryService, MemorySuggestion, OcrCorrectionInput,
    TranslationMemoryInput, UserCorrectionInput,
};
use crate::storage::configured_or_dev_paths;
use serde_json::Value;

fn service() -> Result<LocalMemoryService, String> {
    let paths = configured_or_dev_paths()?;
    LocalMemoryService::open(memory_db_path(&paths))
}

#[tauri::command]
pub async fn export_local_memory() -> Result<Value, String> {
    service()?.export_json()
}

#[tauri::command]
pub async fn import_local_memory(payload: Value) -> Result<(), String> {
    service()?.import_json(&payload)
}

#[tauri::command]
pub async fn upsert_memory_work(work_id: String, title: String) -> Result<(), String> {
    service()?.upsert_work(&work_id, &title)
}

#[tauri::command]
pub async fn record_translation_memory(input: TranslationMemoryInput) -> Result<(), String> {
    service()?.record_translation_memory(input)
}

#[tauri::command]
pub async fn record_user_correction(input: UserCorrectionInput) -> Result<(), String> {
    service()?.record_user_correction(input)
}

#[tauri::command]
pub async fn record_ocr_correction(input: OcrCorrectionInput) -> Result<(), String> {
    service()?.record_ocr_correction(input)
}

#[tauri::command]
pub async fn suggest_memory_translation(
    work_id: String,
    source_text: String,
    glossary_reviewed: bool,
) -> Result<Option<MemorySuggestion>, String> {
    service()?.suggest_translation(&work_id, &source_text, glossary_reviewed)
}
