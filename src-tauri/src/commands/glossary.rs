#[tauri::command]
pub async fn load_glossary(
    app: tauri::AppHandle,
    work_id: String,
) -> Result<crate::glossary::Glossary, String> {
    let storage = crate::storage::service_for_app(&app)?;
    let paths = storage.ensure_base_dirs()?;
    crate::glossary::load(&paths.works, &work_id)
}

#[tauri::command]
pub async fn save_glossary(
    app: tauri::AppHandle,
    glossary: crate::glossary::Glossary,
) -> Result<crate::glossary::Glossary, String> {
    let storage = crate::storage::service_for_app(&app)?;
    let paths = storage.ensure_base_dirs()?;
    storage.check_writable()?;
    crate::glossary::save(&paths.works, &glossary)?;
    Ok(glossary)
}

#[tauri::command]
pub async fn upsert_glossary_entry(
    app: tauri::AppHandle,
    work_id: String,
    entry: crate::glossary::GlossaryEntry,
) -> Result<crate::glossary::Glossary, String> {
    let storage = crate::storage::service_for_app(&app)?;
    let paths = storage.ensure_base_dirs()?;
    storage.check_writable()?;
    let mut glossary = crate::glossary::load(&paths.works, &work_id)?;
    crate::glossary::upsert(&mut glossary, entry);
    crate::glossary::save(&paths.works, &glossary)?;
    Ok(glossary)
}

#[tauri::command]
pub async fn remove_glossary_entry(
    app: tauri::AppHandle,
    work_id: String,
    entry_id: String,
) -> Result<crate::glossary::Glossary, String> {
    let storage = crate::storage::service_for_app(&app)?;
    let paths = storage.ensure_base_dirs()?;
    storage.check_writable()?;
    let mut glossary = crate::glossary::load(&paths.works, &work_id)?;
    crate::glossary::remove(&mut glossary, &entry_id);
    crate::glossary::save(&paths.works, &glossary)?;
    Ok(glossary)
}
