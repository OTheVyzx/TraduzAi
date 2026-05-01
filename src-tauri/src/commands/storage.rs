#[tauri::command]
pub async fn get_storage_paths(
    app: tauri::AppHandle,
) -> Result<crate::storage::StoragePaths, String> {
    let storage = crate::storage::service_for_app(&app)?;
    let paths = storage.ensure_base_dirs()?;
    storage.check_writable()?;
    crate::storage::set_configured_paths(paths.clone());
    Ok(paths)
}
