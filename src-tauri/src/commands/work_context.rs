use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct WorkContextRequest {
    pub title: String,
    #[serde(default = "default_source_language")]
    pub source_language: String,
    #[serde(default = "default_target_language")]
    pub target_language: String,
    #[serde(default)]
    pub synopsis: String,
    #[serde(default)]
    pub genre: Vec<String>,
    #[serde(default)]
    pub characters: Vec<String>,
    #[serde(default)]
    pub terms: Vec<String>,
    #[serde(default)]
    pub factions: Vec<String>,
}

fn default_source_language() -> String {
    "en".to_string()
}

fn default_target_language() -> String {
    "pt-BR".to_string()
}

#[tauri::command]
pub async fn load_or_create_work_context(
    app: tauri::AppHandle,
    request: WorkContextRequest,
) -> Result<crate::work_context::WorkContextProfile, String> {
    let storage = crate::storage::service_for_app(&app)?;
    let paths = storage.ensure_base_dirs()?;
    storage.check_writable()?;
    crate::storage::set_configured_paths(paths.clone());

    let profile = crate::work_context::new_profile(
        &request.title,
        &request.source_language,
        &request.target_language,
        &request.synopsis,
        request.genre,
        request.characters,
        request.terms,
        request.factions,
    );
    crate::work_context::load_or_create_profile(&paths.works, profile)
}
