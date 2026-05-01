use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranslationStyle {
    pub tone: String,
    pub honorifics: String,
    pub names: String,
    pub lore_terms: String,
    pub sound_effects: String,
}

impl Default for TranslationStyle {
    fn default() -> Self {
        Self {
            tone: "natural Brazilian Portuguese".to_string(),
            honorifics: "adapted".to_string(),
            names: "preserve proper names".to_string(),
            lore_terms: "use glossary".to_string(),
            sound_effects: "review".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkContextProfile {
    pub work_id: String,
    pub title: String,
    #[serde(default)]
    pub alt_titles: Vec<String>,
    pub source_language: String,
    pub target_language: String,
    pub status: String,
    pub context_quality: String,
    pub synopsis: String,
    #[serde(default)]
    pub genre: Vec<String>,
    #[serde(default)]
    pub translation_style: TranslationStyle,
    #[serde(default)]
    pub characters: Vec<serde_json::Value>,
    #[serde(default)]
    pub places: Vec<serde_json::Value>,
    #[serde(default)]
    pub factions: Vec<serde_json::Value>,
    #[serde(default)]
    pub terms: Vec<serde_json::Value>,
    #[serde(default)]
    pub forbidden_translations: Vec<serde_json::Value>,
    #[serde(default)]
    pub chapter_memory: Vec<serde_json::Value>,
    pub last_updated: String,
}

pub fn slugify_work_id(title: &str) -> String {
    let mut slug = String::new();
    for ch in title.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        "obra-sem-titulo".to_string()
    } else {
        slug
    }
}

pub fn work_context_path(works_root: &Path, work_id: &str) -> PathBuf {
    works_root.join(work_id).join("work_context.json")
}

pub fn quality_for_profile(profile: &WorkContextProfile) -> String {
    if !profile.synopsis.trim().is_empty()
        || !profile.genre.is_empty()
        || !profile.characters.is_empty()
        || !profile.terms.is_empty()
    {
        "partial".to_string()
    } else {
        "empty".to_string()
    }
}

pub fn risk_level(context_quality: &str, glossary_entries_count: usize) -> String {
    if context_quality == "reviewed" && glossary_entries_count > 0 {
        "low".to_string()
    } else if context_quality == "partial" || glossary_entries_count > 0 {
        "medium".to_string()
    } else {
        "high".to_string()
    }
}

pub fn new_profile(
    title: &str,
    source_language: &str,
    target_language: &str,
    synopsis: &str,
    genre: Vec<String>,
    characters: Vec<String>,
    terms: Vec<String>,
    factions: Vec<String>,
) -> WorkContextProfile {
    let mut profile = WorkContextProfile {
        work_id: slugify_work_id(title),
        title: title.to_string(),
        alt_titles: vec![],
        source_language: source_language.to_string(),
        target_language: target_language.to_string(),
        status: "active".to_string(),
        context_quality: "empty".to_string(),
        synopsis: synopsis.to_string(),
        genre,
        translation_style: TranslationStyle::default(),
        characters: characters
            .into_iter()
            .map(serde_json::Value::String)
            .collect(),
        places: vec![],
        factions: factions
            .into_iter()
            .map(serde_json::Value::String)
            .collect(),
        terms: terms.into_iter().map(serde_json::Value::String).collect(),
        forbidden_translations: vec![],
        chapter_memory: vec![],
        last_updated: current_utc_timestamp(),
    };
    profile.context_quality = quality_for_profile(&profile);
    profile
}

pub fn load_or_create_profile(
    works_root: &Path,
    mut fallback: WorkContextProfile,
) -> Result<WorkContextProfile, String> {
    let path = work_context_path(works_root, &fallback.work_id);
    if path.exists() {
        let raw = fs::read_to_string(&path)
            .map_err(|e| format!("Falha ao ler contexto da obra '{}': {e}", path.display()))?;
        let mut profile: WorkContextProfile = serde_json::from_str(&raw)
            .map_err(|e| format!("Contexto da obra invalido '{}': {e}", path.display()))?;
        if profile.context_quality.trim().is_empty() {
            profile.context_quality = quality_for_profile(&profile);
        }
        return Ok(profile);
    }

    fallback.last_updated = current_utc_timestamp();
    persist_profile(works_root, &fallback)?;
    Ok(fallback)
}

pub fn persist_profile(works_root: &Path, profile: &WorkContextProfile) -> Result<(), String> {
    let path = work_context_path(works_root, &profile.work_id);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "Falha ao criar pasta do contexto da obra '{}': {e}",
                parent.display()
            )
        })?;
    }
    let payload = serde_json::to_string_pretty(profile).map_err(|e| e.to_string())?;
    fs::write(&path, payload)
        .map_err(|e| format!("Falha ao salvar contexto da obra '{}': {e}", path.display()))
}

fn current_utc_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    format!("{secs}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_work_id_creates_stable_safe_id() {
        assert_eq!(
            slugify_work_id("The Regressed Mercenary Has a Plan!"),
            "the-regressed-mercenary-has-a-plan"
        );
        assert_eq!(slugify_work_id("   "), "obra-sem-titulo");
    }

    #[test]
    fn creates_work_context_under_work_folder() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let profile = new_profile(
            "The Regressed Mercenary Has a Plan",
            "en",
            "pt-BR",
            "",
            vec![],
            vec![],
            vec![],
            vec![],
        );

        let loaded = load_or_create_profile(tmp.path(), profile).expect("profile");
        let path = work_context_path(tmp.path(), "the-regressed-mercenary-has-a-plan");

        assert_eq!(loaded.context_quality, "empty");
        assert!(path.is_file());
    }

    #[test]
    fn loads_existing_work_context_without_overwriting_reviewed_data() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let mut existing = new_profile(
            "Solo Leveling",
            "en",
            "pt-BR",
            "A hunter story",
            vec!["action".into()],
            vec!["Sung Jinwoo".into()],
            vec![],
            vec![],
        );
        existing.context_quality = "reviewed".to_string();
        persist_profile(tmp.path(), &existing).expect("persist");

        let fallback = new_profile(
            "Solo Leveling",
            "en",
            "pt-BR",
            "",
            vec![],
            vec![],
            vec![],
            vec![],
        );
        let loaded = load_or_create_profile(tmp.path(), fallback).expect("load");

        assert_eq!(loaded.context_quality, "reviewed");
        assert_eq!(loaded.synopsis, "A hunter story");
    }

    #[test]
    fn risk_level_matches_context_and_glossary_state() {
        assert_eq!(risk_level("empty", 0), "high");
        assert_eq!(risk_level("partial", 0), "medium");
        assert_eq!(risk_level("reviewed", 3), "low");
    }
}
