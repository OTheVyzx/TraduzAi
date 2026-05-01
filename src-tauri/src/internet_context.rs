use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InternetContextSourceState {
    pub source: String,
    pub enabled: bool,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InternetContextConfig {
    pub enabled_sources: Vec<String>,
    pub use_cache: bool,
    pub refresh_cache: bool,
    pub generic_web_enabled: bool,
}

impl Default for InternetContextConfig {
    fn default() -> Self {
        Self {
            enabled_sources: vec![
                "anilist".into(),
                "myanimelist".into(),
                "mangaupdates".into(),
                "kitsu".into(),
                "shikimori".into(),
                "bangumi".into(),
                "wikipedia".into(),
                "wikidata".into(),
                "fandom".into(),
            ],
            use_cache: true,
            refresh_cache: false,
            generic_web_enabled: false,
        }
    }
}

pub fn source_states(config: &InternetContextConfig) -> Vec<InternetContextSourceState> {
    let enabled: BTreeSet<&str> = config.enabled_sources.iter().map(String::as_str).collect();
    [
        "anilist",
        "myanimelist",
        "mangaupdates",
        "kitsu",
        "shikimori",
        "bangumi",
        "wikipedia",
        "wikidata",
        "fandom",
        "generic_web",
    ]
    .iter()
    .map(|source| {
        let is_enabled = if *source == "generic_web" {
            config.generic_web_enabled
        } else {
            enabled.contains(source)
        };
        InternetContextSourceState {
            source: (*source).to_string(),
            enabled: is_enabled,
            status: if is_enabled { "enabled" } else { "disabled" }.to_string(),
        }
    })
    .collect()
}

pub fn cache_path(cache_root: &Path, title: &str) -> PathBuf {
    let mut slug = String::new();
    for ch in title.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    let slug = slug.trim_matches('-');
    cache_root.join(format!(
        "{}.json",
        if slug.is_empty() { "obra-sem-titulo" } else { slug }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internet_context_default_sources_disable_generic_web() {
        let states = source_states(&InternetContextConfig::default());

        assert!(states.iter().any(|state| state.source == "anilist" && state.enabled));
        assert!(states.iter().any(|state| state.source == "fandom" && state.enabled));
        assert!(states.iter().any(|state| state.source == "generic_web" && !state.enabled));
    }

    #[test]
    fn internet_context_cache_path_is_stable() {
        let root = PathBuf::from("cache");
        assert_eq!(
            cache_path(&root, "The Regressed Mercenary Has a Plan!"),
            root.join("the-regressed-mercenary-has-a-plan.json")
        );
        assert_eq!(cache_path(&root, "   "), root.join("obra-sem-titulo.json"));
    }
}
