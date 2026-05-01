#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GlossaryEntry {
    pub id: String,
    pub source: String,
    pub target: String,
    #[serde(rename = "type")]
    pub entry_type: String,
    #[serde(default)]
    pub case_sensitive: bool,
    #[serde(default)]
    pub protect: bool,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub forbidden: Vec<String>,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    #[serde(default = "default_status")]
    pub status: String,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub context_rule: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Glossary {
    pub work_id: String,
    pub version: u32,
    #[serde(default)]
    pub entries: Vec<GlossaryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GlossaryHit {
    pub entry_id: String,
    pub source: String,
    pub target: String,
    pub match_kind: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GlossaryValidation {
    pub ok: bool,
    pub flags: Vec<String>,
}

fn default_confidence() -> f64 {
    1.0
}

fn default_status() -> String {
    "reviewed".to_string()
}

pub fn glossary_path(works_root: &Path, work_id: &str) -> PathBuf {
    works_root.join(work_id).join("glossary.json")
}

pub fn empty_glossary(work_id: &str) -> Glossary {
    Glossary {
        work_id: work_id.to_string(),
        version: 1,
        entries: vec![],
    }
}

pub fn normalize_lookup(value: &str, case_sensitive: bool) -> String {
    let stripped: String = value
        .nfkd()
        .filter(|ch| !('\u{0300}'..='\u{036f}').contains(ch))
        .collect();
    let normalized = stripped.split_whitespace().collect::<Vec<_>>().join(" ");
    if case_sensitive {
        normalized
    } else {
        normalized.to_lowercase()
    }
}

pub fn load(works_root: &Path, work_id: &str) -> Result<Glossary, String> {
    let path = glossary_path(works_root, work_id);
    if !path.exists() {
        return Ok(empty_glossary(work_id));
    }
    let raw = fs::read_to_string(&path)
        .map_err(|e| format!("Falha ao ler glossario '{}': {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("Glossario invalido '{}': {e}", path.display()))
}

pub fn save(works_root: &Path, glossary: &Glossary) -> Result<(), String> {
    let path = glossary_path(works_root, &glossary.work_id);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "Falha ao criar pasta do glossario '{}': {e}",
                parent.display()
            )
        })?;
    }
    let payload = serde_json::to_string_pretty(glossary).map_err(|e| e.to_string())?;
    fs::write(&path, payload)
        .map_err(|e| format!("Falha ao salvar glossario '{}': {e}", path.display()))
}

pub fn upsert(glossary: &mut Glossary, entry: GlossaryEntry) {
    if let Some(existing) = glossary.entries.iter_mut().find(|item| item.id == entry.id) {
        *existing = entry;
    } else {
        glossary.entries.push(entry);
    }
}

pub fn remove(glossary: &mut Glossary, entry_id: &str) {
    glossary.entries.retain(|entry| entry.id != entry_id);
}

pub fn find_exact<'a>(glossary: &'a Glossary, text: &str) -> Option<&'a GlossaryEntry> {
    glossary.entries.iter().find(|entry| {
        normalize_lookup(&entry.source, entry.case_sensitive)
            == normalize_lookup(text, entry.case_sensitive)
    })
}

pub fn find_alias<'a>(glossary: &'a Glossary, text: &str) -> Option<&'a GlossaryEntry> {
    glossary.entries.iter().find(|entry| {
        let needle = normalize_lookup(text, entry.case_sensitive);
        entry
            .aliases
            .iter()
            .any(|alias| normalize_lookup(alias, entry.case_sensitive) == needle)
    })
}

pub fn find_fuzzy<'a>(
    glossary: &'a Glossary,
    text: &str,
    threshold: f64,
) -> Option<(&'a GlossaryEntry, f64)> {
    let needle = normalize_lookup(text, false);
    if needle.len() < 4 || is_common_word(&needle) {
        return None;
    }
    glossary
        .entries
        .iter()
        .filter_map(|entry| {
            let source = normalize_lookup(&entry.source, false);
            if source.len() < 4 || is_common_word(&source) {
                return None;
            }
            let score = similarity(&source, &needle);
            (score >= threshold).then_some((entry, score))
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

pub fn extract_hits(glossary: &Glossary, text: &str) -> Vec<GlossaryHit> {
    let haystack = normalize_lookup(text, false);
    let mut hits = Vec::new();
    for entry in &glossary.entries {
        let source = normalize_lookup(&entry.source, entry.case_sensitive);
        if haystack.contains(&source) {
            hits.push(hit(entry, "exact", 1.0));
            continue;
        }
        if entry
            .aliases
            .iter()
            .any(|alias| haystack.contains(&normalize_lookup(alias, entry.case_sensitive)))
        {
            hits.push(hit(entry, "alias", 1.0));
        }
    }
    hits
}

pub fn validate_translation(
    glossary: &Glossary,
    source: &str,
    translation: &str,
) -> GlossaryValidation {
    let source_hits = extract_hits(glossary, source);
    let translated = normalize_lookup(translation, false);
    let mut flags = Vec::new();
    for entry in source_hits {
        if let Some(full) = glossary
            .entries
            .iter()
            .find(|item| item.id == entry.entry_id)
        {
            for forbidden in &full.forbidden {
                if translated.contains(&normalize_lookup(forbidden, false)) {
                    flags.push(format!("forbidden:{}", full.id));
                }
            }
            if full.protect && !translated.contains(&normalize_lookup(&full.target, false)) {
                flags.push(format!("protect:{}", full.id));
            }
        }
    }
    GlossaryValidation {
        ok: flags.is_empty(),
        flags,
    }
}

pub fn create_candidate(term: &str, page: u32, region_id: &str) -> GlossaryEntry {
    let id = format!(
        "candidate_{}_p{}_{}",
        normalize_lookup(term, false).replace(' ', "_"),
        page,
        normalize_lookup(region_id, false).replace(' ', "_")
    );
    GlossaryEntry {
        id,
        source: term.to_string(),
        target: term.to_string(),
        entry_type: "generic_term".to_string(),
        case_sensitive: false,
        protect: false,
        aliases: vec![],
        forbidden: vec![],
        confidence: 0.35,
        status: "candidate".to_string(),
        notes: "Candidato extraido automaticamente. Revisar antes de confirmar.".to_string(),
        context_rule: String::new(),
    }
}

#[allow(dead_code)]
pub fn export_used_glossary(glossary: &Glossary, hits: &[GlossaryHit]) -> Glossary {
    let ids: std::collections::HashSet<&str> =
        hits.iter().map(|hit| hit.entry_id.as_str()).collect();
    Glossary {
        work_id: glossary.work_id.clone(),
        version: glossary.version,
        entries: glossary
            .entries
            .iter()
            .filter(|entry| ids.contains(entry.id.as_str()))
            .cloned()
            .collect(),
    }
}

fn hit(entry: &GlossaryEntry, match_kind: &str, score: f64) -> GlossaryHit {
    GlossaryHit {
        entry_id: entry.id.clone(),
        source: entry.source.clone(),
        target: entry.target.clone(),
        match_kind: match_kind.to_string(),
        score,
    }
}

fn is_common_word(value: &str) -> bool {
    matches!(
        value,
        "the" | "and" | "for" | "you" | "that" | "this" | "uma" | "com" | "para" | "que"
    )
}

fn similarity(left: &str, right: &str) -> f64 {
    let max_len = left.chars().count().max(right.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    1.0 - (levenshtein(left, right) as f64 / max_len as f64)
}

fn levenshtein(left: &str, right: &str) -> usize {
    let right_chars: Vec<char> = right.chars().collect();
    let mut costs: Vec<usize> = (0..=right_chars.len()).collect();
    for (i, lc) in left.chars().enumerate() {
        let mut last = i;
        costs[0] = i + 1;
        for (j, rc) in right_chars.iter().enumerate() {
            let old = costs[j + 1];
            costs[j + 1] = if lc == *rc {
                last
            } else {
                1 + last.min(costs[j]).min(costs[j + 1])
            };
            last = old;
        }
    }
    costs[right_chars.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Glossary {
        Glossary {
            work_id: "the-regressed-mercenary-has-a-plan".to_string(),
            version: 1,
            entries: vec![
                GlossaryEntry {
                    id: "char_ghislain_perdium".to_string(),
                    source: "Ghislain Perdium".to_string(),
                    target: "Ghislain Perdium".to_string(),
                    entry_type: "character".to_string(),
                    case_sensitive: false,
                    protect: true,
                    aliases: vec!["GHISLAIN PERDIUM".to_string(), "Ghislain".to_string()],
                    forbidden: vec!["Pérdium Ghislain".to_string()],
                    confidence: 1.0,
                    status: "reviewed".to_string(),
                    notes: String::new(),
                    context_rule: String::new(),
                },
                GlossaryEntry {
                    id: "rank_knight".to_string(),
                    source: "Knight".to_string(),
                    target: "Cavaleiro".to_string(),
                    entry_type: "rank".to_string(),
                    case_sensitive: false,
                    protect: false,
                    aliases: vec![],
                    forbidden: vec!["Night".to_string()],
                    confidence: 1.0,
                    status: "reviewed".to_string(),
                    notes: String::new(),
                    context_rule: String::new(),
                },
            ],
        }
    }

    #[test]
    fn exact_alias_and_accent_insensitive_lookup_work() {
        let glossary = fixture();
        assert_eq!(
            find_exact(&glossary, "ghislain perdium").unwrap().id,
            "char_ghislain_perdium"
        );
        assert_eq!(
            find_alias(&glossary, "Ghislain").unwrap().id,
            "char_ghislain_perdium"
        );
        assert_eq!(normalize_lookup("Pérdium", false), "perdium");
    }

    #[test]
    fn fuzzy_avoids_common_short_words_and_finds_close_terms() {
        let glossary = fixture();
        assert!(find_fuzzy(&glossary, "the", 0.8).is_none());
        assert_eq!(
            find_fuzzy(&glossary, "Knigth", 0.65).unwrap().0.id,
            "rank_knight"
        );
    }

    #[test]
    fn forbidden_and_protected_terms_generate_flags() {
        let glossary = fixture();
        let validation = validate_translation(
            &glossary,
            "Ghislain Perdium met a Knight",
            "Pérdium Ghislain encontrou Night",
        );
        assert!(!validation.ok);
        assert!(validation
            .flags
            .iter()
            .any(|flag| flag == "forbidden:char_ghislain_perdium"));
        assert!(validation
            .flags
            .iter()
            .any(|flag| flag == "protect:char_ghislain_perdium"));
        assert!(validation
            .flags
            .iter()
            .any(|flag| flag == "forbidden:rank_knight"));
    }

    #[test]
    fn candidate_generation_and_persistence_work() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let mut glossary = empty_glossary("work");
        upsert(&mut glossary, create_candidate("Mana Core", 2, "r1"));
        save(tmp.path(), &glossary).expect("save");
        let loaded = load(tmp.path(), "work").expect("load");
        assert_eq!(loaded.entries[0].status, "candidate");
        remove(&mut glossary, &loaded.entries[0].id);
        assert!(glossary.entries.is_empty());
    }
}
