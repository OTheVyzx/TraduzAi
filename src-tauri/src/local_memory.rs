use crate::storage::StoragePaths;
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const AUTO_MEMORY_MIN_CONFIDENCE: f64 = 0.80;

#[derive(Debug, Clone)]
pub struct LocalMemoryService {
    db_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationMemoryInput {
    pub work_id: String,
    pub source_text: String,
    pub target_text: String,
    pub context_json: String,
    pub confidence: f64,
    pub confirmed_by_user: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserCorrectionInput {
    pub work_id: String,
    pub page: i64,
    pub region_id: String,
    pub before_text: String,
    pub after_text: String,
    pub correction_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrCorrectionInput {
    pub work_id: String,
    pub raw_text: String,
    pub normalized_text: String,
    pub reason: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemorySuggestion {
    pub target_text: String,
    pub source: String,
    pub confidence: f64,
}

pub fn memory_db_path(paths: &StoragePaths) -> PathBuf {
    paths.memory.join("traduzai_memory.db")
}

impl LocalMemoryService {
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self, String> {
        let db_path = db_path.as_ref().to_path_buf();
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Falha ao criar pasta de memoria local: {e}"))?;
        }
        let service = Self { db_path };
        service.init_schema()?;
        Ok(service)
    }

    fn connection(&self) -> Result<Connection, String> {
        Connection::open(&self.db_path).map_err(|e| format!("Falha ao abrir memoria local: {e}"))
    }

    fn init_schema(&self) -> Result<(), String> {
        let conn = self.connection()?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS works (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              created_at TEXT,
              updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS glossary_entries (
              id TEXT PRIMARY KEY,
              work_id TEXT,
              source TEXT,
              target TEXT,
              type TEXT,
              aliases_json TEXT,
              forbidden_json TEXT,
              protect INTEGER,
              confidence REAL,
              status TEXT,
              created_at TEXT,
              updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS translation_memory (
              id TEXT PRIMARY KEY,
              work_id TEXT,
              source_text TEXT,
              target_text TEXT,
              context_json TEXT,
              confidence REAL,
              confirmed_by_user INTEGER,
              created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS ocr_corrections (
              id TEXT PRIMARY KEY,
              work_id TEXT,
              raw_text TEXT,
              normalized_text TEXT,
              reason TEXT,
              confidence REAL,
              count INTEGER,
              created_at TEXT,
              updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS qa_flags (
              id TEXT PRIMARY KEY,
              work_id TEXT,
              chapter TEXT,
              page INTEGER,
              region_id TEXT,
              type TEXT,
              severity TEXT,
              status TEXT,
              payload_json TEXT,
              created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS user_corrections (
              id TEXT PRIMARY KEY,
              work_id TEXT,
              page INTEGER,
              region_id TEXT,
              before_text TEXT,
              after_text TEXT,
              correction_type TEXT,
              created_at TEXT
            );
            "#,
        )
        .map_err(|e| format!("Falha ao criar schema de memoria local: {e}"))?;
        Ok(())
    }

    pub fn upsert_work(&self, id: &str, title: &str) -> Result<(), String> {
        let now = Utc::now().to_rfc3339();
        self.connection()?
            .execute(
                r#"
                INSERT INTO works (id, title, created_at, updated_at)
                VALUES (?1, ?2, ?3, ?3)
                ON CONFLICT(id) DO UPDATE SET title = excluded.title, updated_at = excluded.updated_at
                "#,
                params![id, title, now],
            )
            .map_err(|e| format!("Falha ao salvar obra na memoria local: {e}"))?;
        Ok(())
    }

    pub fn record_translation_memory(&self, input: TranslationMemoryInput) -> Result<(), String> {
        let now = Utc::now().to_rfc3339();
        self.connection()?
            .execute(
                r#"
                INSERT INTO translation_memory
                (id, work_id, source_text, target_text, context_json, confidence, confirmed_by_user, created_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                "#,
                params![
                    uuid::Uuid::new_v4().to_string(),
                    input.work_id,
                    input.source_text,
                    input.target_text,
                    input.context_json,
                    input.confidence,
                    if input.confirmed_by_user { 1 } else { 0 },
                    now
                ],
            )
            .map_err(|e| format!("Falha ao salvar memoria de traducao: {e}"))?;
        Ok(())
    }

    pub fn record_user_correction(&self, input: UserCorrectionInput) -> Result<(), String> {
        let now = Utc::now().to_rfc3339();
        self.connection()?
            .execute(
                r#"
                INSERT INTO user_corrections
                (id, work_id, page, region_id, before_text, after_text, correction_type, created_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                "#,
                params![
                    uuid::Uuid::new_v4().to_string(),
                    input.work_id,
                    input.page,
                    input.region_id,
                    input.before_text,
                    input.after_text,
                    input.correction_type,
                    now
                ],
            )
            .map_err(|e| format!("Falha ao salvar correcao do usuario: {e}"))?;
        Ok(())
    }

    pub fn record_ocr_correction(&self, input: OcrCorrectionInput) -> Result<(), String> {
        let conn = self.connection()?;
        let now = Utc::now().to_rfc3339();
        let existing: Option<(String, i64)> = conn
            .query_row(
                r#"
                SELECT id, count FROM ocr_corrections
                WHERE work_id = ?1 AND raw_text = ?2 AND normalized_text = ?3
                ORDER BY updated_at DESC LIMIT 1
                "#,
                params![input.work_id, input.raw_text, input.normalized_text],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|e| format!("Falha ao consultar correcao OCR: {e}"))?;

        if let Some((id, count)) = existing {
            conn.execute(
                "UPDATE ocr_corrections SET count = ?1, updated_at = ?2 WHERE id = ?3",
                params![count + 1, now, id],
            )
            .map_err(|e| format!("Falha ao atualizar correcao OCR: {e}"))?;
        } else {
            conn.execute(
                r#"
                INSERT INTO ocr_corrections
                (id, work_id, raw_text, normalized_text, reason, confidence, count, created_at, updated_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1, ?7, ?7)
                "#,
                params![
                    uuid::Uuid::new_v4().to_string(),
                    input.work_id,
                    input.raw_text,
                    input.normalized_text,
                    input.reason,
                    input.confidence,
                    now
                ],
            )
            .map_err(|e| format!("Falha ao salvar correcao OCR: {e}"))?;
        }
        Ok(())
    }

    pub fn suggest_translation(
        &self,
        work_id: &str,
        source_text: &str,
        glossary_reviewed: bool,
    ) -> Result<Option<MemorySuggestion>, String> {
        let conn = self.connection()?;
        let user: Option<String> = conn
            .query_row(
                r#"
                SELECT after_text FROM user_corrections
                WHERE work_id = ?1 AND before_text = ?2 AND correction_type = 'translation'
                ORDER BY created_at DESC LIMIT 1
                "#,
                params![work_id, source_text],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| format!("Falha ao consultar correcao confirmada: {e}"))?;
        if let Some(target_text) = user {
            return Ok(Some(MemorySuggestion {
                target_text,
                source: "user_correction".to_string(),
                confidence: 1.0,
            }));
        }

        if glossary_reviewed {
            return Ok(None);
        }

        conn.query_row(
            r#"
            SELECT target_text, confidence, confirmed_by_user FROM translation_memory
            WHERE work_id = ?1
              AND source_text = ?2
              AND (confirmed_by_user = 1 OR confidence >= ?3)
            ORDER BY confirmed_by_user DESC, confidence DESC, created_at DESC
            LIMIT 1
            "#,
            params![work_id, source_text, AUTO_MEMORY_MIN_CONFIDENCE],
            |row| {
                let confirmed: i64 = row.get(2)?;
                Ok(MemorySuggestion {
                    target_text: row.get(0)?,
                    confidence: row.get(1)?,
                    source: if confirmed == 1 {
                        "confirmed_memory".to_string()
                    } else {
                        "automatic_memory".to_string()
                    },
                })
            },
        )
        .optional()
        .map_err(|e| format!("Falha ao consultar memoria de traducao: {e}"))
    }

    pub fn export_json(&self) -> Result<Value, String> {
        Ok(json!({
            "works": self.query_works()?,
            "glossary_entries": self.query_json_table("glossary_entries")?,
            "translation_memory": self.query_json_table("translation_memory")?,
            "ocr_corrections": self.query_json_table("ocr_corrections")?,
            "qa_flags": self.query_json_table("qa_flags")?,
            "user_corrections": self.query_json_table("user_corrections")?,
        }))
    }

    pub fn import_json(&self, payload: &Value) -> Result<(), String> {
        let conn = self.connection()?;
        for work in payload
            .get("works")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
        {
            conn.execute(
                r#"
                INSERT OR REPLACE INTO works (id, title, created_at, updated_at)
                VALUES (?1, ?2, ?3, ?4)
                "#,
                params![
                    work.get("id").and_then(Value::as_str).unwrap_or(""),
                    work.get("title").and_then(Value::as_str).unwrap_or(""),
                    work.get("created_at").and_then(Value::as_str).unwrap_or(""),
                    work.get("updated_at").and_then(Value::as_str).unwrap_or(""),
                ],
            )
            .map_err(|e| format!("Falha ao importar obras: {e}"))?;
        }
        for item in payload
            .get("ocr_corrections")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
        {
            conn.execute(
                r#"
                INSERT OR REPLACE INTO ocr_corrections
                (id, work_id, raw_text, normalized_text, reason, confidence, count, created_at, updated_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                "#,
                params![
                    item.get("id").and_then(Value::as_str).unwrap_or(""),
                    item.get("work_id").and_then(Value::as_str).unwrap_or(""),
                    item.get("raw_text").and_then(Value::as_str).unwrap_or(""),
                    item.get("normalized_text").and_then(Value::as_str).unwrap_or(""),
                    item.get("reason").and_then(Value::as_str).unwrap_or(""),
                    item.get("confidence").and_then(Value::as_f64).unwrap_or(0.0),
                    item.get("count").and_then(Value::as_i64).unwrap_or(1),
                    item.get("created_at").and_then(Value::as_str).unwrap_or(""),
                    item.get("updated_at").and_then(Value::as_str).unwrap_or(""),
                ],
            )
            .map_err(|e| format!("Falha ao importar correcoes OCR: {e}"))?;
        }
        Ok(())
    }

    fn query_works(&self) -> Result<Vec<Value>, String> {
        let conn = self.connection()?;
        let mut stmt = conn
            .prepare("SELECT id, title, created_at, updated_at FROM works ORDER BY id")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                Ok(json!({
                    "id": row.get::<_, String>(0)?,
                    "title": row.get::<_, String>(1)?,
                    "created_at": row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    "updated_at": row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                }))
            })
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
    }

    fn query_json_table(&self, table: &str) -> Result<Vec<Value>, String> {
        match table {
            "translation_memory" => self.query_translation_memory(),
            "ocr_corrections" => self.query_ocr_corrections(),
            "user_corrections" => self.query_user_corrections(),
            "glossary_entries" | "qa_flags" => Ok(Vec::new()),
            _ => Err(format!("Tabela de memoria desconhecida: {table}")),
        }
    }

    fn query_translation_memory(&self) -> Result<Vec<Value>, String> {
        let conn = self.connection()?;
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, work_id, source_text, target_text, context_json, confidence, confirmed_by_user, created_at
                FROM translation_memory ORDER BY created_at, id
                "#,
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                Ok(json!({
                    "id": row.get::<_, String>(0)?,
                    "work_id": row.get::<_, String>(1)?,
                    "source_text": row.get::<_, String>(2)?,
                    "target_text": row.get::<_, String>(3)?,
                    "context_json": row.get::<_, String>(4)?,
                    "confidence": row.get::<_, f64>(5)?,
                    "confirmed_by_user": row.get::<_, i64>(6)? == 1,
                    "created_at": row.get::<_, String>(7)?,
                }))
            })
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
    }

    fn query_ocr_corrections(&self) -> Result<Vec<Value>, String> {
        let conn = self.connection()?;
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, work_id, raw_text, normalized_text, reason, confidence, count, created_at, updated_at
                FROM ocr_corrections ORDER BY raw_text, normalized_text
                "#,
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                Ok(json!({
                    "id": row.get::<_, String>(0)?,
                    "work_id": row.get::<_, String>(1)?,
                    "raw_text": row.get::<_, String>(2)?,
                    "normalized_text": row.get::<_, String>(3)?,
                    "reason": row.get::<_, String>(4)?,
                    "confidence": row.get::<_, f64>(5)?,
                    "count": row.get::<_, i64>(6)?,
                    "created_at": row.get::<_, String>(7)?,
                    "updated_at": row.get::<_, String>(8)?,
                }))
            })
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
    }

    fn query_user_corrections(&self) -> Result<Vec<Value>, String> {
        let conn = self.connection()?;
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, work_id, page, region_id, before_text, after_text, correction_type, created_at
                FROM user_corrections ORDER BY created_at, id
                "#,
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                Ok(json!({
                    "id": row.get::<_, String>(0)?,
                    "work_id": row.get::<_, String>(1)?,
                    "page": row.get::<_, i64>(2)?,
                    "region_id": row.get::<_, String>(3)?,
                    "before_text": row.get::<_, String>(4)?,
                    "after_text": row.get::<_, String>(5)?,
                    "correction_type": row.get::<_, String>(6)?,
                    "created_at": row.get::<_, String>(7)?,
                }))
            })
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn unique_db_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("traduzai-memory-{}.db", uuid::Uuid::new_v4()))
    }

    fn table_names(path: &std::path::Path) -> Vec<String> {
        let conn = Connection::open(path).unwrap();
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
            .unwrap();
        stmt.query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    }

    #[test]
    fn init_creates_required_tables() {
        let path = unique_db_path();
        let _service = LocalMemoryService::open(&path).unwrap();

        let tables = table_names(&path);
        for table in [
            "glossary_entries",
            "ocr_corrections",
            "qa_flags",
            "translation_memory",
            "user_corrections",
            "works",
        ] {
            assert!(tables.contains(&table.to_string()), "missing {table}");
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn confirmed_user_correction_has_priority_over_automatic_memory() {
        let path = unique_db_path();
        let service = LocalMemoryService::open(&path).unwrap();
        service.upsert_work("work-1", "Obra").unwrap();
        service
            .record_translation_memory(TranslationMemoryInput {
                work_id: "work-1".to_string(),
                source_text: "Knight".to_string(),
                target_text: "Noite".to_string(),
                context_json: "{}".to_string(),
                confidence: 0.99,
                confirmed_by_user: false,
            })
            .unwrap();
        service
            .record_user_correction(UserCorrectionInput {
                work_id: "work-1".to_string(),
                page: 1,
                region_id: "r1".to_string(),
                before_text: "Knight".to_string(),
                after_text: "Cavaleiro".to_string(),
                correction_type: "translation".to_string(),
            })
            .unwrap();

        let suggestion = service
            .suggest_translation("work-1", "Knight", false)
            .unwrap()
            .unwrap();
        assert_eq!(suggestion.target_text, "Cavaleiro");
        assert_eq!(suggestion.source, "user_correction");

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn automatic_memory_requires_confidence_and_does_not_override_reviewed_glossary() {
        let path = unique_db_path();
        let service = LocalMemoryService::open(&path).unwrap();
        service.upsert_work("work-1", "Obra").unwrap();
        service
            .record_translation_memory(TranslationMemoryInput {
                work_id: "work-1".to_string(),
                source_text: "Order".to_string(),
                target_text: "Ordem".to_string(),
                context_json: "{}".to_string(),
                confidence: 0.72,
                confirmed_by_user: false,
            })
            .unwrap();
        assert!(service
            .suggest_translation("work-1", "Order", false)
            .unwrap()
            .is_none());

        service
            .record_translation_memory(TranslationMemoryInput {
                work_id: "work-1".to_string(),
                source_text: "Order".to_string(),
                target_text: "Ordem".to_string(),
                context_json: "{}".to_string(),
                confidence: 0.93,
                confirmed_by_user: false,
            })
            .unwrap();
        assert!(service
            .suggest_translation("work-1", "Order", true)
            .unwrap()
            .is_none());
        assert_eq!(
            service
                .suggest_translation("work-1", "Order", false)
                .unwrap()
                .unwrap()
                .target_text,
            "Ordem"
        );

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn export_import_roundtrips_memory_tables() {
        let path = unique_db_path();
        let service = LocalMemoryService::open(&path).unwrap();
        service.upsert_work("work-1", "Obra").unwrap();
        service
            .record_ocr_correction(OcrCorrectionInput {
                work_id: "work-1".to_string(),
                raw_text: "RAID SOUAD".to_string(),
                normalized_text: "RAID SQUAD".to_string(),
                reason: "mandatory".to_string(),
                confidence: 1.0,
            })
            .unwrap();

        let exported = service.export_json().unwrap();
        let import_path = unique_db_path();
        let imported = LocalMemoryService::open(&import_path).unwrap();
        imported.import_json(&exported).unwrap();

        assert_eq!(
            imported.export_json().unwrap()["ocr_corrections"],
            exported["ocr_corrections"]
        );

        std::fs::remove_file(path).ok();
        std::fs::remove_file(import_path).ok();
    }
}
