use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};

use crate::project_schema;

const REVISION_CONFLICT: &str = "PROJECT_REVISION_CONFLICT";

static ACTIVE_CONSUMER_JOBS: once_cell::sync::Lazy<Mutex<HashSet<String>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashSet::new()));

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReviewDecisionInput {
    pub project_id: String,
    pub owner_id: String,
    pub expected_revision: u64,
    pub actor_kind: String,
    pub actor_id: String,
    pub decision: String,
    pub reason_code: String,
    pub evidence_sha256s: Vec<String>,
    pub idempotency_key: String,
    #[serde(default)]
    pub preference_response: Option<Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProjectEventInput {
    pub job_id: String,
    pub project_id: String,
    pub expected_revision: u64,
    pub project_revision: u64,
    pub sequence: u64,
    pub stage: String,
    pub status: String,
    pub reason_code: String,
    pub payload: Value,
}

fn resolve_project_file(raw_path: &str) -> Result<PathBuf, String> {
    let project_file = project_schema::resolve_project_file(Path::new(raw_path));
    if !project_file.exists() {
        return Err(format!(
            "project.json não encontrado: {}",
            project_file.display()
        ));
    }
    Ok(project_file)
}

fn project_revision(project: &Value) -> u64 {
    project
        .get("project_revision")
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

fn require_revision(project: &Value, expected_revision: u64) -> Result<(), String> {
    let actual = project_revision(project);
    if actual != expected_revision {
        return Err(format!(
            "{REVISION_CONFLICT}: esperado {expected_revision}, atual {actual}"
        ));
    }
    Ok(())
}

fn validate_review_decision(decision: &ReviewDecisionInput) -> Result<(), String> {
    if !matches!(decision.actor_kind.as_str(), "human" | "model" | "system") {
        return Err("actor_kind de revisão inválido".into());
    }
    if decision.project_id.trim().is_empty()
        || decision.owner_id.trim().is_empty()
        || decision.actor_id.trim().is_empty()
        || decision.decision.trim().is_empty()
        || decision.reason_code.trim().is_empty()
        || decision.idempotency_key.trim().is_empty()
    {
        return Err("identidade da decisão de revisão incompleta".into());
    }
    if decision.evidence_sha256s.is_empty()
        || decision
            .evidence_sha256s
            .iter()
            .any(|hash| !is_lower_sha256(hash))
    {
        return Err("decisão de revisão exige evidência SHA-256 válida".into());
    }
    if let Some(response) = &decision.preference_response {
        validate_preference_response(response, decision)?;
    }
    Ok(())
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_sha256_field(value: &Value, field: &str) -> Result<String, String> {
    let digest = value
        .as_str()
        .ok_or_else(|| format!("{field} exige SHA-256"))?;
    if !is_lower_sha256(digest) {
        return Err(format!("{field} exige SHA-256 minúsculo"));
    }
    Ok(digest.to_string())
}

fn canonical_json_sha256(value: &Value) -> Result<String, String> {
    let encoded = serde_json::to_vec(value)
        .map_err(|error| format!("falha ao serializar JSON canônico: {error}"))?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

fn require_safe_relative_path(value: &Value, field: &str) -> Result<PathBuf, String> {
    let raw = value
        .as_str()
        .filter(|candidate| !candidate.is_empty())
        .ok_or_else(|| format!("{field} exige caminho relativo seguro"))?;
    if raw.contains('\\')
        || raw.starts_with('/')
        || raw.contains(':')
        || raw
            .split('/')
            .any(|part| part.is_empty() || matches!(part, "." | ".."))
    {
        return Err(format!("{field} exige caminho relativo seguro"));
    }
    Ok(PathBuf::from(raw))
}

fn read_verified_artifact(
    root: &Path,
    reference: &Value,
    field: &str,
) -> Result<(PathBuf, Vec<u8>), String> {
    let object = reference
        .as_object()
        .filter(|object| {
            object.len() == 2
                && object.contains_key("relative_path")
                && object.contains_key("sha256")
        })
        .ok_or_else(|| format!("{field} exige relative_path e sha256"))?;
    let relative = require_safe_relative_path(&object["relative_path"], field)?;
    let expected = require_sha256_field(&object["sha256"], field)?;
    let path = root.join(relative);
    let canonical_root = root
        .canonicalize()
        .map_err(|error| format!("falha ao resolver raiz de {field}: {error}"))?;
    let canonical_path = path
        .canonicalize()
        .map_err(|error| format!("falha ao resolver {field} em {}: {error}", path.display()))?;
    if !canonical_path.starts_with(&canonical_root) {
        return Err(format!("{field} escapou da raiz autorizada"));
    }
    let bytes = fs::read(&canonical_path).map_err(|error| {
        format!(
            "falha ao ler {field} em {}: {error}",
            canonical_path.display()
        )
    })?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected {
        return Err(format!("{field} falhou na verificação SHA-256"));
    }
    Ok((canonical_path, bytes))
}

fn validate_preference_candidate(
    candidate: &Value,
    owner_id: &str,
    artifact_root: &Path,
) -> Result<(), String> {
    let object = candidate
        .as_object()
        .ok_or_else(|| "renderer candidate precisa ser um objeto".to_string())?;
    if candidate["schema"] != "traduzai.renderer-preference.v1"
        || candidate["preference_profile"] != "uncalibrated"
        || candidate["owner_id"].as_str() != Some(owner_id)
        || candidate["hard_safety_passed"].as_bool() != Some(true)
    {
        return Err("renderer candidate incompatível ou insegura".into());
    }
    for field in [
        "target_sha256",
        "source_sha256",
        "style_sha256",
        "layout_plan_sha256",
        "recipe_sha256",
        "output_sha256",
    ] {
        require_sha256_field(&candidate[field], field)?;
    }
    let target = candidate["target_text"]
        .as_str()
        .filter(|target| !target.is_empty())
        .ok_or_else(|| "renderer candidate exige target_text".to_string())?;
    if format!("{:x}", Sha256::digest(target.as_bytes())) != candidate["target_sha256"] {
        return Err("target_sha256 diverge do texto exato".into());
    }
    let safety = candidate
        .pointer("/metrics/raster_safety")
        .ok_or_else(|| "renderer candidate exige raster_safety".to_string())?;
    if safety["status"] != "pass"
        || safety["outside_authorized_body_px"].as_u64() != Some(0)
        || safety["protected_art_overlap_px"].as_u64() != Some(0)
    {
        return Err("renderer candidate falhou segurança raster".into());
    }
    let (preview_path, _) =
        read_verified_artifact(artifact_root, &candidate["preview_ref"], "preview_ref")?;
    read_verified_artifact(artifact_root, &candidate["context_ref"], "context_ref")?;
    if candidate["preview_ref"]["sha256"] != candidate["output_sha256"] {
        return Err(format!(
            "preview {} diverge do output_sha256",
            preview_path.display()
        ));
    }
    let candidate_id = candidate["candidate_id"]
        .as_str()
        .ok_or_else(|| "renderer candidate exige candidate_id".to_string())?;
    let mut identity = object.clone();
    identity.remove("candidate_id");
    let digest = canonical_json_sha256(&Value::Object(identity))?;
    if candidate_id != format!("renderer-candidate:{}", &digest[..32]) {
        return Err("renderer candidate hash inválido".into());
    }
    Ok(())
}

fn validate_preference_comparison(
    comparison: &Value,
    owner_id: &str,
    artifact_root: &Path,
    expected_comparison_sha256: &str,
) -> Result<(), String> {
    let object = comparison
        .as_object()
        .ok_or_else(|| "renderer comparison precisa ser um objeto".to_string())?;
    if comparison["schema"] != "traduzai.renderer-preference.v1"
        || comparison["preference_profile"] != "uncalibrated"
    {
        return Err("renderer comparison incompatível".into());
    }
    let candidates = comparison["candidates"]
        .as_array()
        .filter(|items| items.len() == 2)
        .ok_or_else(|| "renderer comparison exige duas candidatas".to_string())?;
    for candidate in candidates {
        validate_preference_candidate(candidate, owner_id, artifact_root)?;
    }
    for field in ["target_sha256", "source_sha256", "style_sha256"] {
        if candidates[0][field] != candidates[1][field] {
            return Err(format!("renderer candidates divergem em {field}"));
        }
    }
    let first_id = candidates[0]["candidate_id"]
        .as_str()
        .ok_or_else(|| "renderer candidate_id ausente".to_string())?;
    let second_id = candidates[1]["candidate_id"]
        .as_str()
        .ok_or_else(|| "renderer candidate_id ausente".to_string())?;
    if first_id == second_id {
        return Err("renderer comparison exige candidatas distintas".into());
    }
    let positions = comparison["positions"]
        .as_object()
        .filter(|positions| {
            positions.len() == 2 && positions.contains_key("A") && positions.contains_key("B")
        })
        .ok_or_else(|| "renderer comparison exige posições A e B".to_string())?;
    let displayed = [positions["A"].as_str(), positions["B"].as_str()];
    if !displayed.contains(&Some(first_id))
        || !displayed.contains(&Some(second_id))
        || displayed[0] == displayed[1]
    {
        return Err("renderer comparison positions não vinculam as candidatas".into());
    }
    require_sha256_field(&comparison["randomization_sha256"], "randomization_sha256")?;
    let embedded = require_sha256_field(&comparison["comparison_sha256"], "comparison_sha256")?;
    if embedded != expected_comparison_sha256 {
        return Err("comparison_sha256 diverge do índice".into());
    }
    let mut semantic = object.clone();
    semantic.remove("comparison_sha256");
    if canonical_json_sha256(&Value::Object(semantic))? != embedded {
        return Err("renderer comparison hash inválido".into());
    }
    Ok(())
}

fn validate_preference_response(
    response: &Value,
    decision: &ReviewDecisionInput,
) -> Result<(), String> {
    let object = response
        .as_object()
        .ok_or_else(|| "preference_response precisa ser um objeto".to_string())?;
    let required = [
        "schema",
        "comparison_sha256",
        "choice",
        "selected_candidate_id",
        "displayed_candidate_ids",
        "candidate_recipe_sha256s",
        "candidate_output_sha256s",
        "target_sha256",
        "randomization_sha256",
        "actor_kind",
        "actor_id",
        "recorded_at",
        "training_eligible",
    ];
    if object.len() != required.len() || required.iter().any(|field| !object.contains_key(*field)) {
        return Err("renderer preference response schema incompleto".into());
    }
    if response["schema"] != "traduzai.renderer-preference-response.v1" {
        return Err("renderer preference response schema incompatível".into());
    }
    let choice = response["choice"]
        .as_str()
        .filter(|choice| matches!(*choice, "A" | "B" | "equivalent" | "neither" | "unsure"))
        .ok_or_else(|| "renderer preference choice inválido".to_string())?;
    let positions = response["displayed_candidate_ids"]
        .as_object()
        .filter(|positions| {
            positions.len() == 2 && positions.contains_key("A") && positions.contains_key("B")
        })
        .ok_or_else(|| "renderer preference positions inválidas".to_string())?;
    let candidate_a = positions["A"]
        .as_str()
        .filter(|value| value.starts_with("renderer-candidate:"))
        .ok_or_else(|| "renderer candidate A inválida".to_string())?;
    let candidate_b = positions["B"]
        .as_str()
        .filter(|value| value.starts_with("renderer-candidate:"))
        .ok_or_else(|| "renderer candidate B inválida".to_string())?;
    if candidate_a == candidate_b {
        return Err("renderer preference exige candidatas distintas".into());
    }
    let expected_selected = match choice {
        "A" => Some(candidate_a),
        "B" => Some(candidate_b),
        _ => None,
    };
    if response["selected_candidate_id"].as_str() != expected_selected
        || (expected_selected.is_none() && !response["selected_candidate_id"].is_null())
    {
        return Err("renderer preference selected candidate diverge da escolha".into());
    }
    let mut bound_hashes = Vec::new();
    for field in ["comparison_sha256", "target_sha256", "randomization_sha256"] {
        bound_hashes.push(require_sha256_field(&response[field], field)?);
    }
    for field in ["candidate_recipe_sha256s", "candidate_output_sha256s"] {
        let hashes = response[field]
            .as_array()
            .filter(|hashes| hashes.len() == 2)
            .ok_or_else(|| format!("{field} precisa vincular A e B"))?;
        for digest in hashes {
            bound_hashes.push(require_sha256_field(digest, field)?);
        }
    }
    if bound_hashes
        .iter()
        .any(|digest| !decision.evidence_sha256s.contains(digest))
    {
        return Err("ReviewDecision não vincula toda evidência de preferência".into());
    }
    if response["actor_kind"].as_str() != Some(decision.actor_kind.as_str())
        || response["actor_id"].as_str() != Some(decision.actor_id.as_str())
    {
        return Err("ator da preferência diverge da ReviewDecision".into());
    }
    let recorded_at = response["recorded_at"]
        .as_str()
        .ok_or_else(|| "renderer preference recorded_at inválido".to_string())?;
    chrono::DateTime::parse_from_rfc3339(recorded_at)
        .map_err(|_| "renderer preference recorded_at inválido".to_string())?;
    let training_eligible = response["training_eligible"]
        .as_bool()
        .ok_or_else(|| "training_eligible precisa ser booleano".to_string())?;
    if training_eligible && decision.actor_kind != "human" {
        return Err("somente preferência humana pode ser elegível para treino".into());
    }
    Ok(())
}

fn decision_id(decision: &ReviewDecisionInput) -> Result<String, String> {
    let canonical = serde_json::to_vec(decision)
        .map_err(|error| format!("falha ao serializar decisão de revisão: {error}"))?;
    let digest = format!("{:x}", Sha256::digest(canonical));
    Ok(format!("review-decision:{}", &digest[..32]))
}

fn owner_status(decision: &ReviewDecisionInput) -> &'static str {
    if decision.actor_kind != "human" {
        return "review_required";
    }
    match decision.decision.as_str() {
        "accept_candidate" | "approve" => "approved",
        "reject_candidate" | "reject" => "rejected",
        _ => "review_required",
    }
}

fn find_receipt(project: &Value, operation: &str, idempotency_key: &str) -> Option<Value> {
    project
        .pointer("/integration_v1/idempotency_receipts")
        .and_then(Value::as_array)
        .and_then(|receipts| {
            receipts.iter().find(|receipt| {
                receipt.get("operation").and_then(Value::as_str) == Some(operation)
                    && receipt.get("idempotency_key").and_then(Value::as_str)
                        == Some(idempotency_key)
            })
        })
        .and_then(|receipt| receipt.get("result"))
        .cloned()
}

fn append_receipt(
    project: &mut Value,
    operation: &str,
    idempotency_key: &str,
    result: &Value,
) -> Result<(), String> {
    project
        .as_object_mut()
        .ok_or_else(|| "project.json precisa ser um objeto".to_string())?
        .entry("integration_v1")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "integration_v1 precisa ser um objeto".to_string())?
        .entry("idempotency_receipts")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| "integration_v1.idempotency_receipts precisa ser uma lista".to_string())?
        .push(json!({
            "operation": operation,
            "idempotency_key": idempotency_key,
            "result": result,
        }));
    Ok(())
}

fn append_review_decision(
    project: &mut Value,
    decision: &ReviewDecisionInput,
) -> Result<Value, String> {
    validate_review_decision(decision)?;
    if decision.expected_revision != project_revision(project) {
        return Err(format!(
            "{REVISION_CONFLICT}: decisão esperava {}, projeto está em {}",
            decision.expected_revision,
            project_revision(project)
        ));
    }

    if let Some(response) = &decision.preference_response {
        let entry = project
            .pointer(&format!(
                "/integration_v1/renderer_preference_index/owners/{}",
                escape_json_pointer(&decision.owner_id)
            ))
            .ok_or_else(|| "preferência não corresponde a uma comparação publicada".to_string())?;
        if entry.get("state").and_then(Value::as_str) != Some("pending") {
            return Err("comparação de preferência já foi resolvida".into());
        }
        if entry.get("comparison_sha256") != response.get("comparison_sha256") {
            return Err("preferência diverge do comparison_sha256 publicado".into());
        }
    }

    let root = project
        .as_object_mut()
        .ok_or_else(|| "project.json precisa ser um objeto".to_string())?;
    let integration = root
        .entry("integration_v1")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "integration_v1 precisa ser um objeto".to_string())?;
    let decisions = integration
        .entry("review_decisions")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| "integration_v1.review_decisions precisa ser uma lista".to_string())?;

    let id = decision_id(decision)?;
    let status = owner_status(decision);
    let next_revision = decision.expected_revision + 1;
    let mut persisted = serde_json::to_value(decision)
        .map_err(|error| format!("falha ao persistir decisão de revisão: {error}"))?;
    persisted["decision_id"] = Value::String(id.clone());
    persisted["owner_status"] = Value::String(status.into());
    persisted["project_revision"] = Value::from(next_revision);
    decisions.push(persisted);

    if let Some(response) = &decision.preference_response {
        let entry = integration
            .get_mut("renderer_preference_index")
            .and_then(|index| index.get_mut("owners"))
            .and_then(|owners| owners.get_mut(&decision.owner_id))
            .and_then(Value::as_object_mut)
            .ok_or_else(|| "comparação publicada desapareceu durante a decisão".to_string())?;
        entry.insert("state".into(), Value::String("resolved".into()));
        entry.insert("decision_id".into(), Value::String(id.clone()));
        entry.insert("project_revision".into(), Value::from(next_revision));
        for field in [
            "choice",
            "selected_candidate_id",
            "recorded_at",
            "training_eligible",
        ] {
            if let Some(value) = response.get(field) {
                entry.insert(field.into(), value.clone());
            }
        }
    }

    root.insert("project_revision".into(), Value::from(next_revision));
    let result = json!({
        "decision_id": id,
        "project_revision": next_revision,
        "owner_status": status,
    });
    let integration = root
        .get_mut("integration_v1")
        .and_then(Value::as_object_mut)
        .expect("integration_v1 was initialized above");
    let _ = integration;
    append_receipt(
        project,
        "submit_review_decision",
        &decision.idempotency_key,
        &result,
    )?;
    Ok(result)
}

fn project_identity(project: &Value, project_path: &str) -> Result<String, String> {
    if let Some(value) = project
        .get("project_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        return Ok(value.to_string());
    }
    let digest = canonical_json_sha256(&Value::String(project_path.to_string()))?;
    Ok(format!("project:{}", &digest[..32]))
}

fn project_unit_ids(project: &Value) -> Vec<String> {
    let mut result = Vec::new();
    for (page_index, page) in project
        .get("paginas")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let before = result.len();
        for layer in page
            .get("text_layers")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(identity) = layer
                .get("owner_id")
                .or_else(|| layer.get("id"))
                .and_then(Value::as_str)
                .filter(|identity| !identity.trim().is_empty())
            {
                if !result.iter().any(|existing| existing == identity) {
                    result.push(identity.to_string());
                }
            }
        }
        if result.len() == before {
            let number = page
                .get("numero")
                .and_then(Value::as_u64)
                .unwrap_or(page_index as u64 + 1);
            result.push(format!("page:{number:04}"));
        }
    }
    result
}

fn append_project_event(project: &mut Value, event: Value) -> Result<(), String> {
    project
        .as_object_mut()
        .ok_or_else(|| "project.json precisa ser um objeto".to_string())?
        .entry("integration_v1")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "integration_v1 precisa ser um objeto".to_string())?
        .entry("project_events")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| "integration_v1.project_events precisa ser uma lista".to_string())?
        .push(event);
    Ok(())
}

fn start_job_in_project(
    project: &mut Value,
    project_path: &str,
    chapter_id: &str,
    expected_revision: u64,
    idempotency_key: &str,
) -> Result<Value, String> {
    if let Some(result) = find_receipt(project, "start_consumer_fast", idempotency_key) {
        return Ok(result);
    }
    if chapter_id.trim().is_empty() || idempotency_key.trim().is_empty() {
        return Err("identidade do start_consumer_fast incompleta".into());
    }
    require_revision(project, expected_revision)?;
    if let Some(actual) = project.get("capitulo").and_then(Value::as_u64) {
        if chapter_id
            .parse::<u64>()
            .ok()
            .is_some_and(|value| value != actual)
        {
            return Err(format!(
                "capítulo solicitado {chapter_id} diverge do projeto {actual}"
            ));
        }
    }
    let project_id = project_identity(project, project_path)?;
    let job_digest = canonical_json_sha256(&json!({
        "project_path": project_path,
        "chapter_id": chapter_id,
        "idempotency_key": idempotency_key,
    }))?;
    let job_id = format!("consumer-fast:{}", &job_digest[..32]);
    let next_revision = expected_revision + 1;
    let units = project_unit_ids(project)
        .into_iter()
        .map(|unit_id| {
            (
                unit_id,
                json!({"status": "queued", "terminal": false, "reason_code": ""}),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let state = json!({
        "schema": "traduzai.consumer-fast-job.v1",
        "job_id": job_id,
        "project_id": project_id,
        "project_path": project_path,
        "chapter_id": chapter_id,
        "expected_revision": expected_revision,
        "project_revision": next_revision,
        "sequence": 1,
        "stage": "import",
        "status": "queued",
        "reason_code": "job_queued",
        "units": units,
    });
    project
        .as_object_mut()
        .ok_or_else(|| "project.json precisa ser um objeto".to_string())?
        .entry("integration_v1")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "integration_v1 precisa ser um objeto".to_string())?
        .entry("jobs")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "integration_v1.jobs precisa ser um objeto".to_string())?
        .insert(job_id.clone(), state.clone());
    project["project_revision"] = Value::from(next_revision);
    let result = json!({
        "job_id": job_id,
        "project_revision": next_revision,
        "status": "queued",
    });
    append_project_event(
        project,
        json!({
            "event_id": format!("project-event:{}", &job_digest[..32]),
            "job_id": result["job_id"],
            "project_id": project_id,
            "expected_revision": expected_revision,
            "project_revision": next_revision,
            "sequence": 1,
            "stage": "import",
            "status": "queued",
            "reason_code": "job_queued",
            "payload": {"chapter_id": chapter_id},
        }),
    )?;
    append_receipt(project, "start_consumer_fast", idempotency_key, &result)?;
    Ok(result)
}

fn is_terminal_job_status(status: &str) -> bool {
    matches!(
        status,
        "cancelled" | "failed" | "completed" | "complete" | "complete_with_review"
    )
}

fn control_job_in_project(
    project: &mut Value,
    job_id: &str,
    action: &str,
    terminal_unit_ids: &[String],
    expected_revision: u64,
    idempotency_key: &str,
) -> Result<Value, String> {
    let operation = format!("{action}_consumer_fast");
    if let Some(result) = find_receipt(project, &operation, idempotency_key) {
        return Ok(result);
    }
    require_revision(project, expected_revision)?;
    if !matches!(action, "pause" | "resume" | "cancel" | "retry") {
        return Err("ação de job não suportada".into());
    }
    let next_revision = expected_revision + 1;
    let (project_id, next_sequence, status, event_stage, reason_code) = {
        let job = project
            .pointer_mut(&format!(
                "/integration_v1/jobs/{}",
                escape_json_pointer(job_id)
            ))
            .and_then(Value::as_object_mut)
            .ok_or_else(|| format!("job desconhecido: {job_id}"))?;
        let current_status = job
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("failed");
        if action != "retry" && is_terminal_job_status(current_status) {
            return Err("job terminal não pode mudar de estado".into());
        }
        let (status, reason_code) = match action {
            "pause" if matches!(current_status, "queued" | "running") => {
                ("paused", "pause_requested")
            }
            "resume" if current_status == "paused" => ("queued", "resume_requested"),
            "cancel" => ("cancelled", "job_cancelled"),
            "retry" => ("queued", "retry_requested"),
            "pause" => return Err("somente job em execução pode ser pausado".into()),
            "resume" => return Err("somente job pausado pode ser retomado".into()),
            _ => unreachable!(),
        };
        if action == "retry" {
            if terminal_unit_ids.is_empty() {
                return Err("retry exige pelo menos uma unidade terminal".into());
            }
            let units = job
                .get_mut("units")
                .and_then(Value::as_object_mut)
                .ok_or_else(|| "job sem unidades persistidas".to_string())?;
            for unit_id in terminal_unit_ids {
                let unit = units
                    .get_mut(unit_id)
                    .and_then(Value::as_object_mut)
                    .ok_or_else(|| format!("unidade desconhecida: {unit_id}"))?;
                if unit.get("terminal").and_then(Value::as_bool) != Some(true) {
                    return Err(format!("unidade não é retentável: {unit_id}"));
                }
                unit.insert("status".into(), Value::String("queued".into()));
                unit.insert("terminal".into(), Value::Bool(false));
                unit.insert(
                    "reason_code".into(),
                    Value::String("retry_requested".into()),
                );
            }
        } else if action == "cancel" {
            if let Some(units) = job.get_mut("units").and_then(Value::as_object_mut) {
                for unit in units.values_mut().filter_map(Value::as_object_mut) {
                    if unit.get("terminal").and_then(Value::as_bool) != Some(true) {
                        unit.insert("status".into(), Value::String("cancelled".into()));
                        unit.insert("terminal".into(), Value::Bool(true));
                        unit.insert("reason_code".into(), Value::String("job_cancelled".into()));
                    }
                }
            }
        }
        let next_sequence = job.get("sequence").and_then(Value::as_u64).unwrap_or(0) + 1;
        let project_id = job
            .get("project_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        job.insert("status".into(), Value::String(status.into()));
        job.insert("reason_code".into(), Value::String(reason_code.into()));
        job.insert("project_revision".into(), Value::from(next_revision));
        job.insert("sequence".into(), Value::from(next_sequence));
        (
            project_id,
            next_sequence,
            status.to_string(),
            job.get("stage")
                .and_then(Value::as_str)
                .unwrap_or("analysis")
                .to_string(),
            reason_code.to_string(),
        )
    };
    project["project_revision"] = Value::from(next_revision);
    let result = json!({
        "job_id": job_id,
        "project_revision": next_revision,
        "status": status,
        "stage": event_stage.clone(),
        "reason_code": reason_code.clone(),
        "sequence": next_sequence,
    });
    let event_body = json!({
        "job_id": job_id,
        "project_id": project_id,
        "expected_revision": expected_revision,
        "project_revision": next_revision,
        "sequence": next_sequence,
        "stage": event_stage,
        "status": result["status"],
        "reason_code": reason_code,
    });
    let event_id = format!(
        "project-event:{}",
        &canonical_json_sha256(&event_body)?[..32]
    );
    let payload = json!({"terminal_unit_ids": terminal_unit_ids});
    let mut persisted_event = event_body;
    persisted_event["event_id"] = Value::String(event_id);
    persisted_event["payload_sha256"] = Value::String(canonical_json_sha256(&payload)?);
    persisted_event["payload"] = payload;
    append_project_event(project, persisted_event)?;
    append_receipt(project, &operation, idempotency_key, &result)?;
    Ok(result)
}

fn escape_json_pointer(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

fn persist_event_in_project(
    project: &mut Value,
    event: &ProjectEventInput,
    expected_revision: u64,
    idempotency_key: &str,
) -> Result<Value, String> {
    if let Some(result) = find_receipt(project, "persist_project_event", idempotency_key) {
        return Ok(result);
    }
    require_revision(project, expected_revision)?;
    if event.expected_revision != expected_revision
        || event.project_revision != expected_revision + 1
        || event.job_id.trim().is_empty()
        || event.project_id.trim().is_empty()
        || event.stage.trim().is_empty()
        || event.reason_code.trim().is_empty()
        || !matches!(
            event.status.as_str(),
            "queued"
                | "running"
                | "pausing"
                | "paused"
                | "cancelling"
                | "cancelled"
                | "failed"
                | "blocked"
                | "awaiting_review"
                | "completed"
        )
    {
        return Err("ProjectEvent possui identidade, revisão ou estado inválido".into());
    }
    let last_sequence = project
        .pointer(&format!(
            "/integration_v1/jobs/{}/sequence",
            escape_json_pointer(&event.job_id)
        ))
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("job desconhecido: {}", event.job_id))?;
    if event.sequence <= last_sequence {
        return Err(format!(
            "ProjectEvent sequence não é monotônica: {} <= {last_sequence}",
            event.sequence
        ));
    }
    let job_project_id = project
        .pointer(&format!(
            "/integration_v1/jobs/{}/project_id",
            escape_json_pointer(&event.job_id)
        ))
        .and_then(Value::as_str)
        .unwrap_or("");
    if job_project_id != event.project_id {
        return Err("ProjectEvent pertence a outro projeto".into());
    }
    let current_status = project
        .pointer(&format!(
            "/integration_v1/jobs/{}/status",
            escape_json_pointer(&event.job_id)
        ))
        .and_then(Value::as_str)
        .unwrap_or("failed");
    if is_terminal_job_status(current_status) {
        return Err("job terminal não aceita novos eventos".into());
    }
    let payload_sha256 = canonical_json_sha256(&event.payload)?;
    let body = json!({
        "job_id": event.job_id,
        "project_id": event.project_id,
        "expected_revision": event.expected_revision,
        "project_revision": event.project_revision,
        "sequence": event.sequence,
        "stage": event.stage,
        "status": event.status,
        "reason_code": event.reason_code,
        "payload_sha256": payload_sha256,
    });
    let event_id = format!("project-event:{}", &canonical_json_sha256(&body)?[..32]);
    let mut persisted = body;
    persisted["event_id"] = Value::String(event_id.clone());
    persisted["payload"] = event.payload.clone();
    let job = project
        .pointer_mut(&format!(
            "/integration_v1/jobs/{}",
            escape_json_pointer(&event.job_id)
        ))
        .and_then(Value::as_object_mut)
        .expect("job existence was validated above");
    job.insert(
        "project_revision".into(),
        Value::from(event.project_revision),
    );
    job.insert("sequence".into(), Value::from(event.sequence));
    job.insert("stage".into(), Value::String(event.stage.clone()));
    job.insert("status".into(), Value::String(event.status.clone()));
    job.insert(
        "reason_code".into(),
        Value::String(event.reason_code.clone()),
    );
    project["project_revision"] = Value::from(event.project_revision);
    append_project_event(project, persisted)?;
    let result = json!({
        "event_id": event_id,
        "project_revision": event.project_revision,
    });
    append_receipt(project, "persist_project_event", idempotency_key, &result)?;
    Ok(result)
}

fn job_registry_file(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|error| format!("falha ao localizar dados locais: {error}"))?
        .join("integration_v1")
        .join("job_registry.json"))
}

fn load_job_registry(path: &Path) -> Result<HashMap<String, String>, String> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let bytes =
        fs::read(path).map_err(|error| format!("falha ao ler registro de jobs: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("registro de jobs inválido: {error}"))
}

fn save_job_registry(path: &Path, registry: &HashMap<String, String>) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "registro de jobs sem diretório pai".to_string())?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("falha ao criar diretório do registro de jobs: {error}"))?;
    let temporary = path.with_extension(format!("json.{}.tmp", std::process::id()));
    let bytes = serde_json::to_vec_pretty(registry)
        .map_err(|error| format!("falha ao serializar registro de jobs: {error}"))?;
    fs::write(&temporary, bytes)
        .map_err(|error| format!("falha ao gravar registro de jobs: {error}"))?;
    if path.exists() {
        fs::remove_file(path)
            .map_err(|error| format!("falha ao substituir registro de jobs: {error}"))?;
    }
    fs::rename(&temporary, path)
        .map_err(|error| format!("falha ao publicar registro de jobs: {error}"))
}

fn register_job(app: &AppHandle, job_id: &str, project_file: &Path) -> Result<(), String> {
    let registry_file = job_registry_file(app)?;
    let mut registry = load_job_registry(&registry_file)?;
    registry.insert(
        job_id.to_string(),
        project_file.to_string_lossy().into_owned(),
    );
    save_job_registry(&registry_file, &registry)
}

fn resolve_registered_job(app: &AppHandle, job_id: &str) -> Result<PathBuf, String> {
    let registry = load_job_registry(&job_registry_file(app)?)?;
    let path = registry
        .get(job_id)
        .map(PathBuf::from)
        .ok_or_else(|| format!("job desconhecido no registro durável: {job_id}"))?;
    let project_file = resolve_project_file(&path.to_string_lossy())?;
    let project = project_schema::load_project_value(&project_file)?;
    if project
        .pointer(&format!(
            "/integration_v1/jobs/{}",
            escape_json_pointer(job_id)
        ))
        .is_none()
    {
        return Err(format!("registro de job sem estado no projeto: {job_id}"));
    }
    Ok(project_file)
}

fn latest_job_event(project_file: &Path, job_id: &str) -> Result<Option<Value>, String> {
    let project = project_schema::load_project_value(project_file)?;
    Ok(project
        .pointer("/integration_v1/project_events")
        .and_then(Value::as_array)
        .and_then(|events| {
            events
                .iter()
                .rev()
                .find(|event| event["job_id"].as_str() == Some(job_id))
        })
        .cloned())
}

fn emit_latest_job_event(app: &AppHandle, project_file: &Path, job_id: &str) -> Result<(), String> {
    if let Some(event) = latest_job_event(project_file, job_id)? {
        app.emit("consumer-fast-project-event", event)
            .map_err(|error| format!("falha ao emitir ProjectEvent: {error}"))?;
    }
    Ok(())
}

fn job_runtime_dir(app: &AppHandle, job_id: &str) -> Result<PathBuf, String> {
    let digest = format!("{:x}", Sha256::digest(job_id.as_bytes()));
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|error| format!("falha ao localizar dados locais: {error}"))?
        .join("integration_v1")
        .join("jobs")
        .join(digest))
}

fn set_job_marker(
    app: &AppHandle,
    job_id: &str,
    marker: &str,
    enabled: bool,
) -> Result<(), String> {
    let directory = job_runtime_dir(app, job_id)?;
    let path = directory.join(format!("{marker}.flag"));
    if enabled {
        fs::create_dir_all(&directory)
            .map_err(|error| format!("falha ao criar diretório do job: {error}"))?;
        fs::write(&path, marker)
            .map_err(|error| format!("falha ao gravar marcador {marker}: {error}"))
    } else if path.exists() {
        fs::remove_file(&path)
            .map_err(|error| format!("falha ao remover marcador {marker}: {error}"))
    } else {
        Ok(())
    }
}

fn job_marker_exists(app: &AppHandle, job_id: &str, marker: &str) -> Result<bool, String> {
    Ok(job_runtime_dir(app, job_id)?
        .join(format!("{marker}.flag"))
        .is_file())
}

fn persist_runtime_transition(
    app: &AppHandle,
    project_file: &Path,
    job_id: &str,
    stage: &str,
    status: &str,
    reason_code: &str,
    detail: Value,
) -> Result<Value, String> {
    let result = project_schema::edit_project_value(project_file, |project| {
        let expected_revision = project_revision(project);
        let next_revision = expected_revision + 1;
        let (project_id, next_sequence, terminal_unit_ids) = {
            let job = project
                .pointer_mut(&format!(
                    "/integration_v1/jobs/{}",
                    escape_json_pointer(job_id)
                ))
                .and_then(Value::as_object_mut)
                .ok_or_else(|| format!("job desconhecido: {job_id}"))?;
            let current_status = job
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("failed");
            if is_terminal_job_status(current_status) && current_status != status {
                return Err(format!(
                    "job terminal não aceita transição de runtime: {current_status}"
                ));
            }
            let next_sequence = job.get("sequence").and_then(Value::as_u64).unwrap_or(0) + 1;
            let project_id = job
                .get("project_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            job.insert("project_revision".into(), Value::from(next_revision));
            job.insert("sequence".into(), Value::from(next_sequence));
            job.insert("stage".into(), Value::String(stage.to_string()));
            job.insert("status".into(), Value::String(status.to_string()));
            job.insert("reason_code".into(), Value::String(reason_code.to_string()));
            if status == "failed" {
                if let Some(units) = job.get_mut("units").and_then(Value::as_object_mut) {
                    for unit in units.values_mut().filter_map(Value::as_object_mut) {
                        if unit.get("terminal").and_then(Value::as_bool) != Some(true) {
                            unit.insert("status".into(), Value::String("failed".into()));
                            unit.insert("terminal".into(), Value::Bool(true));
                            unit.insert(
                                "reason_code".into(),
                                Value::String(reason_code.to_string()),
                            );
                        }
                    }
                }
            }
            let terminal_unit_ids = job
                .get("units")
                .and_then(Value::as_object)
                .into_iter()
                .flat_map(|units| units.iter())
                .filter_map(|(unit_id, unit)| {
                    (unit.get("terminal").and_then(Value::as_bool) == Some(true))
                        .then_some(unit_id.clone())
                })
                .collect::<Vec<_>>();
            (project_id, next_sequence, terminal_unit_ids)
        };
        project["project_revision"] = Value::from(next_revision);
        let payload = json!({
            "detail": detail,
            "terminal_unit_ids": terminal_unit_ids,
        });
        let event_body = json!({
            "job_id": job_id,
            "project_id": project_id,
            "expected_revision": expected_revision,
            "project_revision": next_revision,
            "sequence": next_sequence,
            "stage": stage,
            "status": status,
            "reason_code": reason_code,
            "payload_sha256": canonical_json_sha256(&payload)?,
        });
        let event_id = format!(
            "project-event:{}",
            &canonical_json_sha256(&event_body)?[..32]
        );
        let mut event = event_body;
        event["event_id"] = Value::String(event_id.clone());
        event["payload"] = payload;
        append_project_event(project, event)?;
        Ok(json!({
            "event_id": event_id,
            "project_revision": next_revision,
            "sequence": next_sequence,
            "stage": stage,
            "status": status,
            "reason_code": reason_code,
        }))
    })?;
    emit_latest_job_event(app, project_file, job_id)?;
    Ok(result)
}

async fn wait_for_job_boundary(
    app: &AppHandle,
    project_file: &Path,
    job_id: &str,
) -> Result<(), String> {
    loop {
        let project = project_schema::load_project_value(project_file)?;
        let status = project
            .pointer(&format!(
                "/integration_v1/jobs/{}/status",
                escape_json_pointer(job_id)
            ))
            .and_then(Value::as_str)
            .ok_or_else(|| format!("job desconhecido: {job_id}"))?;
        if status == "cancelled" || job_marker_exists(app, job_id, "cancel")? {
            return Err("JOB_CANCELLED".into());
        }
        if is_terminal_job_status(status) {
            return Err("JOB_TERMINAL".into());
        }
        if status != "paused" && !job_marker_exists(app, job_id, "pause")? {
            return Ok(());
        }
        sleep(Duration::from_millis(100)).await;
    }
}

fn physical_pipeline_config(
    project: &Value,
    job_id: &str,
    work_dir: &Path,
    models_dir: &Path,
    logs_dir: &Path,
    pause_file: &Path,
    cancel_file: &Path,
) -> Result<Value, String> {
    let source_path = project
        .get("consumer_source_path")
        .or_else(|| project.get("source_path"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            "SOURCE_PATH_REQUIRED: projeto não referencia a fonte original".to_string()
        })?;
    let obra = project
        .get("obra")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("Obra sem título");
    let capitulo = project.get("capitulo").and_then(Value::as_u64).unwrap_or(1);
    let runtime_job_id: String = job_id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '-'
            }
        })
        .collect();
    Ok(json!({
        "runtime_id": "consumer-fast-v1",
        "job_id": runtime_job_id,
        "integration_job_id": job_id,
        "source_path": source_path,
        "work_dir": work_dir.to_string_lossy(),
        "obra": obra,
        "work_title_user_provided": project
            .get("work_title_user_provided")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        "capitulo": capitulo,
        "idioma_origem": project
            .get("idioma_origem")
            .and_then(Value::as_str)
            .unwrap_or("en"),
        "idioma_destino": project
            .get("idioma_destino")
            .and_then(Value::as_str)
            .unwrap_or("pt-BR"),
        "qualidade": project
            .get("qualidade")
            .and_then(Value::as_str)
            .unwrap_or("alta"),
        "glossario": project.get("glossario").cloned().unwrap_or_else(|| json!({})),
        "engine_preset_id": project.get("engine_preset_id").cloned().unwrap_or(Value::Null),
        // `project.mode` describes how the Studio document was created (for
        // example, a manual or fixture-backed document). It is not authority
        // to downgrade the native Consumer Fast worker. A job dispatched from
        // this production command always executes the real pipeline.
        "mode": "real",
        "contexto": project.get("contexto").cloned().unwrap_or_else(|| json!({
            "sinopse": "", "genero": [], "personagens": [], "aliases": [],
            "termos": [], "relacoes": [], "faccoes": [], "resumo_por_arco": [],
            "memoria_lexical": {}, "fontes_usadas": []
        })),
        "models_dir": models_dir.to_string_lossy(),
        "logs_dir": logs_dir.to_string_lossy(),
        "pause_file": pause_file.to_string_lossy(),
        "cancel_file": cancel_file.to_string_lossy(),
        "owner_graph_mode": "enforce",
        "style_copy_mode": "shadow"
    }))
}

fn physical_pipeline_work_dir(runtime_dir: &Path, revision: u64) -> PathBuf {
    runtime_dir.join(format!("pipeline-staging-r{revision}"))
}

fn resolve_physical_output_project(work_dir: &Path, output_path: &str) -> Result<PathBuf, String> {
    let candidate = if output_path.trim().is_empty() {
        work_dir.join("project.json")
    } else {
        project_schema::resolve_project_file(Path::new(output_path))
    };
    if !candidate.is_file() {
        return Err(format!(
            "PHYSICAL_OUTPUT_MISSING: project.json não encontrado em {}",
            candidate.display()
        ));
    }
    Ok(candidate)
}

fn attach_job_ledger_to_physical_output(
    seed_project_file: &Path,
    output_project_file: &Path,
    job_id: &str,
) -> Result<(), String> {
    let seed = project_schema::load_project_value(seed_project_file)?;
    let integration = seed
        .get("integration_v1")
        .cloned()
        .ok_or_else(|| "projeto de origem não contém ledger integration_v1".to_string())?;
    let seed_metadata: Vec<(&str, Value)> = [
        "id",
        "project_id",
        "obra",
        "capitulo",
        "chapter_id",
        "consumer_source_path",
        "source_path",
        "output_path",
        "work_title_user_provided",
        "idioma_origem",
        "idioma_destino",
        "qualidade",
        "glossario",
        "contexto",
        "engine_preset_id",
        "mode",
    ]
    .into_iter()
    .filter_map(|key| seed.get(key).cloned().map(|value| (key, value)))
    .collect();
    let seed_revision = project_revision(&seed);
    project_schema::edit_project_value(output_project_file, |output| {
        let root = output
            .as_object_mut()
            .ok_or_else(|| "project.json físico precisa ser um objeto".to_string())?;
        root.insert("integration_v1".into(), integration);
        for (key, value) in &seed_metadata {
            root.insert((*key).into(), value.clone());
        }
        if let Some(project_id) = root.get("project_id").cloned() {
            root.insert("id".into(), project_id);
        }
        root.insert("project_revision".into(), Value::from(seed_revision));
        let job = output
            .pointer_mut(&format!(
                "/integration_v1/jobs/{}",
                escape_json_pointer(job_id)
            ))
            .and_then(Value::as_object_mut)
            .ok_or_else(|| format!("job ausente após promoção física: {job_id}"))?;
        if let Some(units) = job.get_mut("units").and_then(Value::as_object_mut) {
            for unit in units.values_mut().filter_map(Value::as_object_mut) {
                unit.insert("status".into(), Value::String("complete".into()));
                unit.insert("terminal".into(), Value::Bool(true));
                unit.insert(
                    "reason_code".into(),
                    Value::String("physical_pipeline_completed".into()),
                );
            }
        }
        Ok(())
    })
}

async fn execute_consumer_job(app: AppHandle, project_file: PathBuf, job_id: String) {
    let outcome = async {
        wait_for_job_boundary(&app, &project_file, &job_id).await?;
        persist_runtime_transition(
            &app,
            &project_file,
            &job_id,
            "import",
            "running",
            "worker_started",
            json!({}),
        )?;

        let project = project_schema::load_project_value(&project_file)?;
        let runtime_dir = job_runtime_dir(&app, &job_id)?;
        let work_dir = physical_pipeline_work_dir(&runtime_dir, project_revision(&project));
        let data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| format!("falha ao localizar dados locais: {error}"))?;
        let models_dir = crate::integration_runtime_adapter::physical_models_dir(&app)?;
        let logs_dir = data_dir.join("logs");
        fs::create_dir_all(&work_dir)
            .map_err(|error| format!("falha ao preparar staging físico: {error}"))?;
        fs::create_dir_all(&models_dir)
            .map_err(|error| format!("falha ao preparar modelos: {error}"))?;
        fs::create_dir_all(&logs_dir)
            .map_err(|error| format!("falha ao preparar logs: {error}"))?;
        let pause_file = runtime_dir.join("pause.flag");
        let cancel_file = runtime_dir.join("cancel.flag");
        let config = physical_pipeline_config(
            &project,
            &job_id,
            &work_dir,
            &models_dir,
            &logs_dir,
            &pause_file,
            &cancel_file,
        )?;
        let config_file = runtime_dir.join("pipeline_config.json");
        fs::write(
            &config_file,
            serde_json::to_vec_pretty(&config)
                .map_err(|error| format!("falha ao serializar config física: {error}"))?,
        )
        .map_err(|error| format!("falha ao persistir config física: {error}"))?;
        let output_path =
            crate::integration_runtime_adapter::run_pipeline_config_file(&app, &config_file)
                .await?;
        let output_project = resolve_physical_output_project(&work_dir, &output_path)?;
        attach_job_ledger_to_physical_output(&project_file, &output_project, &job_id)?;
        register_job(&app, &job_id, &output_project)?;
        persist_runtime_transition(
            &app,
            &output_project,
            &job_id,
            "review",
            "awaiting_review",
            "physical_pipeline_completed",
            json!({
                "final_export_allowed": false,
                "output_project_path": output_project,
                "staging_work_dir": work_dir,
            }),
        )?;
        Ok::<(), String>(())
    }
    .await;

    if let Err(error) = outcome {
        if !matches!(error.as_str(), "JOB_CANCELLED" | "JOB_TERMINAL") {
            let _ = persist_runtime_transition(
                &app,
                &project_file,
                &job_id,
                "persist",
                "failed",
                "worker_stage_failed",
                json!({"error": error}),
            );
        }
    }
    ACTIVE_CONSUMER_JOBS.lock().await.remove(&job_id);
}

async fn spawn_consumer_job(app: AppHandle, project_file: PathBuf, job_id: String) -> bool {
    let mut active = ACTIVE_CONSUMER_JOBS.lock().await;
    if !active.insert(job_id.clone()) {
        return false;
    }
    drop(active);
    tokio::spawn(execute_consumer_job(app, project_file, job_id));
    true
}

fn control_registered_job(
    app: &AppHandle,
    job_id: &str,
    action: &str,
    terminal_unit_ids: &[String],
    expected_revision: u64,
    idempotency_key: &str,
) -> Result<Value, String> {
    let project_file = resolve_registered_job(app, job_id)?;
    let result = project_schema::edit_project_value(&project_file, |project| {
        control_job_in_project(
            project,
            job_id,
            action,
            terminal_unit_ids,
            expected_revision,
            idempotency_key,
        )
    })?;
    emit_latest_job_event(app, &project_file, job_id)?;
    Ok(result)
}

fn export_decision(project: &Value) -> Value {
    let gate = project.pointer("/qa/export_gate").unwrap_or(&Value::Null);
    let status = gate
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("BLOCK");
    let gate_allowed = gate.get("allowed").and_then(Value::as_bool) == Some(true);
    let verified = project.get("verified").and_then(Value::as_bool) == Some(true);
    let completion_status = project
        .get("completion_status")
        .and_then(Value::as_str)
        .unwrap_or("");
    let output_review_state = project
        .get("output_review_state")
        .and_then(Value::as_str)
        .unwrap_or("");
    let blocker_count = project
        .get("blocker_count")
        .and_then(Value::as_u64)
        .or_else(|| gate.get("blocker_count").and_then(Value::as_u64))
        .unwrap_or(0);
    let critical_issue_count = gate
        .get("critical_issue_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let critical_flag_count = gate
        .get("critical_flag_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let review_issue_count = gate
        .get("review_issue_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let review_flag_count = gate
        .get("review_flag_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let issue_count = gate
        .get("issue_count")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| {
            gate.get("issues")
                .and_then(Value::as_array)
                .map(|issues| issues.len() as u64)
                .unwrap_or(0)
        });
    let allowed = verified
        && completion_status == "approved"
        && output_review_state == "approved"
        && status == "PASS"
        && gate_allowed
        && blocker_count == 0
        && critical_issue_count == 0
        && critical_flag_count == 0
        && review_issue_count == 0
        && review_flag_count == 0
        && issue_count == 0;

    json!({
        "allowed": allowed,
        "status": status,
        "gate_allowed": gate_allowed,
        "verified": verified,
        "completion_status": completion_status,
        "output_review_state": output_review_state,
        "blocker_count": blocker_count,
        "critical_issue_count": critical_issue_count,
        "critical_flag_count": critical_flag_count,
        "review_issue_count": review_issue_count,
        "review_flag_count": review_flag_count,
        "issue_count": issue_count,
    })
}

fn record_final_review_approval(
    project: &mut Value,
    expected_revision: u64,
    actor_kind: &str,
    actor_id: &str,
    idempotency_key: &str,
) -> Result<Value, String> {
    if let Some(result) = find_receipt(project, "approve_final_review", idempotency_key) {
        return Ok(result);
    }
    require_revision(project, expected_revision)?;
    if actor_kind != "human" || actor_id.trim().is_empty() || actor_id == "test/synthetic" {
        return Err("a aprovação final exige confirmação humana explícita".into());
    }
    if idempotency_key.trim().is_empty() {
        return Err("idempotency_key é obrigatória".into());
    }

    let gate = project.pointer("/qa/export_gate").unwrap_or(&Value::Null);
    let count = |field: &str| gate.get(field).and_then(Value::as_u64).unwrap_or(0);
    let issue_count = gate
        .get("issue_count")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| {
            gate.get("issues")
                .and_then(Value::as_array)
                .map(|issues| issues.len() as u64)
                .unwrap_or(0)
        });
    let blocker_count = project
        .get("blocker_count")
        .and_then(Value::as_u64)
        .or_else(|| gate.get("blocker_count").and_then(Value::as_u64))
        .unwrap_or(0);
    let clean_gate = gate.get("status").and_then(Value::as_str) == Some("PASS")
        && gate.get("allowed").and_then(Value::as_bool) == Some(true)
        && blocker_count == 0
        && issue_count == 0
        && count("critical_issue_count") == 0
        && count("critical_flag_count") == 0
        && count("review_issue_count") == 0
        && count("review_flag_count") == 0
        && project.get("needs_review").and_then(Value::as_bool) != Some(true);
    if !clean_gate {
        return Err("aprovação final bloqueada: o gate real ainda possui pendências".into());
    }

    let next_revision = expected_revision + 1;
    let approved_at = chrono::Utc::now().to_rfc3339();
    let decision = json!({
        "schema": "traduzai.final-review-decision.v1",
        "decision": "approve",
        "actor_kind": actor_kind,
        "actor_id": actor_id,
        "expected_revision": expected_revision,
        "project_revision": next_revision,
        "approved_at": approved_at,
        "idempotency_key": idempotency_key,
        "gate_sha256": canonical_json_sha256(gate)?,
    });
    let root = project
        .as_object_mut()
        .ok_or_else(|| "project.json precisa ser um objeto".to_string())?;
    root.insert("verified".into(), Value::Bool(true));
    root.insert("completion_status".into(), Value::String("approved".into()));
    root.insert("output_review_state".into(), Value::String("approved".into()));
    root.insert("needs_review".into(), Value::Bool(false));
    root.insert("verified_at".into(), Value::String(approved_at));
    root.insert("project_revision".into(), Value::from(next_revision));
    root.entry("integration_v1")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "integration_v1 precisa ser um objeto".to_string())?
        .entry("final_review_decisions")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| "integration_v1.final_review_decisions precisa ser uma lista".to_string())?
        .push(decision.clone());

    let result = json!({
        "project_revision": next_revision,
        "decision": decision,
        "export_decision": export_decision(project),
    });
    append_receipt(project, "approve_final_review", idempotency_key, &result)?;
    Ok(result)
}

fn record_final_export_in_project(
    project: &mut Value,
    expected_revision: u64,
    idempotency_key: &str,
    destination: &str,
    artifact_sha256: &str,
    artifact_size: u64,
) -> Result<Value, String> {
    if let Some(result) = find_receipt(project, "export_final", idempotency_key) {
        return Ok(result);
    }
    require_revision(project, expected_revision)?;
    if export_decision(project)["allowed"].as_bool() != Some(true) {
        return Err("exportação final bloqueada por pendências reais do projeto".into());
    }
    if !is_lower_sha256(artifact_sha256) || destination.trim().is_empty() {
        return Err("artefato final exige destino e SHA-256 válidos".into());
    }
    let next_revision = expected_revision + 1;
    let export_manifest = json!({
        "schema": "traduzai.final-export-manifest.v1",
        "destination": destination,
        "artifact_sha256": artifact_sha256,
        "artifact_size": artifact_size,
        "source_project_revision": expected_revision,
    });
    let publication_receipt = json!({
        "schema": "traduzai.publication-receipt.v1",
        "export_manifest_sha256": canonical_json_sha256(&export_manifest)?,
        "artifact_sha256": artifact_sha256,
        "published_at": chrono::Utc::now().to_rfc3339(),
        "gate": export_decision(project),
        "human_review_claimed": false,
    });
    let result = json!({
        "project_revision": next_revision,
        "export_manifest": export_manifest,
        "publication_receipt": publication_receipt,
    });
    project
        .as_object_mut()
        .ok_or_else(|| "project.json precisa ser um objeto".to_string())?
        .entry("integration_v1")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "integration_v1 precisa ser um objeto".to_string())?
        .entry("final_exports")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| "integration_v1.final_exports precisa ser uma lista".to_string())?
        .push(result.clone());
    project["project_revision"] = Value::from(next_revision);
    append_receipt(project, "export_final", idempotency_key, &result)?;
    Ok(result)
}

fn file_sha256(path: &Path) -> Result<(String, u64), String> {
    let mut file = fs::File::open(path)
        .map_err(|error| format!("falha ao abrir artefato {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut size = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("falha ao ler artefato {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        size += read as u64;
    }
    Ok((format!("{:x}", hasher.finalize()), size))
}

fn write_diagnostic_archive(destination: &Path, manifest: &Value) -> Result<(), String> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("falha ao preparar diagnóstico: {error}"))?;
    }
    let file = fs::File::create(destination)
        .map_err(|error| format!("falha ao criar diagnóstico: {error}"))?;
    let mut archive = zip::ZipWriter::new(file);
    archive
        .start_file(
            "diagnostic_manifest.json",
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated),
        )
        .map_err(|error| format!("falha ao iniciar diagnóstico: {error}"))?;
    archive
        .write_all(
            &serde_json::to_vec_pretty(manifest)
                .map_err(|error| format!("falha ao serializar diagnóstico: {error}"))?,
        )
        .map_err(|error| format!("falha ao gravar diagnóstico: {error}"))?;
    archive
        .finish()
        .map_err(|error| format!("falha ao finalizar diagnóstico: {error}"))?;
    Ok(())
}

#[derive(Clone, Debug)]
struct PreparedOwnerLayout {
    page_index: usize,
    layer_index: usize,
    page: Value,
    target_text: String,
    request_sha256: String,
    page_sha256: String,
    invalidated_stages: [&'static str; 5],
}

fn merge_json_object(target: &mut Value, patch: &Value, field: &str) -> Result<(), String> {
    let patch = patch
        .as_object()
        .ok_or_else(|| format!("{field} precisa ser um objeto"))?;
    let target = target
        .as_object_mut()
        .ok_or_else(|| format!("{field} persistido precisa ser um objeto"))?;
    for (key, value) in patch {
        target.insert(key.clone(), value.clone());
    }
    Ok(())
}

fn require_layout_bbox(value: &Value) -> Result<(), String> {
    let values = value
        .as_array()
        .filter(|values| values.len() == 4)
        .ok_or_else(|| "layout_bbox exige quatro coordenadas".to_string())?;
    let coords = values
        .iter()
        .map(Value::as_i64)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| "layout_bbox exige coordenadas inteiras".to_string())?;
    if coords[2] <= coords[0] || coords[3] <= coords[1] {
        return Err("layout_bbox precisa ter área positiva".into());
    }
    Ok(())
}

fn prepare_owner_layout(
    project: &Value,
    owner_id: &str,
    layout_request: &Value,
) -> Result<PreparedOwnerLayout, String> {
    if owner_id.trim().is_empty() {
        return Err("owner_id é obrigatório".into());
    }
    let request = layout_request
        .as_object()
        .ok_or_else(|| "layout_request precisa ser um objeto".to_string())?;
    if request
        .get("owner_id")
        .and_then(Value::as_str)
        .is_some_and(|identity| identity != owner_id)
    {
        return Err("layout_request pertence a outro owner".into());
    }
    let pages = project
        .get("paginas")
        .and_then(Value::as_array)
        .ok_or_else(|| "projeto sem páginas".to_string())?;
    let mut found = Vec::new();
    for (page_index, page) in pages.iter().enumerate() {
        for (layer_index, layer) in page
            .get("text_layers")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            let identity = layer
                .get("owner_id")
                .or_else(|| layer.get("id"))
                .and_then(Value::as_str);
            if identity == Some(owner_id) {
                found.push((page_index, layer_index));
            }
        }
    }
    if found.len() != 1 {
        return Err(format!(
            "owner_id precisa identificar exatamente uma camada, encontrou {}",
            found.len()
        ));
    }
    let (page_index, layer_index) = found[0];
    let mut page = pages[page_index].clone();
    let layer = page
        .get_mut("text_layers")
        .and_then(Value::as_array_mut)
        .and_then(|layers| layers.get_mut(layer_index))
        .ok_or_else(|| "camada do owner desapareceu durante preparação".to_string())?;
    let layer = layer
        .as_object_mut()
        .ok_or_else(|| "camada do owner precisa ser um objeto".to_string())?;
    let target_text = if let Some(lines) = request.get("line_breaks") {
        lines
            .as_array()
            .filter(|lines| !lines.is_empty())
            .ok_or_else(|| "line_breaks precisa ser uma lista não vazia".to_string())?
            .iter()
            .map(|line| {
                line.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| "line_breaks exige texto".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?
            .join("\n")
    } else {
        request
            .get("text")
            .or_else(|| request.get("target"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| {
                layer
                    .get("translated")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string()
            })
    };
    if target_text.trim().is_empty() {
        return Err("retypeset exige texto de destino não vazio".into());
    }
    layer.insert("translated".into(), Value::String(target_text.clone()));
    if let Some(bbox) = request.get("layout_bbox") {
        require_layout_bbox(bbox)?;
        layer.insert("layout_bbox".into(), bbox.clone());
    }
    if let Some(style_patch) = request.get("style") {
        let style = layer.entry("style").or_insert_with(|| json!({}));
        merge_json_object(style, style_patch, "style")?;
    }
    if let Some(font_size) = request
        .get("font_size_px")
        .or_else(|| request.get("font_size"))
    {
        let value = font_size
            .as_u64()
            .filter(|value| *value > 0)
            .ok_or_else(|| "font_size precisa ser positivo".to_string())?;
        layer.entry("style").or_insert_with(|| json!({}))["tamanho"] = Value::from(value);
    }
    layer.insert("style_origin".into(), Value::String("editor".into()));
    layer.insert(
        "integration_v1_layout_request".into(),
        layout_request.clone(),
    );
    let request_sha256 = canonical_json_sha256(layout_request)?;
    let page_sha256 = canonical_json_sha256(&page)?;
    Ok(PreparedOwnerLayout {
        page_index,
        layer_index,
        page,
        target_text,
        request_sha256,
        page_sha256,
        invalidated_stages: [
            "layout",
            "rasterize",
            "review",
            "persist",
            "export_decision",
        ],
    })
}

fn project_relative_artifact(project_root: &Path, artifact: &Path) -> Result<String, String> {
    let root = project_root
        .canonicalize()
        .map_err(|error| format!("falha ao resolver raiz do projeto: {error}"))?;
    let artifact = artifact
        .canonicalize()
        .map_err(|error| format!("falha ao resolver artefato raster: {error}"))?;
    let relative = artifact
        .strip_prefix(&root)
        .map_err(|_| "artefato raster escapou da raiz do projeto".to_string())?;
    Ok(relative.to_string_lossy().replace('\\', "/"))
}

fn owner_source_path(project_root: &Path, page: &Value) -> Result<PathBuf, String> {
    let raw = page
        .pointer("/image_layers/base/path")
        .or_else(|| page.get("arquivo_original"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "owner sem imagem-fonte persistida".to_string())?;
    let candidate = project_root.join(raw);
    let canonical_root = project_root
        .canonicalize()
        .map_err(|error| format!("falha ao resolver raiz do projeto: {error}"))?;
    let canonical = candidate
        .canonicalize()
        .map_err(|error| format!("falha ao resolver imagem-fonte: {error}"))?;
    if !canonical.starts_with(canonical_root) {
        return Err("imagem-fonte escapou da raiz do projeto".into());
    }
    Ok(canonical)
}

fn write_recipe(
    project_root: &Path,
    owner_id: &str,
    prepared: &PreparedOwnerLayout,
    source_sha256: &str,
    output_sha256: &str,
    output_relative_path: &str,
    renderer_backend: &str,
) -> Result<(Value, String), String> {
    let runtime_sha256 = canonical_json_sha256(&json!({
        "runtime_id": "tauri-render-preview-page-v1",
        "renderer_backend": renderer_backend,
    }))?;
    let owner_digest = canonical_json_sha256(&Value::String(owner_id.to_string()))?;
    let relative_path = format!(
        "recipes/page_{:04}/{}-{}-{}.json",
        prepared.page_index + 1,
        &owner_digest[..16],
        &prepared.request_sha256[..16],
        &output_sha256[..16]
    );
    let body = json!({
        "output_sha256": output_sha256,
        "relative_path": relative_path,
        "runtime_id": "tauri-render-preview-page-v1",
        "runtime_sha256": runtime_sha256,
        "source_sha256": source_sha256,
        "dependency_hashes": {
            "layout_plan": prepared.page_sha256,
            "layout_request": prepared.request_sha256,
        },
        "owner_id": owner_id,
        "target_sha256": format!("{:x}", Sha256::digest(prepared.target_text.as_bytes())),
        "invalidated_stages": prepared.invalidated_stages,
        "preserved_stages": [
            "source_analysis",
            "detection",
            "ocr",
            "translation",
            "restoration",
        ],
        "stage_invocations": {
            "detection": 0,
            "ocr": 0,
            "translation": 0,
            "restoration": 0,
            "renderer": 1,
        },
        "raster_artifact_ref": {
            "relative_path": output_relative_path,
            "sha256": output_sha256,
        },
    });
    let recipe_sha256 = canonical_json_sha256(&body)?;
    let mut recipe = body;
    recipe["recipe_sha256"] = Value::String(recipe_sha256.clone());
    let recipe_path = project_root.join(&relative_path);
    if let Some(parent) = recipe_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("falha ao criar diretório de receitas: {error}"))?;
    }
    if !recipe_path.exists() {
        fs::write(
            &recipe_path,
            serde_json::to_vec_pretty(&recipe)
                .map_err(|error| format!("falha ao serializar receita: {error}"))?,
        )
        .map_err(|error| format!("falha ao persistir receita: {error}"))?;
    }
    Ok((recipe, relative_path))
}

fn persist_retypeset_in_project(
    project: &mut Value,
    owner_id: &str,
    layout_request: &Value,
    expected_revision: u64,
    idempotency_key: &str,
    rendered_page_sha256: &str,
    raster_relative_path: &str,
    output_sha256: &str,
    recipe: &Value,
    recipe_relative_path: &str,
) -> Result<Value, String> {
    if let Some(result) = find_receipt(project, "retypeset_owner", idempotency_key) {
        return Ok(result);
    }
    require_revision(project, expected_revision)?;
    let mut prepared = prepare_owner_layout(project, owner_id, layout_request)?;
    if prepared.page_sha256 != rendered_page_sha256 {
        return Err("layout preparado diverge da página efetivamente rasterizada".into());
    }
    prepared.page["text_layers"][prepared.layer_index]["render_preview_path"] =
        Value::String(raster_relative_path.to_string());
    let pages = project
        .get_mut("paginas")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "projeto sem páginas persistíveis".to_string())?;
    pages[prepared.page_index] = prepared.page;
    let next_revision = expected_revision + 1;
    let recipe_sha256 = require_sha256_field(&recipe["recipe_sha256"], "recipe_sha256")?;
    if recipe["relative_path"].as_str() != Some(recipe_relative_path)
        || recipe["output_sha256"].as_str() != Some(output_sha256)
    {
        return Err("receita diverge do artefato raster".into());
    }
    let recipe_receipt = json!({
        "recipe_sha256": recipe_sha256,
        "output_sha256": output_sha256,
        "relative_path": recipe_relative_path,
        "runtime_id": recipe["runtime_id"],
    });
    let raster_artifact_ref = json!({
        "relative_path": raster_relative_path,
        "sha256": output_sha256,
    });
    let result = json!({
        "project_revision": next_revision,
        "recipe_receipt": recipe_receipt,
        "raster_artifact_ref": raster_artifact_ref,
    });
    let root = project
        .as_object_mut()
        .ok_or_else(|| "project.json precisa ser um objeto".to_string())?;
    root.insert("project_revision".into(), Value::from(next_revision));
    root.insert("verified".into(), Value::Bool(false));
    root.insert(
        "output_review_state".into(),
        Value::String("review_required".into()),
    );
    let integration = root
        .entry("integration_v1")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "integration_v1 precisa ser um objeto".to_string())?;
    integration
        .entry("recipe_receipts")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| "integration_v1.recipe_receipts precisa ser uma lista".to_string())?
        .push(json!({
            "owner_id": owner_id,
            "project_revision": next_revision,
            "recipe": recipe,
            "receipt": result["recipe_receipt"],
            "raster_artifact_ref": result["raster_artifact_ref"],
        }));
    let qa = root
        .entry("qa")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "qa precisa ser um objeto".to_string())?;
    let gate = qa
        .entry("export_gate")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "qa.export_gate precisa ser um objeto".to_string())?;
    gate.insert("status".into(), Value::String("BLOCK".into()));
    gate.insert("allowed".into(), Value::Bool(false));
    gate.insert("needs_review".into(), Value::Bool(true));
    append_receipt(project, "retypeset_owner", idempotency_key, &result)?;
    Ok(result)
}

#[tauri::command]
pub async fn start_consumer_fast(
    app: AppHandle,
    project_path: String,
    chapter_id: String,
    expected_revision: u64,
    idempotency_key: String,
) -> Result<Value, String> {
    let project_file = resolve_project_file(&project_path)?;
    let canonical_project_file = project_file
        .canonicalize()
        .map_err(|error| format!("falha ao resolver project.json: {error}"))?;
    let canonical_path = canonical_project_file.to_string_lossy().into_owned();
    let result = project_schema::edit_project_value(&canonical_project_file, |project| {
        crate::ensure_runtime_project_identity(project, &canonical_project_file)?;
        start_job_in_project(
            project,
            &canonical_path,
            &chapter_id,
            expected_revision,
            &idempotency_key,
        )
    })?;
    let job_id = result["job_id"]
        .as_str()
        .ok_or_else(|| "start_consumer_fast não produziu job_id".to_string())?;
    register_job(&app, job_id, &canonical_project_file)?;
    emit_latest_job_event(&app, &canonical_project_file, job_id)?;
    set_job_marker(&app, job_id, "pause", false)?;
    set_job_marker(&app, job_id, "cancel", false)?;
    spawn_consumer_job(app, canonical_project_file, job_id.to_string()).await;
    Ok(result)
}

#[tauri::command]
pub async fn cancel_consumer_fast(
    app: AppHandle,
    job_id: String,
    expected_revision: u64,
    idempotency_key: String,
) -> Result<Value, String> {
    let result = control_registered_job(
        &app,
        &job_id,
        "cancel",
        &[],
        expected_revision,
        &idempotency_key,
    )?;
    set_job_marker(&app, &job_id, "cancel", true)?;
    set_job_marker(&app, &job_id, "pause", false)?;
    Ok(result)
}

#[tauri::command]
pub async fn pause_consumer_fast(
    app: AppHandle,
    job_id: String,
    expected_revision: u64,
    idempotency_key: String,
) -> Result<Value, String> {
    let result = control_registered_job(
        &app,
        &job_id,
        "pause",
        &[],
        expected_revision,
        &idempotency_key,
    )?;
    set_job_marker(&app, &job_id, "pause", true)?;
    Ok(result)
}

#[tauri::command]
pub async fn resume_consumer_fast(
    app: AppHandle,
    job_id: String,
    expected_revision: u64,
    idempotency_key: String,
) -> Result<Value, String> {
    let result = control_registered_job(
        &app,
        &job_id,
        "resume",
        &[],
        expected_revision,
        &idempotency_key,
    )?;
    set_job_marker(&app, &job_id, "pause", false)?;
    set_job_marker(&app, &job_id, "cancel", false)?;
    let project_file = resolve_registered_job(&app, &job_id)?;
    let spawned = spawn_consumer_job(app.clone(), project_file.clone(), job_id.clone()).await;
    if !spawned {
        persist_runtime_transition(
            &app,
            &project_file,
            &job_id,
            "import",
            "running",
            "worker_resumed",
            json!({}),
        )?;
    }
    Ok(result)
}

#[tauri::command]
pub async fn retry_consumer_fast(
    app: AppHandle,
    job_id: String,
    terminal_unit_ids: Vec<String>,
    expected_revision: u64,
    idempotency_key: String,
) -> Result<Value, String> {
    let result = control_registered_job(
        &app,
        &job_id,
        "retry",
        &terminal_unit_ids,
        expected_revision,
        &idempotency_key,
    )?;
    set_job_marker(&app, &job_id, "pause", false)?;
    set_job_marker(&app, &job_id, "cancel", false)?;
    let project_file = resolve_registered_job(&app, &job_id)?;
    spawn_consumer_job(app, project_file, job_id).await;
    Ok(result)
}

#[tauri::command]
pub fn persist_project_event(
    app: AppHandle,
    event: ProjectEventInput,
    expected_revision: u64,
    idempotency_key: String,
) -> Result<Value, String> {
    let project_file = resolve_registered_job(&app, &event.job_id)?;
    let result = project_schema::edit_project_value(&project_file, |project| {
        persist_event_in_project(project, &event, expected_revision, &idempotency_key)
    })?;
    emit_latest_job_event(&app, &project_file, &event.job_id)?;
    Ok(result)
}

#[tauri::command]
pub async fn retypeset_owner(
    app: AppHandle,
    project_path: String,
    owner_id: String,
    layout_request: Value,
    expected_revision: u64,
    idempotency_key: String,
) -> Result<Value, String> {
    let project_file = resolve_project_file(&project_path)?;
    let project = project_schema::load_project_value(&project_file)?;
    if let Some(result) = find_receipt(&project, "retypeset_owner", &idempotency_key) {
        return Ok(result);
    }
    require_revision(&project, expected_revision)?;
    let prepared = prepare_owner_layout(&project, &owner_id, &layout_request)?;
    let project_root = project_file
        .parent()
        .ok_or_else(|| "project.json sem diretório pai".to_string())?;
    let source_path = owner_source_path(project_root, &prepared.page)?;
    let (source_sha256, _) = file_sha256(&source_path)?;
    let rendered = crate::integration_runtime_adapter::render_preview_page(
        app,
        crate::integration_runtime_adapter::RenderPreviewConfig {
            project_path: project_file.to_string_lossy().into_owned(),
            page_index: prepared.page_index as u32,
            page: prepared.page.clone(),
            fingerprint: prepared.page_sha256.clone(),
        },
    )
    .await?;
    let raw_output = PathBuf::from(&rendered.output_path);
    let output_path = if raw_output.is_absolute() {
        raw_output
    } else {
        project_root.join(raw_output)
    };
    let (output_sha256, _) = file_sha256(&output_path)?;
    let raster_relative_path = project_relative_artifact(project_root, &output_path)?;
    let (recipe, recipe_relative_path) = write_recipe(
        project_root,
        &owner_id,
        &prepared,
        &source_sha256,
        &output_sha256,
        &raster_relative_path,
        &rendered.renderer_backend,
    )?;
    project_schema::edit_project_value(&project_file, |project| {
        persist_retypeset_in_project(
            project,
            &owner_id,
            &layout_request,
            expected_revision,
            &idempotency_key,
            &prepared.page_sha256,
            &raster_relative_path,
            &output_sha256,
            &recipe,
            &recipe_relative_path,
        )
    })
}

#[tauri::command]
pub fn submit_review_decision(
    project_path: String,
    decision: ReviewDecisionInput,
    expected_revision: u64,
    idempotency_key: String,
) -> Result<Value, String> {
    if decision.expected_revision != expected_revision
        || decision.idempotency_key != idempotency_key
    {
        return Err("identidade da mutação diverge da decisão de revisão".into());
    }
    let project_file = resolve_project_file(&project_path)?;
    project_schema::edit_project_value(&project_file, |project| {
        if let Some(result) = find_receipt(project, "submit_review_decision", &idempotency_key) {
            return Ok(result);
        }
        require_revision(project, expected_revision)?;
        append_review_decision(project, &decision)
    })
}

#[tauri::command]
pub fn decide_export(project_path: String, expected_revision: u64) -> Result<Value, String> {
    let project_file = resolve_project_file(&project_path)?;
    let project = project_schema::load_project_value(&project_file)?;
    require_revision(&project, expected_revision)?;
    Ok(json!({"export_decision": export_decision(&project)}))
}

#[tauri::command]
pub fn approve_final_review(
    project_path: String,
    expected_revision: u64,
    idempotency_key: String,
    actor_kind: String,
    actor_id: String,
) -> Result<Value, String> {
    let project_file = resolve_project_file(&project_path)?;
    project_schema::edit_project_value(&project_file, |project| {
        record_final_review_approval(
            project,
            expected_revision,
            &actor_kind,
            &actor_id,
            &idempotency_key,
        )
    })
}

#[tauri::command]
pub async fn export_final(
    project_path: String,
    destination: String,
    expected_revision: u64,
    idempotency_key: String,
) -> Result<Value, String> {
    let project_file = resolve_project_file(&project_path)?;
    let project = project_schema::load_project_value(&project_file)?;
    if let Some(result) = find_receipt(&project, "export_final", &idempotency_key) {
        return Ok(result);
    }
    require_revision(&project, expected_revision)?;
    if export_decision(&project)["allowed"].as_bool() != Some(true) {
        return Err("exportação final bloqueada por pendências reais do projeto".into());
    }
    let destination_path = PathBuf::from(&destination);
    if let Some(parent) = destination_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("falha ao preparar destino final: {error}"))?;
    }
    crate::integration_runtime_adapter::export_project(
        crate::integration_runtime_adapter::ExportConfig {
            project_path: project_file.to_string_lossy().into_owned(),
            format: "cbz".into(),
            output_path: destination.clone(),
            export_mode: Some("final".into()),
        },
    )
    .await?;
    let (artifact_sha256, artifact_size) = file_sha256(&destination_path)?;
    project_schema::edit_project_value(&project_file, |project| {
        record_final_export_in_project(
            project,
            expected_revision,
            &idempotency_key,
            &destination,
            &artifact_sha256,
            artifact_size,
        )
    })
}

#[tauri::command]
pub fn export_diagnostic(
    project_path: String,
    destination: String,
    expected_revision: u64,
) -> Result<Value, String> {
    let project_file = resolve_project_file(&project_path)?;
    let project = project_schema::load_project_value(&project_file)?;
    require_revision(&project, expected_revision)?;
    let pages = project
        .get("paginas")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let text_units = project
        .get("paginas")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|page| {
            page.get("text_layers")
                .and_then(Value::as_array)
                .map_or(0, Vec::len)
        })
        .sum::<usize>();
    let body = json!({
        "schema": "traduzai.diagnostic-export.v1",
        "project_revision": expected_revision,
        "project_sha256": canonical_json_sha256(&project)?,
        "export_decision": export_decision(&project),
        "page_count": pages,
        "text_unit_count": text_units,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "contains_source_images": false,
        "promotes_final_export_state": false,
    });
    let manifest_sha256 = canonical_json_sha256(&body)?;
    let mut manifest = body;
    manifest["manifest_sha256"] = Value::String(manifest_sha256);
    let destination_path = PathBuf::from(&destination);
    write_diagnostic_archive(&destination_path, &manifest)?;
    let (artifact_sha256, artifact_size) = file_sha256(&destination_path)?;
    Ok(json!({
        "diagnostic_manifest": manifest,
        "artifact_ref": {
            "path": destination,
            "sha256": artifact_sha256,
            "size": artifact_size,
        }
    }))
}

#[tauri::command]
pub fn read_renderer_preference_comparison(
    project_path: String,
    owner_id: String,
    expected_revision: u64,
) -> Result<Value, String> {
    if owner_id.trim().is_empty() {
        return Err("owner_id é obrigatório".into());
    }
    let project_file = resolve_project_file(&project_path)?;
    let project = project_schema::load_project_value(&project_file)?;
    require_revision(&project, expected_revision)?;
    let index = project.pointer("/integration_v1/renderer_preference_index");
    if index.is_some_and(|index| index["schema"] != "traduzai.renderer-preference-index.v1") {
        return Err("renderer preference index incompatível".into());
    }
    let entry = index
        .and_then(|index| index.get("owners"))
        .and_then(Value::as_object)
        .and_then(|owners| owners.get(&owner_id));
    let Some(entry) = entry else {
        return Ok(json!({
            "owner_id": owner_id,
            "project_revision": expected_revision,
            "state": null,
            "comparison": null,
            "artifact_base": null,
        }));
    };
    let state = entry["state"]
        .as_str()
        .ok_or_else(|| "renderer preference entry exige state".to_string())?;
    if state != "pending" {
        return Ok(json!({
            "owner_id": owner_id,
            "project_revision": expected_revision,
            "state": state,
            "comparison": null,
            "artifact_base": null,
        }));
    }
    let expected_comparison_sha256 =
        require_sha256_field(&entry["comparison_sha256"], "comparison_sha256")?;
    let project_root = project_file
        .parent()
        .ok_or_else(|| "project.json sem diretório pai".to_string())?;
    let (comparison_file, comparison_bytes) =
        read_verified_artifact(project_root, &entry["comparison_ref"], "comparison_ref")?;
    let comparison: Value = serde_json::from_slice(&comparison_bytes).map_err(|error| {
        format!(
            "renderer comparison inválida em {}: {error}",
            comparison_file.display()
        )
    })?;
    let artifact_root = comparison_file
        .parent()
        .ok_or_else(|| "renderer comparison sem diretório pai".to_string())?;
    validate_preference_comparison(
        &comparison,
        &owner_id,
        artifact_root,
        &expected_comparison_sha256,
    )?;
    Ok(json!({
        "owner_id": owner_id,
        "project_revision": expected_revision,
        "state": state,
        "comparison": comparison,
        "artifact_base": artifact_root.to_string_lossy(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn preference_candidate(
        owner_id: &str,
        suffix: char,
        preview_sha256: &str,
        context_sha256: &str,
    ) -> Value {
        let digest = suffix.to_string().repeat(64);
        let mut candidate = json!({
            "schema": "traduzai.renderer-preference.v1",
            "owner_id": owner_id,
            "target_text": "AÇÃO TOTAL",
            "target_sha256": "c4830df52d636db475295a8c2c82c9f78e084465a7f711705fdafbbdf21454aa",
            "source_sha256": context_sha256,
            "style_sha256": "2".repeat(64),
            "layout_plan_sha256": digest,
            "recipe_sha256": (if suffix == '3' { "4" } else { "5" }).repeat(64),
            "output_sha256": preview_sha256,
            "preview_ref": {
                "relative_path": if suffix == '3' { "previews/a.svg" } else { "previews/b.svg" },
                "sha256": preview_sha256
            },
            "context_ref": {
                "relative_path": "previews/context.svg",
                "sha256": context_sha256
            },
            "metrics": {
                "raster_safety": {
                    "status": "pass",
                    "outside_authorized_body_px": 0,
                    "protected_art_overlap_px": 0
                }
            },
            "hard_safety_passed": true,
            "preference_profile": "uncalibrated"
        });
        let digest = canonical_json_sha256(&candidate).unwrap();
        candidate["candidate_id"] = Value::String(format!("renderer-candidate:{}", &digest[..32]));
        candidate
    }

    fn preference_comparison(
        owner_id: &str,
        first_preview_sha256: &str,
        second_preview_sha256: &str,
        context_sha256: &str,
    ) -> Value {
        let first = preference_candidate(owner_id, '3', first_preview_sha256, context_sha256);
        let second = preference_candidate(owner_id, '8', second_preview_sha256, context_sha256);
        let mut comparison = json!({
            "schema": "traduzai.renderer-preference.v1",
            "candidates": [first.clone(), second.clone()],
            "positions": {
                "A": first["candidate_id"],
                "B": second["candidate_id"]
            },
            "randomization_sha256": "9".repeat(64),
            "preference_profile": "uncalibrated"
        });
        comparison["comparison_sha256"] =
            Value::String(canonical_json_sha256(&comparison).unwrap());
        comparison
    }

    fn project(revision: u64) -> Value {
        json!({
            "versao": "2.0",
            "app": "traduzai",
            "obra": "Teste",
            "capitulo": 1,
            "paginas": [],
            "project_revision": revision,
        })
    }

    fn decision(actor_kind: &str, revision: u64) -> ReviewDecisionInput {
        ReviewDecisionInput {
            project_id: "project-001".into(),
            owner_id: "owner-001".into(),
            expected_revision: revision,
            actor_kind: actor_kind.into(),
            actor_id: "reviewer-001".into(),
            decision: "accept_candidate".into(),
            reason_code: "visual_review_passed".into(),
            evidence_sha256s: vec!["a".repeat(64)],
            idempotency_key: "review-owner-001-r4".into(),
            preference_response: None,
        }
    }

    #[test]
    fn human_review_is_persisted_once_and_advances_revision() {
        let mut project = project(4);
        let decision = decision("human", 4);
        let first = append_review_decision(&mut project, &decision).unwrap();
        assert_eq!(first["project_revision"], 5);
        assert_eq!(first["owner_status"], "approved");
        assert_eq!(
            project["integration_v1"]["review_decisions"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            find_receipt(
                &project,
                "submit_review_decision",
                &decision.idempotency_key,
            ),
            Some(first)
        );
    }

    #[test]
    fn model_review_never_becomes_human_approval() {
        let mut project = project(4);
        let result = append_review_decision(&mut project, &decision("model", 4)).unwrap();
        assert_eq!(result["owner_status"], "review_required");
    }

    #[test]
    fn preference_response_is_persisted_only_when_fully_evidence_bound() {
        let mut decision = decision("human", 4);
        let hashes = "abcdef1"
            .chars()
            .map(|value| value.to_string().repeat(64))
            .collect::<Vec<_>>();
        decision.evidence_sha256s = hashes.clone();
        decision.preference_response = Some(json!({
            "schema": "traduzai.renderer-preference-response.v1",
            "comparison_sha256": hashes[0],
            "choice": "B",
            "selected_candidate_id": format!("renderer-candidate:{}", "2".repeat(32)),
            "displayed_candidate_ids": {
                "A": format!("renderer-candidate:{}", "1".repeat(32)),
                "B": format!("renderer-candidate:{}", "2".repeat(32)),
            },
            "candidate_recipe_sha256s": [hashes[3], hashes[4]],
            "candidate_output_sha256s": [hashes[5], hashes[6]],
            "target_sha256": hashes[1],
            "randomization_sha256": hashes[2],
            "actor_kind": "human",
            "actor_id": "reviewer-001",
            "recorded_at": "2026-09-27T14:30:00-03:00",
            "training_eligible": false,
        }));

        let mut project = project(4);
        project["integration_v1"] = json!({
            "renderer_preference_index": {
                "schema": "traduzai.renderer-preference-index.v1",
                "owners": {
                    "owner-001": {
                        "state": "pending",
                        "comparison_sha256": hashes[0],
                    }
                }
            }
        });
        append_review_decision(&mut project, &decision).unwrap();
        assert_eq!(
            project["integration_v1"]["review_decisions"][0]["preference_response"]
                ["selected_candidate_id"],
            format!("renderer-candidate:{}", "2".repeat(32))
        );
        assert_eq!(
            project["integration_v1"]["renderer_preference_index"]["owners"]["owner-001"]["state"],
            "resolved"
        );
        assert_eq!(
            project["integration_v1"]["renderer_preference_index"]["owners"]["owner-001"]["choice"],
            "B"
        );

        decision.evidence_sha256s.pop();
        assert!(validate_review_decision(&decision)
            .unwrap_err()
            .contains("evidência"));
    }

    #[test]
    fn stale_review_is_rejected_with_explicit_conflict() {
        let mut project = project(5);
        let error = append_review_decision(&mut project, &decision("human", 4)).unwrap_err();
        assert!(error.contains(REVISION_CONFLICT));
    }

    #[test]
    fn uppercase_sha256_is_rejected_before_persistence() {
        let mut decision = decision("human", 4);
        decision.evidence_sha256s = vec!["A".repeat(64)];
        assert!(validate_review_decision(&decision)
            .unwrap_err()
            .contains("SHA-256"));
    }

    #[test]
    fn export_is_fail_closed_until_all_signals_pass() {
        let mut blocked = project(1);
        blocked["verified"] = Value::Bool(false);
        blocked["qa"] = json!({"export_gate": {"status": "PASS", "allowed": true}});
        assert_eq!(export_decision(&blocked)["allowed"], false);

        let mut allowed = project(1);
        allowed["verified"] = Value::Bool(true);
        allowed["completion_status"] = Value::String("approved".into());
        allowed["output_review_state"] = Value::String("approved".into());
        allowed["qa"] = json!({"export_gate": {"status": "PASS", "allowed": true}});
        assert_eq!(export_decision(&allowed)["allowed"], true);
    }

    #[test]
    fn final_review_approval_requires_a_clean_backend_gate_and_human_actor() {
        let mut blocked = project(4);
        blocked["qa"] = json!({"export_gate": {"status": "BLOCK", "allowed": false}});
        assert!(record_final_review_approval(
            &mut blocked,
            4,
            "human",
            "local-user",
            "approve-final-r4"
        )
        .unwrap_err()
        .contains("gate"));

        let mut allowed = project(4);
        allowed["qa"] = json!({
            "export_gate": {
                "status": "PASS",
                "allowed": true,
                "issue_count": 0,
                "critical_issue_count": 0,
                "critical_flag_count": 0,
                "review_issue_count": 0,
                "review_flag_count": 0
            }
        });
        assert!(record_final_review_approval(
            &mut allowed,
            4,
            "system",
            "test/synthetic",
            "approve-final-r4-system"
        )
        .unwrap_err()
        .contains("humana"));

        let result = record_final_review_approval(
            &mut allowed,
            4,
            "human",
            "local-user",
            "approve-final-r4-human"
        )
        .unwrap();
        assert_eq!(result["project_revision"], 5);
        assert_eq!(allowed["verified"], true);
        assert_eq!(allowed["completion_status"], "approved");
        assert_eq!(allowed["output_review_state"], "approved");
        assert_eq!(export_decision(&allowed)["allowed"], true);
        assert_eq!(
            allowed["integration_v1"]["final_review_decisions"][0]["actor_kind"],
            "human"
        );

        let duplicate = record_final_review_approval(
            &mut allowed,
            4,
            "human",
            "local-user",
            "approve-final-r4-human"
        )
        .unwrap();
        assert_eq!(duplicate, result);
        assert_eq!(allowed["project_revision"], 5);
    }

    #[test]
    fn command_persists_idempotent_review_in_project_json() {
        let temp = tempfile::tempdir().unwrap();
        let project_file = temp.path().join("project.json");
        fs::write(
            &project_file,
            serde_json::to_vec_pretty(&project(4)).unwrap(),
        )
        .unwrap();
        let decision = decision("human", 4);

        let first = submit_review_decision(
            temp.path().to_string_lossy().into_owned(),
            decision.clone(),
            4,
            decision.idempotency_key.clone(),
        )
        .unwrap();
        let duplicate = submit_review_decision(
            project_file.to_string_lossy().into_owned(),
            decision.clone(),
            4,
            decision.idempotency_key.clone(),
        )
        .unwrap();

        assert_eq!(duplicate, first);
        let persisted = project_schema::load_project_value(&project_file).unwrap();
        assert_eq!(persisted["project_revision"], 5);
        assert_eq!(
            persisted["integration_v1"]["review_decisions"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn command_reads_pending_renderer_comparison_with_verified_file_hash() {
        let temp = tempfile::tempdir().unwrap();
        let owner_id = "owner-001";
        let previews = temp.path().join("previews");
        fs::create_dir(&previews).unwrap();
        let first_preview = b"candidate-a";
        let second_preview = b"candidate-b";
        let context = b"context";
        fs::write(previews.join("a.svg"), first_preview).unwrap();
        fs::write(previews.join("b.svg"), second_preview).unwrap();
        fs::write(previews.join("context.svg"), context).unwrap();
        let comparison = preference_comparison(
            owner_id,
            &format!("{:x}", Sha256::digest(first_preview)),
            &format!("{:x}", Sha256::digest(second_preview)),
            &format!("{:x}", Sha256::digest(context)),
        );
        let comparison_bytes = serde_json::to_vec_pretty(&comparison).unwrap();
        let comparison_file = temp.path().join("comparison.json");
        fs::write(&comparison_file, &comparison_bytes).unwrap();

        let mut project = project(4);
        let entry = json!({
            "state": "pending",
            "comparison_ref": {
                "relative_path": "comparison.json",
                "sha256": format!("{:x}", Sha256::digest(&comparison_bytes))
            },
            "comparison_sha256": comparison["comparison_sha256"]
        });
        let mut owners = serde_json::Map::new();
        owners.insert(owner_id.into(), entry);
        project["integration_v1"] = json!({
            "renderer_preference_index": {
                "schema": "traduzai.renderer-preference-index.v1",
                "owners": owners
            }
        });
        fs::write(
            temp.path().join("project.json"),
            serde_json::to_vec_pretty(&project).unwrap(),
        )
        .unwrap();

        let result = read_renderer_preference_comparison(
            temp.path().to_string_lossy().into_owned(),
            owner_id.into(),
            4,
        )
        .unwrap();
        assert_eq!(result["comparison"], comparison);
        assert_eq!(result["state"], "pending");
    }

    #[test]
    fn renderer_comparison_read_is_null_when_owner_has_no_pending_entry() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            temp.path().join("project.json"),
            serde_json::to_vec_pretty(&project(4)).unwrap(),
        )
        .unwrap();

        let result = read_renderer_preference_comparison(
            temp.path().to_string_lossy().into_owned(),
            "owner-missing".into(),
            4,
        )
        .unwrap();
        assert!(result["comparison"].is_null());
    }

    #[test]
    fn renderer_comparison_rejects_project_escape_before_reading() {
        let temp = tempfile::tempdir().unwrap();
        let mut project = project(4);
        project["integration_v1"] = json!({
            "renderer_preference_index": {
                "schema": "traduzai.renderer-preference-index.v1",
                "owners": {
                    "owner-001": {
                        "state": "pending",
                        "comparison_ref": {
                            "relative_path": "../comparison.json",
                            "sha256": "a".repeat(64)
                        },
                        "comparison_sha256": "b".repeat(64)
                    }
                }
            }
        });
        fs::write(
            temp.path().join("project.json"),
            serde_json::to_vec_pretty(&project).unwrap(),
        )
        .unwrap();

        let error = read_renderer_preference_comparison(
            temp.path().to_string_lossy().into_owned(),
            "owner-001".into(),
            4,
        )
        .unwrap_err();
        assert!(error.contains("caminho relativo seguro"));
    }

    #[test]
    fn durable_job_lifecycle_is_revision_bound_idempotent_and_terminal() {
        let mut project = project(4);
        project["project_id"] = Value::String("project-001".into());
        project["capitulo"] = Value::from(57);
        project["paginas"] = json!([{
            "numero": 1,
            "text_layers": [{"id": "owner-001"}, {"id": "owner-002"}]
        }]);

        let started =
            start_job_in_project(&mut project, "N:/p/project.json", "57", 4, "start-r4").unwrap();
        assert_eq!(started["status"], "queued");
        assert_eq!(started["project_revision"], 5);
        let duplicate =
            start_job_in_project(&mut project, "N:/p/project.json", "57", 4, "start-r4").unwrap();
        assert_eq!(duplicate, started);

        let job_id = started["job_id"].as_str().unwrap();
        let paused =
            control_job_in_project(&mut project, job_id, "pause", &[], 5, "pause-r5").unwrap();
        assert_eq!(paused["status"], "paused");
        let resumed =
            control_job_in_project(&mut project, job_id, "resume", &[], 6, "resume-r6").unwrap();
        assert_eq!(resumed["status"], "queued");
        let cancelled =
            control_job_in_project(&mut project, job_id, "cancel", &[], 7, "cancel-r7").unwrap();
        assert_eq!(cancelled["status"], "cancelled");
        assert!(control_job_in_project(
            &mut project,
            job_id,
            "resume",
            &[],
            8,
            "resume-terminal-r8"
        )
        .unwrap_err()
        .contains("terminal"));
    }

    #[test]
    fn project_event_requires_next_revision_and_monotonic_sequence() {
        let mut project = project(4);
        project["project_id"] = Value::String("project-001".into());
        project["capitulo"] = Value::from(57);
        let started =
            start_job_in_project(&mut project, "N:/p/project.json", "57", 4, "start-r4").unwrap();
        let job_id = started["job_id"].as_str().unwrap().to_string();
        let event = ProjectEventInput {
            job_id: job_id.clone(),
            project_id: "project-001".into(),
            expected_revision: 5,
            project_revision: 6,
            sequence: 2,
            stage: "analysis".into(),
            status: "running".into(),
            reason_code: "analysis_started".into(),
            payload: json!({"page_id": "page-001"}),
        };
        let persisted = persist_event_in_project(&mut project, &event, 5, "event-r5").unwrap();
        assert_eq!(persisted["project_revision"], 6);
        let stale = ProjectEventInput {
            expected_revision: 6,
            project_revision: 7,
            ..event
        };
        assert!(
            persist_event_in_project(&mut project, &stale, 6, "event-stale-r6")
                .unwrap_err()
                .contains("sequence")
        );
    }

    #[test]
    fn final_export_receipt_is_persisted_only_after_fail_closed_gate_passes() {
        let digest = "a".repeat(64);
        let mut blocked = project(4);
        let error = record_final_export_in_project(
            &mut blocked,
            4,
            "export-r4",
            "N:/exports/final.cbz",
            &digest,
            123,
        )
        .unwrap_err();
        assert!(error.contains("bloqueada"));
        assert_eq!(blocked["project_revision"], 4);

        let mut allowed = project(4);
        allowed["verified"] = Value::Bool(true);
        allowed["completion_status"] = Value::String("approved".into());
        allowed["output_review_state"] = Value::String("approved".into());
        allowed["qa"] = json!({"export_gate": {"status": "PASS", "allowed": true}});
        let result = record_final_export_in_project(
            &mut allowed,
            4,
            "export-r4",
            "N:/exports/final.cbz",
            &digest,
            123,
        )
        .unwrap();
        assert_eq!(result["project_revision"], 5);
        assert_eq!(result["publication_receipt"]["human_review_claimed"], false);
        assert_eq!(
            record_final_export_in_project(
                &mut allowed,
                4,
                "export-r4",
                "N:/exports/final.cbz",
                &digest,
                123,
            )
            .unwrap(),
            result
        );
    }

    #[test]
    fn physical_dispatch_reuses_one_pipeline_config_and_requires_original_source() {
        let mut value = project(4);
        let missing = physical_pipeline_config(
            &value,
            "job-001",
            Path::new("N:/staging"),
            Path::new("N:/models"),
            Path::new("N:/logs"),
            Path::new("N:/runtime/pause.flag"),
            Path::new("N:/runtime/cancel.flag"),
        )
        .unwrap_err();
        assert!(missing.contains("SOURCE_PATH_REQUIRED"));

        value["consumer_source_path"] = Value::String("N:/source/chapter.cbz".into());
        value["obra"] = Value::String("Obra".into());
        value["capitulo"] = Value::from(57);
        value["mode"] = Value::String("mock".into());
        let config = physical_pipeline_config(
            &value,
            "consumer-fast:job-001",
            Path::new("N:/staging"),
            Path::new("N:/models"),
            Path::new("N:/logs"),
            Path::new("N:/runtime/pause.flag"),
            Path::new("N:/runtime/cancel.flag"),
        )
        .unwrap();

        assert_eq!(config["runtime_id"], "consumer-fast-v1");
        assert_eq!(config["source_path"], "N:/source/chapter.cbz");
        assert_eq!(config["job_id"], "consumer-fast-job-001");
        assert_eq!(config["integration_job_id"], "consumer-fast:job-001");
        assert_eq!(config["mode"], "real");
        assert_eq!(config["owner_graph_mode"], "enforce");
        assert_eq!(config["style_copy_mode"], "shadow");
        assert_eq!(config["pause_file"], "N:/runtime/pause.flag");
        assert_eq!(config["cancel_file"], "N:/runtime/cancel.flag");
    }

    #[test]
    fn physical_retry_uses_revision_isolated_staging_directory() {
        let runtime = Path::new("N:/runtime/job");

        assert_eq!(
            physical_pipeline_work_dir(runtime, 37),
            PathBuf::from("N:/runtime/job/pipeline-staging-r37")
        );
        assert_ne!(
            physical_pipeline_work_dir(runtime, 37),
            physical_pipeline_work_dir(runtime, 38)
        );
    }

    #[test]
    fn physical_output_promotion_carries_job_ledger_and_marks_units_terminal() {
        let temp = tempfile::tempdir().unwrap();
        let seed_file = temp.path().join("seed").join("project.json");
        let output_file = temp.path().join("staging").join("project.json");
        fs::create_dir_all(seed_file.parent().unwrap()).unwrap();
        fs::create_dir_all(output_file.parent().unwrap()).unwrap();
        let mut seed = project(4);
        seed["project_id"] = Value::String("project-001".into());
        seed["obra"] = Value::String("Obra persistida".into());
        seed["capitulo"] = Value::from(57);
        seed["consumer_source_path"] = Value::String("N:/source/chapter.cbz".into());
        seed["paginas"] = json!([{
            "numero": 1,
            "text_layers": [{"id": "owner-001"}]
        }]);
        let started =
            start_job_in_project(&mut seed, &seed_file.to_string_lossy(), "57", 4, "start-r4")
                .unwrap();
        let job_id = started["job_id"].as_str().unwrap();
        project_schema::save_project_value(&seed_file, &mut seed).unwrap();
        let mut output = project(0);
        output["paginas"] = json!([{"numero": 1, "text_layers": []}]);
        project_schema::save_project_value(&output_file, &mut output).unwrap();

        attach_job_ledger_to_physical_output(&seed_file, &output_file, job_id).unwrap();

        let promoted = project_schema::load_project_value(&output_file).unwrap();
        assert_eq!(promoted["project_id"], "project-001");
        assert_eq!(promoted["obra"], "Obra persistida");
        assert_eq!(promoted["capitulo"], 57);
        assert_eq!(promoted["consumer_source_path"], "N:/source/chapter.cbz");
        assert_eq!(promoted["project_revision"], 5);
        assert_eq!(
            promoted["integration_v1"]["jobs"][job_id]["units"]["owner-001"]["status"],
            "complete"
        );
        assert_eq!(
            promoted["integration_v1"]["jobs"][job_id]["units"]["owner-001"]["terminal"],
            true
        );
    }

    #[test]
    fn preparing_owner_retypeset_changes_only_layout_dependents() {
        let mut project = project(7);
        project["paginas"] = json!([{
            "numero": 1,
            "text_layers": [{
                "id": "owner-001",
                "original": "HELLO",
                "translated": "OLÁ",
                "layout_bbox": [10, 20, 110, 90],
                "style": {"tamanho": 24, "fonte": "ComicNeue-Bold.ttf"}
            }]
        }]);
        let source_before = project["paginas"][0]["text_layers"][0]["original"].clone();
        let prepared = prepare_owner_layout(
            &project,
            "owner-001",
            &json!({
                "owner_id": "owner-001",
                "text": "TEXTO CORRIGIDO",
                "layout_bbox": [12, 22, 122, 96],
                "style": {"tamanho": 30}
            }),
        )
        .unwrap();

        assert_eq!(prepared.page_index, 0);
        assert_eq!(
            prepared.page["text_layers"][0]["translated"],
            "TEXTO CORRIGIDO"
        );
        assert_eq!(prepared.page["text_layers"][0]["style"]["tamanho"], 30);
        assert_eq!(
            prepared.page["text_layers"][0]["style"]["fonte"],
            "ComicNeue-Bold.ttf"
        );
        assert_eq!(prepared.page["text_layers"][0]["original"], source_before);
        assert_eq!(project["project_revision"], 7);
        assert_eq!(
            prepared.invalidated_stages,
            [
                "layout",
                "rasterize",
                "review",
                "persist",
                "export_decision"
            ]
        );

        let temp = tempfile::tempdir().unwrap();
        let (recipe, _) = write_recipe(
            temp.path(),
            "owner-001",
            &prepared,
            &"a".repeat(64),
            &"b".repeat(64),
            "render-cache/preview/001.png",
            "koharu_rust",
        )
        .unwrap();
        assert_eq!(recipe["stage_invocations"]["detection"], 0);
        assert_eq!(recipe["stage_invocations"]["ocr"], 0);
        assert_eq!(recipe["stage_invocations"]["renderer"], 1);
        assert_eq!(
            recipe["preserved_stages"],
            json!([
                "source_analysis",
                "detection",
                "ocr",
                "translation",
                "restoration"
            ])
        );
    }
}
