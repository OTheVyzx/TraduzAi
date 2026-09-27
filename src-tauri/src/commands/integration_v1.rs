use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

use crate::commands::project_schema;

const REVISION_CONFLICT: &str = "PROJECT_REVISION_CONFLICT";

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

fn find_receipt(project: &Value, idempotency_key: &str) -> Option<Value> {
    project
        .pointer("/integration_v1/idempotency_receipts")
        .and_then(Value::as_array)
        .and_then(|receipts| {
            receipts.iter().find(|receipt| {
                receipt.get("idempotency_key").and_then(Value::as_str) == Some(idempotency_key)
            })
        })
        .and_then(|receipt| receipt.get("result"))
        .cloned()
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
    integration
        .entry("idempotency_receipts")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| "integration_v1.idempotency_receipts precisa ser uma lista".to_string())?
        .push(json!({
            "operation": "submit_review_decision",
            "idempotency_key": decision.idempotency_key,
            "result": result,
        }));
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
        if let Some(result) = find_receipt(project, &idempotency_key) {
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
            find_receipt(&project, &decision.idempotency_key),
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
        append_review_decision(&mut project, &decision).unwrap();
        assert_eq!(
            project["integration_v1"]["review_decisions"][0]["preference_response"]
                ["selected_candidate_id"],
            format!("renderer-candidate:{}", "2".repeat(32))
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
}
