use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
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
            .any(|hash| hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return Err("decisão de revisão exige evidência SHA-256 válida".into());
    }
    if let Some(response) = &decision.preference_response {
        validate_preference_response(response, decision)?;
    }
    Ok(())
}

fn require_sha256_field(value: &Value, field: &str) -> Result<String, String> {
    let digest = value
        .as_str()
        .ok_or_else(|| format!("{field} exige SHA-256"))?;
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{field} exige SHA-256 minúsculo"));
    }
    Ok(digest.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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
}
