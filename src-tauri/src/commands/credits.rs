use serde::Serialize;

const DEFAULT_STARTING_CREDITS: u32 = 1000;
const CREDITS_SEED_VERSION: u32 = 1;

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
pub struct CreditsInfo {
    pub credits: u32,
    pub weekly_used: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CreditsState {
    credits: u32,
    weekly_used: u32,
    week_start: String,
    seed_version: u32,
}

fn normalize_credits_state(raw: Option<&serde_json::Value>, current_week: &str) -> CreditsState {
    let mut state = CreditsState {
        credits: raw
            .and_then(|data| data.get("credits"))
            .and_then(|value| value.as_u64())
            .unwrap_or(DEFAULT_STARTING_CREDITS as u64) as u32,
        weekly_used: raw
            .and_then(|data| data.get("weekly_used"))
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as u32,
        week_start: raw
            .and_then(|data| data.get("week_start"))
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string(),
        seed_version: raw
            .and_then(|data| data.get("seed_version"))
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as u32,
    };

    if state.seed_version < CREDITS_SEED_VERSION {
        state.credits = state.credits.max(DEFAULT_STARTING_CREDITS);
        state.seed_version = CREDITS_SEED_VERSION;
    }

    if state.week_start != current_week {
        state.week_start = current_week.to_string();
        state.weekly_used = 0;
    }

    state
}

fn credits_state_to_json(state: &CreditsState) -> serde_json::Value {
    serde_json::json!({
        "credits": state.credits,
        "weekly_used": state.weekly_used,
        "week_start": state.week_start,
        "seed_version": state.seed_version,
    })
}

#[tauri::command]
pub async fn get_credits(_app: tauri::AppHandle) -> Result<CreditsInfo, String> {
    let app_data = std::path::PathBuf::from("D:\\traduzai_data");
    let credits_file = app_data.join("credits.json");
    let current_week = current_week_start();

    let raw = if credits_file.exists() {
        let content = std::fs::read_to_string(&credits_file).map_err(|e| e.to_string())?;
        serde_json::from_str::<serde_json::Value>(&content).ok()
    } else {
        None
    };

    let state = normalize_credits_state(raw.as_ref(), &current_week);
    let json = credits_state_to_json(&state);
    std::fs::write(&credits_file, json.to_string()).map_err(|e| e.to_string())?;

    Ok(CreditsInfo {
        credits: state.credits,
        weekly_used: state.weekly_used,
    })
}

fn current_week_start() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let days_since_epoch = secs / 86400;
    let day_of_week = (days_since_epoch + 3) % 7;
    let week_start_days = days_since_epoch - day_of_week;
    format!("{}", week_start_days)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_credits_state_seeds_default_balance_on_first_run() {
        let state = normalize_credits_state(None, "12345");

        assert_eq!(
            state,
            CreditsState {
                credits: 1000,
                weekly_used: 0,
                week_start: "12345".into(),
                seed_version: 1,
            }
        );
    }

    #[test]
    fn normalize_credits_state_upgrades_old_zero_credit_state() {
        let raw = serde_json::json!({
            "credits": 0,
            "weekly_used": 7,
            "week_start": "12345"
        });

        let state = normalize_credits_state(Some(&raw), "12345");

        assert_eq!(state.credits, 1000);
        assert_eq!(state.weekly_used, 7);
        assert_eq!(state.seed_version, 1);
    }

    #[test]
    fn normalize_credits_state_preserves_existing_balance_above_default() {
        let raw = serde_json::json!({
            "credits": 1400,
            "weekly_used": 3,
            "week_start": "12345"
        });

        let state = normalize_credits_state(Some(&raw), "12345");

        assert_eq!(state.credits, 1400);
        assert_eq!(state.weekly_used, 3);
        assert_eq!(state.seed_version, 1);
    }

    #[test]
    fn normalize_credits_state_resets_weekly_usage_when_week_changes() {
        let raw = serde_json::json!({
            "credits": 1000,
            "weekly_used": 14,
            "week_start": "old",
            "seed_version": 1
        });

        let state = normalize_credits_state(Some(&raw), "new");

        assert_eq!(state.credits, 1000);
        assert_eq!(state.weekly_used, 0);
        assert_eq!(state.week_start, "new");
    }
}
