use serde::{Deserialize, Serialize};
use tauri::Manager;

#[tauri::command]
pub async fn restart_app(app: tauri::AppHandle) {
    // Matar processos Python sidecar órfãos antes de reiniciar
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/IM", "traduzai-pipeline.exe"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq traduzai*"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }
    app.restart();
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AppSettings {
    pub ollama_model: String,
    pub ollama_host: String,
    pub idioma_destino: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            ollama_model: "traduzai-translator".to_string(),
            ollama_host: "http://localhost:11434".to_string(),
            idioma_destino: "pt-BR".to_string(),
        }
    }
}

fn settings_path(_app: &tauri::AppHandle) -> Result<std::path::PathBuf, String> {
    let app_data = std::path::PathBuf::from("D:\\traduzai_data");
    Ok(app_data.join("settings.json"))
}

#[tauri::command]
pub async fn save_settings(app: tauri::AppHandle, settings: AppSettings) -> Result<(), String> {
    let path = settings_path(&app)?;
    let json = serde_json::to_string_pretty(&settings).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn load_settings(app: tauri::AppHandle) -> Result<AppSettings, String> {
    let path = settings_path(&app)?;
    if !path.exists() {
        return Ok(AppSettings::default());
    }
    let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    Ok(serde_json::from_str(&content).unwrap_or_default())
}

/// Synchronous helper used by pipeline to read settings without async context.
pub fn load_settings_sync(_app: &tauri::AppHandle) -> AppSettings {
    let path = std::path::PathBuf::from("D:\\traduzai_data").join("settings.json");
    if !path.exists() {
        return AppSettings::default();
    }
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return AppSettings::default(),
    };
    serde_json::from_str(&content).unwrap_or_default()
}

/// Check if Ollama is running and return available models.
/// Tries the configured host first, then common fallbacks.
#[tauri::command]
pub async fn check_ollama(app: tauri::AppHandle) -> Result<serde_json::Value, String> {
    let settings = load_settings_sync(&app);
    let configured_host = settings.ollama_host.trim_end_matches('/').to_string();

    // Build fallback list: configured host + common alternatives
    let mut hosts = vec![configured_host.clone()];
    if !hosts.contains(&"http://127.0.0.1:11434".to_string()) {
        hosts.push("http://127.0.0.1:11434".to_string());
    }
    if !hosts.contains(&"http://localhost:11434".to_string()) {
        hosts.push("http://localhost:11434".to_string());
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .map_err(|e| e.to_string())?;

    for host in &hosts {
        let url = format!("{host}/api/tags");
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                let data: serde_json::Value = resp.json().await.unwrap_or_default();
                let models: Vec<String> = data["models"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|m| m["name"].as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                let has_translator = models
                    .iter()
                    .any(|m| m.contains("traduzai-translator") || m.contains("mangatl-translator"));
                return Ok(serde_json::json!({
                    "running": true,
                    "models": models,
                    "has_translator": has_translator,
                }));
            }
        }
    }

    Ok(serde_json::json!({
        "running": false,
        "models": [],
        "has_translator": false,
    }))
}

/// Create the traduzai-translator Ollama model by opening a visible terminal window.
/// Returns the two commands the user will see running.
#[tauri::command]
pub async fn create_translator_model(app: tauri::AppHandle) -> Result<String, String> {
    let modelfile = if cfg!(debug_assertions) {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default()
            .join("pipeline")
            .join("models")
            .join("Modelfile")
    } else {
        app.path()
            .resource_dir()
            .map_err(|e| e.to_string())?
            .join("models")
            .join("Modelfile")
    };

    if !modelfile.exists() {
        return Err(format!(
            "Modelfile não encontrado em: {}",
            modelfile.display()
        ));
    }

    let modelfile_str = modelfile.to_string_lossy().to_string();

    // Write a PS1 script to %TEMP% so we can run it in a visible window
    let script = format!(
        r#"$Host.UI.RawUI.WindowTitle = 'TraduzAi - Setup do modelo'
Write-Host '========================================' -ForegroundColor Cyan
Write-Host ' TraduzAi - Criando modelo de tradução  ' -ForegroundColor Cyan
Write-Host '========================================' -ForegroundColor Cyan
Write-Host ''
Write-Host 'Passo 1/2: Baixando qwen2.5:3b (~1.9 GB)...' -ForegroundColor Yellow
Write-Host '(isso pode demorar dependendo da sua internet)' -ForegroundColor Gray
Write-Host ''
ollama pull qwen2.5:3b
if ($LASTEXITCODE -ne 0) {{
    Write-Host ''
    Write-Host 'ERRO: falha ao baixar qwen2.5:3b.' -ForegroundColor Red
    Write-Host 'Verifique se o Ollama esta rodando e tente novamente.' -ForegroundColor Red
    Read-Host 'Pressione Enter para fechar'
    exit 1
}}
Write-Host ''
Write-Host 'Passo 2/2: Criando traduzai-translator...' -ForegroundColor Yellow
ollama create traduzai-translator -f '{modelfile_str}'
if ($LASTEXITCODE -eq 0) {{
    Write-Host ''
    Write-Host 'Modelo criado com sucesso!' -ForegroundColor Green
    Write-Host 'Voce ja pode fechar esta janela e usar o app.' -ForegroundColor Green
}} else {{
    Write-Host ''
    Write-Host 'ERRO ao criar o modelo.' -ForegroundColor Red
}}
Write-Host ''
Read-Host 'Pressione Enter para fechar'
"#
    );

    let tmp_script = std::env::temp_dir().join("traduzai_setup_model.ps1");
    std::fs::write(&tmp_script, script).map_err(|e| e.to_string())?;

    // Open a new visible PowerShell window running the script
    std::process::Command::new("cmd")
        .args([
            "/c",
            "start",
            "powershell.exe",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            &tmp_script.to_string_lossy(),
        ])
        .spawn()
        .map_err(|e| format!("Não foi possível abrir terminal PowerShell: {e}"))?;

    Ok(format!(
        "Terminal aberto!\n\nComandos que serão executados:\n  ollama pull qwen2.5:3b\n  ollama create traduzai-translator -f {modelfile_str}\n\nAcompanhe o progresso na janela PowerShell que abriu."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_settings_default_values() {
        let settings = AppSettings::default();
        assert_eq!(settings.ollama_model, "traduzai-translator");
        assert_eq!(settings.idioma_destino, "pt-BR");
    }
}
