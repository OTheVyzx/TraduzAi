use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

static CONFIGURED_PATHS: once_cell::sync::OnceCell<StoragePaths> = once_cell::sync::OnceCell::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageMode {
    Dev,
    #[allow(dead_code)]
    Production,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StoragePaths {
    pub mode: StorageMode,
    pub root: PathBuf,
    pub works: PathBuf,
    pub memory: PathBuf,
    pub logs: PathBuf,
    pub exports: PathBuf,
    pub debug: PathBuf,
    pub fixtures: PathBuf,
    pub models: PathBuf,
    pub projects: PathBuf,
    pub settings: PathBuf,
}

#[derive(Debug, Clone)]
pub struct StorageService {
    mode: StorageMode,
    repo_root: Option<PathBuf>,
    app_data_root: Option<PathBuf>,
}

impl StorageService {
    pub fn dev(repo_root: impl Into<PathBuf>) -> Self {
        Self {
            mode: StorageMode::Dev,
            repo_root: Some(repo_root.into()),
            app_data_root: None,
        }
    }

    #[allow(dead_code)]
    pub fn production(app_data_root: impl Into<PathBuf>) -> Self {
        Self {
            mode: StorageMode::Production,
            repo_root: None,
            app_data_root: Some(app_data_root.into()),
        }
    }

    pub fn paths(&self) -> StoragePaths {
        let root = match self.mode {
            StorageMode::Dev => self
                .repo_root
                .as_ref()
                .expect("dev storage precisa do repo root")
                .join("data"),
            StorageMode::Production => self
                .app_data_root
                .as_ref()
                .expect("production storage precisa do appDataDir")
                .join("TraduzAI"),
        };

        let (debug, fixtures) = match self.mode {
            StorageMode::Dev => {
                let repo_root = self
                    .repo_root
                    .as_ref()
                    .expect("dev storage precisa do repo root");
                (repo_root.join("debug"), repo_root.join("fixtures"))
            }
            StorageMode::Production => (root.join("debug"), root.join("fixtures")),
        };

        StoragePaths {
            mode: self.mode,
            works: root.join("works"),
            memory: root.join("memory"),
            logs: root.join("logs"),
            exports: root.join("exports"),
            models: root.join("models"),
            projects: root.join("projects"),
            settings: root.join("settings.json"),
            debug,
            fixtures,
            root,
        }
    }

    pub fn ensure_base_dirs(&self) -> Result<StoragePaths, String> {
        let paths = self.paths();
        for dir in [
            &paths.root,
            &paths.works,
            &paths.memory,
            &paths.logs,
            &paths.exports,
            &paths.debug,
            &paths.fixtures,
            &paths.models,
            &paths.projects,
        ] {
            fs::create_dir_all(dir)
                .map_err(|e| format!("Falha ao criar pasta de storage '{}': {e}", dir.display()))?;
        }
        Ok(paths)
    }

    pub fn check_writable(&self) -> Result<(), String> {
        let paths = self.paths();
        let probe = paths.root.join(".traduzai_write_test");
        fs::write(&probe, b"ok").map_err(|e| {
            format!(
                "Storage do TraduzAI sem permissao de escrita em '{}': {e}",
                paths.root.display()
            )
        })?;
        fs::remove_file(&probe).map_err(|e| {
            format!(
                "Storage do TraduzAI escreveu, mas nao conseguiu limpar '{}': {e}",
                probe.display()
            )
        })?;
        Ok(())
    }
}

pub fn repo_root_from_current_dir(current_dir: &Path) -> PathBuf {
    if current_dir
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("src-tauri"))
    {
        current_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| current_dir.to_path_buf())
    } else {
        current_dir.to_path_buf()
    }
}

pub fn set_configured_paths(paths: StoragePaths) {
    let _ = CONFIGURED_PATHS.set(paths);
}

pub fn configured_or_dev_paths() -> Result<StoragePaths, String> {
    if let Some(paths) = CONFIGURED_PATHS.get() {
        return Ok(paths.clone());
    }

    let current_dir = std::env::current_dir()
        .map_err(|e| format!("Falha ao localizar pasta atual do app: {e}"))?;
    Ok(StorageService::dev(repo_root_from_current_dir(&current_dir)).paths())
}

#[cfg(debug_assertions)]
pub fn service_for_app(app: &tauri::AppHandle) -> Result<StorageService, String> {
    let _ = app;
    let current_dir = std::env::current_dir()
        .map_err(|e| format!("Falha ao localizar pasta atual do app: {e}"))?;
    Ok(StorageService::dev(repo_root_from_current_dir(
        &current_dir,
    )))
}

#[cfg(not(debug_assertions))]
pub fn service_for_app(app: &tauri::AppHandle) -> Result<StorageService, String> {
    use tauri::Manager;

    let app_data = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Falha ao localizar appDataDir do TraduzAI: {e}"))?;
    Ok(StorageService::production(app_data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dev_paths_stay_inside_repo_roots() {
        let repo_root = std::path::PathBuf::from(r"C:\repo\TraduzAi");
        let service = StorageService::dev(repo_root.clone());
        let paths = service.paths();

        assert_eq!(paths.mode, StorageMode::Dev);
        assert_eq!(paths.root, repo_root.join("data"));
        assert_eq!(paths.works, repo_root.join("data").join("works"));
        assert_eq!(paths.memory, repo_root.join("data").join("memory"));
        assert_eq!(paths.logs, repo_root.join("data").join("logs"));
        assert_eq!(paths.exports, repo_root.join("data").join("exports"));
        assert_eq!(paths.debug, repo_root.join("debug"));
        assert_eq!(paths.fixtures, repo_root.join("fixtures"));
    }

    #[test]
    fn production_paths_use_app_data_traduzai_root() {
        let app_data = std::path::PathBuf::from(r"C:\Users\Ana\AppData\Roaming");
        let service = StorageService::production(app_data.clone());
        let paths = service.paths();

        assert_eq!(paths.mode, StorageMode::Production);
        assert_eq!(paths.root, app_data.join("TraduzAI"));
        assert_eq!(paths.works, app_data.join("TraduzAI").join("works"));
        assert_eq!(paths.memory, app_data.join("TraduzAI").join("memory"));
        assert_eq!(paths.logs, app_data.join("TraduzAI").join("logs"));
        assert_eq!(paths.exports, app_data.join("TraduzAI").join("exports"));
        assert_eq!(paths.debug, app_data.join("TraduzAI").join("debug"));
        assert_eq!(paths.fixtures, app_data.join("TraduzAI").join("fixtures"));
    }

    #[test]
    fn ensure_base_dirs_creates_all_central_dirs() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let service = StorageService::dev(tmp.path().join("repo"));

        service.ensure_base_dirs().expect("cria diretorios");
        let paths = service.paths();

        assert!(paths.works.is_dir());
        assert!(paths.memory.is_dir());
        assert!(paths.logs.is_dir());
        assert!(paths.exports.is_dir());
        assert!(paths.debug.is_dir());
        assert!(paths.fixtures.is_dir());
    }

    #[test]
    fn check_writable_writes_and_removes_probe_file() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let service = StorageService::production(tmp.path().join("appdata"));

        service.ensure_base_dirs().expect("cria diretorios");
        service.check_writable().expect("storage gravavel");

        assert!(!service.paths().root.join(".traduzai_write_test").exists());
    }

    #[test]
    fn creation_failure_returns_clear_error() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let blocked = tmp.path().join("blocked");
        std::fs::write(&blocked, "not a dir").expect("arquivo bloqueador");
        let service = StorageService::dev(blocked);

        let err = service.ensure_base_dirs().expect_err("deve falhar");
        assert!(err.contains("Falha ao criar pasta de storage"), "{err}");
    }

    #[test]
    fn repo_root_resolver_handles_src_tauri_current_dir() {
        let current = std::path::PathBuf::from(r"C:\repo\TraduzAi\src-tauri");
        assert_eq!(
            repo_root_from_current_dir(&current),
            std::path::PathBuf::from(r"C:\repo\TraduzAi")
        );
    }
}
