use super::manager::{
    match_response_id, parse_runtime_response, read_bounded_runtime_line,
    retry_after_seconds_from_error, runtime_exit_message, windows_java_compatible_path,
    RuntimeLaunchSpec, RuntimeReply,
};
use super::protocol::{parse_request_line, ProtocolRequest, PROTOCOL_VERSION};
use super::sandbox::SandboxLimits;

#[test]
fn parses_a_versioned_runtime_request() {
    let request = parse_request_line(
        r#"{"v":1,"id":"req-1","method":"runtime.hello","deadlineMs":15000,"params":{}}"#,
    )
    .unwrap();

    assert_eq!(
        request,
        ProtocolRequest {
            version: PROTOCOL_VERSION,
            id: "req-1".to_string(),
            method: "runtime.hello".to_string(),
            deadline_ms: 15_000,
            params: serde_json::json!({}),
        }
    );
}

#[test]
fn rejects_an_incompatible_protocol_and_unbounded_deadline() {
    let incompatible = parse_request_line(
        r#"{"v":2,"id":"req-1","method":"runtime.hello","deadlineMs":15000,"params":{}}"#,
    )
    .unwrap_err();
    assert_eq!(incompatible.code, "PROTOCOL_INCOMPATIBLE");

    let unbounded = parse_request_line(
        r#"{"v":1,"id":"req-1","method":"source.search","deadlineMs":999999,"params":{}}"#,
    )
    .unwrap_err();
    assert_eq!(unbounded.code, "INVALID_REQUEST");
}

#[test]
fn discovers_only_a_complete_packaged_runtime() {
    let root = tempfile::tempdir().unwrap();
    assert!(RuntimeLaunchSpec::from_resource_root(root.path()).is_none());

    let java = root.path().join("source-runtime/jre/bin/java.exe");
    let libs = root.path().join("source-runtime/broker/lib");
    std::fs::create_dir_all(java.parent().unwrap()).unwrap();
    std::fs::create_dir_all(&libs).unwrap();
    std::fs::write(&java, b"fixture").unwrap();
    std::fs::write(libs.join("runtime-broker.jar"), b"fixture").unwrap();

    let spec =
        RuntimeLaunchSpec::from_resource_root(root.path()).expect("runtime should be complete");
    assert_eq!(spec.program, java);
    let classpath = spec.classpath.to_string_lossy();
    assert!(classpath.contains("runtime-broker.jar"));
    assert!(!classpath.ends_with("lib/*"));
}

#[test]
fn packaged_runtime_classpath_filters_stale_kotlin_versions() {
    let root = tempfile::tempdir().unwrap();
    let java = root.path().join("source-runtime/jre/bin/java.exe");
    let libs = root.path().join("source-runtime/broker/lib");
    std::fs::create_dir_all(java.parent().unwrap()).unwrap();
    std::fs::create_dir_all(&libs).unwrap();
    std::fs::write(&java, b"fixture").unwrap();
    for jar in [
        "runtime-broker-0.1.0.jar",
        "kotlin-stdlib-2.1.21.jar",
        "kotlin-stdlib-2.4.10.jar",
        "kotlinx-coroutines-core-jvm-1.10.2.jar",
        "kotlinx-coroutines-core-jvm-1.11.0.jar",
        "sqlite-jdbc-3.49.1.0.jar",
    ] {
        std::fs::write(libs.join(jar), b"fixture").unwrap();
    }

    let spec = RuntimeLaunchSpec::from_resource_root(root.path()).unwrap();
    let classpath = spec.classpath.to_string_lossy();
    assert!(classpath.contains("kotlin-stdlib-2.4.10.jar"));
    assert!(classpath.contains("kotlinx-coroutines-core-jvm-1.11.0.jar"));
    assert!(!classpath.contains("kotlin-stdlib-2.1.21.jar"));
    assert!(!classpath.contains("kotlinx-coroutines-core-jvm-1.10.2.jar"));
    assert!(classpath.contains("sqlite-jdbc-3.49.1.0.jar"));
}

#[test]
fn rejects_a_response_for_another_request() {
    let error = match_response_id(
        "req-expected",
        r#"{"v":1,"id":"req-other","ok":true,"result":{}}"#,
    )
    .unwrap_err();
    assert!(error.contains("req-other"));
}

#[test]
fn keeps_valid_domain_errors_separate_from_transport_failures() {
    let reply = parse_runtime_response(
        "req-1",
        r#"{"v":1,"id":"req-1","ok":false,"error":{"code":"INVALID_INDEX","message":"Índice inválido"}}"#,
    )
    .expect("a correlated protocol error is still a valid broker response");

    assert_eq!(
        reply,
        RuntimeReply::DomainError("INVALID_INDEX: Índice inválido".to_string())
    );
}

#[test]
fn preserves_retry_after_metadata_for_the_download_scheduler() {
    let reply = parse_runtime_response(
        "req-retry",
        r#"{"v":1,"id":"req-retry","ok":false,"error":{"code":"IMAGE_HTTP_ERROR","message":"HTTP 429","retryable":true,"retryAfterSeconds":7}}"#,
    )
    .unwrap();
    let RuntimeReply::DomainError(error) = reply else {
        panic!("expected domain error");
    };
    assert_eq!(retry_after_seconds_from_error(&error), Some(7));
}

#[tokio::test]
async fn accepts_a_large_catalog_but_rejects_a_response_flood_before_unbounded_growth() {
    use tokio::io::{AsyncWriteExt, BufReader};

    let payload = format!("{{\"catalog\":\"{}\"}}\n", "x".repeat(2 * 1024 * 1024));
    let (mut writer, reader) = tokio::io::duplex(payload.len() + 16);
    let write = tokio::spawn(async move {
        writer.write_all(payload.as_bytes()).await.unwrap();
    });
    let mut reader = BufReader::new(reader);
    let line = read_bounded_runtime_line(&mut reader, 8 * 1024 * 1024)
        .await
        .unwrap();
    write.await.unwrap();
    assert!(line.len() > 2 * 1024 * 1024);

    let oversized = format!("{}\n", "y".repeat(1025));
    let (mut writer, reader) = tokio::io::duplex(oversized.len() + 16);
    let write = tokio::spawn(async move {
        writer.write_all(oversized.as_bytes()).await.unwrap();
    });
    let mut reader = BufReader::new(reader);
    let error = read_bounded_runtime_line(&mut reader, 1024)
        .await
        .unwrap_err();
    write.await.unwrap();
    assert!(error.starts_with("PAYLOAD_TOO_LARGE"));
}

#[test]
fn runtime_exit_diagnostics_include_status_and_bounded_stderr() {
    let message = runtime_exit_message(Some(1), "java.lang.IllegalStateException: falha interna");
    assert!(message.contains("exit=1"));
    assert!(message.contains("IllegalStateException"));

    let bounded = runtime_exit_message(None, &"x".repeat(10_000));
    assert!(bounded.len() < 5_000);
}

#[test]
fn removes_windows_verbatim_prefix_before_passing_a_wildcard_classpath_to_java() {
    let drive = windows_java_compatible_path(std::path::PathBuf::from(
        r"\\?\N:\TraduzAI\source-runtime\broker\lib\*",
    ));
    assert_eq!(
        drive,
        std::path::PathBuf::from(r"N:\TraduzAI\source-runtime\broker\lib\*")
    );

    let unc = windows_java_compatible_path(std::path::PathBuf::from(
        r"\\?\UNC\server\share\broker\lib\*",
    ));
    assert_eq!(
        unc,
        std::path::PathBuf::from(r"\\server\share\broker\lib\*")
    );
}

#[test]
fn sandbox_limits_are_mandatory_and_bounded() {
    let limits = SandboxLimits::default();
    assert_eq!(limits.max_processes, 9);
    assert!(limits.process_memory_bytes <= 512 * 1024 * 1024);
    assert!(limits.job_memory_bytes <= 768 * 1024 * 1024);
    assert!((1..=10_000).contains(&limits.cpu_rate));
}

#[tokio::test]
async fn packaged_runtime_handshake_runs_inside_the_windows_job() {
    let Some(root) = std::env::var_os("TRADUZAI_TEST_PACKAGED_RUNTIME_ROOT") else {
        return;
    };
    let spec = RuntimeLaunchSpec::from_resource_root(std::path::Path::new(&root))
        .expect("packaged runtime fixture should exist");
    let manager = super::manager::SourceRuntimeManager::default();
    let response = manager
        .request(&spec, "runtime.hello", serde_json::json!({}), 5_000)
        .await
        .expect("packaged runtime should answer");
    assert_eq!(
        response.pointer("/result/workerIsolation"),
        Some(&serde_json::json!(true))
    );
    assert!(manager.shutdown().await.unwrap());
}

#[tokio::test]
async fn packaged_runtime_survives_a_domain_error() {
    let Some(root) = std::env::var_os("TRADUZAI_TEST_PACKAGED_RUNTIME_ROOT") else {
        return;
    };
    let spec = RuntimeLaunchSpec::from_resource_root(std::path::Path::new(&root))
        .expect("packaged runtime fixture should exist");
    let manager = super::manager::SourceRuntimeManager::default();

    let error = manager
        .request(&spec, "method.does-not-exist", serde_json::json!({}), 5_000)
        .await
        .unwrap_err();
    assert!(
        error.contains("METHOD_NOT_FOUND"),
        "unexpected error: {error}"
    );

    let response = manager
        .request(&spec, "runtime.hello", serde_json::json!({}), 5_000)
        .await
        .expect("broker must remain alive after a domain error");
    assert_eq!(
        response.pointer("/result/protocol"),
        Some(&serde_json::json!(1))
    );
    assert!(manager.shutdown().await.unwrap());
}

#[tokio::test]
async fn packaged_runtime_previews_a_live_repository_when_requested() {
    let (Some(root), Some(url)) = (
        std::env::var_os("TRADUZAI_TEST_PACKAGED_RUNTIME_ROOT"),
        std::env::var_os("TRADUZAI_TEST_REPOSITORY_URL"),
    ) else {
        return;
    };
    let mut spec = RuntimeLaunchSpec::from_resource_root(std::path::Path::new(&root))
        .expect("packaged runtime fixture should exist");
    let data = tempfile::tempdir().expect("runtime data fixture should exist");
    spec.data_dir = Some(data.path().join("source-runtime"));
    let manager = super::manager::SourceRuntimeManager::default();
    let handshake = manager
        .request(&spec, "runtime.hello", serde_json::json!({}), 5_000)
        .await
        .expect("the UI handshake should start the packaged runtime");
    assert_eq!(
        handshake.pointer("/result/protocol"),
        Some(&serde_json::json!(1))
    );
    let response = manager
        .request(
            &spec,
            "repository.preview",
            serde_json::json!({ "url": url.to_string_lossy() }),
            30_000,
        )
        .await
        .expect("the live repository should be previewable through the Tauri manager");

    assert_eq!(response.get("ok"), Some(&serde_json::json!(true)));
    let extensions = response
        .pointer("/result/extensions")
        .and_then(serde_json::Value::as_array)
        .expect("preview should contain extensions");
    assert!(!extensions.is_empty());
    assert!(extensions[0].pointer("/sources/0/id").is_some());
    assert!(manager.shutdown().await.unwrap());
}
