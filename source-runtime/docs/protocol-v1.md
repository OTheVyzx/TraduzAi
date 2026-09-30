# Source Runtime protocol v1

Transport is UTF-8 JSON Lines over inherited stdin/stdout. Stderr is redacted diagnostics only. Each line is capped at 1 MiB; larger data is exchanged through opaque records or bounded staging manifests.

## Request

```json
{"v":1,"id":"req-1","method":"runtime.hello","deadlineMs":15000,"params":{}}
```

- `v` must equal `1`.
- `id` is a non-empty unique request id.
- `method` is a namespaced method.
- `deadlineMs` is in `1..120000`.
- `params` is always an object.

## Response

```json
{"v":1,"id":"req-1","ok":true,"result":{}}
{"v":1,"id":"req-1","ok":false,"error":{"code":"SOURCE_TIMEOUT","message":"...","retryable":true}}
```

Exactly one terminal response is emitted for each accepted id. Cancelamento por job e descarte de respostas tardias são parte do contrato reservado, mas ainda não estão implementados no vertical slice.

## Event

```json
{"v":1,"event":"job.progress","jobId":"job-1","payload":{"completed":1,"total":10}}
```

Events nunca substituem respostas terminais. O transporte de eventos JVM ainda é um gate aberto; hoje o Tauri emite apenas o estado de lifecycle do runtime.

## Required handshake result

`runtime.hello` retorna protocolo `1`, API mínima Mihon `1.6`, suporte JAR, `workerIsolation=true`, `webView=false` e sandbox obrigatória. O anúncio de APK é diagnóstico de fallback planejado: a instalação devolve `APK_CONVERSION_UNAVAILABLE` enquanto dex2jar/apksig não forem integrados. O build atual aplica Job Object; AppContainer/token restrito e ACL continuam gates de release.

## Error codes implementados ou reservados

- `INVALID_REQUEST`, `PROTOCOL_INCOMPATIBLE`, `PAYLOAD_TOO_LARGE`
- `RUNTIME_UNAVAILABLE`, `RUNTIME_CRASHED`, `RUNTIME_BUSY`
- `JOB_CANCELLED`, `JOB_TIMEOUT`
- `REPOSITORY_UNTRUSTED`, `REPOSITORY_INVALID`
- `PACKAGE_UNSIGNED`, `SIGNER_MISMATCH`, `DOWNGRADE_BLOCKED`, `PACKAGE_QUARANTINED`
- `EXTENSION_INCOMPATIBLE`, `WEBVIEW_REQUIRED`, `SOURCE_CRASHED`, `SOURCE_TIMEOUT`
- `NOT_FOUND`, `STORAGE_LIMIT`, `INTEGRITY_FAILED`, `PATH_REJECTED`

Errors from one source are local to that source and do not discard healthy results from other sources.
