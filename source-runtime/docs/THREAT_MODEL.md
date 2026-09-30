# Source Runtime threat model

## Assets

Protected assets are TraduzAI projects and images, credentials and tokens, other extensions' cookies/preferences, reader history, private network services, and host availability.

## Trust boundaries

- React is untrusted input and can only call typed Tauri commands.
- Rust/Tauri is the local authority for paths, process creation, sandbox policy and promotion of staged files.
- The broker JVM is trusted runtime code but never loads extension classes.
- Every extension package is arbitrary third-party code and executes in a package-specific worker.
- Repository metadata and all remote responses are untrusted.

## Mandatory release controls

Esta lista define gates de release, não uma declaração de que todos já foram implementados. Na entrega atual, Job Object, limites de payload/prazo, bloqueio de rede privada no cliente HTTP, quarentena JAR e SHA-256 estão ativos. AppContainer/token restrito, ACL por extensão, regra de rede no nível do SO, apksig e pacote assinado continuam abertos; portanto o runtime atual é experimental e não deve ser distribuído como sandbox completa.

- Zero repositories by default; preview does not execute code.
- Explicit initial trust pins the repository signing fingerprint.
- APK/JAR signature, signer, package, version and SHA-256 are checked before loading.
- Downloads use quarantine, bounded parsing, disposable smoke worker and atomic activation.
- Workers receive read access only to the private runtime image and their active package, and write access only to package data plus a job staging directory.
- Workers never receive project paths or TraduzAI secrets.
- Windows AppContainer or restricted token, package ACL, Job Object and resource limits are release gates.
- Public internet is allowed. Loopback, private, link-local and cloud metadata networks are denied unless a package receives explicit self-hosted-source permission.
- No public switch disables the sandbox.
- Cookies, authorization headers, titles and private URLs are redacted from logs.

## Failure policy

A timeout, malformed response, crash or quota violation terminates only the responsible worker/job. Repeated crashes open a circuit breaker. No partial installation, download or translation import is promoted.

## Out of scope for v1

WebView/JCEF, remote server access, OPDS, SyncYomi and execution while the Studio is closed are not release claims.
