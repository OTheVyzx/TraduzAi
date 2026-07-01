# TraduzAI site security and LGPD audit - 2026-05-31

Scope: manual security review of the web/SaaS surface in `server/`, `site/`, `worker/`, dependency manifests, legal pages, and local secret hygiene. This is not a formal legal opinion.

## Executive summary

The site is not production-ready from a security/LGPD risk perspective. The highest-risk issues are:

1. A real `VAST_API_KEY` and worker token are present in `.env.txt`, which is not ignored by `.gitignore`.
2. Project CBZ/JPG export can read server-local files if attacker-controlled `project.json` final-image fields point outside the workspace.
3. `drive_link` accepts arbitrary HTTP(S) URLs and the backend downloads them server-side, creating SSRF risk.
4. Project import and bundle materialization unzip archives without the same file-count and expanded-size limits used for normal uploads.
5. Worker authentication is one global bearer token; token compromise can claim queued jobs across tenants.
6. Worker artifact uploads and editor bitmap writes do not enforce size limits, allowing storage/CPU exhaustion.
7. Legal documents still use placeholder contact/controlador data, so the LGPD rights channel is not operational.

The code has some good foundations: Argon2 password hashing, httpOnly session cookies, SameSite cookies, organization access checks before most project/job/artifact reads, path traversal guards via `safe_path`, upload extension/magic checks for normal uploads, and npm production audits currently showing zero known vulnerabilities.

## Findings

### P0 - Secrets committed or staged locally in `.env.txt`

Evidence:
- `.env.txt:7` contains `VAST_API_KEY`.
- `.env.txt:12` contains `TRADUZAI_WORKER_TOKEN`.
- `.gitignore:65-67` ignores `.env` and `.env.local`, but not `.env.txt`.

Impact: anyone with access to this workspace, backups, zip exports of the repo, or an accidental commit can control/cost your Vast.ai account or impersonate a worker. Treat these tokens as compromised.

Fix:
- Rotate the Vast API key and worker token immediately.
- Delete `.env.txt` or move it outside the repo.
- Add `.env.*`, `*.env`, and `.env.txt` to `.gitignore`, while keeping `.env.example` explicitly allowed.
- Search git history before publishing. If already committed, rewrite history and revoke keys.

### P0 - Project export can read arbitrary server files through project image paths

Evidence:
- `server/projects/export_api.py:93` iterates project pages when `translated/` has no final images.
- `server/projects/export_api.py:97` does `path = root / rel` from `_page_final_image_rel(page)` without `safe_path`.
- `server/projects/export_api.py:160` writes the selected `Path` into the CBZ/JPG zip.
- `_page_final_image_rel()` accepts `arquivo_traduzido`, `rendered_path`, `translated_path`, and `image_layers.*.path` values from project JSON.

Impact: an authenticated project owner can import or save a project whose final-image field references `../../...` or another path accepted by `Path`; then exporting CBZ/JPG can include server-local files in the download. This is a direct confidentiality risk.

Fix:
- Resolve every project asset through `safe_path(root, rel)`.
- Require `path.is_file()` and `path.suffix.lower() in IMAGE_SUFFIXES`.
- Ignore or reject invalid final-image paths instead of joining raw `root / rel`.
- Add regression tests for `../../` and absolute paths in `arquivo_traduzido`, `rendered_path`, and `image_layers`.

### P1 - SSRF via arbitrary `drive_link`

Evidence:
- `server/jobs/uploads.py:76` downloads a URL supplied as `drive_link`.
- `server/jobs/uploads.py:88` calls `urllib.request.urlopen`.
- `server/jobs/uploads.py:144-157` only special-cases Google Drive, then returns any other HTTP(S) URL.

Impact: an authenticated user can make the server request internal services, cloud metadata endpoints, local admin panels, or private network resources. Even with magic-file validation, this can leak reachability and consume bandwidth, and can become worse behind cloud/LAN deployments.

Fix:
- Restrict `drive_link` to `drive.google.com` and expected Google Drive URL shapes only, or remove arbitrary URL download.
- Resolve DNS and block private/link-local/loopback IP ranges.
- Enforce redirect limits and re-check every redirect target.
- Keep low timeouts and file size limits.

### P1 - Zip import/materialization can exhaust disk and memory

Evidence:
- `server/projects/api.py:44-60` imports uploaded ZIPs with `await file.read()` and extracts each entry without max file count or expanded-size accounting.
- `server/projects/workspace.py:145-156` extracts stored `bundle_zip` without expanded-size or count limits.
- Normal job uploads do have limits in `server/jobs/uploads.py:128-139`, but those checks are not reused here.

Impact: an authenticated user can upload/import a zip bomb or huge project archive, exhausting RAM/disk and potentially taking the service down. This is also a data protection availability issue under LGPD security expectations.

Fix:
- Reuse one archive validator for every ZIP/CBZ path: normal upload, project import, and bundle materialization.
- Stream upload to temp file instead of `await file.read()`.
- Enforce `max_file_mb`, `max_files_per_job`, `max_zip_expanded_mb`, per-entry max size, and total extracted byte count.
- Reject nested archives unless explicitly supported.

### P1 - Unbounded worker artifact uploads

Evidence:
- `server/workers/api.py:187-229` streams worker artifacts to temp files but increments `size` without enforcing any max.

Impact: a leaked worker token or compromised worker can fill storage with large artifacts. The current global worker Bearer token makes this especially sensitive.

Fix:
- Enforce per-artifact size limits by artifact kind.
- Enforce total job artifact quota.
- Prefer per-worker tokens hashed in DB, not one global shared token.
- Log and alert on unusually large artifacts.

### P1 - Global worker token can claim jobs across tenants

Evidence:
- `server/workers/auth.py:15` validates one shared `settings.worker_token` for all worker endpoints.
- `server/workers/api.py:83` allows a token holder to register/reuse a worker identity by name.
- `server/workers/api.py:119` exposes claim-job to any authenticated worker token holder.
- `server/queue.py:35` selects queued jobs by status and mode, not by worker/org/tenant binding.

Impact: if the worker token leaks, a rogue worker can register, claim queued jobs from any organization, download user input, upload artifacts, and mark jobs failed/completed. This combines badly with the `.env.txt` secret leak and Vast bootstrap distribution.

Fix:
- Replace the global token with per-worker tokens stored hashed in DB.
- Bind workers to allowed queues/orgs or a trusted worker pool.
- Enforce server-side capability and org constraints during `claim_next`.
- Add token rotation/revocation and audit worker registration/claim events.

### P1 - User-controlled `project_config` can select local worker executable

Evidence:
- `server/jobs/api.py:80` accepts arbitrary JSON `project_config`.
- `server/workers/api.py:139` returns that config to workers.
- `worker/runner.py:150` forwards `vision_worker_path`/`_vision_worker_path`.
- `pipeline/main.py:2455` reads the worker path from config.
- `pipeline/vision_stack/runtime.py:1276` resolves and executes that path with `subprocess.run`.

Impact: this is not shell injection, because subprocess uses argv lists, but it is still an untrusted executable-selection boundary. If an attacker can point to a malicious or abusable executable already present on a worker, a SaaS job can drive worker process execution.

Fix:
- Do not accept executable paths from tenant-controlled `project_config`.
- Configure worker binaries only from trusted worker/server config.
- Add strict schema validation for `project_config`, with an allowlist of accepted user fields.
- If dynamic binaries are required, enforce fixed directory, extension, ownership, signature/hash allowlist, and no tenant override.

### P1 - Legal/LGPD contact channel is a placeholder

Evidence:
- `site/src/legal/Legal.tsx:6` sets `CONTACT_EMAIL = "[EMAIL_CONTATO]"`.
- Privacy policy/controller section uses this placeholder at `site/src/legal/Legal.tsx:308-313`.
- Rights request section uses this placeholder at `site/src/legal/Legal.tsx:428-429`.

Impact: users cannot exercise LGPD rights through a real channel. ANPD small-agent rules allow flexibility, but still require accessible treatment information and a channel for data subjects.

Fix:
- Replace with a monitored real mailbox, e.g. `privacidade@...` and `legal@...`.
- Identify the controller with legal name/CPF/CNPJ or business identity, not just the brand.
- Define internal SLA and evidence trail for rights requests.

### P2 - Logout does not revoke the session row

Evidence:
- `server/auth_api.py:190-196` expects `session_id` as a normal parameter, not `Cookie`, so browser logout normally deletes only the client cookie and leaves the DB session valid until expiry.

Impact: stolen session cookies remain valid after logout for up to 14 days.

Fix:
- Change logout to read `session_id: str | None = Cookie(default=None)`.
- Delete the DB session and set cookie deletion with the same `secure`, `samesite`, `httponly`, and path settings.

### P2 - User deletion does not remove materialized project workspace

Evidence:
- `site/src/legal/Legal.tsx:338` and `site/src/legal/Legal.tsx:405` state job content can be removed by the user and content is removed after deletion.
- `server/jobs/api.py:237-246` deletes registered storage artifacts and marks the job `deleted`.
- `server/projects/workspace.py:24` materializes projects under `storage/projects/{job_id}`.
- No corresponding workspace `rmtree` was found in the delete flow.

Impact: uploaded images, OCR text, translations, previews, exports, and project JSON can remain on disk after the UI/policy says the project is deleted. This is a LGPD retention and user-trust issue.

Fix:
- On job deletion, remove the materialized `projects/{job_id}` workspace after validating it is inside the storage root.
- Add tests that materialize a project, delete it, and assert artifacts plus workspace files are gone.
- Align the privacy policy with actual backup/retention windows.

### P2 - Missing production HTTP security headers

Evidence:
- `server/app.py:42-48` configures CORS, but there is no middleware for `Content-Security-Policy`, `X-Frame-Options`/`frame-ancestors`, `X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`, or HSTS.

Impact: increases blast radius of XSS, clickjacking, MIME sniffing, and downgrade mistakes.

Fix:
- Add response-header middleware.
- Start CSP in report-only mode, then enforce.
- In production behind HTTPS, enable HSTS at proxy/CDN.

### P2 - Rate limiting is in-memory and per-IP only

Evidence:
- `server/auth_api.py:22` uses module-global `_attempts`.
- `server/auth_api.py:35-43` limits only by request client IP.
- `server/auth_api.py:88-96` appends attempts only for invalid credentials.

Impact: limits reset on restart and do not work correctly across multiple processes/instances. Shared NATs can lock out legitimate users; distributed attackers can bypass it.

Fix:
- Use Redis/database-backed rate limiting keyed by IP plus normalized email.
- Add progressive delay or account lock warning.
- Record auth failures in audit logs.

### P2 - Editor bitmap writes can accept very large base64 payloads

Evidence:
- `server/projects/editor_api.py:34-35` accepts `png_data`.
- `server/projects/workspace.py:131-141` decodes and writes the payload without byte, pixel, or image validation.

Impact: authenticated users can consume CPU/memory/disk and store invalid/non-PNG bytes under `.png`.

Fix:
- Limit request body size and decoded bytes.
- Validate PNG signature and dimensions with Pillow.
- Enforce max width/height per page and per project storage quota.

### P2 - Frontend allows remote project image URLs

Evidence:
- `site/src/editor/webProjectAdapter.ts:40` and nearby logic preserve `data:`, `blob:`, `file:`, and `http(s):` asset paths.
- `site/src/App.tsx:1594` can render those URLs in `<img>`.

Impact: an imported or modified project can force the browser to load external images, exposing IP/referrer and enabling tracking or misleading visual content. No direct React XSS was found in the reviewed code.

Fix:
- Restrict project image paths to server-served `/api/projects/{id}/assets/...` URLs.
- If remote URLs are a feature, proxy them through server-side validation and add an explicit privacy notice.

### P2 - Python server dependencies are unpinned

Evidence:
- `requirements-server.txt:1-11` lists package names without versions or hashes.

Impact: production builds are not reproducible and may silently pick up vulnerable or breaking releases.

Fix:
- Pin exact versions in a lock file.
- Run `pip-audit` in CI.
- Separate runtime and dev/test dependencies.

### P3 - Privacy claims exceed implemented evidence

Evidence:
- `site/src/legal/Legal.tsx:333` says access logs are kept for 6 months.
- `site/src/legal/Legal.tsx:407-409` describes retention rules.
- `site/src/legal/Legal.tsx:437-445` describes incident notification.

Impact: the policy promises processes that are not clearly implemented in code. This is not just cosmetic; privacy notices should match actual operations.

Fix:
- Implement access-log retention/deletion policies and documented incident response.
- Add data deletion/export endpoint for account-level rights requests, or document the manual workflow.
- Keep a Registro de Operacoes de Tratamento and operator inventory.

### P3 - Cookie consent UI does not match policy promises

Evidence:
- `site/src/legal/CookieBanner.tsx:18` stores only `accepted`/`rejected` in `localStorage`, with no timestamp or expiry.
- `site/src/legal/Legal.tsx:495` states the consent preference lasts 12 months.
- `site/src/legal/Legal.tsx:515` says the user can change cookie choices via a footer link, but no reset/revoke UI was found.

Impact: current technical risk is low because no Google Analytics loader was found, but the legal text and UI behavior diverge. If analytics is added later, consent revocation will be incomplete.

Fix:
- Store consent with timestamp/version and expire it after the promised period.
- Add a visible cookie preferences control on `/legal/cookies` and/or footer.
- Load analytics only after accepted consent.

### P3 - Outbound text/metadata sharing needs explicit SaaS policy gate

Evidence:
- `pipeline/translator/translate.py:427`, `:464`, and `:627` show OCR text can be translated through external providers such as Google Translate or configured Ollama hosts.
- `server/projects/setup_api.py:98` and `:266` send work-title queries to external metadata providers.

Impact: images are not sent on the translation path observed, but OCR text, manga/work names, and metadata queries can leave the service. That must be clearly disclosed and controllable for a privacy-sensitive SaaS product.

Fix:
- Add per-job/user-visible outbound-processing consent or mode selection.
- Provide an offline/local-only mode.
- Audit and document which external services receive text, metadata, IPs, and identifiers.

## LGPD checklist

Required before public launch:

- Real controller identity and privacy contact published.
- Data inventory: account, session, uploaded images, OCR text, translations, logs, Google OAuth profile, analytics, billing if added.
- Legal basis mapped per processing activity.
- Retention/deletion implemented and documented.
- Data subject request workflow with evidence trail.
- Incident response plan, including ANPD/titular notification criteria.
- Operator list and contracts: hosting, Google OAuth, analytics, Vast.ai/GPU workers, email provider, payment provider if added.
- Security baseline: access control, strong secrets management, encryption in transit, backups, least privilege, logging, vulnerability management.
- Policy simplificada de seguranca da informacao, especially if operating as agente de tratamento de pequeno porte.

## Tooling results

- `npm audit --omit=dev --json` at repo root: 0 known production vulnerabilities.
- `cd site; npm audit --omit=dev --json`: 0 known production vulnerabilities.
- `pip-audit`, `bandit`, `safety`, and `cargo-audit` were not installed in this environment, so Python/Rust advisory scanning was not completed.
- Agent-assisted targeted test run from the upload/storage/worker review: `python -m pytest server/tests/test_worker_api.py server/tests/test_project_workspace.py server/tests/test_project_exports.py server/tests/test_artifacts_zip.py server/tests/test_storage_contract.py worker/tests/test_runner_artifacts.py worker/tests/test_worker_run_once.py -q` produced 22 passed and 1 failed. The failure was `server/tests/test_project_workspace.py::test_materialize_preview_and_asset_serving` returning 404 in the current dirty checkout and was not treated as proof against the findings.

## Source notes

Primary legal/regulatory references checked on 2026-05-31:

- ANPD guide/checklist page for information security for small processing agents.
- ANPD Resolution CD/ANPD No. 2/2022, especially small-agent duties, data subject channel, simplified records, security measures, and simplified security policy.
- LGPD text via Planalto search result snippets for security measures, incident communication, and sanctions.
