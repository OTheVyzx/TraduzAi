# Landing Demo Backend + Vast.ai Plan

> **For Claude/Codex:** implement task-by-task. Do not connect the landing directly to Vast.ai from the browser. Keep the first backend implementation behind the existing server, queue, worker, and artifact contracts.

**Goal:** Turn the visual landing demo into a real one-page translation flow: the user uploads one image on the public landing, the backend creates a controlled demo job, a worker on the cheapest eligible Vast.ai GPU processes the page using the existing fast-page pipeline, and the landing displays the translated image directly.

**Architecture:** Add a public, rate-limited demo API over the existing SaaS backend. Reuse `Job`, `Artifact`, `JobEvent`, `server.queue.enqueue`, `worker.run_fast_page_job`, and `server.vast.orchestrator.ensure_worker_available`. Treat Vast.ai as backend infrastructure only; the frontend never sees Vast API credentials or offer selection.

**Primary implementation files:**

- `site/src/App.tsx` - landing UI upload/status/result integration.
- `site/src/projectApi.ts` or new `site/src/demoApi.ts` - browser API client.
- `server/app.py` - register the demo router.
- `server/demo_api.py` - new public demo endpoints.
- `server/jobs/uploads.py` - reuse upload helpers where safe; add image-only validation if needed.
- `server/vast/orchestrator.py` - reuse Vast autostart and cheapest eligible offer logic.
- `worker/__main__.py` - preserve worker claim/upload behavior.
- `worker/runner.py` - reuse `run_fast_page_job` and `translated_image` artifact streaming.
- `server/tests/` and `worker/tests/` - coverage for job creation, validation, status, artifacts, and Vast orchestration.

---

## Non-Negotiable Contracts

- The browser must call only the TraduzAI backend, never Vast.ai directly.
- `VAST_API_KEY`, `TRADUZAI_WORKER_TOKEN`, template IDs, and worker bootstrap env vars stay server-side.
- Demo upload accepts only one raster image: PNG, JPG/JPEG, or WEBP.
- Demo upload rejects ZIP, CBZ, PDF, SVG, archives, multi-file requests, and unknown MIME.
- The demo flow creates a normal queued job and stores the input as `Artifact(kind="input_original")`.
- The worker result must be exposed as `Artifact(kind="translated_image")`.
- Keep the existing worker contract: worker registers, claims, downloads input, uploads artifacts, posts events, completes or fails the job.
- Use `run_fast_page_job` by default for the demo. Do not fork a second worker runtime unless the fast-page path proves incompatible.
- The job must have a timeout and cleanup path. Demo jobs cannot leave Vast instances running indefinitely.
- The UI/terms must make remote GPU processing explicit before production release because this differs from the desktop-local promise.

---

## Phase 0: Backend Reality Check

**Objective:** verify current backend behavior before adding endpoints.

Tasks:

1. Confirm `server/jobs/api.py` still enqueues non-manual jobs and calls `ensure_worker_available`.
2. Confirm `server/workers/api.py` still returns `input_download_url` and accepts `translated_image` artifacts.
3. Confirm `worker/__main__.py` still routes non-mock work through `run_fast_page_job` when `TRADUZAI_FAST_PAGE_SERVER=1`.
4. Confirm `server/vast/orchestrator.py` still chooses the cheapest eligible offer by sorting `dph_total` ascending.
5. Run existing tests:
   - `python -m pytest server/tests/test_vast_orchestrator_unit.py`
   - `python -m pytest server/tests/test_worker_api.py`
   - `python -m pytest worker/tests/test_worker_run_once.py`

Validation gate:

- No implementation until the current queue/worker/Vast contracts are confirmed or documented as changed.

Rollback:

- No runtime changes in this phase.

---

## Phase 1: Public Demo API

**Objective:** add a public endpoint that safely creates a single-page demo job.

Add `server/demo_api.py` with:

- `POST /api/demo/page`
- `GET /api/demo/page/{job_id}`

`POST /api/demo/page` request:

- multipart form
- `file`: required image
- optional `src_lang`, default `en`
- optional `dst_lang`, default `pt-BR`
- required `remote_processing_consent=true` before production

`POST /api/demo/page` behavior:

1. Validate image-only upload.
2. Create or resolve a demo organization/user identity.
3. Create a `Job` with:
   - `status="queued"`
   - `mode="real"`
   - `obra="Demo landing"`
   - `capitulo="1"`
   - `src_lang="en"`
   - `dst_lang="pt-BR"`
   - `page_count=1`
   - `config_json` containing demo flags.
4. Store the uploaded image as `Artifact(kind="input_original")`.
5. Add `JobEvent(stage="queue", kind="status", message="Demo enfileirada")`.
6. Call `enqueue(settings, job_id)`.
7. Call `ensure_worker_available(settings)` when Vast autostart is enabled.
8. Return `{ job: { id, status } }`.

Suggested `config_json`:

```json
{
  "mode": "auto",
  "demo_page": true,
  "runtime_profile": "fast",
  "export_mode": "single_page"
}
```

`GET /api/demo/page/{job_id}` response:

```json
{
  "job": {
    "id": "...",
    "status": "queued|claimed|running|uploading_results|completed|failed|cancelled",
    "progress": 0,
    "message": "Processando...",
    "error_message": null,
    "result_image_url": null,
    "artifacts": []
  }
}
```

Validation gate:

- A test can create a demo job with a PNG and see a queued job plus `input_original`.
- Invalid file types return 422.
- No authenticated user is required for the demo endpoint if product decision is public demo.

Rollback:

- Remove only `server/demo_api.py` and router registration. Normal `/api/jobs` remains untouched.

---

## Phase 2: Image Validation and Abuse Limits

**Objective:** prevent expensive or unsafe public uploads.

Add config fields in `server/config.py`:

- `demo_enabled`
- `demo_max_image_mb`
- `demo_max_pixels`
- `demo_max_jobs_per_ip_hour`
- `demo_global_concurrency`
- `demo_job_timeout_seconds`
- `demo_retention_hours`

Validation rules:

- Check declared content type.
- Check extension.
- Inspect file signature with Pillow or a minimal image parser.
- Reject images over size/pixel limits.
- Normalize or strip metadata only if needed by storage policy.
- Store source IP/user-agent hash in audit or demo metadata for rate limiting.

Concurrency/rate limiting:

- Start simple with SQLite-backed counters or a lightweight table if available.
- If no table is added initially, enforce global concurrency by counting queued/running demo jobs.
- Return 429 when limits are exceeded.

Validation gate:

- Tests cover PNG/JPG/WEBP accepted.
- Tests cover ZIP/PDF/SVG rejected.
- Tests cover max size and max concurrency.

Rollback:

- Turn off with `TRADUZAI_DEMO_ENABLED=0`.

---

## Phase 3: Worker Compatibility

**Objective:** make the existing worker process demo jobs without a new runtime.

Preferred path:

- Keep `job.mode="real"`.
- Put `demo_page=true` inside `project_config`.
- Let worker capabilities remain `{"mode": ["mock", "real"]}`.
- Let `pipeline_mode_for_job` keep returning `auto`.
- Use `run_fast_page_job` when `TRADUZAI_FAST_PAGE_SERVER=1`.

Only change worker code if needed:

- If the pipeline needs a special mode, add a config flag read inside `worker/runner.py`, not a new public job mode.
- If artifact upload races with status polling, rely on existing streamed page artifact callback.

Validation gate:

- Worker unit test proves a demo job uploads `translated_image`.
- `GET /api/demo/page/{job_id}` returns `result_image_url` when artifact exists.

Rollback:

- Because the job mode remains `real`, rollback is limited to removing demo endpoint use.

---

## Phase 4: Vast.ai Autostart for Demo

**Objective:** ensure the cheapest eligible GPU worker starts when a demo job is queued.

Existing Vast settings to reuse:

- `VAST_API_KEY`
- `VAST_AUTOSTART=1`
- `VAST_OFFER_AUTO=1`
- `VAST_TEMPLATE_HASH`
- `VAST_WORKER_API_URL`
- `TRADUZAI_WORKER_TOKEN`
- `VAST_OFFER_MAX_DPH`
- `VAST_OFFER_MIN_GPU_RAM_GB`
- `VAST_DISK_GB`
- `VAST_IDLE_STOP_MINUTES`

Optional demo-specific settings:

- `VAST_DEMO_OFFER_MAX_DPH`
- `VAST_DEMO_MIN_GPU_RAM_GB`
- `VAST_DEMO_DISK_GB`
- `VAST_DEMO_IDLE_STOP_MINUTES`
- `VAST_DEMO_LABEL`

Implementation approach:

1. Reuse `ensure_worker_available(settings)` first.
2. Add demo-specific offer settings only if normal Vast settings are too broad or too expensive.
3. Store the selected offer summary as a `JobEvent` payload for observability.
4. Do not block the HTTP response until the Vast instance is fully warm; return queued status and let polling continue.

Validation gate:

- Unit tests with fake Vast client prove:
  - missing API key is reported cleanly
  - cheapest eligible offer is chosen
  - max price cap is respected
  - worker bootstrap config is required

Rollback:

- Set `VAST_AUTOSTART=0`; jobs remain queued for a manually started worker.

---

## Phase 5: Status, Events, and Frontend Integration

**Objective:** connect the landing visual UI to real backend state.

Frontend API:

- Add `site/src/demoApi.ts`.
- `createDemoPageJob(file)` calls `POST /api/demo/page`.
- `getDemoPageJob(jobId)` polls `GET /api/demo/page/{job_id}`.

Landing behavior:

1. User selects an image.
2. User clicks `Traduzir`.
3. UI uploads file and stores returned `job.id`.
4. Poll every 1.5-3 seconds.
5. Update timeline from backend status/events.
6. When completed, render `result_image_url`.
7. On failed/cancelled/timeout, show retry.

Progress mapping:

- `queued`: 10
- `claimed`: 20
- `running`: 45
- event `page_completed` or `uploading_results`: 85
- `completed`: 100
- `failed`: error state

Validation gate:

- Playwright flow can upload image, mock backend status transitions, and display result.
- No UI call contains Vast credentials.

Rollback:

- Feature flag the real API call and fall back to visual-only demo.

---

## Phase 6: Cleanup, Timeout, and Cost Control

**Objective:** avoid leaked storage, stuck jobs, and idle GPU spend.

Backend cleanup:

- Add janitor task or command:
  - expire demo jobs older than `demo_job_timeout_seconds`
  - delete demo artifacts after `demo_retention_hours`
  - mark stuck demo jobs failed with `error_code="demo_timeout"`
  - call `stop_idle_worker_if_needed(settings)` when there is no queued/running work

Cost controls:

- enforce `VAST_OFFER_MAX_DPH` or demo-specific max DPH
- keep global demo concurrency low
- record Vast action in job events
- optionally require login before using real GPU in production

Validation gate:

- Test stuck jobs are failed.
- Test old demo artifacts are deleted.
- Test idle stop is called only when no queued/running jobs exist.

Rollback:

- Disable janitor separately if it interferes with normal jobs.

---

## Phase 7: Legal and Product Guardrails

**Objective:** make remote processing clear and consistent with product promises.

Required copy before production:

- "Para este teste online, sua página será processada temporariamente em uma GPU remota e apagada após o job."

Backend requirement:

- `remote_processing_consent=true` must be submitted for public demo jobs.
- Store consent in job config or audit event.

Docs/legal:

- Update site privacy/terms if the public demo is enabled.
- Keep desktop-local wording separate from SaaS/landing demo wording.

Validation gate:

- Demo endpoint rejects missing consent in production mode.

Rollback:

- Keep endpoint disabled until legal copy is merged.

---

## Test Matrix

Server:

- `POST /api/demo/page` creates queued job from PNG.
- JPG and WEBP are accepted.
- ZIP, CBZ, PDF, SVG, and text files are rejected.
- Oversized image is rejected.
- Rate limit returns 429.
- Demo disabled returns 404 or 403.
- Status returns `result_image_url` after `translated_image` artifact exists.
- Vast autostart is called once per queued demo job when enabled.
- Vast autostart failure does not lose the queued job.

Worker:

- Demo-shaped job is claimed as `real`.
- Fast-page runner uploads `translated_image`.
- Failure uploads runner log if available and marks job failed.

Frontend:

- Upload field accepts only images.
- Create job request sends the selected image.
- Polling updates status.
- Completed job displays translated result.
- Failed job shows retry.
- Mobile first viewport does not overlap controls or cookie banner-critical actions.

---

## Suggested Implementation Order

1. `server/demo_api.py` with mocked/no-op Vast call under tests.
2. Router registration in `server/app.py`.
3. Image-only validation and demo config in `server/config.py`.
4. Status endpoint returning artifacts.
5. Tests for server demo creation/status.
6. Landing API client and real upload/polling.
7. Worker compatibility tests for demo-shaped job.
8. Vast event logging and cleanup/timeout.
9. Legal consent gate and copy.

---

## Open Decisions

- Public anonymous demo or login-required demo?
- Final retention window for uploaded pages and translated results.
- Whether to add demo-specific Vast pricing/GPU settings or reuse global Vast settings.
- Whether to stream status via SSE later or keep polling for v1.
- Whether a demo result should create a recoverable full project or stay as a disposable one-page artifact.

