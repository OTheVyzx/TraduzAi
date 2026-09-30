import { beforeEach, describe, expect, it, vi } from "vitest";

const { invoke, listen } = vi.hoisted(() => ({
  invoke: vi.fn(),
  listen: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({ invoke }));
vi.mock("@tauri-apps/api/event", () => ({ listen }));

import {
  STUDIO_PROJECT_EVENT_CHANNEL,
  createTauriStudioIpcClient,
  subscribeToProjectEvents,
} from "../studioIpcTauri";

describe("Tauri Studio IPC v1 transport", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    invoke.mockResolvedValue({ job_id: "job-001", project_revision: 2 });
  });

  it("passes canonical arguments directly to the Tauri command", async () => {
    const client = createTauriStudioIpcClient();

    await client.startConsumerFast({
      projectPath: "N:/project/project.json",
      chapterId: "chapter-1",
      expectedRevision: 1,
      idempotencyKey: "start-r1",
    });

    expect(invoke).toHaveBeenCalledWith("start_consumer_fast", {
      projectPath: "N:/project/project.json",
      chapterId: "chapter-1",
      expectedRevision: 1,
      idempotencyKey: "start-r1",
    });
  });

  it("subscribes to the canonical channel and forwards a valid ProjectEvent", async () => {
    let callback: ((event: { payload: unknown }) => void) | undefined;
    const unlisten = vi.fn();
    listen.mockImplementation(async (_channel, handler) => {
      callback = handler;
      return unlisten;
    });
    const onEvent = vi.fn();

    await subscribeToProjectEvents(onEvent);
    const payload = {
      job_id: "job-001",
      project_id: "project-001",
      expected_revision: 1,
      project_revision: 1,
      sequence: 0,
      stage: "analysis",
      status: "running",
      reason_code: "provider_started",
      payload: {},
    };
    callback?.({ payload });

    expect(listen).toHaveBeenCalledWith(STUDIO_PROJECT_EVENT_CHANNEL, expect.any(Function));
    expect(onEvent).toHaveBeenCalledWith(payload);
  });

  it("rejects malformed event payloads without updating job state", async () => {
    let callback: ((event: { payload: unknown }) => void) | undefined;
    listen.mockImplementation(async (_channel, handler) => {
      callback = handler;
      return vi.fn();
    });
    const onEvent = vi.fn();
    const onInvalid = vi.fn();

    await subscribeToProjectEvents(onEvent, onInvalid);
    callback?.({ payload: { job_id: "job-001", status: "completed" } });

    expect(onEvent).not.toHaveBeenCalled();
    expect(onInvalid).toHaveBeenCalledWith(expect.objectContaining({
      message: expect.stringContaining("ProjectEvent"),
    }));
  });
});
