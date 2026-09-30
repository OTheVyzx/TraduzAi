import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import { createStudioIpcClient } from "./studioIpcClient";
import type { ProjectEvent } from "./studioIpcV1";

export const STUDIO_PROJECT_EVENT_CHANNEL = "consumer-fast-project-event";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNonNegativeInteger(value: unknown): value is number {
  return Number.isInteger(value) && Number(value) >= 0;
}

export function parseProjectEvent(value: unknown): ProjectEvent {
  if (
    !isRecord(value)
    || typeof value.job_id !== "string"
    || typeof value.project_id !== "string"
    || !isNonNegativeInteger(value.expected_revision)
    || !isNonNegativeInteger(value.project_revision)
    || !isNonNegativeInteger(value.sequence)
    || typeof value.stage !== "string"
    || typeof value.status !== "string"
    || typeof value.reason_code !== "string"
    || !isRecord(value.payload)
  ) {
    throw new Error("ProjectEvent inválido recebido do backend integrado");
  }

  return value as unknown as ProjectEvent;
}

export function createTauriStudioIpcClient() {
  return createStudioIpcClient({
    invoke: <T>(command: string, args: Record<string, unknown>) => invoke<T>(command, args),
  });
}

export function subscribeToProjectEvents(
  onEvent: (event: ProjectEvent) => void,
  onInvalid: (error: Error) => void = () => undefined,
): Promise<UnlistenFn> {
  return listen<unknown>(STUDIO_PROJECT_EVENT_CHANNEL, ({ payload }) => {
    try {
      onEvent(parseProjectEvent(payload));
    } catch (error) {
      onInvalid(error instanceof Error ? error : new Error(String(error)));
    }
  });
}
