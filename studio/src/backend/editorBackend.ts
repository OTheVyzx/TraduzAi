import type { StudioPage, StudioProject, StudioTextLayer, ImageLayerKey } from "../project/studioProject";
import type {
  FluxGenerateConfig,
  FluxGenerateResult,
  FluxProviderStatus,
} from "../ai/fluxContract";
import type { StudioRecoverySnapshot } from "../autosave/recovery";

export type BitmapLayerKey = Exclude<ImageLayerKey, "base" | "rendered"> | "rendered";

export interface EditorPagePayload {
  project_file?: string;
  project_dir?: string;
  page_index: number;
  total_pages: number;
  page: StudioPage;
  project: StudioProject;
}

export interface BitmapRegionConfig {
  project_path: string;
  page_index: number;
  layer_key: BitmapLayerKey;
  width: number;
  height: number;
  png_data: string;
  dirty_bbox?: [number, number, number, number] | null;
}

export interface GeneratedAssetConfig {
  project_path: string;
  page_index: number;
  asset_id: string;
  png_data: string;
}

export interface DeleteGeneratedAssetsConfig {
  project_path: string;
  page_index: number;
  asset_ids: string[];
}

export interface StudioLiteModelStatus {
  status: "ready" | "missing" | "error" | string;
  path?: string | null;
  message?: string | null;
  [key: string]: unknown;
}

export interface StudioLiteDetection {
  bbox: [number, number, number, number];
  score?: number | null;
  label?: string | null;
}

export interface StudioLiteDetectResult {
  detections: StudioLiteDetection[];
  mask_path?: string | null;
  message?: string | null;
  model?: StudioLiteModelStatus | null;
}

export interface StudioLiteInpaintResult {
  inpaint_path: string;
  before_inpaint_path?: string | null;
  bbox?: [number, number, number, number] | null;
  message?: string | null;
}

export interface StudioEditorBackend {
  loadProject(config: { project_path: string }): Promise<StudioProject>;
  saveProjectJson(config: { project_path: string; project_json: StudioProject }): Promise<void>;
  mutateProject<T>(config: {
    project_path: string;
    mutate: (project: StudioProject) => T | Promise<T>;
  }): Promise<{ project: StudioProject; result: T }>;
  saveRecoverySnapshot(config: { project_path: string; snapshot: StudioRecoverySnapshot }): Promise<void>;
  loadRecoverySnapshot(config: { project_path: string }): Promise<StudioRecoverySnapshot | null>;
  clearRecoverySnapshot(config: { project_path: string }): Promise<void>;
  loadEditorPage(config: { project_path: string; page_index: number }): Promise<EditorPagePayload>;
  createEditorTextLayer(config: {
    project_path: string;
    page_index: number;
    layout_bbox: [number, number, number, number];
  }): Promise<StudioTextLayer>;
  patchEditorTextLayer(config: {
    project_path: string;
    page_index: number;
    layer_id: string;
    patch: Record<string, unknown>;
  }): Promise<StudioTextLayer>;
  deleteEditorTextLayer(config: { project_path: string; page_index: number; layer_id: string }): Promise<void>;
  setEditorLayerVisibility(config: {
    project_path: string;
    page_index: number;
    layer_kind: "image" | "text";
    layer_key?: ImageLayerKey | null;
    layer_id?: string | null;
    visible: boolean;
  }): Promise<void>;
  updateBitmapLayer(config: BitmapRegionConfig): Promise<string>;
  saveGeneratedAsset(config: GeneratedAssetConfig): Promise<string>;
  deleteGeneratedAssets(config: DeleteGeneratedAssetsConfig): Promise<void>;
  fluxProviderStatus(): Promise<FluxProviderStatus>;
  generateFluxFill(config: FluxGenerateConfig): Promise<FluxGenerateResult>;
  cancelFluxFill(jobId: string): Promise<boolean>;
  studioLiteModelStatus?(): Promise<StudioLiteModelStatus>;
  studioLiteDetectPage?(config: {
    project_path: string;
    page_index: number;
    boxes_only?: boolean;
  }): Promise<StudioLiteDetectResult>;
  studioLiteBuildMask?(config: {
    project_path: string;
    page_index: number;
    detections?: StudioLiteDetection[];
    bboxes?: [number, number, number, number][];
    padding?: number;
  }): Promise<string>;
  studioLiteInpaintRegion?(config: {
    project_path: string;
    page_index: number;
    bbox?: [number, number, number, number] | null;
    mask_path?: string | null;
  }): Promise<StudioLiteInpaintResult>;
}

const projectMutationTails = new Map<string, Promise<void>>();
const PROJECT_WRITE_QUIESCENCE_TIMEOUT_MS = 15_000;

interface StudioProjectWriteGate {
  epoch: number;
  frozen: boolean;
  activeWriters: number;
  idleWaiters: Set<() => void>;
}

export interface StudioProjectWriteToken {
  readonly projectPath: string;
  readonly key: string;
  readonly epoch: number;
}

export interface StudioProjectWriteLease {
  readonly token: StudioProjectWriteToken;
  readonly id: string;
  active: boolean;
}

const projectWriteGates = new Map<string, StudioProjectWriteGate>();

function mutationKey(projectPath: string) {
  let normalized = projectPath.trim().replace(/\\/g, "/");
  if (/^memory:\/\//i.test(normalized)) return normalized.replace(/\/+$/, "");
  normalized = normalized.replace(/\/+$/, "");
  if (!/\.json$/i.test(normalized)) normalized = `${normalized}/project.json`;
  const drive = normalized.match(/^([A-Za-z]:)\//)?.[1] ?? "";
  const unc = normalized.startsWith("//");
  const absolute = normalized.startsWith("/");
  const withoutRoot = drive ? normalized.slice(drive.length + 1) : unc ? normalized.slice(2) : absolute ? normalized.slice(1) : normalized;
  const segments: string[] = [];
  for (const segment of withoutRoot.split("/")) {
    if (!segment || segment === ".") continue;
    if (segment === "..") {
      if (segments.length > 0 && segments.at(-1) !== "..") segments.pop();
      else if (!drive && !unc && !absolute) segments.push("..");
      continue;
    }
    segments.push(segment);
  }
  const rooted = drive ? `${drive}/${segments.join("/")}` : unc ? `//${segments.join("/")}` : absolute ? `/${segments.join("/")}` : segments.join("/");
  return drive || unc ? rooted.toLocaleLowerCase() : rooted;
}

function writeGate(key: string) {
  let gate = projectWriteGates.get(key);
  if (!gate) {
    gate = { epoch: 0, frozen: false, activeWriters: 0, idleWaiters: new Set() };
    projectWriteGates.set(key, gate);
  }
  return gate;
}

function assertCurrentWriteToken(token: StudioProjectWriteToken, gate = writeGate(token.key)) {
  if (gate.epoch !== token.epoch) throw new Error("Esta operacao pertence a uma sessao antiga do projeto e foi cancelada");
}

function finishStudioProjectWriter(gate: StudioProjectWriteGate) {
  gate.activeWriters = Math.max(0, gate.activeWriters - 1);
  if (gate.activeWriters !== 0) return;
  for (const resolve of gate.idleWaiters) resolve();
  gate.idleWaiters.clear();
}

async function waitForStudioProjectWriters(gate: StudioProjectWriteGate) {
  if (gate.activeWriters === 0) return;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let resolveIdle!: () => void;
  const idle = new Promise<void>((resolve) => { resolveIdle = resolve; gate.idleWaiters.add(resolve); });
  const timeout = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => {
      gate.idleWaiters.delete(resolveIdle);
      reject(new Error("As operacoes do editor nao terminaram a tempo. Aguarde e tente salvar novamente"));
    }, PROJECT_WRITE_QUIESCENCE_TIMEOUT_MS);
  });
  try { await Promise.race([idle, timeout]); } finally { if (timeoutId !== null) clearTimeout(timeoutId); }
}

function enqueueStudioProjectMutation<T>(key: string, operation: () => Promise<T>): Promise<T> {
  const previous = projectMutationTails.get(key) ?? Promise.resolve();
  const running = previous.catch(() => undefined).then(operation);
  const tail = running.then(() => undefined, () => undefined);
  projectMutationTails.set(key, tail);
  return running.finally(() => { if (projectMutationTails.get(key) === tail) projectMutationTails.delete(key); });
}

/**
 * Serializa todas as escritas do mesmo projeto dentro do processo do Studio.
 * A operacao recebe a vez mesmo quando a mutacao anterior falha.
 */
export async function runSerializedStudioProjectMutation<T>(
  projectPath: string,
  operation: () => Promise<T>,
): Promise<T> {
  const key = mutationKey(projectPath);
  return enqueueStudioProjectMutation(key, operation);
}

export function captureStudioProjectWriteToken(projectPath: string): StudioProjectWriteToken {
  const key = mutationKey(projectPath);
  return { projectPath, key, epoch: writeGate(key).epoch };
}

export async function runStudioProjectWrite<T>(token: StudioProjectWriteToken, operation: (lease: StudioProjectWriteLease) => Promise<T>): Promise<T> {
  const gate = writeGate(token.key);
  assertCurrentWriteToken(token, gate);
  if (gate.frozen) throw new Error("O projeto esta sendo salvo em outro local. Aguarde a operacao terminar");
  const lease: StudioProjectWriteLease = { token, id: crypto.randomUUID(), active: true };
  gate.activeWriters += 1;
  try { return await operation(lease); } finally { lease.active = false; finishStudioProjectWriter(gate); }
}

export async function runStudioProjectWriteLease<T>(lease: StudioProjectWriteLease, operation: () => Promise<T>): Promise<T> {
  if (!lease.active) throw new Error("O lease de escrita do Studio ja foi encerrado");
  assertCurrentWriteToken(lease.token);
  return operation();
}

async function runExclusiveStudioProjectGate<T>({ token, targetPath, transition, operation }: {
  token: StudioProjectWriteToken; targetPath?: string; transition: boolean; operation: () => Promise<T>;
}) {
  const sourceGate = writeGate(token.key);
  assertCurrentWriteToken(token, sourceGate);
  const targetKey = targetPath ? mutationKey(targetPath) : token.key;
  const targetGate = writeGate(targetKey);
  const gates = sourceGate === targetGate ? [sourceGate] : [sourceGate, targetGate];
  if (gates.some((gate) => gate.frozen)) throw new Error("Ja existe uma transicao exclusiva em andamento para este projeto");
  for (const gate of gates) gate.frozen = true;
  try {
    await Promise.all(gates.map(waitForStudioProjectWriters));
    assertCurrentWriteToken(token, sourceGate);
    const result = await operation();
    if (transition) for (const gate of gates) gate.epoch += 1;
    return result;
  } finally { for (const gate of gates) gate.frozen = false; }
}

export function runExclusiveStudioProjectCheckpoint<T>(token: StudioProjectWriteToken, operation: () => Promise<T>) {
  return runExclusiveStudioProjectGate({ token, transition: false, operation });
}

export function runExclusiveStudioProjectTransition<T>(token: StudioProjectWriteToken, targetPath: string, operation: () => Promise<T>) {
  return runExclusiveStudioProjectGate({ token, targetPath, transition: true, operation });
}

let configuredBackend: StudioEditorBackend | null = null;

export function configureStudioEditorBackend(backend: StudioEditorBackend | null) {
  configuredBackend = backend;
}

export function getStudioEditorBackend() {
  if (!configuredBackend) {
    throw new Error("Studio editor backend is not configured");
  }
  return configuredBackend;
}
