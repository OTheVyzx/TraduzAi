import { execFileSync } from "node:child_process";

function normalizeWindowsPath(value) {
  return String(value ?? "")
    .replace(/\//g, "\\")
    .toLowerCase();
}

function readField(processInfo, key) {
  return processInfo?.[key] ?? processInfo?.[key[0].toUpperCase() + key.slice(1)];
}

export function shouldCleanupDevProcess(processInfo, workspaceRoot, currentPid) {
  const processId = Number(readField(processInfo, "processId") ?? 0);
  if (!processId || processId === currentPid) {
    return false;
  }

  const commandLine = normalizeWindowsPath(readField(processInfo, "commandLine"));
  const name = normalizeWindowsPath(readField(processInfo, "name"));
  const workspace = normalizeWindowsPath(workspaceRoot);

  if (!name.endsWith("node.exe")) {
    return false;
  }

  if (commandLine.includes("adobe creative cloud")) {
    return false;
  }

  return (
    commandLine.includes("scripts\\run-tauri.mjs")
    || commandLine.includes("scripts\\run-vite.mjs")
    || commandLine.includes(`${workspace}\\node_modules\\@tauri-apps\\cli\\tauri.js dev`)
    || commandLine.includes(`${workspace}\\node_modules\\vite\\bin\\vite.js`)
  );
}

export function collectWorkspaceDevProcessIds(processList, workspaceRoot, currentPid) {
  return processList
    .filter((processInfo) => shouldCleanupDevProcess(processInfo, workspaceRoot, currentPid))
    .map((processInfo) => Number(readField(processInfo, "processId")))
    .filter((processId) => Number.isInteger(processId) && processId > 0);
}

export function buildProcessSnapshotScript() {
  return [
    "$ErrorActionPreference = 'Stop'",
    "Get-CimInstance Win32_Process |",
    "Where-Object { $_.Name -eq 'node.exe' } |",
    "Select-Object ProcessId, Name, CommandLine |",
    "ConvertTo-Json -Compress",
  ].join("\n");
}

function readWindowsProcessSnapshot() {
  const raw = execFileSync(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-Command", buildProcessSnapshotScript()],
    { encoding: "utf8" },
  ).trim();

  if (!raw) {
    return [];
  }

  const parsed = JSON.parse(raw);
  return Array.isArray(parsed) ? parsed : [parsed];
}

export function buildProcessStopScript(processIds) {
  const joinedIds = processIds.join(",");
  return [
    "$ErrorActionPreference = 'SilentlyContinue'",
    `$ids = @(${joinedIds})`,
    "foreach ($id in $ids) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }",
  ].join("; ");
}

function stopWindowsProcesses(processIds) {
  if (processIds.length === 0) {
    return;
  }

  execFileSync(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-Command", buildProcessStopScript(processIds)],
    { encoding: "utf8" },
  );
}

export async function cleanupWorkspaceDevProcesses(workspaceRoot, currentPid = process.pid) {
  if (process.platform !== "win32") {
    return [];
  }

  const snapshot = readWindowsProcessSnapshot();
  const processIds = collectWorkspaceDevProcessIds(snapshot, workspaceRoot, currentPid);

  if (processIds.length === 0) {
    return [];
  }

  stopWindowsProcesses(processIds);
  return processIds;
}
