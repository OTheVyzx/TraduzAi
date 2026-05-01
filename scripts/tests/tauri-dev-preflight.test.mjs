import test from "node:test";
import assert from "node:assert/strict";

import {
  buildProcessStopScript,
  buildProcessSnapshotScript,
  collectWorkspaceDevProcessIds,
  shouldCleanupDevProcess,
} from "../tauri-dev-preflight.mjs";

const workspaceRoot = "D:\\TraduzAi";

test("detecta wrappers e subprocessos de dev do workspace atual", () => {
  assert.equal(
    shouldCleanupDevProcess(
      {
        processId: 222,
        name: "node.exe",
        commandLine: "node scripts/run-tauri.mjs dev",
      },
      workspaceRoot,
      999,
    ),
    true,
  );

  assert.equal(
    shouldCleanupDevProcess(
      {
        processId: 333,
        name: "node.exe",
        commandLine: "\"C:\\Program Files\\nodejs\\node.exe\" D:\\TraduzAi\\node_modules\\vite\\bin\\vite.js",
      },
      workspaceRoot,
      999,
    ),
    true,
  );

  assert.equal(
    shouldCleanupDevProcess(
      {
        ProcessId: 444,
        Name: "node.exe",
        CommandLine: "node  scripts/run-vite.mjs",
      },
      workspaceRoot,
      999,
    ),
    true,
  );
});

test("ignora o processo atual e processos fora do workspace", () => {
  assert.equal(
    shouldCleanupDevProcess(
      {
        processId: 999,
        name: "node.exe",
        commandLine: "node scripts/run-tauri.mjs dev",
      },
      workspaceRoot,
      999,
    ),
    false,
  );

  assert.equal(
    shouldCleanupDevProcess(
      {
        processId: 444,
        name: "node.exe",
        commandLine: "\"C:\\Program Files\\Adobe\\Adobe Creative Cloud Experience\\libs\\node.exe\" something",
      },
      workspaceRoot,
      999,
    ),
    false,
  );

  assert.equal(
    shouldCleanupDevProcess(
      {
        processId: 555,
        name: "node.exe",
        commandLine: "\"C:\\Program Files\\nodejs\\node.exe\" D:\\OutroProjeto\\node_modules\\vite\\bin\\vite.js",
      },
      workspaceRoot,
      999,
    ),
    false,
  );
});

test("coleta apenas os pids que devem ser limpos", () => {
  const processIds = collectWorkspaceDevProcessIds(
    [
      {
        processId: 111,
        name: "node.exe",
        commandLine: "node scripts/run-tauri.mjs dev",
      },
      {
        processId: 222,
        name: "node.exe",
        commandLine: "\"C:\\Program Files\\nodejs\\node.exe\" D:\\TraduzAi\\node_modules\\@tauri-apps\\cli\\tauri.js dev",
      },
      {
        processId: 333,
        name: "node.exe",
        commandLine: "\"C:\\Program Files\\nodejs\\node.exe\" D:\\OutroProjeto\\node_modules\\vite\\bin\\vite.js",
      },
    ],
    workspaceRoot,
    999,
  );

  assert.deepEqual(processIds, [111, 222]);
});

test("gera scripts PowerShell com separadores válidos", () => {
  const snapshotScript = buildProcessSnapshotScript();
  assert.match(snapshotScript, /\$ErrorActionPreference = 'Stop'\s+Get-CimInstance/);
  assert.doesNotMatch(snapshotScript, /;\s*\|/);
  assert.doesNotMatch(snapshotScript, /\|\s*;/);
  assert.match(buildProcessStopScript([11, 22]), /\$ids = @\(11,22\);/);
});
