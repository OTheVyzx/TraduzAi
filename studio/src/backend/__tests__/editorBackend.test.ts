import { describe, expect, it } from "vitest";
import {
  captureStudioProjectWriteToken,
  runExclusiveStudioProjectCheckpoint,
  runExclusiveStudioProjectTransition,
  runStudioProjectWrite,
  runStudioProjectWriteLease,
} from "../editorBackend";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => { resolve = resolvePromise; });
  return { promise, resolve };
}

describe("Studio project write identity", () => {
  it("maps directory and project.json aliases to the same gate", () => {
    const directory = captureStudioProjectWriteToken("N:\\capitulo");
    const projectFile = captureStudioProjectWriteToken("n:/capitulo/./project.json");
    expect(directory.key).toBe(projectFile.key);
  });

  it("maps case-variant UNC aliases to the same Windows gate", () => {
    const directory = captureStudioProjectWriteToken("\\\\SERVIDOR\\Projetos\\Capitulo");
    const projectFile = captureStudioProjectWriteToken("//servidor/projetos/capitulo/./project.json");
    expect(directory.key).toBe(projectFile.key);
  });

  it("keeps leading parent segments distinct in relative project paths", () => {
    expect(captureStudioProjectWriteToken("../capitulo/project.json").key)
      .not.toBe(captureStudioProjectWriteToken("capitulo/project.json").key);
  });

  it("lets an admitted writer finish nested leased work before a checkpoint", async () => {
    const token = captureStudioProjectWriteToken("memory://nested-lease-checkpoint");
    const writerEntered = deferred<void>();
    const allowNestedWrite = deferred<void>();
    const order: string[] = [];
    const writer = runStudioProjectWrite(token, async (lease) => {
      order.push("writer");
      writerEntered.resolve(undefined);
      await allowNestedWrite.promise;
      await runStudioProjectWriteLease(lease, async () => { order.push("nested"); });
    });
    await writerEntered.promise;
    const checkpoint = runExclusiveStudioProjectCheckpoint(token, async () => { order.push("checkpoint"); });
    await Promise.resolve();
    expect(order).toEqual(["writer"]);
    allowNestedWrite.resolve(undefined);
    await Promise.all([writer, checkpoint]);
    expect(order).toEqual(["writer", "nested", "checkpoint"]);
  });

  it("freezes aliases together and invalidates both after a path transition", async () => {
    const sourceToken = captureStudioProjectWriteToken("N:\\projetos\\capitulo");
    const aliasToken = captureStudioProjectWriteToken("n:/projetos/capitulo/./project.json");
    const writerEntered = deferred<void>();
    const allowWriter = deferred<void>();
    const writer = runStudioProjectWrite(sourceToken, async () => {
      writerEntered.resolve(undefined);
      await allowWriter.promise;
    });
    await writerEntered.promise;
    const transition = runExclusiveStudioProjectTransition(
      sourceToken,
      "N:\\projetos\\copia\\project.json",
      async () => undefined,
    );
    await expect(runStudioProjectWrite(aliasToken, async () => undefined))
      .rejects.toThrow("sendo salvo em outro local");
    allowWriter.resolve(undefined);
    await Promise.all([writer, transition]);
    await expect(runStudioProjectWrite(aliasToken, async () => undefined))
      .rejects.toThrow("sessao antiga");
  });

  it("rejects reuse of a lease after its top-level writer ends", async () => {
    const token = captureStudioProjectWriteToken("memory://expired-lease");
    let capturedLease: Parameters<typeof runStudioProjectWriteLease>[0] | null = null;
    await runStudioProjectWrite(token, async (lease) => { capturedLease = lease; });
    await expect(runStudioProjectWriteLease(capturedLease!, async () => undefined))
      .rejects.toThrow("ja foi encerrado");
  });
});
