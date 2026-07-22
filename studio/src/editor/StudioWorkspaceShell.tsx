import { useEffect, useState } from "react";
import { Languages, Paintbrush } from "lucide-react";
import { useEditorStore } from "../../../src/editor-shared";
import type { StudioProject } from "../project/studioProject";
import { StudioSharedEditor } from "./StudioSharedEditor";
import {
  readStudioWorkspace,
  requestWorkspaceClose,
  writeStudioWorkspace,
  type StudioWorkspace,
} from "./studioWorkspace";

export function StudioWorkspaceSwitcher({
  workspace,
  onChange,
}: {
  workspace: StudioWorkspace;
  onChange: (workspace: StudioWorkspace) => void;
}) {
  return (
    <div className="studio-workspace-switcher" aria-label="Área de trabalho">
      <button type="button" aria-pressed={workspace === "translation"} onClick={() => onChange("translation")}>
        <Languages size={12} /> Tradução
      </button>
      <button type="button" aria-pressed={workspace === "editing"} onClick={() => onChange("editing")}>
        <Paintbrush size={12} /> Edição
      </button>
    </div>
  );
}

export function StudioWorkspaceShell({
  project,
  projectPath,
  onBack,
  storage = typeof window === "undefined" ? null : window.localStorage,
  confirmDiscard = () => window.confirm("Há alterações não salvas. Descartar e voltar para a biblioteca?"),
}: {
  project: StudioProject;
  projectPath: string;
  onBack: () => void;
  storage?: Storage | null;
  confirmDiscard?: () => boolean;
}) {
  const [workspace, setWorkspace] = useState<StudioWorkspace>(() => (
    readStudioWorkspace(project, projectPath, storage)
  ));

  useEffect(() => {
    setWorkspace(readStudioWorkspace(project, projectPath, storage));
  }, [project, projectPath, storage]);

  const changeWorkspace = (nextWorkspace: StudioWorkspace) => {
    setWorkspace(nextWorkspace);
    writeStudioWorkspace(projectPath, nextWorkspace, storage);
  };

  const returnToLibrary = () => {
    const editorStore = useEditorStore.getState();
    if (!requestWorkspaceClose(editorStore.dirty, confirmDiscard)) return;
    editorStore.resetEditor();
    onBack();
  };

  return (
    <StudioSharedEditor
      project={project}
      projectPath={projectPath}
      workspace={workspace}
      onBack={returnToLibrary}
      workspaceSwitcher={<StudioWorkspaceSwitcher workspace={workspace} onChange={changeWorkspace} />}
    />
  );
}
