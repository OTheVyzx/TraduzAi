import type { StudioEditorBackend } from "../../backend/editorBackend";
import type { StudioPage, StudioScene } from "../../project/studioProject";
import { projectStudioSceneToPage } from "../../store/studioSceneStore";

export async function persistStudioScene({
  backend,
  projectPath,
  pageIndex,
  scene,
}: {
  backend: StudioEditorBackend;
  projectPath: string;
  pageIndex: number;
  scene: StudioScene;
}): Promise<StudioPage> {
  const { result } = await backend.mutateProject({
    project_path: projectPath,
    mutate: (project) => {
      const page = project.paginas[pageIndex];
      if (!page) throw new Error(`Página ${pageIndex + 1} não encontrada`);
      const projectedPage = projectStudioSceneToPage(page, scene);
      project.paginas[pageIndex] = projectedPage;
      return JSON.parse(JSON.stringify(projectedPage)) as StudioPage;
    },
  });
  return result;
}
