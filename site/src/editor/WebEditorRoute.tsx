import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { Editor } from "../../../src/pages/Editor";
import { configureEditorBackend } from "../../../src/lib/editorBackend";
import { useAppStore } from "../../../src/lib/stores/appStore";
import { useEditorStore } from "../../../src/lib/stores/editorStore";
import { projectApi } from "../projectApi";
import { httpEditorBackend } from "./httpEditorBackend";
import { normalizeWebProject } from "./webProjectAdapter";

function numericPage(value: string | null) {
  const parsed = Number(value ?? "0");
  if (!Number.isFinite(parsed) || parsed < 0) return 0;
  return Math.floor(parsed);
}

export function WebEditorRoute() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [ready, setReady] = useState(false);
  const pageIndex = numericPage(searchParams.get("page"));

  const projectQuery = useQuery({
    queryKey: ["editor-shared-project", id],
    queryFn: () => projectApi.getProject(id!),
    enabled: Boolean(id),
    retry: false,
  });

  const project = useMemo(() => {
    if (!id || !projectQuery.data?.project) return null;
    return normalizeWebProject(id, projectQuery.data.project);
  }, [id, projectQuery.data?.project]);

  useEffect(() => {
    if (!id || !project) return;
    configureEditorBackend(httpEditorBackend);
    useEditorStore.getState().resetEditor();
    useEditorStore.setState({ currentPageIndex: Math.min(pageIndex, Math.max(0, project.paginas.length - 1)) });
    useAppStore.getState().setProject(project);
    setReady(true);
    return () => {
      setReady(false);
      useAppStore.getState().setProject(null);
      useEditorStore.getState().resetEditor();
      configureEditorBackend(null);
    };
  }, [id, pageIndex, project]);

  if (projectQuery.isLoading || !ready) {
    return (
      <div className="flex h-screen items-center justify-center bg-bg-primary text-sm text-text-muted">
        Carregando editor
      </div>
    );
  }

  if (projectQuery.isError || !id) {
    const message = projectQuery.error instanceof Error ? projectQuery.error.message : "Nao foi possivel abrir o editor";
    return (
      <div className="flex h-screen items-center justify-center bg-bg-primary px-6 text-center">
        <div className="space-y-3">
          <p className="text-sm text-status-error">{message}</p>
          <button className="text-sm text-brand hover:underline" onClick={() => navigate("/projects")}>
            Voltar aos projetos
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="web-editor-fullscreen">
      <Editor
        onBack={() => navigate(`/projects/${id}/preview`)}
        emptyBackLabel="Voltar aos projetos"
      />
    </div>
  );
}
