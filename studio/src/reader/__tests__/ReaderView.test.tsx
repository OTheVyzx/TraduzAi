import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { DownloadActions, DownloadsPanel, HistoryPanel, ReaderView, RepositoryOverlay, RepositoryPanel } from "../ReaderView";
import { ReaderLanguageDialog } from "../ReaderLanguageDialog";
import { createEmptyReaderDocument } from "../readerModel";

describe("ReaderView", () => {
  it("renders reader navigation separately from translation projects", () => {
    const html = renderToStaticMarkup(createElement(ReaderView));
    expect(html).toContain("Descobrir");
    expect(html).toContain("Biblioteca do leitor");
    expect(html).toContain("Atualizações");
    expect(html).toContain("Downloads");
    expect(html).toContain("Histórico");
    expect(html).toContain("Nenhum repositório configurado");
    expect(html).not.toContain("ID AniList");
  });

  it("always exposes a way back from the repository form", () => {
    const html = renderToStaticMarkup(createElement(RepositoryPanel, {
      preview: null,
      url: "",
      busy: false,
      configured: true,
      onUrlChange: () => undefined,
      onPreview: () => undefined,
      onTrust: () => undefined,
      onClose: () => undefined,
    }));

    expect(html).toContain('aria-label="Voltar ao catálogo"');
    expect(html).toContain('title="Voltar"');
    expect(html).toContain("Adicionar repositório");
  });

  it("uses the shared floating contract for repository and language dialogs", () => {
    const repositoryHtml = renderToStaticMarkup(createElement(RepositoryOverlay, {
      preview: null,
      url: "https://repo.invalid/index.min.json",
      busy: false,
      configured: true,
      onUrlChange: () => undefined,
      onPreview: () => undefined,
      onTrust: () => undefined,
      onClose: () => undefined,
    }));
    const languageHtml = renderToStaticMarkup(createElement(ReaderLanguageDialog, {
      open: true,
      languages: [{ code: "pt-BR", label: "PortuguÃªs (Brasil)", count: 1 }],
      allowed: ["pt-BR"],
      onCancel: () => undefined,
      onApply: () => undefined,
    }));

    expect(repositoryHtml).toContain("studio-reader-repository-backdrop studio-floating-layer");
    expect(languageHtml).toContain("reader-language-backdrop studio-floating-layer");
  });

  it("renders queued progress and pause resume retry controls", () => {
    const document = createEmptyReaderDocument();
    const downloads = [
      { jobId: "queued", mangaId: "m1", chapterId: "c1", status: "queued" as const, pageCount: 10, downloadedPages: 3, pages: [] },
      { jobId: "paused", mangaId: "m2", chapterId: "c2", status: "paused" as const, pageCount: 8, downloadedPages: 4, pages: [] },
      { jobId: "failed", mangaId: "m3", chapterId: "c3", status: "failed" as const, pageCount: 6, downloadedPages: 2, pages: [], error: "Falha temporária" },
    ];
    const html = renderToStaticMarkup(createElement(DownloadsPanel, {
      downloads,
      document,
      onRead: () => undefined,
      onPause: () => undefined,
      onResume: () => undefined,
      onRetry: () => undefined,
      onRemove: () => undefined,
    }));

    expect(html).toContain("3 de 10 páginas");
    expect(html).toContain("Pausar");
    expect(html).toContain("Continuar");
    expect(html).toContain("Tentar novamente");
    expect(html).toContain("Falha temporária");
  });

  it("describes downloaded-page preparation honestly until a processing job exists", () => {
    const html = renderToStaticMarkup(createElement(DownloadActions, {
      download: {
        jobId: "done",
        mangaId: "m1",
        chapterId: "c1",
        status: "completed" as const,
        pageCount: 1,
        downloadedPages: 1,
        pages: [],
      },
      busy: false,
      onRead: () => undefined,
      onTranslate: () => undefined,
      onPause: () => undefined,
      onResume: () => undefined,
      onRetry: () => undefined,
      onRemove: () => undefined,
    }));

    expect(html).toContain("Preparar no Studio");
    expect(html).not.toContain("Traduzir capítulo");
  });

  it("does not report an empty queue while persisted downloads are still loading", () => {
    const html = renderToStaticMarkup(createElement(DownloadsPanel, {
      downloads: [],
      loaded: false,
      document: createEmptyReaderDocument(),
      onRead: () => undefined,
      onPause: () => undefined,
      onResume: () => undefined,
      onRetry: () => undefined,
      onRemove: () => undefined,
    }));

    expect(html).toContain("Carregando downloads");
    expect(html).not.toContain("Fila de downloads vazia");
  });

  it("renders the manga cover at the left of download and history rows", () => {
    const document = {
      ...createEmptyReaderDocument(),
      manga: [{
        id: "m1", extensionPackage: "ext.flix", sourceId: "20", sourceName: "MangaFlix", mangaUrl: "/solo", title: "Solo Max-Level Newbie", thumbnailUrl: "https://covers.invalid/solo.webp",
        chapters: [{ id: "c1", url: "/chapter/1", name: "Capítulo 1", read: false, bookmarked: false, lastPageRead: 4, pageCount: 16 }], categoryIds: [], favorite: true, autoUpdate: false, autoDownload: false,
      }],
    };
    const download = { jobId: "done", mangaId: "m1", chapterId: "c1", status: "completed" as const, pageCount: 16, downloadedPages: 16, pages: [] };
    const downloadsHtml = renderToStaticMarkup(createElement(DownloadsPanel, { downloads: [download], document, onRead: () => undefined, onPause: () => undefined, onResume: () => undefined, onRetry: () => undefined, onRemove: () => undefined }));
    const historyHtml = renderToStaticMarkup(createElement(HistoryPanel, { history: [{ mangaId: "m1", chapterId: "c1", lastPage: 4, pageCount: 16, read: false, updatedAt: "2026-04-12T22:58:00.000Z" }], document, downloads: [download], onResume: () => undefined }));

    expect(downloadsHtml).toContain('class="reader-download-cover"');
    expect(downloadsHtml).toContain('src="https://covers.invalid/solo.webp"');
    expect(historyHtml).toContain('class="reader-history-cover"');
    expect(historyHtml).toContain('src="https://covers.invalid/solo.webp"');
    expect(historyHtml).toContain("12/04/2026");
  });
});
