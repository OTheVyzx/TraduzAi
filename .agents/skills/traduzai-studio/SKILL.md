---
name: traduzai-studio
description: Use when Studio actions ignore a lasso or mask, return empty detections, claim success without changing the project, lose text aliases or metadata, or differ between browser and Tauri runtimes.
---

# TraduzAI Studio

## Princípio

Trate o Studio como app separado e confirme o contrato real de cada ação. Não chame `pipeline/main.py` nem o pipeline automático para “resolver” lacunas do Studio.

## Fluxo ativo

Siga nesta ordem:

`studio/src/App.tsx` → `StudioSharedEditor` → `src/pages/Editor.tsx` + `editorStore` → alias Vite `currentEditorBackend` → `createLegacyEditorBackendAdapter` → `TauriStudioEditorBackend` → comandos Rust `studio_lite_*` → `pipeline/studio_lite/worker.py`.

O `projectStore` importa, salva e recarrega o projeto. Não investigue o editor legado `studio/src/editor/StudioEditor.tsx` antes desse caminho.

## Escopo da ação

A UI prioriza: lasso → texto selecionado (exceto detect) → máscara somente para inpaint → confirmação da página inteira. A máscara nunca limita detect, OCR ou tradução.

Status atual:

| Ação | Realidade |
|---|---|
| Inpaint regional | Implementado com bbox/máscara, OpenCV Telea e ROI; grava `editor_cache/studio_lite/inpaint/` e persiste `image_layers.inpaint` no `project.json`. |
| Detect/caixas | O adapter descarta bbox/máscara; `detect_page()` retorna vazio mesmo com modelo/ONNX disponíveis. Não alegue detecção funcional. |
| OCR/tradução | `ocrPage` e `translatePage` retornam mensagens “ainda não conectado”; são no-op. |
| Processar texto/região | `runProcessRegion`/`processBlock` fabricam overlay/mensagem, sem OCR, tradução ou alteração de texto; são placeholders. |

## Contratos e artefatos

Os únicos comandos Lite são `model_status`, `detect_page`, `build_mask` e `inpaint_region`, isolados em `pipeline/studio_lite` e `editor_cache/studio_lite/{requests,detections,masks,inpaint}`. Inspecione request JSON, stdout/erro do worker, PNG gerado e depois o `project.json`; a primeira divergência define o owner.

O modelo editável é `paginas[] + image_layers + text_layers`. `textos` é alias regenerado; adapters preservam campos extras por spread. O schema Studio aceita extras. Projetos v12 usam `legacy.paginas` quando presente; sem isso, `pages[].regions` é apenas projeção com aviso e pode perder metadados analíticos.

## Testes e REDs obrigatórios

Rode:

```powershell
npm --prefix studio test -- src/backend/__tests__/editorBackendCompat.test.ts src/project/__tests__/adapters.test.ts src/store/__tests__/projectStore.test.ts
python -m pytest pipeline/tests/test_studio_lite_worker.py -q
```

Antes de conectar ações, escreva REDs para: falha explícita em vez de sucesso vazio/no-op; detect respeitar ou rejeitar seleção; inpaint persistir o path e não alterar fora da ROI; round-trip manter `text_layers`/`textos` e extras. Não transforme placeholder em sucesso verde.

## Última verificação

- Data: 2026-07-24
- Commit: `c0801348`
