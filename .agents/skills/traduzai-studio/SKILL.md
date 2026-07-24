---
name: traduzai-studio
description: Use when Studio actions ignore a lasso or mask, return empty detections, claim success without changing the project, lose text aliases or metadata, or differ between browser and Tauri runtimes.
---

# TraduzAI Studio

## Princípio

Trate o Studio como app separado e confirme o contrato real de cada ação. Não chame `pipeline/main.py` nem o pipeline automático para “resolver” lacunas do Studio.

## Fluxo ativo

Siga nesta ordem:

`studio/src/App.tsx` → `StudioSharedEditor` → `src/pages/Editor.tsx` + `editorStore` → alias Vite `currentEditorBackend` → `createLegacyEditorBackendAdapter` → backend híbrido. No browser ou em paths `memory://`, ele usa `MemoryStudioEditorBackend`; em Tauri com projeto em disco, usa `TauriStudioEditorBackend` e, somente para ações Lite conectadas, comandos `studio_lite_*` → `pipeline/studio_lite/worker.py`.

## Escopo da ação

A UI prioriza lasso → texto selecionado (exceto detect) → máscara somente para inpaint → confirmação da página inteira. Máscara nunca limita detect, OCR ou tradução.

Status atual:

| Ação | Realidade |
|---|---|
| Inpaint regional | Implementado com bbox/máscara, OpenCV Telea e ROI; grava `editor_cache/studio_lite/inpaint/` e persiste `image_layers.inpaint` no `project.json`. |
| Detect/caixas | O adapter descarta bbox/máscara antes de chamar Studio Lite; `detect_page()` retorna vazio mesmo com modelo/ONNX disponíveis. Não alegue detecção funcional. |
| OCR/tradução por lasso | `Editor` → `runMaskedActionFromLasso` → `editorStore` → `runPageActionWithOptionalMask`. O fallback do adapter devolve apenas `changed_assets: ["project_json"]` e “ação não conectada”; não chama worker nem altera `text_layers`. `ocrPage`/`translatePage` não são a primeira divergência desse fluxo. |
| Processar texto/região | `runProcessRegion`/`processBlock` fabricam overlay/mensagem, sem OCR, tradução ou alteração de texto; são placeholders. |

## Contratos e artefatos

O lasso rasteriza sua seleção em `layers/mask/<pagina>.png`. Os comandos Lite são `model_status`, `detect_page`, `build_mask` e `inpaint_region`, isolados em `pipeline/studio_lite`; no cache do projeto, siga somente rotas reais da ação: `editor_cache/studio_lite/requests/`, `masks/` e `inpaint/`. Detect pode criar a pasta `detections/` sem materializar JSON: não trate o artefato como garantido.

O modelo editável é `paginas[] + image_layers + text_layers`. `textos` é alias regenerado; adapters preservam campos extras por spread. O schema Studio aceita extras. Projetos v12 usam `legacy.paginas` quando presente; sem isso, `pages[].regions` é apenas projeção com aviso e pode perder metadados analíticos.

## Testes e REDs obrigatórios

Rode:

```powershell
npm test -- --run src/lib/stores/__tests__/editorBitmapTools.test.ts
npm --prefix studio test -- src/backend/__tests__/editorBackendCompat.test.ts src/project/__tests__/adapters.test.ts src/store/__tests__/projectStore.test.ts
python -m pytest pipeline/tests/test_studio_lite_worker.py -q
```

`editorBitmapTools` cobre lasso/store e execução de página inteira, mas não a confirmação visual. Antes de conectar ações, escreva REDs para: OCR e translate por lasso falharem explicitamente em vez do fallback “sucesso”/no-op; detect respeitar ou rejeitar seleção; UI exigir segunda confirmação sem alvo; inpaint persistir path e respeitar ROI; e Studio nunca invocar comandos do pipeline automático. Não transforme placeholder em sucesso verde.

## Checklist de diagnóstico

1. Confirme checkout, projeto e backend memória/Tauri.
2. Trace `Editor` → store → adapter; confira bbox/mask no request.
3. Confira saída do worker somente se houve comando Lite.
4. Recarregue `project.json` e valide layer/path/texto; mensagem não prova mutação.
5. Antes do commit, valide UTF-8 estrito, `git diff --check` e o auditor dos skills.

## Última verificação

- Data: 2026-07-24
- Commit: `c0801348`
