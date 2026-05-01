# Fase 5 - Presets de projeto

Data: 2026-05-01

## Objetivo
Reduzir configuracoes manuais e preparar parametros de pipeline por tipo de obra/projeto.

## Implementado
- Modulo `src/lib/projectPresets.ts` com presets iniciais:
  - Manhwa/Webtoon colorido;
  - Manga preto e branco;
  - Manhua colorido;
  - Baloes pequenos;
  - Scanlation clean;
  - Traducao natural BR;
  - Traducao mais literal;
  - SFX preservar;
  - SFX traduzir parcial.
- Cada preset define:
  - OCR sensitivity;
  - limpeza de OCR;
  - estilo de traducao;
  - escala de fonte;
  - margem do balao;
  - modo de SFX;
  - modo de inpaint;
  - modo de QA.
- UI no Setup para escolher preset e ver descricao/parametros.
- Criacao de preset customizado a partir do preset atual.
- Preset salvo no projeto e enviado no payload de `startPipeline`.
- Pipeline Rust repassa `preset` ao Python.
- `project.json` gerado pelo Python preserva `preset`.

## Arquivos alterados nesta fase
- `src/lib/projectPresets.ts`
- `src/lib/__tests__/projectPresets.test.ts`
- `src/lib/stores/appStore.ts`
- `src/lib/tauri.ts`
- `src/pages/Home.tsx`
- `src/pages/Setup.tsx`
- `src/pages/Processing.tsx`
- `src-tauri/src/commands/pipeline.rs`
- `pipeline/main.py`
- `e2e/editor-rebuild.spec.ts`

## Comandos rodados
- `npx vitest run src/lib/__tests__/projectPresets.test.ts`
- `npx playwright test --grep "@presets"`
- `cargo check`
- `npm run build`

## Falhas encontradas
- Nenhuma falha nos gates da Fase 5.
- `cargo check` manteve avisos de `dead_code` no modulo `internet_context`, ja existentes desde fases anteriores.

## Resultado
- Fase 5 aprovada.
- Vitest presets: 3 passed.
- Playwright `@presets`: 1 passed.
- `cargo check`: passou.
- `npm run build`: passou.

## Proximo ponto de retomada
Continuar na Fase 6: Memoria da obra.
