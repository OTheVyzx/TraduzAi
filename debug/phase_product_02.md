# Fase 2 - Setup profissional e prevencao de erro do usuario

Data: 2026-05-01

## Objetivo
Fortalecer o Setup para mostrar o estado do contexto da obra, alertar antes de traducao sem contexto e persistir o risco no `work_context`.

## Implementado
- Card permanente de estado do contexto no Setup:
  - obra;
  - contexto vazio/parcial;
  - glossario;
  - memoria da obra;
  - risco.
- Alertas separados para:
  - nenhuma obra selecionada;
  - obra selecionada com glossario vazio.
- Acoes do alerta:
  - Buscar obra;
  - Buscar contexto online;
  - Revisar glossario;
  - Continuar sem contexto.
- Persistencia explicita de `work_context` para o caso sem obra:
  - `selected`;
  - `internet_context_loaded`;
  - `glossary_loaded`;
  - `user_ignored_warning`;
  - `risk_level`.
- Testes unitarios do calculo de risco e dos tipos de alerta.
- Testes Playwright `@setup` para:
  - sem obra;
  - obra sem glossario;
  - contexto online aplicado reduzindo risco.

## Arquivos alterados nesta fase
- `src/lib/workContextProfile.ts`
- `src/lib/__tests__/workContextProfile.test.ts`
- `src/pages/Setup.tsx`
- `e2e/editor-rebuild.spec.ts`

## Comandos rodados
- `npx vitest run src/lib/__tests__/workContextProfile.test.ts`
- `npx playwright test --grep "@setup"`
- `npm run build`

## Falhas encontradas
- Primeira execucao de `npx playwright test --grep "@setup"` falhou no caso sem obra.
- Causa: `handleStart()` usava fallback para `project.obra` quando o campo da obra era apagado, reaproveitando o valor antigo `Fixture E2E`.

## Correcao aplicada
- O fluxo do Setup passou a usar `obraSearch.trim()` como fonte autoritativa do campo editavel no inicio do projeto e no alerta.

## Resultado
- Fase 2 aprovada.
- `npx vitest run src/lib/__tests__/workContextProfile.test.ts`: 5 passed.
- `npx playwright test --grep "@setup"`: 3 passed.
- `npm run build`: passou.

## Proximo ponto de retomada
Continuar na Fase 3: Glossario como centro do produto.
