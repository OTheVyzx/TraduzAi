# Fase 7 - Onboarding e tutorial

Data: 2026-05-01

## Objetivo
Evitar que o usuario novo se perca no fluxo principal.

## Implementado
- Fluxo de onboarding:
  - Importe um capitulo;
  - Selecione a obra;
  - Busque contexto online;
  - Revise o glossario;
  - Traduza;
  - Corrija alertas;
  - Exporte.
- Modal de primeira abertura na Home.
- Botao Ajuda para abrir o tutorial novamente.
- Botao para pular tutorial com persistencia em `localStorage`.
- Checklist no Setup:
  - capitulo importado;
  - obra selecionada;
  - contexto online carregado;
  - glossario revisado;
  - traducao iniciada;
  - revisao final;
  - export.

## Arquivos alterados nesta fase
- `src/lib/onboarding.ts`
- `src/lib/__tests__/onboarding.test.ts`
- `src/pages/Home.tsx`
- `src/pages/Setup.tsx`
- `e2e/editor-rebuild.spec.ts`

## Comandos rodados
- `npx vitest run src/lib/__tests__/onboarding.test.ts`
- `npx playwright test --grep "@onboarding"`
- `npm run build`

## Falhas encontradas
- Nenhuma falha nos gates da Fase 7.

## Resultado
- Fase 7 aprovada.
- Vitest onboarding: 2 passed.
- Playwright `@onboarding`: 1 passed.
- `npm run build`: passou.

## Proximo ponto de retomada
Continuar na Fase 8: Export profissional.
