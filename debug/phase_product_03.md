# Fase 3 - Glossario como centro do produto

Data: 2026-05-01

## Objetivo
Transformar o glossario em camada central de consistencia, com candidatos online revisaveis e termos revisados alimentando o pipeline.

## Implementado
- Modulo `src/lib/glossaryCenter.ts` com regras de dominio:
  - conversao de candidato online para entrada de glossario;
  - filtro de candidatos rejeitados sem `force_refresh`;
  - export apenas de termos `reviewed` para uso no pipeline;
  - deteccao de conflitos;
  - regra de warning para candidato sem revisao;
  - flags criticas para `forbidden`.
- UI central no Setup com abas:
  - Revisados;
  - Candidatos online;
  - Detectados no capitulo;
  - Rejeitados;
  - Conflitos.
- Acoes por candidato:
  - Confirmar;
  - Editar;
  - Rejeitar;
  - Aplicar em todas as ocorrencias;
  - Adicionar forbidden;
  - Transformar em nome protegido.
- Confirmar candidato cria termo `reviewed`, atualiza `contexto.glossario` e atualiza contador/risco em `work_context`.
- Rejeitar candidato move o termo para a aba Rejeitados e impede que continue aparecendo como sugestao pendente na sessao.
- Entrada `GlossaryEntry` passou a aceitar `sources`, preservando a origem de candidatos.

## Arquivos alterados nesta fase
- `src/lib/glossaryCenter.ts`
- `src/lib/__tests__/glossaryCenter.test.ts`
- `src/lib/tauri.ts`
- `src/pages/Setup.tsx`
- `src-tauri/src/glossary.rs`
- `e2e/editor-rebuild.spec.ts`

## Comandos rodados
- `npx vitest run src/lib/__tests__/glossaryCenter.test.ts src/lib/__tests__/internetContext.test.ts src/lib/__tests__/workContextProfile.test.ts`
- `cargo test glossary --lib`
- `npx playwright test --grep "@glossary-center"`
- `npm run build`

## Falhas encontradas
- Nenhuma falha nos gates da Fase 3.

## Resultado
- Fase 3 aprovada.
- Vitest focado: 12 passed.
- Rust focado: 6 passed.
- Playwright `@glossary-center`: 1 passed.
- `npm run build`: passou.

## Proximo ponto de retomada
Continuar na Fase 4: Modo Revisao Profissional.
