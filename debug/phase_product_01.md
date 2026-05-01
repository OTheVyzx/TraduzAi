# Phase Product 01

Data: 2026-05-01

## Fase
Contexto online da obra.

## Resultado
- Criado pacote `pipeline/context/internet_context` com modelos, normalizador, cache, merge e resolvedor offline-safe.
- Cache por titulo implementado e validado.
- Fontes sao plugaveis e testes usam fontes falsas, sem internet.
- Glossario revisado pelo usuario nao e sobrescrito por candidato online.
- Criado utilitario frontend `src/lib/internetContext.ts`.
- Setup ganhou painel "Contexto online", fontes habilitadas, Generic Web desligado, resultados por fonte, candidatos e acao "Aplicar alta confianca".
- Fallback E2E de `searchWork`/`enrichWorkContext` evita rede no Playwright.
- Backend Rust ganhou contrato testavel de configuracao/cache em `src-tauri/src/internet_context.rs`.

## Arquivos alterados
- `pipeline/context/__init__.py`
- `pipeline/context/internet_context/*`
- `pipeline/tests/test_internet_context_resolver.py`
- `src/lib/internetContext.ts`
- `src/lib/__tests__/internetContext.test.ts`
- `src/lib/tauri.ts`
- `src/lib/stores/appStore.ts`
- `src/pages/Setup.tsx`
- `e2e/editor-rebuild.spec.ts`
- `src-tauri/src/internet_context.rs`
- `src-tauri/src/lib.rs`

## Testes e comandos
- `.\\venv\\Scripts\\python.exe -m pytest tests/test_internet_context_resolver.py -q` passou com 3 testes.
- `npx vitest run src/lib/__tests__/internetContext.test.ts` passou com 3 testes.
- `cargo test internet_context --lib` passou com 2 testes.
- `npx playwright test --grep "@internet-context"` passou com 1 teste.
- `npm run build` passou.

## Falhas e correcoes
- RED esperado: pacote `context.internet_context` nao existia.
- Playwright falhou por seletores ambiguos; asserts foram escopados ao painel correto.
- Playwright revelou que o modal antigo de glossario sem contexto bloqueava a aplicacao dos candidatos online; corrigido para nao abrir o modal apos contexto online carregado.

## Proxima fase
Fase 2: Setup profissional e prevencao de erro do usuario.
