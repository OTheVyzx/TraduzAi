# TraduzAI

![Build](https://img.shields.io/badge/build-local-blue)
![Tests](https://img.shields.io/badge/tests-pytest%20%7C%20vitest%20%7C%20playwright-green)
![Tauri](https://img.shields.io/badge/Tauri-v2-7C5CFF)
![React](https://img.shields.io/badge/React-19-22D3EE)
![Python](https://img.shields.io/badge/Python-3.12-3776AB)
![Status](https://img.shields.io/badge/status-v0.2.0--beta-orange)

TraduzAI e um app desktop local para traduzir, revisar e exportar manga, manhwa e manhua EN -> PT-BR com IA, contexto de obra, glossario, memoria, editor visual e QA antes do export.

## O que e

O app importa um capitulo, detecta regioes de texto, executa OCR, traduz com contexto, remove texto original, recria o typesetting e entrega um projeto editavel. O usuario revisa glossario, alertas de QA e export antes de finalizar.

O TraduzAI nao fornece obras, nao hospeda conteudo e nao distribui material protegido. Ele processa arquivos locais do usuario.

## Principais recursos

- Contexto online da obra com cache e candidatos revisaveis.
- Glossario central com termos reviewed, candidates, rejected e conflicts.
- Memoria da obra para manter consistencia entre capitulos.
- Presets de projeto para perfis de traducao e typesetting.
- Pipeline local com OCR, traducao, inpaint, typesetting e export.
- Editor visual com camadas, propriedades, mascara, preview e atalhos.
- QA profissional com flags, agrupamento, ignorar com motivo e bloqueio de export limpo.
- Export em modos clean, with warnings, debug e review package.
- Site publico em `site/` com landing, download, docs, legal e roadmap.

## Demonstracao

Assets sinteticos do site:

- `site/public/assets/hero-app-mockup.png`
- `site/public/assets/before-after-demo.png`
- `site/public/assets/context-panel.png`
- `site/public/assets/glossary-panel.png`
- `site/public/assets/qa-report.png`
- `site/public/assets/editor-view.png`

## Como funciona

```text
Importar capitulo
-> selecionar obra e contexto
-> revisar glossario
-> detectar/OCR/traduzir
-> inpaint
-> typesetting
-> QA
-> editor visual
-> export
```

## Stack

- Frontend: React 19, TypeScript, Tailwind CSS, Zustand.
- Desktop: Tauri v2, Rust, Tokio, Serde.
- Pipeline: Python 3.12, OCR, traducao, OpenCV/inpaint, renderer/typesetting.
- Testes: Vitest, Pytest, Cargo tests/check, Playwright.
- Site: Vite, React, Tailwind em `site/`.

## Instalacao

Requisitos de desenvolvimento:

- Node.js 20+
- Rust stable
- Python 3.12
- Windows 10/11 recomendado para o fluxo atual

```bash
npm install
cd pipeline
python -m venv venv
.\\venv\\Scripts\\python.exe -m pip install -r requirements.txt
cd ..
```

## Desenvolvimento

```bash
npm run tauri dev
```

Pipeline isolado:

```bash
cd pipeline
.\\venv\\Scripts\\python.exe main.py config.json
```

Site:

```bash
cd site
npm install
npm run dev
```

## Testes

```bash
npm run build
cd pipeline
.\\venv\\Scripts\\python.exe -m pytest -q
cd ..
cargo check
npx playwright test
```

Checks focados comuns:

```bash
npx vitest run src/lib/__tests__/glossaryCenter.test.ts
npx playwright test --grep "@setup"
cargo test glossary --lib
```

## Formatos principais

- `project.json`: projeto editavel e reimportavel.
- `work_context.json`: contexto normalizado da obra.
- `glossary.json`: termos revisados, candidatos, rejeitados e conflitos.
- `memory`: historico local da obra para consistencia.
- `qa_report`: flags e decisoes de revisao.

## Feature flags

Algumas funcionalidades experimentais ficam desativadas por padrao.

### Lab

O modulo Lab esta oculto por padrao.

Para ativar em ambiente de desenvolvimento:

```env
VITE_ENABLE_LAB=1
```

Sem essa variavel, a rota `/lab/*` redireciona para a tela inicial e o item Lab nao aparece no menu lateral.

## Stack de traducao e OCR

### Traducao

O pipeline usa traducao automatica com fallback local:

- Google Translate via `deep-translator`
- Fallback local via Ollama, usando modelo configuravel
- Glossario/contexto da obra quando disponivel

### OCR e deteccao

O pipeline utiliza um stack visual com deteccao/OCR e fallback legado quando necessario.

Componentes principais:

- `pipeline/vision_stack/` — stack ativo de deteccao/OCR/inpaint
- `pipeline/ocr/` — entrada OCR ativa
- `pipeline/ocr_legacy/` — fallback legado

## Roadmap

- v0.2.0-beta: fluxo local testavel, site, docs, QA/export/editor melhorados.
- Instalador Windows e checksums publicados.
- Export avancado e pacote de revisao.
- Modo lote.
- macOS/Linux.
- Integracao comercial somente depois de validacao legal e produto.

## Aviso legal

O TraduzAI e uma ferramenta local de edicao, OCR, traducao e typesetting. Ele nao hospeda, distribui ou fornece obras protegidas por direitos autorais. O usuario e responsavel pelos arquivos utilizados e por possuir direito de processa-los.

## Privacidade

As imagens e capitulos ficam no computador do usuario. O app so acessa a internet para traducao textual e busca de contexto quando esses recursos estiverem ativados. Nenhuma pagina e enviada para fontes de contexto.
