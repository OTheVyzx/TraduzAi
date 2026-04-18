# TraduzAi - Contexto Enxuto

> Ultima atualizacao: 2026-04-17
> Objetivo: retomar o projeto com o minimo de leitura.
> Historico detalhado: `context.archive.md`

## Snapshot do projeto

- App desktop Tauri v2 para traducao automatica de manga/manhwa/manhua EN -> PT-BR.
- Stack:
  - Frontend: React 19 + TypeScript + Tailwind + Zustand + Framer Motion
  - Backend: Rust + Tauri v2
  - Pipeline IA: Python 3.12
- Fluxo:
  - `src/` -> `invoke()` -> `src-tauri/`
  - Rust spawna `pipeline/main.py`
  - Python responde por JSON lines no stdout

## Regras fixas

- UI sempre em portugues brasileiro.
- Tema escuro e o unico tema.
- Processamento local: nenhuma imagem vai para servidor.
- Traducao: Google Translate publico como primario; Ollama local como fallback.
- 1 credito = 1 pagina.
- Free tier: 2 capitulos/semana, reset na segunda-feira.

## Arquivos mais importantes

- `src/lib/tauri.ts`: bindings Tauri
- `src/lib/stores/appStore.ts`: estado global
- `src-tauri/src/commands/`: comandos Rust expostos ao frontend
- `pipeline/main.py`: orquestrador do pipeline
- `pipeline/translator/translate.py`: traducao e reparos contextuais
- `pipeline/vision_stack/runtime.py`: OCR/inpaint runtime
- `pipeline/layout/balloon_layout.py`: deteccao/enriquecimento de baloes
- `pipeline/typesetter/renderer.py`: layout e renderizacao do texto
- `fonts/font-map.json`: mapa de fontes por tipo

## Estado atual

- Rebrand MangáTL -> TraduzAi ja aplicado em nomes, logs e paths principais.
- Editor pos-processamento existe e permite editar texto/layout com retypeset de pagina.
- Traducao em lote existe e processa varios capitulos em sequencia.
- Suite Python ja ficou verde em 2026-04-12; mudancas posteriores usaram testes focados.

## Ultimas mudancas relevantes

### 2026-04-17 - Baloes duplos conectados: modelo em 4 etapas visuais

- `pipeline/layout/balloon_layout.py`
  - baloes conectados agora expõem explicitamente:
    - `ocr_text_bbox` = azul
    - `connected_text_groups` = verde
    - `connected_lobe_bboxes` = vermelho
    - `connected_position_bboxes` = laranja
  - `balloon_subregions` continua como alias do vermelho
  - `connected_focus_bboxes` continua como alias do laranja
  - novas confiancas:
    - `connected_detection_confidence`
    - `connected_group_confidence`
    - `connected_position_confidence`
- Regra oficial do caso conectado:
  - o vermelho define o espaco do lobo
  - o verde define onde o texto tende a viver
  - o laranja define a caixa final de posicionamento
  - o tamanho do texto continua sendo decidido pelo vermelho, nao pelo laranja
- `pipeline/typesetter/renderer.py`
  - `connected_position_bboxes` passou a ser entrada de primeira classe
  - balanceamento de split usa `connected_text_groups` antes de cair para caixas mais grossas
  - `max_width` e `max_height` dos lobos continuam presos no vermelho
  - o score conectado penaliza desalinhamento dentro do laranja
- `debug_pipeline_test/debug_connected_pipeline.py`
  - agora gera debug em 4 etapas:
    - `step_connected_1_ocr_bbox`
    - `step_connected_2_text_groups`
    - `step_connected_3_lobes`
    - `step_connected_4_position_boxes`
- Validacao:
  - `python -m unittest pipeline.tests.test_layout_analysis pipeline.tests.test_typesetting_layout pipeline.tests.test_vision_stack_runtime`
  - 137 testes OK, 1 skip

### 2026-04-17 - Reasoner local via Ollama para posicionamento humano

- `pipeline/layout/balloon_layout.py`
  - baloes conectados agora podem chamar um `reasoner` local via Ollama depois do azul/verde/vermelho
  - o reasoner refina apenas o laranja (`connected_position_bboxes`)
  - o sizing continua preso no vermelho; o laranja so move/centraliza melhor o bloco
  - validadores novos impedem:
    - bbox sair do lobo vermelho
    - bbox crescer ou encolher demais
    - perda do stagger vertical natural entre os grupos verdes
  - se o modelo falhar, demorar demais ou responder lixo, o pipeline volta para a heuristica sem quebrar a pagina
- `pipeline/main.py`
  - cada pagina agora recebe config opcional de reasoner:
    - `connected_balloon_reasoner`
    - `connected_balloon_reasoner_enabled`
    - `connected_balloon_ollama_host`
    - `connected_balloon_ollama_model`
    - `connected_balloon_ollama_use_image`
    - `connected_balloon_ollama_timeout_sec`
    - `connected_balloon_ollama_temperature`
- defaults atuais:
  - provider: `ollama`
  - modelo preferido: `qwen2.5`
  - modelos com visao como `gemma4` podem entrar como fallback/segunda tentativa se entregarem resposta valida
- `debug_pipeline_test/debug_connected_pipeline.py`
  - passou a registrar:
    - `connected_position_reasoner`
    - `connected_reasoner_model`
    - `connected_reasoner_notes`
- Validacao:
  - `python -m unittest pipeline.tests.test_layout_analysis`
  - `python -m unittest pipeline.tests.test_typesetting_layout pipeline.tests.test_vision_stack_runtime pipeline.tests.test_main_emit`
  - 140 testes OK, 1 skip

### 2026-04-16 - Claude Code subagents (dev-time)

- `.claude/agents/typesetter-expert.md`: subagente para `pipeline/typesetter/` e `pipeline/layout/balloon_layout.py`, carrega constraints FreeType/Windows.
- `.claude/agents/vision-stack-expert.md`: subagente para `pipeline/vision_stack/`, `pipeline/ocr/`, `pipeline/inpainter/`.
- `.claude/skills/run-pipeline-debug/SKILL.md`: skill que executa `run_full_debug.py` e reporta por stage.
- Escopo: dev-time somente. Nao confundir com os agentes do Lab (`lab/critics/*`), que sao runtime do app.

### 2026-04-16 - Lab: design de agentes reais + seletor de capitulos

- Plano em `docs/plans/2026-04-16-lab-agents-e-seletor-capitulos-design.md`.
- Fase 1: seletor livre EN/PT-BR + modo `explicit` no `LabChapterScope`.
- Fase 2: critics rule-based locais (OCR, translation, typeset, inpaint) + `lab/planner.py` com `Proposal` estruturado.
- Fase 3: `lab/coders/ollama_coder.py` (default, local, zero custo).
- Fase 4: `lab/coders/claude_code_coder.py` (opt-in, usa `claude -p` ou `claude-agent-sdk`).
- Dry-run obrigatorio em todos os coders; usuario aprova patch manualmente pela UI.

### 2026-04-16 - Baloes conectados

- `pipeline/vision_stack/runtime.py`
  - `MANGATL_DISABLE_WHITE_BALLOON_WHITENING=1` agora desliga so o cleanup agressivo do text box; limpeza leve continua.
- `pipeline/typesetter/renderer.py`
  - lobos conectados carregam vies vertical por lobo
  - quando o split semantico precisa dividir 1 bloco em 2, o renderer aplica stagger para preservar leitura diagonal
- Resultado esperado:
  - menos quadrados brancos agressivos
  - diagonal dos textos em baloes conectados preservada
- Validacao:
  - `python -m unittest pipeline.tests.test_typesetting_layout pipeline.tests.test_vision_stack_runtime`
  - 100 testes OK, 1 skip

### 2026-04-15 - Regressao em baloes conectados

- Renderer ganhou deduplicacao de blocos para evitar texto duplicado.
- Corte entre lobos passou a usar o espaco em branco real entre grupos OCR, nao o centro geometrico bruto.
- Overlap artificial foi removido em favor de gaps mais fortes.
- Lobos passaram a usar melhor o volume util sem estourar o centro.

### 2026-04-12 - Base estavel

- Benchmark real do capitulo 82 chegou a `score_after 99.3`.
- `translate.py` ganhou reparo pontual com Google quando backend local volta vazio ou igual ao ingles.
- OCR/runtime passaram a filtrar melhor watermark/creditos (`scan`, `toon`, variantes).
- Suite `pipeline/tests` ficou verde nessa rodada: 169 testes OK.

## Pendencias e cuidado atual

- Baloes conectados continuam sendo a area mais sensivel do pipeline.
- A suite completa de `pipeline.tests.test_vision_stack_runtime` ja teve falhas antigas de fixture/expectativa em algumas rodadas; para mudancas locais, priorizar testes focados primeiro.
- Mudancas cross-layer devem respeitar os contratos TS <-> Rust <-> Python, especialmente em `src/lib/tauri.ts`.

## Comandos uteis

```bash
npm run tauri dev
cd pipeline && python main.py config.json
python -m unittest pipeline.tests.test_typesetting_layout pipeline.tests.test_vision_stack_runtime
```

## Backups

- Backup atual preservado: `D:/TraduzAi v0.27/`
- Backup anterior removido: `D:/TraduzAi v0.26/`

## Fluxo `cntbk`

Quando o usuario pedir `cntbk`:

1. Atualizar este `context.md`
2. Criar um novo backup versionado do projeto
3. Excluir o backup versionado anterior

## Regra para manter este arquivo compacto

- Manter aqui apenas:
  - snapshot do projeto
  - estado atual
  - ultimas 3 a 5 mudancas realmente relevantes
  - riscos/pendencias ativas
- Mover historico longo para `context.archive.md`
- Evitar logs extensos, listas de arquivos enormes e narrativa de tentativas intermediarias
