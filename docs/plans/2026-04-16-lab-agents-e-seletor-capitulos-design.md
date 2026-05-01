# Lab: agentes reais + seletor livre de capítulos

> Design — 2026-04-16

## Contexto

Hoje o TraduzAi roda 100% local em produção (Google Translate + Ollama + PaddleOCR + LaMa + FT2Font). O único custo Claude que o usuário paga é quando conversa com o Claude Code pra evoluir o pipeline — e isso está esgotando limites rápido demais.

O Lab já tem ossatura sólida: orquestra `pipeline/main.py` sobre capítulos EN, compara com corpus PT-BR de referência, calcula 6 métricas reais e emite snapshot pra UI. Mas os agentes internos (`ocr_critic`, `translation_critic`, `inpaint_critic`, `typeset_critic`, `improvement_planner`, `eval_judge`) são **stubs** — só emitem status events sem produzir análise real. O `reviewer_result_for()` devolve veredito genérico baseado só no verde/vermelho do benchmark.

Além disso, os diretórios EN/PT-BR do Lab estão **hardcoded** em `default_lab_dirs()` (Rust) apontando pra `exemplos/exemploen` e `exemplos/exemploptbr`. Os modos de escopo (`all`, `first_n`, `range`) não permitem escolher capítulos arbitrários nem trocar a pasta-base.

**Outcome desejado:** o Lab vira o "upgrader" do programa. Usuário seleciona quais capítulos EN + referência PT-BR quer analisar, roda o Lab, e recebe propostas concretas de melhoria. Claude Code só é consumido quando o Lab explicitamente precisar de um patch de código — caso contrário, tudo é local.

---

## Arquitetura proposta

### Camada 1 — Critics reais (100% local, rule-based, zero custo)

Substitui os stubs atuais. Cada critic recebe os artefatos do capítulo (`project.json`, imagens output, imagens referência, benchmark) e devolve `list[Finding]` estruturado.

**Arquivos novos:**
- `lab/critics/__init__.py` — registra critics disponíveis
- `lab/critics/base.py` — `Finding` dataclass, `Critic` protocol
- `lab/critics/ocr_critic.py` — flageia confidence < 0.6, watermarks não filtrados (regex `(?i)scan|toon|lagoon|asura|discord\.gg`), boxes com IOU > 0.3 entre si, texto com sequências `111111`
- `lab/critics/translation_critic.py` — flageia `original == traduzido` (tradução não rolou), artefatos UTF-8 (`VocÃª`, `Ã©`), ratio `len(traduzido)/len(original)` fora de [0.5, 2.0], terminologia inconsistente (mesma palavra EN → traduções diferentes em páginas consecutivas)
- `lab/critics/typeset_critic.py` — flageia occupancy fora de [0.18, 0.72], font < 14px em balões > 50px de altura, texto extrapolando bbox (via `_estimate_effective_text_block`), lobos conectados com desequilíbrio severo de palavras
- `lab/critics/inpaint_critic.py` — compara variância local antes/depois do inpaint por região → flageia texto residual e spill de máscara

**Dataclass `Finding`:**
```python
@dataclass
class Finding:
    critic_id: str              # "ocr_critic"
    chapter_number: int
    page_index: int
    bbox: list[int] | None      # [x1,y1,x2,y2] da região problemática
    issue_type: str             # "low_confidence", "watermark_leaked", "encoding_artifact", ...
    severity: str               # "info" | "warning" | "error"
    evidence: dict              # dados brutos pra debug
    suggested_fix: str          # texto human-readable do que fazer
    suggested_file: str = ""    # arquivo no codebase se aplicável
    suggested_anchor: str = ""  # símbolo/função alvo
```

Cada critic é independente e testável isoladamente.

### Camada 2 — Improvement Planner (local, agregação)

**Arquivo novo:** `lab/planner.py`

Responsabilidades:
1. Recebe `list[Finding]` de todos os critics + `aggregate_benchmark` da rodada.
2. Agrupa findings por `issue_type` × `suggested_file`.
3. Calcula prioridade: `severity_weight × findings_count × metric_gap`.
4. Pra cada cluster de alta prioridade, gera uma `Proposal` estruturada:
   ```python
   @dataclass
   class Proposal:
       proposal_id: str
       title: str                  # "Endurecer filtro de watermark para 'Scanlator'"
       motivation: str             # 3 linhas explicando o problema
       target_file: str            # "pipeline/ocr/postprocess.py"
       target_anchor: str          # "WATERMARK_PATTERNS"
       change_kind: str            # "regex_add" | "threshold_tune" | "logic_fix"
       local_patch_hint: dict      # diff sugerido pra mudanças simples (regex/threshold)
       needs_coder: bool           # True se mudança é complexa (lógica nova)
       expected_metric_gain: dict  # {"term_consistency": +3.2}
       findings_sample: list[Finding]  # até 5 exemplos
   ```
5. Se `change_kind in {"regex_add", "threshold_tune"}` e o arquivo tem um "registry" claro (lista de regexes, dict de thresholds), gera o patch **localmente** (sem LLM) — 80% dos casos.
6. Se `change_kind == "logic_fix"`, marca `needs_coder=True` pra passar pra Camada 3.

### Camada 3 — Coder (invocado sob demanda)

**Dois coders plugáveis, usuário escolhe.**

**Arquivo novo:** `lab/coders/base.py` — protocol `Coder`:
```python
class Coder(Protocol):
    def propose_patch(self, proposal: Proposal, repo_root: Path) -> PatchProposal: ...
```

**`lab/coders/ollama_coder.py`** — default, 100% local:
- Usa Ollama com Qwen2.5-Coder-7B (ou DeepSeek-Coder-V2 se disponível)
- System prompt carrega constraints do projeto (do `CLAUDE.md`): FT2Font, sem ProcessPool, etc.
- Input: proposta + conteúdo atual do arquivo alvo
- Output: patch unified-diff + rationale
- Custo: zero

**`lab/coders/claude_code_coder.py`** — opt-in, usuário paga:
- Invoca Claude Code CLI não-interativo: `claude -p "<prompt>" --output-format stream-json`
- Alternativa: usar `claude-agent-sdk` (Python package) pra interação estruturada
- Input: proposta + paths relevantes (Claude Code lê os arquivos via seus próprios tools)
- Output: patch + rationale
- Custo: API calls, mas só quando o Lab explicitamente chama

**Seleção:** campo `coder_strategy` no `LabRunnerConfig`:
- `"ollama"` (default) — sempre Ollama
- `"claude_code"` — sempre Claude Code
- `"ollama_with_claude_fallback"` — tenta Ollama, se confidence < threshold ou patch inválido, escala pra Claude

**Dry-run obrigatório:** coders nunca aplicam patch automaticamente. Retornam `PatchProposal(patch_unified_diff, files_affected, rationale)` que vai pra UI. Humano revisa e aprova via botão "Aplicar patch".

### Camada 4 — Seletor livre de capítulos

Remove hardcoding, adiciona seleção manual EN + PT-BR.

**Rust (`src-tauri/src/commands/lab.rs`):**
- Novos commands:
  - `pick_lab_source_dir() -> String` — abre folder picker, retorna path
  - `pick_lab_reference_dir() -> String`
  - `pick_lab_source_files() -> Vec<String>` — abre multi-file picker (.cbz)
  - `pick_lab_reference_files() -> Vec<String>`
  - `set_lab_dirs(source_dir, reference_dir) -> LabSnapshot` — atualiza snapshot, refaz `discover_reference_pairs()`
- Novo modo no `LabChapterScope`:
  - `mode: "explicit"`, campo novo `chapter_numbers: Vec<u32>` — exatamente esses capítulos
- `default_lab_dirs()` vira fallback apenas: se não houver config persistida, aponta pra `exemplos/exemploen`, caso contrário lê do storage
- Persistência: adicionar `last_source_dir` e `last_reference_dir` num `lab_preferences.json` em `D:/traduzai_data/lab/`

**Python (`lab/runner.py`):**
- Já aceita `selected_chapters` no config — `apply_chapter_scope` já suporta via filtro na main. Mudança mínima: aceitar também `mode: "explicit"` sem exigir `start_chapter`/`end_chapter`.

**Frontend (`src/pages/Lab.tsx`):**
- Novo bloco "Fontes":
  - Botão "Selecionar pasta EN" → `pick_lab_source_dir` → mostra path
  - Botão "Selecionar pasta PT-BR" → `pick_lab_reference_dir`
  - Alternativa: "Adicionar arquivos avulsos" → multi-picker
  - Após selecionar, chama `set_lab_dirs` e atualiza lista de capítulos pareados
- Substituir o seletor de escopo atual por checkboxes:
  - Lista `available_chapter_pairs` com checkbox por capítulo
  - "Selecionar todos", "Nenhum", "Inverter"
  - Campo de busca (filtra por número)
- Botão "Iniciar Lab" envia `chapter_scope = { mode: "explicit", chapter_numbers: [...] }`

### Camada 5 — Integração com UI existente

A UI já renderiza `agents[]`, `proposals[]`, `reviews[]`, `benchmarks[]`. Mudanças:

- `LabProposal` (Rust) ganha campos novos opcionais: `motivation`, `target_file`, `change_kind`, `expected_metric_gain` (JSON), `patch_proposal` (opcional, com diff)
- UI mostra `proposal.motivation` + lista de findings amostrais
- Novo botão "Gerar patch" por proposta → dispara Camada 3 → atualiza com `patch_proposal`
- Modal de review de patch com:
  - Diff colorizado (biblioteca `diff2html` ou similar)
  - Lista de arquivos afetados
  - Rationale
  - Botões "Aplicar", "Pedir novamente", "Rejeitar"

---

## Critérios de verificação

1. **Seletor livre:** abrir Lab, clicar "Selecionar pasta EN", apontar pra qualquer diretório com CBZs, mesmo pra PT-BR. Lista de capítulos pareados aparece. Selecionar 2 capítulos arbitrários (ex: 5 e 23), rodar Lab, ver rodada concluída com esses 2 capítulos apenas.
2. **Critics reais:** após rodar o Lab, a UI mostra pelo menos 1 finding concreto por critic (page_index, bbox, motivo). Os findings têm `suggested_fix` human-readable, não placeholders.
3. **Planner:** propostas têm `motivation` específica (não mais "Aprimoramento guiado pelo benchmark real"), `target_file` real, e `change_kind` classificado.
4. **Coder Ollama:** botão "Gerar patch" numa proposta chama Ollama, retorna diff válido. Aplicar o patch em branch local não quebra `pytest pipeline/tests/`.
5. **Coder Claude Code:** mesma coisa, porém via `claude -p`. Custo aparece apenas quando botão é clicado — zero custo durante a rodada do Lab em si.
6. **Testes:** `lab/tests/test_critics_*.py` (4 arquivos) + `test_planner.py` + `test_coders.py` verdes. Rust `cargo test` continua verde com os novos testes de scope `explicit`.

---

## Ordem de execução sugerida (fases)

**Fase 1 — Seletor livre (UI + backend)** — 1-2 dias
- Rust: commands novos + modo `explicit`
- Python: aceitar explicit sem range
- Frontend: pickers + checkbox list
- Entrega: user consegue rodar Lab sobre qualquer corpus

**Fase 2 — Critics + Planner locais** — 2-3 dias
- `lab/critics/` completo
- `lab/planner.py`
- Substituir stubs em `lab/runner.py`
- Testes
- Entrega: propostas reais aparecem na UI após cada rodada, zero custo

**Fase 3 — Coder Ollama** — 1-2 dias
- `lab/coders/ollama_coder.py` + base
- Integração UI (botão "Gerar patch" + modal diff)
- Prompt tuning com constraints do projeto
- Entrega: patches locais sugeridos, user aprova/rejeita

**Fase 4 — Coder Claude Code (opcional)** — 1 dia
- `lab/coders/claude_code_coder.py` via CLI `claude -p`
- Seletor de coder na UI
- Entrega: user pode pagar API Claude só quando quiser maior qualidade de patch

---

## Caminhos críticos

- `lab/runner.py` — onde os stubs são substituídos (func `reviewer_result_for`, `build_proposal`)
- `src-tauri/src/commands/lab.rs` — linhas 389-426 (`apply_chapter_scope`) + 460-465 (`default_lab_dirs`)
- `src/pages/Lab.tsx` — onde está a UI de escopo atual
- `CLAUDE.md` — constraints a serem injetadas nos prompts dos coders (FT2Font, sem ProcessPool, etc.)

---

## Fora de escopo (follow-ups)

- **Promoção automática de patches** — por ora, tudo humano-aprovado
- **Re-run automático pós-patch** — user roda o Lab de novo manualmente pra confirmar ganho
- **Learning loop** — o planner poderia memorizar quais tipos de fix produziram ganho real e priorizar similar no futuro
- **Multi-worker** — rodar vários capítulos em paralelo (hoje é serial)
