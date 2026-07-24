# TraduzAI Skills Modernization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Versionar, modernizar e sincronizar oito skills do TraduzAI, com UTF-8 válido e referências verificadas contra o checkout atual.

**Architecture:** `N:\TraduzAI\.agents\skills` será a fonte canônica e `C:\Users\PICHAU\.agents\skills` continuará sendo o diretório instalado. Um auditor PowerShell validará estrutura, encoding e referências; um sincronizador copiará somente as oito skills conhecidas e oferecerá modo `-Check` sem escrita.

**Tech Stack:** Markdown, PowerShell 7/Windows PowerShell, Git, pytest/Vitest apenas para coleta focada dos testes citados.

---

## Regras de execução

- Use `@writing-skills` antes de redigir ou revisar os `SKILL.md`.
- Preserve todas as alterações preexistentes do checkout, especialmente em `DEBUGM/`.
- Não altere frontend, Rust ou Python; use esses arquivos somente como fonte de verdade.
- Faça commits somente dos arquivos listados em cada tarefa.
- Grave Markdown e scripts como UTF-8.
- Não remova nenhuma skill de `C:\Users\PICHAU\.agents\skills`.

### Task 1: Criar o auditor estrutural das skills

**Files:**
- Create: `scripts/Test-TraduzAISkills.ps1`

**Step 1: Criar o auditor com a lista fechada de skills**

O script deve aceitar `-SkillsRoot`, com default para `<repo>\.agents\skills`, e validar estes nomes:

```powershell
$ExpectedSkills = @(
    'mangatl-dev',
    'traduzai-detect',
    'traduzai-ocr',
    'traduzai-inpaint',
    'traduzai-typesetting',
    'traduzai-pipeline',
    'traduzai-translation',
    'traduzai-studio'
)
```

Para cada arquivo, usar `System.Text.UTF8Encoding($false, $true)` e falhar quando:

- `SKILL.md` não existir;
- UTF-8 for inválido;
- o frontmatter não tiver `name` igual ao diretório;
- `description` estiver ausente;
- faltar a seção `## Última verificação`;
- o texto contiver `Ã`, `Â`, `â€` ou `�`.

Ao final, imprimir uma linha por skill e sair com código `1` se houver qualquer erro.

**Step 2: Executar e confirmar a falha inicial**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Test-TraduzAISkills.ps1
```

Expected: FAIL informando que as oito skills canônicas ainda não existem.

**Step 3: Validar sintaxe do script**

Run:

```powershell
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
  (Resolve-Path 'scripts/Test-TraduzAISkills.ps1'),
  [ref]$null,
  [ref]$errors
) | Out-Null
if ($errors.Count) { $errors | Format-List; exit 1 }
```

Expected: exit code `0`.

**Step 4: Commit**

```powershell
git add -- scripts/Test-TraduzAISkills.ps1
git commit -m "test: audit TraduzAI skill documents"
```

### Task 2: Modernizar a skill roteadora `mangatl-dev`

**Files:**
- Create: `.agents/skills/mangatl-dev/SKILL.md`
- Read: `src/lib/tauri.ts`
- Read: `src/lib/stores/appStore.ts`
- Read: `src-tauri/src/commands/`
- Read: `pipeline/main.py`
- Read: `studio/src/App.tsx`
- Read: `studio/src/editor/StudioSharedEditor.tsx`

**Step 1: Confirmar os entrypoints atuais**

Run:

```powershell
$paths = @(
  'src/lib/tauri.ts',
  'src/lib/stores/appStore.ts',
  'src-tauri/src/commands',
  'pipeline/main.py',
  'studio/src/App.tsx',
  'studio/src/editor/StudioSharedEditor.tsx'
)
$paths | ForEach-Object { if (-not (Test-Path $_)) { throw "Ausente: $_" } }
```

Expected: PASS.

**Step 2: Escrever a skill mestre**

O documento deve conter:

- frontmatter `name: mangatl-dev`;
- mapa React → Tauri/Rust → sidecar Python;
- separação explícita entre pipeline automático e Studio;
- roteamento para as sete skills especialistas;
- contratos IPC, JSON lines e `project.json`;
- regra de começar pelo entrypoint/caller real;
- regra de preservar checkouts sujos;
- validação cruzada por camada;
- `## Última verificação` com `2026-07-24` e o commit-base observado.

**Step 3: Executar o auditor**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Test-TraduzAISkills.ps1
```

Expected: ainda FAIL pelas sete skills restantes, mas `mangatl-dev` deve aparecer como válida.

**Step 4: Commit**

```powershell
git add -- .agents/skills/mangatl-dev/SKILL.md
git commit -m "docs: modernize TraduzAI master skill"
```

### Task 3: Modernizar as quatro skills de estágio existentes

**Files:**
- Create: `.agents/skills/traduzai-detect/SKILL.md`
- Create: `.agents/skills/traduzai-ocr/SKILL.md`
- Create: `.agents/skills/traduzai-inpaint/SKILL.md`
- Create: `.agents/skills/traduzai-typesetting/SKILL.md`
- Read: `pipeline/vision_stack/detector.py`
- Read: `pipeline/vision_stack/runtime.py`
- Read: `pipeline/vision_stack/ocr.py`
- Read: `pipeline/vision_stack/inpainter.py`
- Read: `pipeline/strip/process_bands.py`
- Read: `pipeline/typesetter/renderer.py`
- Read: `pipeline/layout/balloon_layout.py`
- Read: `pipeline/typesetter/style_policy.py`

**Step 1: Confirmar arquivos e testes citáveis**

Verificar com `Test-Path` todos os arquivos acima e os testes já citados nas skills instaladas. Remover do novo documento qualquer referência que não exista.

Expected: todos os caminhos mantidos na documentação existem.

**Step 2: Escrever `traduzai-detect`**

Incluir `_vision_blocks`, relação com bandas, falsos positivos/negativos, consumers imediatos, artefatos de detecção e testes focados.

**Step 3: Escrever `traduzai-ocr`**

Incluir OCR primário/fallback, normalização, reviewers, aliases de idioma, `confidence`, `tipo`, `skip_processing` e o texto efetivamente encaminhado à tradução.

**Step 4: Escrever `traduzai-inpaint`**

Incluir máscaras, proteção de arte, balões brancos/escuros/translúcidos/texturizados, reconstrução local, fallbacks e os artefatos `debug/e2e/06_mask_segmentation` e `debug_inpaint`.

**Step 5: Escrever `traduzai-typesetting`**

Incluir FT2Font, execução serial, style-copy com confiança, render plans, balões conectados, overflow e validação visual.

**Step 6: Executar o auditor**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Test-TraduzAISkills.ps1
```

Expected: as cinco skills existentes passam; o comando ainda falha somente pelas três novas.

**Step 7: Confirmar coleta dos testes Python citados**

Run:

```powershell
pipeline/venv/Scripts/python.exe -m pytest --collect-only -q `
  pipeline/tests/test_vision_stack_detector.py `
  pipeline/tests/test_vision_stack_runtime.py `
  pipeline/tests/test_vision_stack_ocr.py `
  pipeline/tests/test_primary_ocr_routing.py `
  pipeline/tests/test_ocr_reviewer.py `
  pipeline/tests/test_contextual_reviewer.py `
  pipeline/tests/test_vision_stack_inpainter.py `
  pipeline/tests/test_inpainting_profile.py `
  pipeline/tests/test_mask_builder.py `
  pipeline/tests/test_typesetting_layout.py `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_layout_analysis.py
```

Expected: collection succeeds without import or path errors.

**Step 8: Commit**

```powershell
git add -- .agents/skills/traduzai-detect/SKILL.md `
  .agents/skills/traduzai-ocr/SKILL.md `
  .agents/skills/traduzai-inpaint/SKILL.md `
  .agents/skills/traduzai-typesetting/SKILL.md
git commit -m "docs: modernize TraduzAI pipeline stage skills"
```

### Task 4: Criar `traduzai-pipeline`

**Files:**
- Create: `.agents/skills/traduzai-pipeline/SKILL.md`
- Read: `pipeline/main.py`
- Read: `pipeline/strip/run.py`
- Read: `pipeline/strip/process_bands.py`
- Read: `pipeline/strip/scheduler.py`
- Read: `pipeline/strip/reassemble.py`
- Read: `pipeline/qa/export_gate.py`

**Step 1: Traçar o fluxo automático real**

Registrar a cadeia `pipeline/main.py` → `strip.run.run_chapter` → `process_bands.process_band` → reassembly → QA/export gate, confirmando os nomes diretamente no código.

**Step 2: Escrever a skill**

Incluir:

- quando usar esta skill junto a uma especialista de estágio;
- ordem do pipeline por bandas;
- contratos de progresso, conclusão, `completion_status` e `export_gate`;
- telemetria e artefatos `debug/e2e`;
- diferença entre smoke/helper pass e resultado visual completo;
- testes focados existentes encontrados por `rg --files pipeline/tests`.

**Step 3: Executar o auditor**

Expected: `traduzai-pipeline` passa; faltam somente tradução e Studio.

**Step 4: Commit**

```powershell
git add -- .agents/skills/traduzai-pipeline/SKILL.md
git commit -m "docs: add TraduzAI pipeline skill"
```

### Task 5: Criar `traduzai-translation`

**Files:**
- Create: `.agents/skills/traduzai-translation/SKILL.md`
- Read: `pipeline/translator/translate.py`
- Read: `pipeline/translator/context.py`
- Read: `pipeline/glossary/`
- Read: `pipeline/main.py`

**Step 1: Mapear os backends e fallbacks atuais**

Usar `rg` para localizar Google Translate, Ollama, normalização, contexto e o campo de texto enviado ao tradutor.

**Step 2: Escrever a skill**

Incluir:

- distinção entre OCR incorreto e tradução incorreta;
- texto de origem, contexto, glossário e resultado traduzido;
- backend primário, fallback local e diagnóstico de disponibilidade;
- contratos de idioma e aliases;
- testes focados existentes, sem inventar nomes.

**Step 3: Executar o auditor**

Expected: sete skills passam; falta apenas Studio.

**Step 4: Commit**

```powershell
git add -- .agents/skills/traduzai-translation/SKILL.md
git commit -m "docs: add TraduzAI translation skill"
```

### Task 6: Criar `traduzai-studio`

**Files:**
- Create: `.agents/skills/traduzai-studio/SKILL.md`
- Read: `studio/src/App.tsx`
- Read: `studio/src/editor/StudioSharedEditor.tsx`
- Read: `src/pages/Editor.tsx`
- Read: `studio/src/backend/editorBackend.ts`
- Read: `studio/src/backend/editorBackendCompat.ts`
- Read: `studio/src/store/projectStore.ts`
- Read: `pipeline/studio_lite/worker.py`

**Step 1: Confirmar o entrypoint visual ativo**

Run:

```powershell
rg -n "StudioSharedEditor|from .*src/pages/Editor|<Editor" `
  studio/src/App.tsx `
  studio/src/editor/StudioSharedEditor.tsx
```

Expected: `App.tsx` carrega `StudioSharedEditor`, que reutiliza `src/pages/Editor.tsx`.

**Step 2: Escrever a skill**

Incluir:

- Studio como app separado do pipeline automático;
- entrypoint visual autoritativo;
- `StudioEditorBackend`, adapter legado e stores;
- `project.json`, `image_layers`, `text_layers` e persistência;
- seleção/lasso/máscara e métodos existentes do editor store;
- `studio_lite` como worker dedicado;
- testes do Studio via Vitest.

**Step 3: Executar o auditor completo**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Test-TraduzAISkills.ps1
```

Expected: PASS para as oito skills.

**Step 4: Executar coleta/testes estruturais do Studio citados**

Usar o script definido em `studio/package.json`. Se houver `test`, executar os arquivos citados pela skill; caso contrário, documentar o comando real encontrado no checkout.

Expected: os testes citados são descobertos e executáveis.

**Step 5: Commit**

```powershell
git add -- .agents/skills/traduzai-studio/SKILL.md
git commit -m "docs: add TraduzAI Studio skill"
```

### Task 7: Criar o sincronizador seguro

**Files:**
- Create: `scripts/Sync-TraduzAISkills.ps1`
- Create: `scripts/tests/Sync-TraduzAISkills.Tests.ps1`

**Step 1: Escrever o teste de cópia em diretórios temporários**

O teste deve:

1. criar origem e destino dentro de um diretório temporário explícito;
2. criar oito fixtures mínimas válidas;
3. executar o sincronizador com `-SourceRoot` e `-DestinationRoot`;
4. verificar igualdade byte a byte;
5. criar uma nona skill no destino e confirmar que ela não foi removida;
6. alterar uma cópia e confirmar que `-Check` retorna erro;
7. remover o temporário em `finally`, após validar que o caminho resolvido permanece dentro da raiz temporária.

**Step 2: Executar o teste e confirmar a falha**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/tests/Sync-TraduzAISkills.Tests.ps1
```

Expected: FAIL porque `scripts/Sync-TraduzAISkills.ps1` ainda não existe.

**Step 3: Implementar o sincronizador mínimo**

Parâmetros:

```powershell
param(
    [string]$SourceRoot,
    [string]$DestinationRoot = 'C:\Users\PICHAU\.agents\skills',
    [switch]$Check
)
```

O script deve reutilizar a lista fechada de oito nomes, validar todos os arquivos de origem antes da primeira escrita, copiar somente `SKILL.md` e comparar SHA-256. Em `-Check`, não deve criar diretórios nem escrever arquivos.

**Step 4: Executar o teste e confirmar sucesso**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/tests/Sync-TraduzAISkills.Tests.ps1
```

Expected: PASS cobrindo cópia, preservação de skill externa e detecção de divergência.

**Step 5: Commit**

```powershell
git add -- scripts/Sync-TraduzAISkills.ps1 scripts/tests/Sync-TraduzAISkills.Tests.ps1
git commit -m "feat: synchronize TraduzAI skills safely"
```

### Task 8: Sincronizar as skills instaladas e validar o conjunto

**Files:**
- Modify externally: `C:\Users\PICHAU\.agents\skills\mangatl-dev\SKILL.md`
- Modify externally: `C:\Users\PICHAU\.agents\skills\traduzai-detect\SKILL.md`
- Modify externally: `C:\Users\PICHAU\.agents\skills\traduzai-ocr\SKILL.md`
- Modify externally: `C:\Users\PICHAU\.agents\skills\traduzai-inpaint\SKILL.md`
- Modify externally: `C:\Users\PICHAU\.agents\skills\traduzai-typesetting\SKILL.md`
- Create externally: `C:\Users\PICHAU\.agents\skills\traduzai-pipeline\SKILL.md`
- Create externally: `C:\Users\PICHAU\.agents\skills\traduzai-translation\SKILL.md`
- Create externally: `C:\Users\PICHAU\.agents\skills\traduzai-studio\SKILL.md`

**Step 1: Confirmar que o modo check detecta o estado antigo**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Sync-TraduzAISkills.ps1 -Check
```

Expected: FAIL indicando as cinco divergentes e as três ausentes.

**Step 2: Sincronizar**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Sync-TraduzAISkills.ps1
```

Expected: oito skills copiadas ou confirmadas, sem remoções.

**Step 3: Confirmar igualdade**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Sync-TraduzAISkills.ps1 -Check
```

Expected: PASS para as oito skills.

**Step 4: Executar auditoria nas duas raízes**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Test-TraduzAISkills.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Test-TraduzAISkills.ps1 `
  -SkillsRoot 'C:\Users\PICHAU\.agents\skills'
```

Expected: ambos passam.

**Step 5: Verificação final do Git**

Run:

```powershell
git diff --check
git status --short -- .agents/skills scripts docs/plans
```

Expected: nenhum erro de whitespace; somente alterações deliberadas, se alguma ainda não tiver sido commitada. As alterações preexistentes fora desses caminhos permanecem intactas.

### Task 9: Revisão final de utilidade e fronteiras

**Files:**
- Review: `.agents/skills/*/SKILL.md`
- Review: `scripts/Test-TraduzAISkills.ps1`
- Review: `scripts/Sync-TraduzAISkills.ps1`
- Review: `docs/plans/2026-07-24-traduzai-skills-modernization-design.md`

**Step 1: Revisar roteamento sem sobreposição**

Confirmar que cada sintoma tem uma skill principal e, quando necessário, uma combinação explícita com `mangatl-dev` ou `traduzai-pipeline`.

**Step 2: Revisar alegações contra o código**

Para cada entrypoint, campo de schema, artefato e teste citado, usar `rg` ou `Test-Path` e corrigir qualquer afirmação não confirmada.

**Step 3: Rodar a bateria documental final**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Test-TraduzAISkills.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Sync-TraduzAISkills.ps1 -Check
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/tests/Sync-TraduzAISkills.Tests.ps1
```

Expected: todos os comandos passam.

**Step 4: Commit de correções finais, se necessário**

```powershell
git add -- .agents/skills scripts/Test-TraduzAISkills.ps1 `
  scripts/Sync-TraduzAISkills.ps1 scripts/tests/Sync-TraduzAISkills.Tests.ps1
git commit -m "docs: finalize TraduzAI skill set"
```

Não criar commit vazio se a revisão não exigir correções.
