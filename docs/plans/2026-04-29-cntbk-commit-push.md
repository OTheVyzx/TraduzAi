# CNTBK + Commit + Push Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** atualizar `context.md`, gerar um novo backup versionado, remover o backup versionado anterior e publicar no Git apenas as mudanças intencionais do lote atual.

**Architecture:** o fluxo vai acontecer em três blocos: `cntbk`, saneamento de escopo Git e publicação. Primeiro atualizamos o contexto e rodamos o backup do projeto no padrão já existente em `scripts/backup_project.py`. Depois filtramos o worktree para separar código útil de artefatos locais pesados. Por fim, validamos, criamos um commit único e fazemos `push` no branch atual.

**Tech Stack:** PowerShell, Git, Python 3.12, `scripts/backup_project.py`

---

### Task 1: Atualizar `context.md`

**Files:**
- Modify: `D:\TraduzAi\context.md`

**Step 1: Ler o contexto atual e o status recente do projeto**

Run:
```powershell
Get-Content D:\TraduzAi\context.md -TotalCount 260
git status --short
```

Expected: contexto atual carregado e lista completa das mudanças locais.

**Step 2: Atualizar o cabeçalho e o snapshot com o estado mais recente**

Registrar pelo menos:
- lote de correções em `traduzido24`
- filtros para `sfx`/marca/watermark
- correções de `connected balloon`
- estado final da página `051`
- branch atual `fix/connected-balloon-rendering`

**Step 3: Revisar o diff do contexto**

Run:
```powershell
git diff -- D:\TraduzAi\context.md
```

Expected: somente atualização de contexto, sem lixo acidental.

---

### Task 2: Executar `cntbk`

**Files:**
- Use: `D:\TraduzAi\scripts\backup_project.py`
- Output: `D:\TraduzAi\backups\traduzai_backup_YYYYMMDD_HHMMSS.zip`

**Step 1: Confirmar o backup versionado atual**

Run:
```powershell
Get-ChildItem D:\TraduzAi\backups | Sort-Object LastWriteTime -Descending
```

Expected: o backup anterior aparece antes da execução.

**Step 2: Rodar o backup no fluxo oficial do projeto**

Run:
```powershell
D:\TraduzAi\pipeline\venv\Scripts\python.exe D:\TraduzAi\scripts\backup_project.py
```

Expected:
- novo `traduzai_backup_*.zip` criado
- backup versionado anterior removido
- `context.md` tocado/atualizado no processo

**Step 3: Verificar o resultado**

Run:
```powershell
Get-ChildItem D:\TraduzAi\backups | Sort-Object LastWriteTime -Descending
```

Expected: apenas um backup versionado atual em `backups/`.

---

### Task 3: Fechar o escopo do commit

**Files:**
- Review: todo o worktree atual

**Step 1: Separar código útil de artefatos locais**

Run:
```powershell
git status --short
```

Classificar em dois grupos:
- incluir: código, testes, docs/plans relevantes, `context.md`, backup versionado novo se a equipe quiser versionar remoção/adição em `backups/`
- excluir: `NOV/`, `DEBUGR/`, `logs/`, `.bench_*`, `tmp_*`, `outapp/`, outputs de teste e outros artefatos locais pesados

**Step 2: Montar a lista de staging explícita**

Preferir `git add` por caminho, nunca `git add .`.

Exemplo de blocos prováveis para este lote:
```powershell
git add context.md
git add docs/plans/2026-04-29-cntbk-commit-push.md
git add pipeline/main.py
git add pipeline/ocr/postprocess.py
git add pipeline/vision_stack/runtime.py
git add pipeline/layout/balloon_layout.py
git add pipeline/typesetter/renderer.py
git add pipeline/tests/test_main_emit.py
git add pipeline/tests/test_vision_stack_runtime.py
git add pipeline/tests/test_layout_analysis.py
git add pipeline/tests/test_typesetting_renderer.py
```

**Step 3: Conferir staged vs unstaged**

Run:
```powershell
git status --short
git diff --cached --stat
```

Expected: só o lote intencional pronto para commit; artefatos locais permanecem unstaged.

---

### Task 4: Validar antes do commit

**Files:**
- Test: `D:\TraduzAi\pipeline\tests\...`

**Step 1: Rodar a suíte focada que cobre o lote**

Run:
```powershell
D:\TraduzAi\pipeline\venv\Scripts\python.exe -m pytest `
  D:\TraduzAi\pipeline\tests\test_main_emit.py `
  D:\TraduzAi\pipeline\tests\test_vision_stack_runtime.py `
  D:\TraduzAi\pipeline\tests\test_layout_analysis.py `
  D:\TraduzAi\pipeline\tests\test_typesetting_renderer.py -q
```

Expected: tudo verde.

**Step 2: Opcionalmente registrar um sanity check real**

Usar a página `051` do projeto já validado para confirmar que:
- `KEUK?!` ficou fora do pipeline
- o balão inferior ficou separado em duas falas

---

### Task 5: Criar commit

**Files:**
- staged changes only

**Step 1: Revisar a mensagem**

Mensagem sugerida:
```text
fix: preserve skipped sfx and split worker-merged white balloons
```

**Step 2: Commitar**

Run:
```powershell
git commit -m "fix: preserve skipped sfx and split worker-merged white balloons"
```

Expected: commit criado no branch atual `fix/connected-balloon-rendering`.

**Step 3: Conferir o estado após o commit**

Run:
```powershell
git status --short
git log -1 --stat
```

Expected: somente sobras locais não incluídas continuam fora do commit.

---

### Task 6: Fazer push

**Files:**
- Branch: `fix/connected-balloon-rendering`
- Remote: `origin`

**Step 1: Confirmar branch e remote**

Run:
```powershell
git branch --show-current
git remote -v
```

Expected: branch atual correto e `origin` acessível.

**Step 2: Push**

Run:
```powershell
git push origin fix/connected-balloon-rendering
```

Expected: branch remoto atualizado sem erro.

**Step 3: Registrar o hash final**

Run:
```powershell
git rev-parse HEAD
```

Expected: hash final para referência no handoff.

