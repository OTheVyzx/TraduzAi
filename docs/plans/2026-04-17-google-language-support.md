# Google Language Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Liberar todos os idiomas do backend Google Translate para origem e destino no TraduzAi, com OCR em melhor esforco para origens fora do suporte forte atual.

**Architecture:** O sidecar Python sera a fonte de verdade da lista de idiomas suportados. O backend Rust expora essa lista ao frontend, e o pipeline passara a normalizar codigos de idioma para Google, PaddleOCR e EasyOCR em um util central. O frontend deixara de usar listas fixas e passara a carregar idiomas dinamicamente.

**Tech Stack:** React 19, TypeScript, Zustand, Tauri v2, Rust, Python 3.12, deep-translator, PaddleOCR, EasyOCR

---

### Task 1: Expor idiomas do Google no sidecar

**Files:**
- Modify: `pipeline/translator/translate.py`
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_translate_context.py`

**Step 1: Write the failing test**

Adicionar teste para garantir que a funcao que lista idiomas:
- retorna uma colecao nao vazia
- contem pelo menos `en`, `pt`, `es`, `de`, `ru`, `ja`, `ko`, `zh-CN`
- ordena/normaliza os dados para consumo do app

**Step 2: Run test to verify it fails**

Run: `.\pipeline\venv\Scripts\python -m pytest pipeline/tests/test_translate_context.py -k languages -v`

**Step 3: Write minimal implementation**

- Criar helper em `translate.py` para listar idiomas suportados pelo `GoogleTranslator().get_supported_languages(as_dict=True)`
- Adicionar modo CLI em `pipeline/main.py` para emitir essa lista

**Step 4: Run test to verify it passes**

Run: `.\pipeline\venv\Scripts\python -m pytest pipeline/tests/test_translate_context.py -k languages -v`

**Step 5: Commit**

```bash
git add pipeline/translator/translate.py pipeline/main.py pipeline/tests/test_translate_context.py
git commit -m "feat: expose google translate languages"
```

### Task 2: Expor idiomas ao frontend via Tauri

**Files:**
- Modify: `src-tauri/src/commands/settings.rs`
- Modify: `src-tauri/src/lib.rs`
- Modify: `src/lib/tauri.ts`

**Step 1: Write the failing test**

Adicionar teste Rust pequeno para o formato default/serializacao, ou teste TypeScript para shape da resposta, conforme padrao mais barato do repo.

**Step 2: Run test to verify it fails**

Run: `cargo test settings --manifest-path src-tauri/Cargo.toml`

**Step 3: Write minimal implementation**

- Criar command Tauri `load_supported_languages`
- Fazer o Rust chamar o sidecar com o modo de listagem
- Expor o tipo e binding em `src/lib/tauri.ts`

**Step 4: Run test to verify it passes**

Run: `cargo test settings --manifest-path src-tauri/Cargo.toml`

**Step 5: Commit**

```bash
git add src-tauri/src/commands/settings.rs src-tauri/src/lib.rs src/lib/tauri.ts
git commit -m "feat: expose supported languages to frontend"
```

### Task 3: Trocar listas fixas do frontend por idiomas dinamicos

**Files:**
- Modify: `src/pages/Settings.tsx`
- Modify: `src/pages/Setup.tsx`
- Modify: `src/pages/Home.tsx`

**Step 1: Write the failing test**

Adicionar teste de unidade/componente se a base permitir; senao validar com um helper puro TypeScript para:
- aplicar defaults carregados
- aceitar origem e destino dinamicos
- nao forcar `idioma_origem` para `en`

**Step 2: Run test to verify it fails**

Run: comando de testes frontend existente, ou validar helper isolado

**Step 3: Write minimal implementation**

- Carregar idiomas suportados ao abrir `Settings` e `Setup`
- Usar `default` salvo para destino
- Deixar origem e destino totalmente selecionaveis na UI
- Ajustar `Home` para parar de criar projetos com origem fixa em `en`

**Step 4: Run test to verify it passes**

Run: comando de testes frontend relevante

**Step 5: Commit**

```bash
git add src/pages/Settings.tsx src/pages/Setup.tsx src/pages/Home.tsx
git commit -m "feat: load dynamic language lists in ui"
```

### Task 4: Normalizar idiomas para traducao e OCR

**Files:**
- Modify: `pipeline/translator/translate.py`
- Modify: `pipeline/vision_stack/ocr.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/ocr_legacy/detector.py`
- Optionally modify: `pipeline/ocr/postprocess.py`
- Test: `pipeline/tests/test_translate_context.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

**Step 1: Write the failing test**

Adicionar testes para:
- variantes como `en-GB` e `pt-BR`
- idiomas latinos adicionais (`es`, `de`, `fr`, etc.)
- idiomas cirilicos (`ru`)
- fallback melhor esforco do OCR quando o idioma nao tiver modelo dedicado

**Step 2: Run test to verify it fails**

Run: `.\pipeline\venv\Scripts\python -m pytest pipeline/tests/test_translate_context.py pipeline/tests/test_vision_stack_runtime.py -k language -v`

**Step 3: Write minimal implementation**

- Criar helpers de normalizacao
- Remover fallback silencioso para ingles no Google translator
- Mapear codigos do Google para Paddle/EasyOCR
- Ajustar o legado para respeitar `idioma_origem`

**Step 4: Run test to verify it passes**

Run: `.\pipeline\venv\Scripts\python -m pytest pipeline/tests/test_translate_context.py pipeline/tests/test_vision_stack_runtime.py -k language -v`

**Step 5: Commit**

```bash
git add pipeline/translator/translate.py pipeline/vision_stack/ocr.py pipeline/vision_stack/runtime.py pipeline/ocr_legacy/detector.py pipeline/tests/test_translate_context.py pipeline/tests/test_vision_stack_runtime.py
git commit -m "feat: normalize languages across translation and ocr"
```

### Task 5: Verificacao final

**Files:**
- Modify if needed: arquivos acima

**Step 1: Run targeted tests**

Run: `.\pipeline\venv\Scripts\python -m pytest pipeline/tests/test_translate_context.py pipeline/tests/test_vision_stack_runtime.py -v`

**Step 2: Run Rust tests**

Run: `cargo test --manifest-path src-tauri/Cargo.toml`

**Step 3: Smoke-check the TypeScript app**

Run: `npm run build`

**Step 4: Fix remaining issues**

Aplicar os ajustes minimos para deixar os checks verdes.

**Step 5: Commit**

```bash
git add .
git commit -m "feat: support dynamic google languages across app"
```
