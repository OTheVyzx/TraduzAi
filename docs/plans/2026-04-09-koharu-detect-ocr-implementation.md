# Koharu Detect + OCR Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** colocar o par padrao do Koharu (`comic-text-bubble-detector` + `paddle-ocr-vl-1.5`) como caminho principal de `detect + OCR` no MangáTL, com reaproveitamento no pipeline automatico e no editor.

**Architecture:** um worker Rust dedicado roda `detect + OCR` e devolve JSON normalizado para o shape atual do MangáTL. O sidecar Python continua orquestrando o pipeline, mas delega esse passo ao worker. O editor chama o mesmo worker por comandos Tauri.

**Tech Stack:** Rust, Tauri v2, serde, tokio, Python 3.12, unittest, JSON IPC

---

### Task 1: Criar a base documental do port

**Files:**
- Create: `docs/plans/2026-04-09-koharu-detect-ocr-design.md`
- Create: `docs/plans/2026-04-09-koharu-detect-ocr-implementation.md`

**Step 1: Salvar o design aprovado**

Expected: o design descreve arquitetura, contrato, rollout e testes

**Step 2: Salvar o plano de implementacao**

Expected: o plano descreve tarefas pequenas e testaveis

### Task 2: Criar o contrato Rust do worker de vision

**Files:**
- Modify: `src-tauri/Cargo.toml`
- Create: `src-tauri/src/vision/mod.rs`
- Create: `src-tauri/src/vision/types.rs`
- Create: `src-tauri/src/vision/worker.rs`
- Create: `src-tauri/src/bin/mangatl-vision.rs`

**Step 1: Write the failing test**

Criar testes Rust para:
- serializar request `page`
- serializar request `region`
- desserializar response com `text_blocks`

**Step 2: Run test to verify it fails**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml vision_worker_contract -- --nocapture`
Expected: FAIL por simbolos e modulos ausentes

**Step 3: Write minimal implementation**

Criar:
- tipos serde do contrato
- worker CLI minimo
- binario que le request de arquivo ou stdin

**Step 4: Run test to verify it passes**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml vision_worker_contract -- --nocapture`
Expected: PASS

### Task 3: Portar o detector padrao do Koharu

**Files:**
- Modify: `src-tauri/src/vision/mod.rs`
- Create: `src-tauri/src/vision/detect.rs`
- Test: `src-tauri/src/vision/detect.rs`

**Step 1: Write the failing test**

Criar testes para:
- imagem sintetica com um bloco simples
- saida contendo `text_blocks`
- saida contendo `bubble_regions`

**Step 2: Run test to verify it fails**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml vision_detect -- --nocapture`
Expected: FAIL

**Step 3: Write minimal implementation**

Implementar:
- adaptador do `comic-text-bubble-detector`
- normalizacao de bbox
- montagem de `bubble_regions`

**Step 4: Run test to verify it passes**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml vision_detect -- --nocapture`
Expected: PASS

### Task 4: Portar o OCR padrao do Koharu

**Files:**
- Create: `src-tauri/src/vision/ocr.rs`
- Test: `src-tauri/src/vision/ocr.rs`

**Step 1: Write the failing test**

Criar testes para:
- crop por bloco
- saida OCR anexada ao bloco
- preservacao de ordem dos blocos

**Step 2: Run test to verify it fails**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml vision_ocr -- --nocapture`
Expected: FAIL

**Step 3: Write minimal implementation**

Implementar:
- adaptador do `paddle-ocr-vl-1.5`
- recorte por bloco
- merge do texto OCR no bloco detectado

**Step 4: Run test to verify it passes**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml vision_ocr -- --nocapture`
Expected: PASS

### Task 5: Ligar detect + OCR no engine do worker

**Files:**
- Create: `src-tauri/src/vision/engine.rs`
- Modify: `src-tauri/src/vision/worker.rs`
- Modify: `src-tauri/src/bin/mangatl-vision.rs`

**Step 1: Write the failing test**

Criar testes para:
- fluxo completo `page`
- fluxo completo `region`
- resposta com `timings`

**Step 2: Run test to verify it fails**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml vision_engine -- --nocapture`
Expected: FAIL

**Step 3: Write minimal implementation**

Implementar:
- load da imagem
- detect
- crops por bloco
- OCR
- normalizacao de resposta

**Step 4: Run test to verify it passes**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml vision_engine -- --nocapture`
Expected: PASS

### Task 6: Integrar o worker ao sidecar Python

**Files:**
- Modify: `src-tauri/src/commands/pipeline.rs`
- Modify: `pipeline/main.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

**Step 1: Write the failing test**

Criar testes Python para:
- montar request do worker
- chamar worker fake
- cair para fallback quando o worker falhar

**Step 2: Run test to verify it fails**

Run: `D:\\mangatl\\pipeline\\venv\\Scripts\\python.exe -m unittest discover -s D:\\mangatl\\pipeline\\tests -p \"test_vision_stack_runtime.py\" -v`
Expected: FAIL nos novos casos

**Step 3: Write minimal implementation**

Implementar:
- resolver caminho do worker no Rust
- repassar para o Python na config
- chamar o worker em `run_detect_ocr(...)`
- usar fallback atual por pagina em caso de erro

**Step 4: Run test to verify it passes**

Run: `D:\\mangatl\\pipeline\\venv\\Scripts\\python.exe -m unittest discover -s D:\\mangatl\\pipeline\\tests -p \"test_vision_stack_runtime.py\" -v`
Expected: PASS

### Task 7: Integrar o worker ao editor

**Files:**
- Modify: `src-tauri/src/commands/pipeline.rs`
- Modify: `src-tauri/src/lib.rs`
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/stores/editorStore.ts`

**Step 1: Write the failing test**

Criar testes Rust para:
- reocr por pagina
- reocr por regiao

**Step 2: Run test to verify it fails**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml editor_reocr -- --nocapture`
Expected: FAIL

**Step 3: Write minimal implementation**

Implementar:
- comando Tauri que chama o worker em modo `page`
- comando Tauri que chama o worker em modo `region`
- atualizacao do projeto no shape atual

**Step 4: Run test to verify it passes**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml editor_reocr -- --nocapture`
Expected: PASS

### Task 8: Validar o caso real e registrar contexto

**Files:**
- Modify: `context.md`
- Output: `testes/...`

**Step 1: Rodar caso real da pagina**

Run: worker/pipeline sobre `T:\\para testes\\nov tradu\\Tradutor automatico MMM\\nao_traduzidos\\Ursaring\\Ursaring (mangabuddy)_Chapter 82_787dd0\\002__002.jpg`
Expected: detectar e OCRizar a pagina sem regressao estrutural

**Step 2: Validar editor**

Run: reocr por pagina e por regiao
Expected: saida equivalente ao automatico

**Step 3: Registrar**

Esperado:
- entrada adicionada em `context.md`
- caminhos de saida anotados
- fallback remanescente documentado
