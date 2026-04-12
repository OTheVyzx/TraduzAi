# Koharu Inpaint Port Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** portar o motor de inpaint do Koharu para o MangáTL, em Rust, e usá-lo tanto no pipeline automático quanto no modo de edição.

**Architecture:** o novo inpaint vira um núcleo Rust compartilhado. O sidecar Python continua orquestrando OCR/tradução/typeset, mas delega o passo de inpaint a um worker Rust. O app Tauri chama a mesma biblioteca diretamente para edição e crop parcial.

**Tech Stack:** Rust, Tauri v2, serde, tokio, image, imageproc, Python wrapper, JSON IPC

---

### Task 1: Criar o documento de design e o plano de execução

**Files:**
- Create: `docs/plans/2026-04-09-koharu-inpaint-design.md`
- Create: `docs/plans/2026-04-09-koharu-inpaint-port.md`

**Step 1: Validar o escopo**

- confirmar que o alvo inclui:
  - pipeline automática
  - edição parcial
  - worker Rust
  - comportamento Koharu-style

**Step 2: Salvar os documentos**

Run: n/a
Expected: ambos os arquivos presentes em `docs/plans/`

---

### Task 2: Preparar a base Rust do novo motor

**Files:**
- Modify: `src-tauri/Cargo.toml`
- Create: `src-tauri/src/inpaint/mod.rs`
- Create: `src-tauri/src/inpaint/types.rs`
- Create: `src-tauri/src/inpaint/geometry.rs`
- Create: `src-tauri/src/inpaint/balloon.rs`
- Create: `src-tauri/src/inpaint/mask.rs`
- Create: `src-tauri/src/inpaint/engine.rs`
- Create: `src-tauri/src/bin/mangatl-inpaint.rs`

**Step 1: Escrever teste falho de geometria**

- criar teste para `enlarge_window`
- criar teste para localizar blocos dentro de crop parcial

**Step 2: Rodar o teste**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml enlarge_window -- --nocapture`
Expected: FAIL por símbolos ainda inexistentes

**Step 3: Implementar tipos e geometria mínimos**

- request/response JSON
- bbox
- janela ampliada
- localização de blocos no crop

**Step 4: Rodar o teste novamente**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml geometry -- --nocapture`
Expected: PASS

---

### Task 3: Portar a heurística de balão do Koharu

**Files:**
- Modify: `src-tauri/src/inpaint/balloon.rs`
- Test: `src-tauri/src/inpaint/balloon.rs`

**Step 1: Escrever teste falho de extração de balão**

- caso com balão elíptico simples
- caso com máscara de texto dentro do balão
- caso garantindo que a arte fora do balão não entra no fill

**Step 2: Rodar o teste**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml balloon -- --nocapture`
Expected: FAIL

**Step 3: Implementar o porte**

- `extract_balloon_mask`
- `non_text_mask`
- `median_rgb`
- `color_stddev`
- `try_fill_balloon`

**Step 4: Rodar o teste**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml balloon -- --nocapture`
Expected: PASS

---

### Task 4: Implementar engine block-aware sem fallback ao Python

**Files:**
- Modify: `src-tauri/src/inpaint/engine.rs`
- Create: `src-tauri/src/inpaint/lama.rs`
- Create: `src-tauri/src/inpaint/aot.rs`

**Step 1: Escrever teste falho de block-aware page**

- garantir que o engine processa bloco por bloco
- garantir que cada crop é costurado na imagem final

**Step 2: Rodar o teste**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml inpaint_engine -- --nocapture`
Expected: FAIL

**Step 3: Implementar roteamento do engine**

- tentar `try_fill_balloon`
- usar backend default configurável
- costurar o resultado no canvas final

**Step 4: Rodar o teste**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml inpaint_engine -- --nocapture`
Expected: PASS

---

### Task 5: Criar o worker Rust para o sidecar Python

**Files:**
- Modify: `src-tauri/src/bin/mangatl-inpaint.rs`
- Modify: `src-tauri/src/inpaint/types.rs`

**Step 1: Escrever teste falho do contrato JSON**

- request com imagem, máscara, blocos, backend e output
- response com status e caminho gerado

**Step 2: Rodar o teste**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml worker -- --nocapture`
Expected: FAIL

**Step 3: Implementar CLI**

- ler JSON de arquivo ou stdin
- carregar imagem/máscara
- executar engine
- gravar saída
- imprimir response JSON

**Step 4: Rodar o teste**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml worker -- --nocapture`
Expected: PASS

---

### Task 6: Integrar o worker Rust ao pipeline Python automático

**Files:**
- Modify: `src-tauri/src/commands/pipeline.rs`
- Modify: `pipeline/main.py`
- Modify: `pipeline/inpainter/lama.py`
- Modify: `pipeline/vision_stack/inpainter.py`
- Test: `pipeline/tests/test_vision_stack_inpainter.py`

**Step 1: Escrever teste falho do wrapper Python**

- o wrapper deve montar request
- deve chamar o worker Rust
- deve retornar o arquivo final

**Step 2: Rodar o teste**

Run: `D:\\mangatl\\pipeline\\venv\\Scripts\\python.exe -m unittest -v test_vision_stack_inpainter`
Expected: FAIL

**Step 3: Implementar integração**

- `pipeline.rs` inclui caminho do worker na config JSON
- `main.py` repassa a informação
- `lama.py` vira wrapper do worker Rust
- `vision_stack/inpainter.py` passa a usar o wrapper como caminho principal

**Step 4: Rodar o teste**

Run: `D:\\mangatl\\pipeline\\venv\\Scripts\\python.exe -m unittest -v test_vision_stack_inpainter`
Expected: PASS

---

### Task 7: Integrar o motor Rust ao modo de edição

**Files:**
- Modify: `src-tauri/src/commands/pipeline.rs`
- Modify: `src-tauri/src/lib.rs`
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/stores/editorStore.ts`
- Create: comando Tauri de inpaint parcial/página

**Step 1: Escrever teste falho do comando de edição**

- comando de página inteira
- comando de crop parcial

**Step 2: Rodar o teste**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml edit_inpaint -- --nocapture`
Expected: FAIL

**Step 3: Implementar os comandos**

- comando de página inteira usando o engine Rust
- comando de crop parcial localizando blocos do `project.json`
- persistência do `inpainted`

**Step 4: Rodar o teste**

Run: `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml edit_inpaint -- --nocapture`
Expected: PASS

---

### Task 8: Validar o caso real da 002__002

**Files:**
- Modify: `context.md`
- Output: `testes/...`

**Step 1: Rodar a página real no automático**

Run: worker/pipeline sobre `T:\\para testes\\nov tradu\\Tradutor automatico MMM\\nao_traduzidos\\Ursaring\\Ursaring (mangabuddy)_Chapter 82_787dd0\\002__002.jpg`
Expected: imagem limpa gerada

**Step 2: Rodar o mesmo caso via fluxo de edição**

Run: comando Tauri de página/crop
Expected: resultado visual equivalente

**Step 3: Registrar no contexto**

- descrever backend final
- descrever fallback remanescente, se existir
- salvar caminhos das imagens geradas

---

### Task 9: Remover o Python como caminho principal do inpaint

**Files:**
- Modify: `pipeline/inpainter/lama.py`
- Modify: `pipeline/vision_stack/inpainter.py`
- Modify: `context.md`

**Step 1: Confirmar que o automático já depende do Rust**

- revisar logs e testes

**Step 2: Rebaixar o Python para fallback**

- deixar o caminho antigo só como contingência temporária

**Step 3: Rodar verificação final**

Run:
- `cargo test --manifest-path D:\\mangatl\\src-tauri\\Cargo.toml`
- `npx tsc --noEmit`
- `D:\\mangatl\\pipeline\\venv\\Scripts\\python.exe -m py_compile D:\\mangatl\\pipeline\\main.py D:\\mangatl\\pipeline\\inpainter\\lama.py D:\\mangatl\\pipeline\\vision_stack\\inpainter.py`

Expected:
- PASS no Rust
- PASS no TypeScript
- PASS no py_compile
