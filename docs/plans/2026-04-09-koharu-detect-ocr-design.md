# Koharu Detect + OCR Design

**Date:** 2026-04-09

**Goal:** portar o par padrao de `detect + OCR` do Koharu para o MangáTL, usando o mesmo comportamento-base no processamento automático e no modo de edição.

**Decision:** seguir a opcao 3 escolhida pelo usuario:
- worker Rust dedicado para `detect + OCR`
- contrato JSON normalizado para o formato atual do MangáTL
- fallback automatico para o stack atual quando o worker nao puder rodar

---

## Contexto atual

Hoje o MangáTL faz `detect + OCR` pelo sidecar Python:

- detector principal em `pipeline/vision_stack/detector.py`
- OCR principal em `pipeline/vision_stack/ocr.py`
- orquestracao em `pipeline/vision_stack/runtime.py`

O editor consome o resultado salvo em `project.json`, mas nao compartilha um motor unico de reexecucao de `detect + OCR`.

No Koharu, o par padrao e:

- detector: `comic-text-bubble-detector`
- OCR: `paddle-ocr-vl-1.5`

Esse fluxo e resolvido em Rust e retorna blocos com OCR ja anexado.

---

## O que sera copiado do Koharu

O port precisa preservar estes comportamentos:

1. `Bubble-aware detection`
   O detector padrao devolve blocos de texto e regioes de balao no mesmo passe.

2. `OCR over cropped text blocks`
   O OCR recebe crops por bloco de texto, nao a pagina inteira.

3. `Shared engine`
   O mesmo motor atende:
   - pipeline automatica
   - reexecucao por pagina no editor
   - reexecucao parcial por regiao, quando o editor pedir

4. `Normalized output`
   O resultado final continua no shape esperado pelo MangáTL:
   - `text_blocks`
   - `bubble_regions`
   - hints opcionais como `font_hints`

5. `Controlled fallback`
   Se o worker Rust falhar, a pagina cai para o detector/OCR atuais do MangáTL sem abortar o job inteiro.

---

## Decisao de arquitetura

O port entra como um worker Rust separado do core Tauri.

Arquitetura alvo:

- `src-tauri` continua como app host
- novo binario Rust dedicado a `detect + OCR`
- o sidecar Python chama esse worker por JSON
- o editor reutiliza o mesmo worker por comando Tauri

Isso evita:

- misturar crates pesados do Koharu no boot da app
- depender da pasta externa `D:\koharu` em tempo de execucao
- duplicar a logica em Python e Rust

---

## Estrutura alvo

### Worker Rust

Novo binario:

- `src-tauri/src/bin/mangatl-vision.rs`

Novos modulos:

- `src-tauri/src/vision/mod.rs`
- `src-tauri/src/vision/types.rs`
- `src-tauri/src/vision/detect.rs`
- `src-tauri/src/vision/ocr.rs`
- `src-tauri/src/vision/engine.rs`
- `src-tauri/src/vision/worker.rs`

Responsabilidades:

- `types.rs`
  request/response JSON, bboxes, text blocks, bubble regions

- `detect.rs`
  porta do `comic-text-bubble-detector`

- `ocr.rs`
  porta do `paddle-ocr-vl-1.5`

- `engine.rs`
  fluxo completo:
  - carregar imagem
  - detectar
  - recortar blocos
  - OCR por bloco
  - normalizar saida

- `worker.rs`
  CLI/stdin/stdout para o sidecar Python e para comandos Tauri

### Integracao no pipeline

- `pipeline/vision_stack/runtime.py`
  passa a tentar o worker primeiro em `run_detect_ocr(...)`

- `pipeline/main.py`
  recebe o caminho do worker na config e repassa para o runtime

- `src-tauri/src/commands/pipeline.rs`
  resolve e injeta o caminho do worker no sidecar Python

### Integracao no editor

- novos comandos Tauri para:
  - reexecutar `detect + OCR` da pagina
  - reexecutar `detect + OCR` em uma regiao

O editor continua salvando no shape atual do projeto.

---

## Fluxo e contrato

Request para o worker:

- caminho da imagem
- modo:
  - `page`
  - `region`
- regiao opcional
- perfil/qualidade
- paths de modelos, quando necessario

Response normalizada:

- `status`
- `text_blocks`
- `bubble_regions`
- `timings`
- `warnings`

Cada `text_block` volta pronto para o MangáTL:

- bbox
- texto OCR
- confianca
- tipo de balao
- metadados auxiliares

---

## Rollout e fallback

- o worker Koharu-style vira o caminho padrao
- o Python atual continua como fallback por pagina
- projetos antigos seguem abrindo sem migracao
- se faltar modelo ou houver falha do worker, o job continua no stack atual

O fallback so pode ser removido depois de validacao em paginas reais e no editor.

---

## Testes necessarios

### Rust

- contrato JSON do worker
- detect fake/controlado
- OCR fake/controlado
- fluxo `page`
- fluxo `region`

### Python

- wrapper do worker
- fallback por excecao
- `run_detect_ocr(...)` usando o worker

### Tauri/editor

- comando de reocr por pagina
- comando de reocr por regiao

### Validacao real

- pagina `002__002.jpg`
- pagina com balao branco simples
- pagina com balao texturizado
- comparativo entre automatico e editor
