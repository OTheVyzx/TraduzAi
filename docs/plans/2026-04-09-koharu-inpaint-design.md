# Koharu Inpaint Port Design

**Date:** 2026-04-09

**Goal:** substituir o inpaint atual do MangáTL por um motor com comportamento equivalente ao do Koharu, cobrindo tanto o processamento automático quanto o modo de edição.

**Decision:** seguir a opção 3 pedida pelo usuário, com motor de inpaint movido para Rust e reutilizado por dois chamadores:
- pipeline automática orquestrada pelo sidecar Python
- fluxo de edição disparado pelo app Tauri

---

## Contexto atual

Hoje o MangáTL tem duas realidades:

- o pipeline automática roda no sidecar Python e chama `pipeline/vision_stack/runtime.py` e `pipeline/inpainter/lama.py`
- o editor do app ainda é orientado a `project.json` e retypeset, sem um motor de inpaint compartilhado em Rust

O principal problema dessa arquitetura é que o inpaint ficou preso ao Python e às heurísticas locais do projeto, enquanto o Koharu tem um motor unificado com:

- janela ampliada por bloco de texto
- tentativa de `fill balloon` antes do modelo
- processamento block-aware
- reutilização da mesma lógica para documento inteiro e para crops parciais

---

## O que será copiado do Koharu

O porte deve preservar estes comportamentos do Koharu:

1. `Block-aware inpainting`
   Cada bloco de texto é processado em uma janela expandida, em vez de inpaintar a página inteira de uma vez.

2. `Balloon-first strategy`
   Antes do modelo neural, o motor tenta detectar um balão simples e preenchê-lo diretamente a partir do fundo, evitando artefatos de painel atravessando o balão.

3. `Shared engine`
   O mesmo núcleo de inpaint atende:
   - documento/página inteira
   - crop parcial do editor

4. `Segment mask as source of truth`
   O motor recebe máscara explícita de segmentação/inpaint e não depende só de bbox expandido.

5. `Model backend pluggable`
   O motor precisa permitir backend Koharu-style para `Lama` e `AOT`, com seleção configurável.

---

## Decisão de arquitetura

O motor novo ficará em Rust dentro do `src-tauri`, não no sidecar Python.

Arquitetura proposta:

- criar um módulo interno Rust de inpaint inspirado no Koharu
- expor esse módulo por duas superfícies:
  - uma biblioteca interna para comandos Tauri
  - um binário worker para ser chamado pelo sidecar Python
- manter o Python como orquestrador do pipeline completo, mas remover dele a responsabilidade do inpaint

Isso preserva o investimento atual no pipeline Python sem duplicar o motor de limpeza entre linguagens.

---

## Estrutura alvo

### Núcleo Rust

Novo conjunto de módulos em `src-tauri/src/inpaint/`:

- `mod.rs`
- `types.rs`
- `geometry.rs`
- `balloon.rs`
- `mask.rs`
- `lama.rs`
- `aot.rs`
- `engine.rs`
- `worker.rs`

Responsabilidades:

- `types.rs`
  contratos JSON/serde para blocos, máscaras, requests e responses

- `geometry.rs`
  utilidades de bbox, enlarge-window e localização por crop

- `balloon.rs`
  porte das heurísticas do Koharu para:
  - extrair máscara de balão por borda/contorno
  - construir `non_text_mask`
  - `try_fill_balloon`

- `mask.rs`
  binarização, recorte, limpeza e fusão de máscaras

- `lama.rs`
  backend neural inspirado no Koharu `Lama`, adaptado ao nosso empacotamento

- `aot.rs`
  backend AOT equivalente ao Koharu

- `engine.rs`
  roteamento:
  - block-aware whole-page
  - crop partial inpaint
  - fallback entre `fill balloon`, `AOT`, `Lama`

- `worker.rs`
  entrada JSON/CLI para ser chamada pelo Python

### Binário worker

Novo binário Rust em `src-tauri/src/bin/mangatl-inpaint.rs`.

Esse worker:

- recebe um JSON de request
- carrega imagem, máscara e blocos
- executa o motor Rust
- grava a saída no caminho pedido
- responde em JSON por stdout

### Integração automática

No automático:

- `src-tauri/src/commands/pipeline.rs` passa para o sidecar Python o caminho do worker Rust
- `pipeline/main.py` inclui esse caminho na config
- `pipeline/inpainter/lama.py` e `pipeline/vision_stack/inpainter.py` deixam de ser o motor principal e passam a ser apenas wrapper do worker Rust

### Integração no editor

No editor:

- novo comando Tauri para inpaint de página inteira via motor Rust
- novo comando Tauri para inpaint parcial por região/crop
- o editor passa a poder recalcular a camada `images/` usando o mesmo motor do automático

---

## Fluxo de dados

### Processamento automático

1. Python detecta/OCR e gera blocos com bbox/máscaras
2. Python serializa request de inpaint para o worker Rust
3. Worker Rust processa os blocos localmente
4. Python recebe os caminhos gerados e segue para typeset

### Edição parcial

1. Frontend envia crop/região atual
2. Rust carrega `project.json`, `source`, máscara e blocos da página
3. Rust localiza os blocos que intersectam a região
4. Rust processa só o crop com o mesmo motor
5. Rust costura o patch no `inpainted` base
6. Frontend atualiza a visualização

---

## Estratégia de backend

O Koharu usa `aot-inpainting` como inpainter padrão e oferece `lama-manga` também.

No MangáTL, a porta completa deve suportar:

- `aot-inpainting` como default
- `lama-manga` como alternativa
- fallback controlado por config/flag

Na primeira fase de implementação:

- o motor Rust nasce com a interface dos dois backends
- o caminho do backend default é decidido em configuração
- a seleção fica centralizada em Rust, não no Python

---

## Compatibilidade e rollout

Para reduzir risco:

- manter o stack Python atual como fallback temporário
- permitir desativar o worker Rust por flag de ambiente enquanto a porta estabiliza
- registrar no `context.md` quando o fallback puder ser removido

---

## Testes necessários

### Unitários Rust

- enlarge-window
- localization de blocos para crop
- `extract_balloon_mask`
- `try_fill_balloon`
- processamento block-aware preservando dimensões

### Integração Rust

- request/response do worker
- documento inteiro
- crop parcial

### Integração Python

- wrapper do worker
- fallback quando worker falha
- pipeline automática usando worker Rust

### Validação real

- `002__002.jpg` problemática
- comparativo automático vs edição
- páginas com balão branco simples
- páginas com balão texturizado vermelho

---

## Riscos

1. `Complexidade de build`
   Portar o stack Koharu para dentro do nosso `src-tauri` aumenta dependências Rust e tempo de compilação.

2. `Empacotamento`
   O worker Rust precisa funcionar em dev e produção sem depender de `D:\koharu`.

3. `Drift de comportamento`
   Se só parte da lógica for portada, podemos ficar com “meio Koharu, meio MangáTL”, que é o cenário a evitar.

4. `Editor sem partial inpaint nativo hoje`
   Será preciso criar a superfície de edição parcial no Tauri, não só plugar o motor.

---

## Critério de pronto

Essa migração só conta como concluída quando:

- o automático usa o worker Rust por padrão
- o editor consegue usar o mesmo motor para página e crop parcial
- o fallback Python deixa de ser o caminho normal
- o caso real da `002__002.jpg` fica limpo sem a borda do quadro atravessando o balão
- os testes Rust/Python cobrirem o comportamento novo
