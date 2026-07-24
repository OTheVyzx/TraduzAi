# Modernização das skills do TraduzAI

**Data:** 2026-07-24
**Status:** aprovado para planejamento de implementação

## Objetivo

Transformar as skills locais do TraduzAI em documentação operacional confiável, versionada junto ao projeto e sincronizável com o diretório de skills do usuário. A atualização deve corrigir o mojibake existente, refletir os entrypoints e contratos atuais e cobrir as áreas importantes que hoje não possuem uma skill especializada.

## Escopo

O trabalho moderniza as cinco skills existentes:

- `mangatl-dev`
- `traduzai-detect`
- `traduzai-ocr`
- `traduzai-inpaint`
- `traduzai-typesetting`

E cria três skills:

- `traduzai-pipeline`
- `traduzai-translation`
- `traduzai-studio`

Não faz parte deste trabalho alterar o comportamento do frontend, do backend Rust ou do pipeline Python. As skills devem descrever o código real, sem introduzir novos contratos de runtime.

## Fonte canônica e instalação

As cópias canônicas ficarão em:

```text
N:\TraduzAI\.agents\skills\<nome>\SKILL.md
```

As cópias consumidas pelo Codex continuarão em:

```text
C:\Users\PICHAU\.agents\skills\<nome>\SKILL.md
```

Um script PowerShell versionado no repositório sincronizará somente as oito skills conhecidas. O script deverá:

1. resolver caminhos absolutos e validar que a origem está dentro de `N:\TraduzAI\.agents\skills`;
2. criar apenas os diretórios de destino necessários;
3. copiar os arquivos em UTF-8;
4. oferecer modo de verificação sem escrita;
5. comparar origem e destino depois da cópia;
6. nunca remover outras skills instaladas.

O repositório será a fonte da verdade. Alterações futuras devem ser feitas primeiro na cópia canônica e depois sincronizadas.

## Estrutura comum das skills

Cada `SKILL.md` usará uma estrutura consistente:

1. frontmatter com `name` e `description`;
2. gatilhos de uso baseados em sintomas e arquivos;
3. entrypoint e cadeia de chamadas autoritativa;
4. arquivos sob posse e limites explícitos;
5. contratos de entrada e saída;
6. invariantes e armadilhas conhecidas;
7. artefatos de diagnóstico que devem ser consultados;
8. testes focados que existem no checkout;
9. impacto nas camadas vizinhas;
10. checklist de encerramento;
11. data e commit da última verificação.

As skills devem orientar investigação antes de edição, preservar checkouts sujos e evitar alegações de sucesso baseadas apenas em testes auxiliares quando o resultado visual continuar incorreto.

## Responsabilidade de cada skill

### `mangatl-dev`

Será a skill roteadora e de arquitetura. Deve mapear React/Tauri/Rust/Python, IPC, stores, `project.json`, pipeline automático, Studio separado e regras globais. Não deve repetir os detalhes técnicos das skills especialistas.

### `traduzai-detect`

Cobrirá detecção de regiões e balões, `_vision_blocks`, runtime visual e a relação com o processamento por bandas. Deve encaminhar problemas de reconhecimento para OCR e problemas de máscara para inpaint.

### `traduzai-ocr`

Cobrirá OCR primário e fallback, normalização, reviewers, confiança, classificação, idioma e `skip_processing`. Deve documentar o shape consumido por tradução, layout e inpaint.

### `traduzai-inpaint`

Cobrirá máscaras, proteção de arte, balões brancos, escuros, translúcidos e texturizados, reconstrução local, LaMA/ONNX e fallbacks clássicos. Deve apontar os artefatos atuais de `debug/e2e/06_mask_segmentation` e `debug_inpaint`.

### `traduzai-typesetting`

Cobrirá layout, medição, fontes, renderização FT2Font, style-copy, balões conectados, render plans e validação visual. Deve preservar as restrições de execução serial no Windows.

### `traduzai-pipeline`

Cobrirá `pipeline/main.py`, `pipeline/strip/run.py`, `process_bands.py`, scheduler, reassembly, telemetria, artefatos E2E, QA, `completion_status` e `export_gate`. Será a skill principal para bugs que atravessam mais de um estágio do pipeline automático.

### `traduzai-translation`

Cobrirá `pipeline/translator/`, contexto, glossário, normalização, Google Translate, Ollama e contratos de texto. Não tratará OCR ruim como problema de tradução sem primeiro conferir o texto de origem enviado ao tradutor.

### `traduzai-studio`

Cobrirá o aplicativo Studio separado do pipeline automático. O caminho visual autoritativo será `studio/src/App.tsx` → `StudioSharedEditor` → `src/pages/Editor.tsx`, incluindo backend, adapters, stores, schema, `studio_lite` e testes próprios.

## Roteamento e sobreposição

`mangatl-dev` decidirá qual especialista carregar. Para problemas multietapa do processamento automático, `traduzai-pipeline` será carregada junto da especialista do estágio afetado. `traduzai-studio` não será usada para alterar o pipeline automático sem autorização explícita. Tradução, OCR e typesetting manterão fronteiras separadas mesmo quando compartilham campos do `project.json`.

## Validação

A implementação será validada em quatro níveis:

1. todos os oito arquivos devem ser UTF-8 válido e não conter sequências típicas de mojibake;
2. todos os caminhos e comandos citados devem existir ou estar explicitamente marcados como condicionais;
3. os testes citados devem ser coletáveis pelo pytest, Vitest, Cargo ou comando correspondente;
4. o modo de verificação do sincronizador deve confirmar igualdade entre a fonte canônica e as cópias instaladas.

Como as mudanças são documentais, não é necessário executar toda a suíte do produto. A validação deve testar referências, codificação e sincronização, além de verificações focadas de coleta dos testes mencionados.

## Tratamento de erros

O sincronizador falhará sem alterar destinos quando a origem estiver ausente, fora da raiz autorizada ou contiver um `SKILL.md` inválido. Em uma falha parcial de cópia, deverá retornar erro e identificar a skill divergente; não apagará skills nem tentará limpar diretórios externos.

Referências que não puderem ser confirmadas no checkout serão removidas ou marcadas como condicionais. Nenhum caminho lembrado será apresentado como atual sem verificação local.

## Sequência de entrega

1. criar a fonte canônica e o verificador estrutural;
2. modernizar `mangatl-dev` e as quatro especialistas atuais;
3. criar as três skills novas;
4. criar e testar o sincronizador;
5. sincronizar as oito skills instaladas;
6. executar a auditoria final de encoding, caminhos, comandos e igualdade.
