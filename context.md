# Contexto Atual do Projeto

Última atualização: 2026-04-29

## Resumo
- Branch ativa: `fix/connected-balloon-rendering`
- Foco recente: estabilização de balões conectados, remoção de falsos positivos, preservação de `SFX`/marca/watermark fora do pipeline, correções de persistência/export e melhoria do preview original/traduzido.
- Estado geral: o lote crítico de páginas problemáticas do capítulo 1 melhorou bastante, com destaque para `006`, `017`, `018`, `019`, `020`, `021` e `051`.

## Mudanças Recentes

### Pipeline de layout/typeset
- `pipeline/layout/balloon_layout.py`
  - reduziu falsos positivos de `connected_balloon` em balões simples;
  - melhorou separação de balões brancos falsamente compartilhados;
  - refinou tratamento de caixas amplas para evitar merge conectado indevido.
- `pipeline/typesetter/renderer.py`
  - passou a respeitar `skip_processing`;
  - deixou de renderizar textos marcados como `SFX`, marca ou watermark pulados pelo OCR/postprocess.

### OCR / Remapeamento / Reprocessamento
- `pipeline/ocr/postprocess.py`
  - filtragem mais forte para ruído curto;
  - `SFX`, marca e watermark curtos saem do fluxo antes de virar `text_layer`.
- `pipeline/main.py`
  - remapeamento entre OCR antigo/novo passou a priorizar geometria em vez de índice;
  - quando o OCR novo funde falas antigas, as traduções podem ser combinadas;
  - quando o OCR novo separa uma fala antes fundida, o texto antigo pode ser redistribuído por sentença.

### Inpaint / Vision runtime / Strip
- `pipeline/vision_stack/runtime.py`
  - heurísticas melhores para preservar separação de blocos brancos diagonais/empilhados;
  - melhorias em balões brancos e casos com resíduo;
  - casos como a página `051` passaram a manter blocos distintos em vez de fundir tudo.
- `pipeline/inpainter/__init__.py`
  - caminho `strip` passou a usar o mesmo round principal de inpaint, em vez do inpaint clássico simplificado.
- `pipeline/strip/process_bands.py`
  - preservação de metadata mais rica (`text_pixel_bbox`, `line_polygons`, `ocr_source`, `ocr_confidence`, `_vision_blocks`).

### Export / Persistência / Preview
- `pipeline/strip/run.py`, `pipeline/strip/types.py`, `pipeline/main.py`
  - separação real entre `originals`, `images` e `translated`;
  - preservação distinta de original, imagem limpa e render final.
- `src/pages/Preview.tsx`
  - toggle do preview passou a usar `image_layers.base` e `image_layers.rendered` quando disponíveis.
- `src/pages/previewImage.ts`
  - helper central para resolver a imagem correta de preview.

## Casos Reais Importantes

### Lote `traduzido24_fix_all` / `traduzido24_fix_all_v6`
- Página `006`
  - falso positivo de balão duplo conectado corrigido.
- Página `017`
  - conteúdo perdido recuperado e melhor separado visualmente.
- Página `018`
  - corrigido o caso de “dois type” no mesmo balão.
- Página `019`
  - removido merge conectado falso; `NONE.` corrigido para `NENHUMA.` no contexto dessa página.
- Página `020`
  - balão branco gigante deixou de engolir bloco vizinho.
- Página `021`
  - layout estabilizado após corrigir herança de caixa ampla errada.
- Página `051`
  - `KEUK?!` preservado como arte original;
  - bloco inferior voltou a ter duas falas separadas;
  - página ficou utilizável, ainda sendo um bom caso de regressão para teste futuro.

## Regras/Heurísticas Atuais Importantes
- Pular `SFX`, marca e watermark:
  - não fazer OCR útil;
  - não entrar no inpaint;
  - não fazer typeset por cima.
- Priorizar geometria e caixas refinadas antes de fundir blocos.
- Evitar reaproveitamento de tradução por índice quando a geometria mudou.

## Testes e Validação
- Rodadas focadas recentes passaram com sucesso em lotes como:
  - `test_layout_analysis.py`
  - `test_typesetting_renderer.py`
  - `test_main_emit.py`
  - `test_vision_stack_runtime.py`
  - `test_strip_process_bands.py`
  - `test_strip_inpaint_complete.py`
- Antes do fluxo atual de backup/commit, os lotes focados mais recentes estavam verdes, incluindo uma rodada de `201 passed` no conjunto principal de regressões do pipeline.

## Pendências Conhecidas
- Ainda vale manter regressão dedicada para a página `051`, porque ela continua sendo um caso sensível de OCR/layout.
- O worktree local está sujo com muitos artefatos de validação (`NOV/`, logs, benches e saídas temporárias), então commits devem ser feitos por caminho explícito.

## Convenção Operacional
- `cntbk` significa:
  - atualizar este `context.md`;
  - criar um novo backup versionado do projeto;
  - excluir o backup versionado anterior.
