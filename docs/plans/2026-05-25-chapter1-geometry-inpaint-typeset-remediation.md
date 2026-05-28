# Chapter 1 Geometry, Inpaint, Typeset Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Corrigir as falhas em que textos aparecem fora do balao, baloes ficam vazios, residuos de texto ingles permanecem apos inpaint, e o debug/QA permite exportar saidas visualmente quebradas.

**Architecture:** A correcao deve preservar o pipeline automatico atual, mas tornar o contrato de coordenadas explicito entre strip/banda, pagina reagrupada e pagina final. A renderizacao final so pode usar caixas em coordenada da pagina final; fast fill so pode remover blocos do inpaint real quando a mascara realmente cobrir o texto e passar em verificacao residual.

**Tech Stack:** Python 3.12, OpenCV, PaddleOCR/debug E2E, `pipeline/strip`, `pipeline/inpainter`, `pipeline/typesetter`, `pipeline/qa`, `project.json`.

---

## Contexto Do Bug

Run analisada:

- `N:/TraduzAI/DEBUGM/runs/2026-05-24_task10_real_validation_20260524_174315/chapter1_full`

Saida final com problema:

- `N:/TraduzAI/DEBUGM/runs/2026-05-24_task10_real_validation_20260524_174315/chapter1_full/translated`

Artefatos relevantes:

- `debug/e2e/05_layout_geometry/bbox_coordinate_audit.json`
- `debug/e2e/09_typeset/render_plan_raw.jsonl`
- `debug/e2e/09_typeset/render_plan_final.jsonl`
- `debug/e2e/11_qa_export_gate/qa_issues.jsonl`
- `debug_inpaint/page_003_band_035/metadata.json`
- `debug_inpaint/page_003_band_046/metadata.json`
- `pipeline.log`
- `project.json`

## Causa Raiz

### Problema 1: Coordenadas mistas entre banda e pagina final

**Sintoma:**

Textos aparecem no topo da pagina, no rosto/personagem, ou fora do balao correto. O balao real fica vazio ou com texto fantasma.

**Evidencia:**

`page_002_band_005` no `project.json` mostrou:

- `text_pixel_bbox`: `y ~= 5655`, coordenada da pagina final.
- `balloon_bbox` em `textos`: `y ~= 5606`, coordenada da pagina final.
- `bubble_inner_bbox`: `y ~= 230`, coordenada local da banda.
- `safe_text_box`: `y ~= 242`, coordenada local da banda.
- `render_bbox`: `y ~= 246`, coordenada local da banda.

O texto `POR FAVOR, PELO BEM DA CRIANCA.` foi renderizado em `y ~= 246`, nao em `y ~= 5600`.

**Causa no codigo:**

`pipeline/strip/run.py` tem dois helpers de remapeamento:

- `_shift_text_geometry_y`
- `_shift_text_geometry_xy`

Eles deslocam `bbox`, `balloon_bbox`, `text_pixel_bbox`, `render_bbox`, `safe_text_box`, etc., mas nao deslocam:

- `bubble_mask_bbox`
- `bubble_inner_bbox`

Depois, o typesetter prioriza `bubble_inner_bbox` como area segura em `pipeline/typesetter/renderer.py`, entao ele calcula o layout com coordenada local da banda dentro de uma pagina final.

**Solucao:**

Shiftar `bubble_mask_bbox` e `bubble_inner_bbox` em todos os helpers de remapeamento de geometria. Depois disso, garantir que `safe_text_box`, `render_bbox`, `_debug_safe_text_box` e `_render_debug.*bbox` sejam recalculados ou validados na mesma coordenada.

### Problema 2: Rerender final em pagina inteira usa geometria suja

**Sintoma:**

Algumas etapas intermediarias parecem corretas, mas o `translated/*.jpg` final sai errado.

**Causa no codigo:**

`pipeline/main.py` faz `sync_final_page_space_typeset` e chama:

```python
typesetter_mod.render_band_image(clean_rgb, {"texts": page_texts, "_coordinate_space": "main_final_page_space_typeset"})
```

Esse passo rerenderiza paginas finais com `page_texts`. Se qualquer texto ainda tiver `bubble_inner_bbox` local, o output final fica errado mesmo que a banda isolada pareca correta.

**Solucao:**

Antes de `main_final_page_space_typeset`, rodar uma validacao de coordenadas por texto. Se houver mistura local/global, converter antes de renderizar ou bloquear o rerender daquela pagina com flag critica. Esse passo nao deve confiar no auditor atual, porque ele nao cobre `bubble_*`.

### Problema 3: Auditor de coordenadas nao cobre as chaves que quebraram

**Sintoma:**

`bbox_coordinate_audit.json` marcou `all_consistent: true`, mesmo com coordenadas claramente mistas.

**Causa no codigo:**

`pipeline/debug_tools/bbox.py` lista `BBOX_KEYS`, mas nao inclui:

- `bubble_mask_bbox`
- `bubble_inner_bbox`

O auditor tambem nao marca como critica a diferenca entre `target_bbox` global e `safe_text_box/render_bbox` local quando a causa vem de `bubble_inner_bbox`.

**Solucao:**

Adicionar `bubble_mask_bbox` e `bubble_inner_bbox` no auditor e criar regra especifica:

- Se `target_bbox`/`balloon_bbox` estao em pagina final e `bubble_inner_bbox`/`safe_text_box`/`render_bbox` estao perto de `0..band_height`, marcar `layout_bbox_coordinate_mismatch`.
- Propagar essa flag para `qa_flags` da layer e para o export gate.

### Problema 4: Fast solid fill apaga pouco e pula inpaint real

**Sintoma:**

Ficam residuos de texto ingles em baloes brancos/conectados, como `I'M STARVING`, `SOME.`, textos de titulo e caixas escuras.

**Evidencia:**

`debug_inpaint/page_003_band_035/metadata.json` e `page_003_band_046/metadata.json` mostram:

- `used_fast_solid_fill: true`
- `used_real_inpaint: false`
- `raw_mask_pixels: 0`
- `expanded_mask_pixels: 0`
- `fast_fill_without_raw_mask: true`
- `post_cleanup_skipped_reason: fast_solid_fill`

**Causa no codigo:**

`pipeline/inpainter/__init__.py` usa `_solid_text_fill_mask`. Quando existem `line_polygons`, a mascara vem de geometria de linha e recebe pouco padding. Depois `_apply_fast_solid_balloon_fill` remove o bloco da lista de `vision_blocks`. Como nao sobram blocos, o inpaint real nao roda e o post-cleanup pode ser pulado.

**Solucao:**

Fast solid so pode remover um bloco do inpaint real quando cumprir tres condicoes:

1. Mascara cobre texto com dilatacao suficiente dentro do balao.
2. Verificacao residual nao encontra pixels escuros/claros suspeitos onde havia texto.
3. A area preenchida e coerente com `text_pixel_bbox` e/ou OCR mask.

Se qualquer condicao falhar, manter o bloco para inpaint real ou executar retry real com uma mascara expandida e limitada ao interior do balao.

### Problema 5: Projeto/render plan perde rastreabilidade em blocos agrupados

**Sintoma:**

O texto final existe na imagem, mas `project.json` nem sempre tem `render_bbox` correto para a layer que realmente foi renderizada. Em `page_003_band_035`, `TODAY?` foi combinado no bloco de `ocr_001`, enquanto `ocr_003` aparece separado no projeto.

**Causa provavel:**

O typesetter consolida blocos de render, mas o materializador de projeto preserva layers OCR originais sem mapear todos os filhos renderizados para o bloco final.

**Solucao:**

Cada render block final deve registrar:

- `trace_id`
- `source_trace_ids`
- `render_bbox`
- `safe_text_box`
- `target_bbox`
- `connected_children`, quando houver

O `project.json` deve receber esse bloco consolidado ou propagar `render_bbox`/`qa_flags` para todas as source layers relacionadas.

### Problema 6: QA/export gate permite saida visualmente ruim

**Sintoma:**

`pipeline.log` registrou `TEXT_OVERFLOW`, `TEXT_CLIPPED`, `render_outside_balloon`, mas o debug terminou com `PASS`.

**Causa no codigo:**

`pipeline/qa/export_gate.py` bloqueia apenas flags `critical`. Hoje `TEXT_OVERFLOW` e `TEXT_CLIPPED` sao `high`, entao entram como `needs_review`, mas nao bloqueiam.

**Solucao:**

Nao transformar todo `TEXT_OVERFLOW` em critical globalmente. Em vez disso, criar bloqueios criticos especificos:

- `layout_bbox_coordinate_mismatch`
- `render_outside_balloon`
- `render_on_art_suspected`
- `text_residual_after_inpaint`
- `fast_fill_unverified_residual`
- `page_space_rerender_mixed_coordinates`

Esses casos devem gerar `BLOCK` em debug/strict e aparecer em `visual_blockers.jsonl`.

## Plano De Execucao

### Task 1: Criar testes que reproduzem coordenada mista

**Problema a resolver:** impedir que `bubble_inner_bbox` local sobreviva depois do remapeamento para pagina final.

**Arquivos:**

- Testar: `pipeline/tests/test_strip_balloon_bbox_propagation.py`
- Modificar depois: `pipeline/strip/run.py`

**Passos:**

1. Criar teste com um texto em banda local:
   - `band_y_top = 5420`
   - `bubble_inner_bbox = [513, 230, 649, 313]`
   - `bubble_mask_bbox = [501, 218, 661, 325]`
2. Chamar `_shift_text_geometry_y(text, 5420)`.
3. Esperar:
   - `bubble_inner_bbox == [513, 5650, 649, 5733]`
   - `bubble_mask_bbox == [501, 5638, 661, 5745]`
4. Rodar teste e confirmar falha antes do patch.

Comando:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_strip_balloon_bbox_propagation.py -q
```

### Task 2: Corrigir remapeamento de geometria

**Problema a resolver:** todas as bboxes derivadas precisam acompanhar o mesmo deslocamento.

**Arquivos:**

- Modificar: `pipeline/strip/run.py`

**Solucao:**

Adicionar aos loops de `_shift_text_geometry_y` e `_shift_text_geometry_xy`:

- `bubble_mask_bbox`
- `bubble_inner_bbox`
- `balloon_inner_bbox`, se existir no contrato

Garantir que listas aninhadas e `_render_debug` continuem sendo shiftadas.

**Validacao:**

Rodar o teste da Task 1.

### Task 3: Corrigir auditor de coordenadas

**Problema a resolver:** o debug precisa detectar exatamente esse bug.

**Arquivos:**

- Modificar: `pipeline/debug_tools/bbox.py`
- Testar: `pipeline/tests/test_derived_bbox_coordinate_audit.py`

**Solucao:**

1. Incluir `bubble_mask_bbox` e `bubble_inner_bbox` em `BBOX_KEYS`.
2. Adicionar teste onde:
   - `target_bbox` e `balloon_bbox` estao em `y ~= 5600`.
   - `bubble_inner_bbox`, `safe_text_box` ou `render_bbox` estao em `y ~= 230`.
3. Esperar finding:
   - `blocker: derived_bbox_coordinate_mismatch`
   - `severity: critical`
   - flag sugerida: `layout_bbox_coordinate_mismatch`

**Validacao:**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_derived_bbox_coordinate_audit.py -q
```

### Task 4: Bloquear rerender final com coordenada mista

**Problema a resolver:** impedir que `translated/*.jpg` seja regravado com texto em posicao errada.

**Arquivos:**

- Modificar: `pipeline/main.py`
- Possivelmente criar helper em: `pipeline/strip/run.py` ou `pipeline/debug_tools/bbox.py`
- Testar: `pipeline/tests/test_main_emit.py` ou novo teste focado

**Solucao:**

Antes de `sync_final_page_space_typeset`, validar cada page text:

- Se `target_bbox` ou `balloon_bbox` tem y global e `safe_text_box/render_bbox/bubble_inner_bbox` tem y local, marcar:
  - `page_space_rerender_mixed_coordinates`
  - `layout_bbox_coordinate_mismatch`
- Nao renderizar a pagina com geometria mista.

Depois da Task 2, esse bloqueio deve ser apenas rede de seguranca.

**Validacao:**

Criar teste que monta `page_texts` mistos e confirma que o rerender nao acontece ou que a flag critica e propagada.

### Task 5: Recalcular/normalizar safe boxes apos remapeamento

**Problema a resolver:** mesmo com `bubble_inner_bbox` shiftado, `safe_text_box` e `render_bbox` antigos podem ficar obsoletos.

**Arquivos:**

- Modificar: `pipeline/typesetter/renderer.py`
- Modificar: `pipeline/strip/run.py`
- Testar: `pipeline/tests/test_typesetting_renderer.py`

**Solucao:**

1. Quando a coordenada final for pagina, `render_band_image` deve rejeitar safe area fora da vizinhanca do `target_bbox`.
2. Se `safe_text_box` nao intersecta `target_bbox`, ignorar safe box antigo e recalcular a partir de `balloon_bbox`/`bubble_inner_bbox` correto.
3. Registrar em `_render_debug`:
   - `layout_safe_reason`
   - `coordinate_space_validation`
   - `recomputed_safe_box: true/false`

**Validacao:**

Teste com `target_bbox` global e safe box local deve gerar safe box global correta.

### Task 6: Tornar fast solid verificavel antes de remover blocos

**Problema a resolver:** fast solid nao pode pular inpaint real com residuos.

**Arquivos:**

- Modificar: `pipeline/inpainter/__init__.py`
- Testar: `pipeline/tests/test_vision_stack_inpainter.py`

**Solucao:**

Alterar `_apply_fast_solid_balloon_fill`:

1. Construir mascara de preenchimento a partir de `line_polygons` dilatada por 3-6 px, limitada ao interior do balao.
2. Comparar coverage contra `text_pixel_bbox`.
3. Se coverage for baixo, rejeitar fast solid com motivo `insufficient_fast_solid_text_coverage`.
4. Depois de preencher, rodar residual check na regiao de texto expandida.
5. Se houver residual, nao remover o bloco de `vision_blocks`; deixar ir para inpaint real.

**Validacao:**

Teste deve cobrir:

- fast solid bom em balao branco liso.
- fast solid rejeitado quando mascara cobre pouco.
- bloco permanece em `vision_blocks` quando residual check falha.

### Task 7: Corrigir metadata de inpaint para nao mascarar falhas

**Problema a resolver:** hoje `raw_mask_pixels: 0` e `expanded_mask_pixels: 0` coexistem com `used_fast_solid_fill: true`, e o QA nao bloqueia.

**Arquivos:**

- Modificar: `pipeline/inpainter/__init__.py`
- Modificar: `pipeline/debug_tools/masks.py`
- Testar: `pipeline/tests/test_mask_chain_debug.py`

**Solucao:**

Quando fast fill roda sem mascara raw/expanded:

- Se residual check nao foi executado com sucesso, adicionar `fast_fill_unverified_residual`.
- Se residual check detectou texto, adicionar `text_residual_after_inpaint`.
- Se fast solid foi aceito, gravar `fast_fill_verified: true`.

**Validacao:**

`inpaint_decision.json` precisa diferenciar:

- fast fill verificado
- fast fill suspeito
- fallback para inpaint real

### Task 8: Propagar flags criticas do debug para project.json

**Problema a resolver:** debug detecta warning, mas projeto/gate nao bloqueiam.

**Arquivos:**

- Modificar: `pipeline/main.py`
- Modificar: `pipeline/qa/export_gate.py`
- Modificar: `pipeline/qa/translation_qa.py`
- Testar: `pipeline/tests/test_qa_flag_propagation_v2.py`

**Solucao:**

Adicionar critical flags especificas:

- `page_space_rerender_mixed_coordinates`
- `fast_fill_unverified_residual`
- `text_residual_after_inpaint`

Garantir que essas flags aparecam em:

- layer `qa_flags`
- `qa.summary`
- `qa.export_gate`
- `debug/e2e/11_qa_export_gate/visual_blockers.jsonl`

### Task 9: Corrigir rastreabilidade de render blocks agrupados

**Problema a resolver:** bloco renderizado nao corresponde claramente a todas as OCR layers originais.

**Arquivos:**

- Modificar: `pipeline/typesetter/renderer.py`
- Modificar: `pipeline/main.py`
- Testar: `pipeline/tests/test_render_plan_trace_integrity.py` ou criar novo teste

**Solucao:**

Para cada render block final:

- Preservar `source_trace_ids`.
- Se render consolidado cobre varios textos, propagar `render_bbox` e `qa_flags` para todos os filhos, ou criar layer consolidada unica no `project.json`.
- Garantir que `render_plan_final.jsonl` e `project.json` tenham os mesmos `trace_ids` e bboxes finais.

**Validacao:**

Teste com `page_003_band_035` simulado:

- `ocr_001` + `ocr_003`
- render block consolidado
- `project.json` rastreia ambos sem perder `render_bbox`.

### Task 10: Reexecutar Chapter 1 com debug completo

**Problema a resolver:** provar que a correcao resolve a run real.

**Entrada:**

- `C:/Users/PICHAU/Downloads/Chapter 1`

**Saida sugerida:**

- `N:/TraduzAI/DEBUGM/runs/2026-05-25_chapter1_geometry_inpaint_fix_v1`

**Comando base:**

Usar o mesmo runner/config da run analisada, mantendo:

- `runtime_profile: balanced`
- `TRADUZAI_STRIP_FAST_SOLID_INPAINT=1`
- `TRADUZAI_STRIP_FAST_WHITE_INPAINT=0`
- `TRADUZAI_STRIP_FAST_DARK_PANEL_FILL=0`
- `TRADUZAI_PAGE_CLEANUP_RERENDER=0`
- `TRADUZAI_PADDLE_FULL_PAGE=1`
- `TRADUZAI_STRIP_DETECT_FULL_PAGE=1`

**Validacao visual obrigatoria:**

Comparar:

- `translated/001.jpg`
- `translated/002.jpg`
- `translated/003.jpg`
- `translated/006.jpg`

Checar especificamente:

- `POR FAVOR, PELO BEM DA CRIANCA.` no balao correto, nao no topo.
- `ESTOU MORRENDO DE FOME...` sem texto ingles residual.
- Balao duplo conectado sem `SOME.` residual.
- Caixas azuis/pretas com texto apagado e texto novo no lugar correto.

**Validacao automatica:**

- `bbox_coordinate_audit.json` deve ter `all_consistent: true`.
- `qa_issues.jsonl` nao deve conter coordenada mista.
- `visual_blockers.jsonl` deve ficar vazio apenas se realmente nao houver blocker.
- `qa_report.json` deve ser `PASS` sem blockers criticos.

### Task 11: Reexecutar matriz curta de regressao

**Problema a resolver:** garantir que a correcao nao vale so para Chapter 1.

**Capitulos:**

- `C:/Users/PICHAU/Downloads/Chapter 1`
- `C:/Users/PICHAU/Downloads/Articuno (comick)_Ch. 61 OFFICIAL TRANSLATION`
- `C:/Users/PICHAU/Downloads/Chapter 39`
- `D:/Mihon pra pc/downloads/mangas/Manhwatop (EN)/The God of Death/Chapter 2.cbz`
- Primeiros 2 capitulos de `D:/Mihon pra pc/downloads/mangas/Manhwatop (EN)/1 Second`

**Validacao:**

Gerar contact sheets finais e revisar:

- baloes brancos
- baloes conectados
- caixas pretas/azuis
- textos inclinados/rotacionados
- signos/texto em fundo solido

### Task 12: Atualizar documentacao de debug

**Problema a resolver:** tornar o bug rastreavel se voltar.

**Arquivos:**

- Modificar: `docs/debug/e2e_pipeline_debug_guide.md`

**Adicionar:**

- Como identificar coordenada mista.
- Quais chaves precisam estar em page-space.
- Como ler `render_plan_final.jsonl`.
- Como interpretar `fast_fill_without_raw_mask`.
- Quando `PASS` com warnings ainda deve ser tratado como falha visual.

## Ordem Recomendada

1. Task 1-3: corrigem e provam geometria.
2. Task 4-5: protegem o rerender final.
3. Task 6-7: corrigem residuos de inpaint/fast solid.
4. Task 8-9: fazem QA e rastreabilidade parar de mentir.
5. Task 10-11: validam visualmente em capitulos reais.
6. Task 12: documenta o processo.

## Criterios De Pronto

- Nenhuma layer final com `target_bbox` global e `safe_text_box/render_bbox/bubble_inner_bbox` local.
- `bbox_coordinate_audit.json` pega qualquer regressao desse tipo.
- `translated/*.jpg` bate com `render_plan_final.jsonl`.
- Fast solid nao deixa residuos evidentes de texto.
- Blocos agrupados mantem `source_trace_ids` e render bbox rastreavel.
- `visual_blockers.jsonl` bloqueia falhas reais em debug.
- Chapter 1 passa na comparacao visual dos pontos reportados pelo usuario.

## Riscos

- Shiftar `bubble_*` pode revelar testes antigos que assumiam coordenada local depois do remapeamento. Esses testes devem ser atualizados para separar claramente banda-local de pagina-final.
- Fast solid mais conservador pode aumentar tempo de inpaint em alguns casos. O ganho correto e qualidade primeiro; performance pode ser recuperada depois com cache e coalescing seguro.
- Promover flags para critical indiscriminadamente pode bloquear demais. Por isso o plano cria flags criticas especificas para coordenada mista e residual verificado.
