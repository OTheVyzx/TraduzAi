# Pure Inpaint, Band Ownership and Card Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Tornar o pipeline automático atomicamente seguro, sem fills sólidos no modo padrão, sem duplicação entre bandas e com cards legíveis e completos.

**Architecture:** A correção preserva o pipeline por bandas, mas adiciona uma arbitragem em coordenadas de página antes da tradução e um retry adaptativo para cards cortados. O inpaint passa a obedecer uma política única `pure`, o renderer resolve cards como grupos hierárquicos e o QA visual final passa a alimentar o export gate.

**Tech Stack:** Python 3.12, NumPy, OpenCV, PIL, AOT/LaMA/Telea, pytest, artefatos JSON/JSONL do pipeline automático.

---

## Regras de execução

- Use `@mangatl-dev`, `@traduzai-detect`, `@traduzai-ocr`, `@traduzai-inpaint`, `@traduzai-typesetting` e `@traduzai-pipeline` conforme o owner de cada tarefa.
- Preserve todas as alterações preexistentes. Os arquivos deste plano já possuem mudanças locais extensas; leia o diff relevante antes de cada patch.
- Não use reset, checkout destrutivo ou limpeza ampla de `DEBUGM/` e `.codex-tmp/`.
- Escreva primeiro o teste que reproduz o defeito e confirme a falha antes do patch.
- Faça commits somente dos arquivos listados na tarefa corrente.
- Não paralelize FT2Font/typesetting.
- Não declare sucesso com testes auxiliares apenas; confira os artefatos visuais do run final.
- Use um `work_dir` novo para cada validação E2E, evitando artefatos antigos.

## Baseline conhecido

Run de reprodução:

```text
N:\TraduzAI\.codex-tmp\mythic_ch40_card_resolved4_full_20260724
```

Baseline focado executado em 2026-07-24:

```text
3 passed, 1 failed
```

Falha existente:

```text
tests/test_typesetting_renderer.py::TypesettingRendererTests::test_render_text_block_flags_below_minimum_legible_and_blocks_export
Expected: BLOCK
Actual: PASS
```

### Task 1: Fixar o baseline e a matriz de casos

**Files:**
- Create: `pipeline/tests/fixtures/mythic_ch40_visual_cases.json`
- Modify: `pipeline/tests/regression/test_visual_regression_manifest.py`
- Read: `.codex-tmp/mythic_ch40_card_resolved4_full_20260724/debug/e2e/`

**Step 1: Criar o manifesto dos casos**

Registrar, sem copiar imagens para o Git, estas expectativas:

```json
{
  "run_id": "2026-07-24T02-43-27Z_regressed_genius_creates_mythic_items_ea1037",
  "bad_bands": [25, 26, 28, 32, 40, 41, 42, 43, 45],
  "control_bands": [35, 39],
  "required_issue_codes": [
    "cross_band_duplicate",
    "render_bbox_missing",
    "fit_below_minimum_legible",
    "inpaint_texture_flattened"
  ]
}
```

**Step 2: Adicionar teste de schema do manifesto**

O teste deve exigir IDs únicos, disjunção entre casos ruins/controles e códigos não vazios.

**Step 3: Rodar o teste**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/regression/test_visual_regression_manifest.py -q
```

Expected: PASS.

**Step 4: Commit**

```powershell
git add -- pipeline/tests/fixtures/mythic_ch40_visual_cases.json pipeline/tests/regression/test_visual_regression_manifest.py
git commit -m "test: record mythic chapter visual failure matrix"
```

### Task 2: Estender a arbitragem de ownership entre bandas

**Files:**
- Modify: `pipeline/strip/run.py:59-162`
- Modify: `pipeline/tests/test_strip_ocr_debug_artifacts.py:126`

**Step 1: Escrever testes que falham**

Adicionar:

```python
def test_reconcile_overlapping_bands_suppresses_equivalent_full_duplicate():
    # DID IT GO WELL? versus DID IT GO well?, bboxes quase iguais.
    # Apenas o owner mais central/suportado deve sobreviver.


def test_reconcile_overlapping_bands_quarantines_unsupported_edge_fragment():
    # OCR em y=0 sem overlap com candidato detectado da banda.
    # O registro não pode seguir para tradução/inpaint.


def test_reconcile_overlapping_bands_keeps_repeated_text_in_distinct_geometry():
    # Mesma fala em balões distantes não pode ser deduplicada.
```

**Step 2: Confirmar as falhas**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_strip_ocr_debug_artifacts.py -k "equivalent_full_duplicate or unsupported_edge_fragment or repeated_text_in_distinct_geometry" -q
```

Expected: 3 FAIL.

**Step 3: Implementar score de owner em page-space**

Em `pipeline/strip/run.py`, manter o caminho de prefixo existente e adicionar uma segunda decisão estreita:

```python
owner_score = (
    detector_support * 4.0
    + bbox_interior_ratio * 2.0
    + ocr_confidence
    + completeness_score
    - edge_touch_penalty * 3.0
)
```

Requisitos:

- comparar somente bandas sobrepostas da mesma página;
- exigir forte overlap geométrico para duplicata textual;
- normalizar caixa, espaços e pontuação;
- registrar `cross_band_owner_trace_id` no vencedor;
- registrar o perdedor em `cross_band_suppressed_trace_ids`;
- remover o perdedor de `texts` e do `_vision_blocks` correspondente;
- para fragmento de borda sem candidato, usar `route_action: review_required`, `route_reason: cross_band_unsupported_edge_fragment` e impedir ação destrutiva.

**Step 4: Rodar testes de reconciliação**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_strip_ocr_debug_artifacts.py -k "reconcile_overlapping or candidate_text_matching" -q
```

Expected: PASS, incluindo o teste de prefixo já existente.

**Step 5: Commit**

```powershell
git add -- pipeline/strip/run.py pipeline/tests/test_strip_ocr_debug_artifacts.py
git commit -m "fix: arbitrate OCR ownership across overlapping bands"
```

### Task 3: Adicionar recorte adaptativo para cards tocando a borda

**Files:**
- Modify: `pipeline/strip/run.py:5049-5353`
- Modify: `pipeline/strip/bands.py:10-68`
- Modify: `pipeline/vision_stack/runtime.py:13520-13642`
- Modify: `pipeline/tests/test_strip_ocr_debug_artifacts.py`
- Modify: `pipeline/tests/test_strip_process_bands.py`

**Step 1: Escrever testes que falham**

Cobrir:

```python
def test_visual_card_edge_recall_expands_only_own_band_and_keeps_id(): ...
def test_visual_card_retry_is_limited_to_once(): ...
def test_visual_card_expansion_stops_at_source_page_boundary(): ...
def test_normal_speech_band_does_not_receive_card_expansion(): ...
```

**Step 2: Confirmar falha**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_strip_ocr_debug_artifacts.py pipeline/tests/test_strip_process_bands.py -k "visual_card_edge or card_expansion" -q
```

Expected: FAIL.

**Step 3: Implementar a decisão de expansão**

Adicionar helper puro que receba banda, page bounds e OCR result. Ele deve retornar `None` ou novos `y_top/y_bottom` quando:

- o registro tem `visual_card_ocr_recall` ou card context confiável;
- bbox toca até 8 px da borda;
- há cluster visual de card;
- a banda ainda não possui `_adaptive_edge_retry_done`.

Expandir 160 px adicionais apenas na direção necessária, reaplicar `attach_band_slices` naquele band e repetir OCR/review uma vez antes da tradução.

Persistir:

```text
adaptive_original_y_top
adaptive_original_y_bottom
adaptive_final_y_top
adaptive_final_y_bottom
adaptive_edge_retry_reason
```

**Step 4: Rodar testes focados de detect/OCR**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_strip_detect.py pipeline/tests/test_strip_ocr_debug_artifacts.py pipeline/tests/test_strip_process_bands.py -k "band_to_page or card or edge or overlap" -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add -- pipeline/strip/run.py pipeline/strip/bands.py pipeline/vision_stack/runtime.py pipeline/tests/test_strip_ocr_debug_artifacts.py pipeline/tests/test_strip_process_bands.py
git commit -m "fix: adapt band crops for edge-clipped visual cards"
```

### Task 4: Tornar `pure` a política padrão de inpaint

**Files:**
- Modify: `pipeline/runtime_profiles.py`
- Modify: `pipeline/main.py:8243-8248`
- Modify: `pipeline/inpainter/__init__.py:1335-1368`
- Modify: `pipeline/inpainter/__init__.py:10924-12012`
- Modify: `pipeline/strip/run.py:3214-3318`
- Modify: `pipeline/tests/test_runtime_profiles.py`
- Modify: `pipeline/tests/test_vision_stack_inpainter.py`

**Step 1: Escrever testes da política**

Adicionar:

```python
def test_default_runtime_profile_selects_pure_inpaint(): ...
def test_explicit_fast_profile_does_not_override_pure_without_opt_in(): ...
def test_pure_mode_skips_visual_item_card_forced_cleanup(): ...
def test_pure_mode_skips_dark_geometry_solid_fill(): ...
def test_pure_mode_disables_metadata_and_white_post_fill_defaults(): ...
```

**Step 2: Confirmar falhas**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_runtime_profiles.py pipeline/tests/test_vision_stack_inpainter.py -k "pure_mode or pure_inpaint" -q
```

Expected: FAIL.

**Step 3: Criar fonte única da política**

Usar:

```text
TRADUZAI_INPAINT_POLICY=pure
```

`runtime_profiles.py` deve aplicar `pure` quando a configuração não declarar política. Uma função única em `inpainter/__init__.py` deve responder se alterações diretas são permitidas.

No modo `pure`, transformar em no-op todos os caminhos de atribuição sólida, incluindo:

- `_apply_fast_solid_balloon_fill`;
- `_apply_fast_white_balloon_fill`;
- `_apply_fast_dark_panel_text_fill`;
- `_apply_fast_local_balloon_fill`;
- `_apply_dark_panel_text_fills`;
- `_apply_visual_item_card_contract_cleanup`;
- `_apply_dark_visual_text_geometry_cleanup`;
- `text_contract_direct_fill` e force-fill residual.

Não basta alterar os defaults de ambiente. Os próprios helpers precisam honrar a política para cobrir todos os callers.

**Step 4: Preservar AOT/LaMA/Telea**

Confirmar que `pure` ainda:

- chama AOT/LaMA quando a action mask é segura;
- permite Telea restrito como fallback;
- mantém o clamp fora da expanded mask;
- registra `inpaint_policy`, `engine` e `fallback_used` no decision artifact.

**Step 5: Rodar testes de inpaint**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest tests/test_runtime_profiles.py tests/test_vision_stack_inpainter.py -k "pure or final_inpaint_clamp or translucent_balloon or visual_item_card" -q
} finally { Pop-Location }
```

Expected: PASS. Atualizar o teste antigo que exigia forced solid cleanup para exigir zero alterações sólidas em `pure`, mantendo uma variante explícita somente se um perfil fast opt-in continuar suportado.

**Step 6: Commit**

```powershell
git add -- pipeline/runtime_profiles.py pipeline/main.py pipeline/inpainter/__init__.py pipeline/strip/run.py pipeline/tests/test_runtime_profiles.py pipeline/tests/test_vision_stack_inpainter.py
git commit -m "fix: make pure model inpaint the automatic default"
```

### Task 5: Detectar flattening e garantir máscara estrita

**Files:**
- Modify: `pipeline/inpainter/__init__.py`
- Modify: `pipeline/qa/translation_qa.py`
- Modify: `pipeline/qa/export_gate.py`
- Modify: `pipeline/tests/test_inpaint_debug_residual.py`
- Modify: `pipeline/tests/test_export_gate.py`

**Step 1: Escrever teste sintético de gradiente**

Criar imagem com gradiente/textura, glifos centrais e um resultado artificialmente preenchido por cor única. O detector deve produzir `inpaint_texture_flattened`.

Também criar um resultado AOT/Telea com continuidade suficiente que não deve ser marcado.

**Step 2: Confirmar falha**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_inpaint_debug_residual.py pipeline/tests/test_export_gate.py -k "texture_flattened" -q
```

Expected: FAIL.

**Step 3: Implementar métricas**

Dentro da action mask e de um anel local, comparar before/after:

- desvio-padrão de luminância;
- magnitude média de gradiente;
- número de cores/quantização;
- presença de bordas retangulares coincidentes com bbox.

Marcar somente quando houver queda forte de textura e grande região quase uniforme. Persistir as métricas no `inpaint_decision.json`.

**Step 4: Tornar a flag crítica**

Adicionar `inpaint_texture_flattened` às severidades críticas e aos links para before/mask/after.

**Step 5: Rodar testes**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_inpaint_debug_residual.py pipeline/tests/test_export_gate.py -k "texture or residual or fast_fill" -q
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add -- pipeline/inpainter/__init__.py pipeline/qa/translation_qa.py pipeline/qa/export_gate.py pipeline/tests/test_inpaint_debug_residual.py pipeline/tests/test_export_gate.py
git commit -m "test: block flattened inpaint output"
```

### Task 6: Implementar atomicidade entre limpeza e render

**Files:**
- Modify: `pipeline/strip/process_bands.py:7702-8279`
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/tests/test_strip_process_bands.py`
- Modify: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Escrever testes que falham**

Cobrir:

```python
def test_failed_render_restores_original_pixels_for_trace_mask(): ...
def test_unsafe_mask_never_leaves_empty_dark_rectangle(): ...
def test_successful_render_keeps_cleaned_pixels_and_translation(): ...
```

**Step 2: Confirmar falhas**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_strip_process_bands.py pipeline/tests/test_typesetting_renderer.py -k "restores_original_pixels or empty_dark_rectangle or successful_render_keeps" -q
```

Expected: FAIL.

**Step 3: Expor resultado final por trace**

O typesetter deve registrar para cada trace:

```text
render_completed
render_bbox
fit_status
minimum_legible_font_px
```

No copyback, para trace sem render seguro, restaurar da `band.original_slice` somente a action mask daquele trace. Marcar:

```text
route_action: review_required
route_reason: atomic_inpaint_render_rollback
qa_flags: [pure_inpaint_unresolved]
```

**Step 4: Rodar testes de estágio**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_strip_process_bands.py pipeline/tests/test_typesetting_renderer.py -k "copy_back or rollback or minimum_legible" -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add -- pipeline/strip/process_bands.py pipeline/typesetter/renderer.py pipeline/tests/test_strip_process_bands.py pipeline/tests/test_typesetting_renderer.py
git commit -m "fix: make inpaint and translated render atomic"
```

### Task 7: Resolver layout hierárquico de cards

**Files:**
- Modify: `pipeline/layout/balloon_layout.py:999-1082`
- Modify: `pipeline/typesetter/renderer.py:6261-6391`
- Modify: `pipeline/tests/test_typesetting_layout.py`
- Modify: `pipeline/tests/test_typesetting_renderer.py:1313`

**Step 1: Escrever testes de papéis e layout**

Cobrir card completo com título, nota, corpo e rodapé:

```python
def test_item_card_assigns_stable_roles_without_merging_payloads(): ...
def test_item_card_joint_layout_preserves_all_translated_text(): ...
def test_item_card_title_never_renders_below_minimum(): ...
def test_item_card_rows_do_not_overlap_after_ptbr_expansion(): ...
```

**Step 2: Confirmar falhas**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_typesetting_layout.py pipeline/tests/test_typesetting_renderer.py -k "item_card.*role or item_card.*joint or item_card.*minimum or item_card.*overlap" -q
```

Expected: FAIL.

**Step 3: Classificar filhos no layout**

Em `_assign_visual_item_card_groups`, preservar textos independentes e adicionar `card_panel_role`. Usar ordem vertical e proporção geométrica como evidência primária; conteúdo textual é apenas desempate conservador.

**Step 4: Substituir slots independentes por solver conjunto**

O solver deve:

- trabalhar no `card_panel_bbox` completo;
- reservar áreas por papel;
- manter o centro horizontal do original;
- preservar a ordem vertical;
- manter um gap mínimo;
- testar wrapping/tamanho para todos os filhos antes de aceitar;
- falhar o grupo inteiro quando um filho obrigatório ficar ilegível.

**Step 5: Rodar testes de renderer**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_typesetting_layout.py pipeline/tests/test_typesetting_renderer.py -k "visual_item_card or below_minimum or wraps_long_text" -q
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add -- pipeline/layout/balloon_layout.py pipeline/typesetter/renderer.py pipeline/tests/test_typesetting_layout.py pipeline/tests/test_typesetting_renderer.py
git commit -m "fix: lay out item cards as hierarchical groups"
```

### Task 8: Preservar o contrato de legibilidade até o export gate

**Files:**
- Modify: `pipeline/typesetter/renderer.py:13101-13215`
- Modify: `pipeline/main.py:1869-1959`
- Modify: `pipeline/main.py:6996-7016`
- Modify: `pipeline/qa/export_gate.py:664-702`
- Modify: `pipeline/tests/test_typesetting_renderer.py:2945`
- Modify: `pipeline/tests/test_main_emit.py`
- Modify: `pipeline/tests/test_qa_flag_propagation_v2.py`

**Step 1: Manter o teste de baseline vermelho**

Rodar:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_typesetting_renderer.py::TypesettingRendererTests::test_render_text_block_flags_below_minimum_legible_and_blocks_export -q
```

Expected antes do patch: FAIL, `PASS != BLOCK`.

**Step 2: Adicionar teste de round-trip**

Um plano com `font_size_final=6`, `minimum_legible_font_px=12` e `fit_status=below_minimum_legible` deve preservar os três campos em:

```text
render_plan_raw -> project layer -> render_plan_final -> export gate
```

**Step 3: Corrigir hidratação**

- Não criar tentativa `ok` somente porque existe `render_bbox`.
- Não remover `fit_below_minimum_legible` por contenção geométrica.
- Copiar `font_size_final` e `minimum_legible_font_px` ao project layer.
- Remover a flag apenas quando um novo fit real, com tamanho suficiente, tiver sido calculado.

**Step 4: Corrigir severidade**

`fit_below_minimum_legible` não resolvido deve permanecer crítico. Exceções de nota do tradutor só podem ser review quando o tamanho final ainda respeitar o mínimo específico de nota.

**Step 5: Rodar testes**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_typesetting_renderer.py::TypesettingRendererTests::test_render_text_block_flags_below_minimum_legible_and_blocks_export pipeline/tests/test_main_emit.py pipeline/tests/test_qa_flag_propagation_v2.py -k "fit_below_minimum or render_metadata" -q
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add -- pipeline/typesetter/renderer.py pipeline/main.py pipeline/qa/export_gate.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_main_emit.py pipeline/tests/test_qa_flag_propagation_v2.py
git commit -m "fix: preserve minimum legibility through export QA"
```

### Task 9: Integrar o QA visual final ao export gate

**Files:**
- Modify: `pipeline/main.py:7835-8161`
- Modify: `pipeline/qa/export_gate.py:851-970`
- Modify: `pipeline/tests/test_main_emit.py:1608-1784`
- Modify: `pipeline/tests/test_export_gate.py`

**Step 1: Escrever testes que falham**

Adicionar casos para:

```python
def test_export_gate_blocks_final_visual_render_bbox_missing(): ...
def test_export_gate_blocks_translated_crop_mismatch(): ...
def test_export_gate_links_visual_failure_to_trace_ids(): ...
def test_visual_control_band_pass_does_not_create_issue(): ...
```

**Step 2: Confirmar falhas**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_main_emit.py pipeline/tests/test_export_gate.py -k "final_visual.*export_gate or translated_crop_mismatch" -q
```

Expected: FAIL.

**Step 3: Consumir rows antes do gate**

`collect_export_blocking_issues` deve ler `project.qa.post_rerender_final_visual_contract.qa.rows`, derivar página pelo `band_id`, resolver layers pelos `trace_ids` e criar issues deduplicadas.

Não duplicar uma issue já presente no layer. Manter no payload:

```text
source: post_rerender_final_visual_contract
band_id
trace_ids
flags
metrics
artifact_links
```

**Step 4: Rodar testes de QA**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_main_emit.py pipeline/tests/test_export_gate.py -k "final_rerender_visual or export_gate" -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add -- pipeline/main.py pipeline/qa/export_gate.py pipeline/tests/test_main_emit.py pipeline/tests/test_export_gate.py
git commit -m "fix: enforce final visual QA at export gate"
```

### Task 10: Corrigir cor e identidade dos artefatos de debug

**Files:**
- Modify: `pipeline/inpainter/__init__.py:7496-7511`
- Modify: `pipeline/debug_tools/recorder.py:90-104`
- Modify: `pipeline/tests/test_inpaint_debug_residual.py`
- Modify: `pipeline/tests/test_strip_ocr_debug_artifacts.py`

**Step 1: Escrever teste RGB/BGR**

Salvar a mesma matriz colorida pelos dois caminhos e reler com OpenCV/PIL. Os pixels devem representar as mesmas cores.

**Step 2: Escrever teste de proveniência**

`debug_inpaint/metadata.json` deve conter `run_id`, `band_id`, `color_space` e dimensões. Artefato com run ID diferente não pode ser usado pelo QA atual.

**Step 3: Confirmar falhas**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_inpaint_debug_residual.py pipeline/tests/test_strip_ocr_debug_artifacts.py -k "color_space or debug_provenance" -q
```

Expected: FAIL.

**Step 4: Unificar escrita**

Escolher uma convenção explícita para arrays internos e converter somente na fronteira do writer. Não manter um helper chamado `_save_rgb` se o caller fornece BGR sem conversão.

**Step 5: Rodar testes**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_inpaint_debug_residual.py pipeline/tests/test_strip_ocr_debug_artifacts.py -k "color_space or debug_provenance or visual_debug" -q
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add -- pipeline/inpainter/__init__.py pipeline/debug_tools/recorder.py pipeline/tests/test_inpaint_debug_residual.py pipeline/tests/test_strip_ocr_debug_artifacts.py
git commit -m "fix: make inpaint debug colors and provenance reliable"
```

### Task 11: Executar a suíte focada integrada

**Files:**
- Read: todos os arquivos modificados nas Tasks 1-10

**Step 1: Rodar testes dos owners**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_strip_ocr_debug_artifacts.py `
    tests/test_strip_process_bands.py `
    tests/test_runtime_profiles.py `
    tests/test_vision_stack_inpainter.py `
    tests/test_inpaint_debug_residual.py `
    tests/test_typesetting_layout.py `
    tests/test_typesetting_renderer.py `
    tests/test_export_gate.py `
    tests/test_qa_flag_propagation_v2.py `
    -q
} finally { Pop-Location }
```

Expected: PASS.

**Step 2: Rodar testes do contrato principal**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_main_emit.py -k "final_rerender_visual or export_gate or render_metadata" -q
```

Expected: PASS.

**Step 3: Validar skills e diff**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Test-TraduzAISkills.ps1
git diff --check
```

Expected: PASS, sem novos erros de whitespace.

### Task 12: Rodar o capítulo 40 em workdir novo

**Files:**
- Create: `.codex-tmp/mythic_ch40_pure_inpaint_validation_20260724_config.json`
- Generate: `.codex-tmp/mythic_ch40_pure_inpaint_validation_20260724/`

**Step 1: Criar configuração de validação**

Usar a mesma fonte e opções do run de reprodução, mudando somente `work_dir` e adicionando:

```json
"inpaint_policy": "pure"
```

**Step 2: Executar**

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
$env:TRADUZAI_SKIP_LOCAL_VENV_REEXEC='1'
pipeline\venv\Scripts\python.exe pipeline\main.py .codex-tmp\mythic_ch40_pure_inpaint_validation_20260724_config.json
```

Expected: execução técnica completa e `project.json` persistido. `BLOCK` é aceitável somente se houver issue real ainda não corrigida; não interpretar exit 0 como aprovação visual.

**Step 3: Auditar contratos automaticamente**

Exigir:

```text
fast_fill_mask_pixels == 0
dark_panel_fill_count == 0
visual_item_card_forced_contract_cleanup ausente
fit_below_minimum_legible ausente em layers visíveis
inpaint decision presente para todo layer renderizável
nenhuma colisão cross-band
```

**Step 4: Inspecionar visualmente**

Gerar/abrir sheet de antes/depois para bands:

```text
25, 26, 28, 32, 40, 41, 42, 43, 45
```

Verificar controles:

```text
35, 39
```

Critérios:

- `CORREU BEM?` aparece uma vez;
- `IMLAK...` não aparece;
- nenhum retângulo vazio;
- cards sem faixas sólidas;
- título legível e próximo do tamanho original;
- todos os textos do card presentes;
- arte fora da máscara preservada.

**Step 5: Registrar relatório**

Criar relatório em:

```text
docs/reports/2026-07-24-mythic-ch40-pure-inpaint-validation.md
```

Incluir caminhos absolutos Windows dos outputs e resultados do QA.

**Step 6: Commit do relatório e ajustes finais já validados**

Adicionar somente os arquivos de código/teste/documentação intencionais. Não adicionar `.codex-tmp`.

### Task 13: Executar regressão cruzada em outras categorias

**Files:**
- Read: casos conhecidos de balão translúcido, texto sobre arte e cards dos capítulos já usados
- Modify: `docs/reports/2026-07-24-mythic-ch40-pure-inpaint-validation.md`

**Step 1: Selecionar casos existentes**

Incluir pelo menos:

- balão branco simples;
- balão translúcido;
- texto sobre arte;
- painel escuro;
- card colorido;
- referência positiva `case16_final_validation_20260722`;
- referência positiva `case41_typeset_validation_20260722`.

**Step 2: Rodar sem substituir outputs anteriores**

Cada caso recebe diretório novo e config própria.

**Step 3: Comparar QA e imagem**

Falhar a validação se:

- texto original continuar visível;
- houver alteração fora da máscara;
- aparecer região mais chapada que o original;
- texto novo sair da safe box;
- qualquer layer perder conteúdo;
- caso positivo piorar visualmente.

**Step 4: Atualizar relatório e commit**

```powershell
git add -- docs/reports/2026-07-24-mythic-ch40-pure-inpaint-validation.md
git commit -m "docs: validate pure inpaint across visual categories"
```

## Encerramento

Antes de declarar conclusão:

1. confirmar todos os nodeids focados verdes;
2. confirmar o teste de baseline `below_minimum_legible` verde;
3. confirmar o run novo visualmente;
4. confirmar ausência de fills sólidos em metadata;
5. confirmar export gate coerente com o QA visual;
6. listar alterações preexistentes ainda presentes no checkout;
7. fornecer links absolutos Windows para relatório, sheet e imagens finais.
