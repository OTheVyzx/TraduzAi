---
name: traduzai-inpaint
description: Use when TraduzAI leaves source glyphs, erases line art, smears textures, paints the wrong balloon, changes pixels outside a mask, or chooses the wrong white, dark, translucent, or textured cleanup strategy.
---

# TraduzAI Inpaint

## Owner e fluxo ativo

Owner ativo: `pipeline/inpainter/__init__.py`, `pipeline/inpainter/mask_builder.py`, `pipeline/vision_stack/inpainter.py` e `pipeline/vision_stack/runtime.py`. `process_bands.py` contém o wrapper.

`pipeline/main.py` → `pipeline/strip/run.py::run_chapter` → `process_band` → `_run_inpaint_stage` → `inpaint_band_image`.

Entrada: banda RGB e página traduzida com `texts`, `_vision_blocks`, bboxes, geometria/linha, perfil, `route_action` e evidência de máscara. Saída: banda limpa, contratos `texts`/`_vision_blocks` atualizados e métricas `_strip_*`. Typesetting/render, copyback e QA consomem o resultado.

## Cadeia de máscaras e proteção

- **raw mask:** evidência de glifo antes da expansão.
- **expanded/final action mask:** máscara enviada ao modelo e ao clamp principal; retries/reconstruções podem ampliá-la.
- **effective limit mask:** calculada depois para cleanup e diagnóstico, não para a inferência principal.

A cadeia ativa é: raw → `_augment_inpaint_masks_from_texts` → `_constrain_translucent_balloon_action_masks` → `_expand_strip_real_inpaint_mask`/`expand_text_mask` → `_constrain_translucent_balloon_action_masks` → modelo → clamp/reconstrução. Clips de balão, line polygons e densidade protegem arte. `protection_mask` em `06_mask_segmentation` é debug derivado; **não é entrada do modelo**.

## Estratégias

- Branco/solid seguro: fast white/solid fill ou Telea local; preserve contorno.
- Escuro/colorido: use evidência de glifo e amostra local; não transforme painel em retângulo sólido sem contrato confiável.
- Translúcido/texturizado/gradiente: evite fill opaco; use reconstrução/continuação local restrita à action mask.
- O preset seleciona `aot-inpainting` (default) ou `lama-manga`. LaMA prefere sessão ONNX; carregadores/checkpoints alternativos são tentados e, se o modelo falhar, fallbacks clássicos/OpenCV Telea podem limpar localmente com qualidade inferior.
- Fallbacks também podem reconstruir `_vision_blocks` de `texts` ou máscara por geometria; sem máscara confiável, preserve arte e registre skip/review.

## Quick reference: primeira divergência

| Primeira evidência incorreta | Investigue |
|---|---|
| raw mask em `debug/e2e/06_mask_segmentation` | OCR/line geometry ou `mask_builder` |
| raw correta, expanded/action larga ou curta | augment, constraint, expansão e perfil |
| clamp/cleanup diverge | final action, effective limit e fallback |
| máscaras corretas, decisão/resultado errado em `08_inpaint` | fast fill, real inpaint, reconstrução local |
| `08_inpaint` correto, imagem final errada | typesetting/copyback/QA; use `traduzai-pipeline` + especialista |

`debug_inpaint` guarda before/raw/expanded/effective/after. `06_mask_segmentation` mostra cadeia global/per-texto; `08_inpaint`, decisão, engine, skips e pixels alterados.

## Fronteiras e diagnóstico

Detect é dona da geometria candidata; OCR, do texto/rota/evidência; inpaint, da máscara e reconstrução; typesetting, do texto novo. Em falha multietapa automática, use `traduzai-pipeline` com a especialista afetada; `mangatl-dev` coordena owners.

Encontre a primeira divergência antes de alterar expansão, classifier, perfil ou engine. Resíduo pode nascer em OCR/máscara; blur não corrige causa.

## Testes focados

```powershell
Push-Location pipeline
try {
  python -m pytest tests/test_inpaint_mask_geometry.py::test_build_inpaint_mask_rejects_large_no_line_art_component tests/test_vision_stack_inpainter.py::VisionStackInpainterTests::test_translucent_balloon_does_not_receive_a_second_runtime_mask_expansion -q
  python -m pytest tests/test_vision_stack_inpainter.py::VisionStackInpainterTests::test_final_inpaint_clamp_restores_artifacts_outside_expanded_mask tests/test_vision_stack_inpainter.py::VisionStackInpainterTests::test_translucent_balloon_uses_local_text_over_art_reconstruction -q
} finally { Pop-Location }
```

## Checklist de encerramento

- Primeira divergência e owner registrados; raw/expanded/action/effective comparadas.
- Before/after comparados visualmente; arte fora do limite e perfil preservados.
- Consumers, nodeids, auditor, UTF-8 e `git diff --check` validados.
- Checkout sujo preservado e alterações preexistentes reportadas.

## Última verificação

- Data: 2026-07-24
- Runtime-base: `c0801348`
