---
name: traduzai-inpaint
description: Use when TraduzAI leaves source glyphs, erases line art, smears textures, paints the wrong balloon, changes pixels outside a mask, or chooses the wrong white, dark, translucent, or textured cleanup strategy.
---

# TraduzAI Inpaint

## Owner e fluxo ativo

Owner: `pipeline/inpainter/__init__.py`, `pipeline/inpainter/mask_builder.py`, `pipeline/inpainter/region_strategy.py` e `pipeline/vision_stack/inpainter.py`. `pipeline/strip/process_bands.py` contém o wrapper, não o algoritmo.

`pipeline/main.py` → `pipeline/strip/run.py::run_chapter` → `process_band` → `_run_inpaint_stage` → `inpaint_band_image`.

Entrada: banda RGB e página traduzida com `texts`, `_vision_blocks`, bboxes, geometria/linha, perfil, `route_action` e evidência de máscara. Saída: banda limpa, contratos `texts`/`_vision_blocks` atualizados e métricas `_strip_*`. Typesetting/render, copyback e QA consomem o resultado.

## Máscaras e proteção de arte

- **raw mask:** evidência de glifo antes da expansão.
- **expanded mask:** envelope de limpeza após expansão/clips.
- **action/final action mask:** pixels autorizados para fast fill, modelo, retries e cleanup.
- **effective limit mask:** limite final usado pelo clamp; mudanças externas são restauradas.

Clips de balão, line polygons, bboxes locais e limites de densidade evitam apagar bordas/arte. `protection_mask` em `06_mask_segmentation` é visualização derivada de `expanded_mask` fora do limite efetivo; **não é entrada do modelo**.

## Estratégias

- Branco/solid seguro: fast white/solid fill ou Telea local; preserve contorno.
- Escuro/colorido: use evidência de glifo e amostra local; não transforme painel em retângulo sólido sem contrato confiável.
- Translúcido/texturizado/gradiente: evite fill opaco; prefira LaMa/AOT conforme estratégia e reconstrução/continuação local restrita à action mask.
- Blocos seguros restantes usam real inpaint. Fallbacks podem reconstruir `_vision_blocks` de `texts`, recuperar máscara por geometria ou fazer fill local; sem máscara confiável, preserve arte e registre skip/review.

## Quick reference: primeira divergência

| Primeira evidência incorreta | Investigue |
|---|---|
| raw mask em `debug/e2e/06_mask_segmentation` | OCR/line geometry ou `mask_builder` |
| raw correta, expanded larga/curta | expansão, clip, densidade e perfil |
| action/effective mask diverge | estratégia, fallback e clamp |
| máscaras corretas, decisão/resultado errado em `08_inpaint` | fast fill, real inpaint, reconstrução local |
| `08_inpaint` correto, imagem final errada | typesetting/copyback/QA; use `traduzai-pipeline` + especialista |

`debug_inpaint` guarda before/raw/expanded/effective/after e mudanças externas. `06_mask_segmentation` mostra cadeia global/per-texto. `08_inpaint` registra decisão, skips, engine e pixels alterados.

## Fronteiras e diagnóstico

Detect é dona da geometria candidata; OCR, do texto/rota/evidência; inpaint, da máscara e reconstrução; typesetting, do texto novo. Em falha multietapa automática, use `traduzai-pipeline` com a especialista afetada; `mangatl-dev` coordena owners.

Encontre a primeira divergência antes de alterar expansão, classifier, perfil ou engine. Resíduo pode nascer em OCR/máscara; blur não corrige causa.

## Testes focados

```powershell
Push-Location pipeline
try {
  python -m pytest tests/test_inpaint_region_strategy.py::test_translucent_balloon_profile_requires_lama_strategy tests/test_inpaint_mask_geometry.py::test_build_inpaint_mask_rejects_large_no_line_art_component -q
  python -m pytest tests/test_vision_stack_inpainter.py::VisionStackInpainterTests::test_final_inpaint_clamp_restores_artifacts_outside_expanded_mask tests/test_vision_stack_inpainter.py::VisionStackInpainterTests::test_translucent_balloon_uses_local_text_over_art_reconstruction -q
} finally { Pop-Location }
```

## Checklist de encerramento

- Primeira divergência e owner registrados; raw/expanded/action/effective comparadas.
- Arte fora do limite preservada e perfil de balão correto.
- Consumers, nodeids, auditor, UTF-8 e `git diff --check` validados.

## Última verificação

- Data: 2026-07-24
- Runtime-base: `c0801348`
