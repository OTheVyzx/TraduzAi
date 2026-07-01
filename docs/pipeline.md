# Pipeline

## Etapas

1. Importacao e normalizacao de paginas.
2. Deteccao de baloes/regioes.
3. OCR.
4. Contexto e glossario.
5. Traducao.
6. Inpaint.
7. Typesetting.
8. QA.
9. Export.

## Progresso

O pipeline informa progresso por JSON lines. A UI mostra etapa atual, paginas por minuto, flags e estimativas.

## Regras

- Glossario reviewed tem prioridade maxima.
- Contexto online entra como candidato, nunca como revisado automatico.
- QA deve bloquear export limpo quando houver flag critica/alta ativa.
- Imagens do usuario nao sao enviadas para fontes de contexto.

## Rota SFX manhwa

SFX Hangul de manhwa usa a rota `translate_sfx_inpaint_render`. Essa rota adapta a onomatopeia para PT-BR, preserva metadados em `text_layers[].sfx`, gera mascara de glifo conservadora e so permite inpaint automatico quando o gate visual marca `sfx.inpaint_allowed=true`.

Quando a traducao/adaptacao e desconhecida, a camada vira `route_action=review_required`. Quando a adaptacao existe mas o inpaint e inseguro, a camada permanece na rota SFX com `sfx.inpaint_allowed=false` e o export gate retorna `REVIEW`, nao `BLOCK`, para revisao visual manual.

Flags como `sfx_render_missing`, `sfx_render_outside_source_region`, `sfx_inpaint_damaged_art_risk`, `sfx_translation_unknown` e `sfx_style_low_confidence` sao sinais de revisao e nao disparam rerun de OCR.

## Detector visual SFX

O detector visual roda antes da traducao para encontrar SFX estilizado que o OCR comum nao le. Ele emite candidatos paralelos em `_sfx_visual_candidates` com `content_class=sfx`, `detector=sfx_visual` e `route_action=review_required`. A promocao para `text_layers` acontece no pipeline/editor; candidatos sem texto reconhecido ficam em revisao com `sfx_script_unknown`.

Quando um probe regional reconhece Hangul, o candidato pode virar `translate_sfx_inpaint_render`. Kana, CJK nao Hangul ou texto vazio permanecem em revisao. O detector nunca faz inpaint diretamente; ele apenas fornece bbox/evidencia para `sfx.mask`, `sfx.inpaint_gate` e debug sheets.

Benchmarks visuais locais devem ficar fora do git em:

```text
data/sfx_benchmarks/manhwa/
```

Para resumir esse corpus opcional:

```bash
cd pipeline
.\\venv\\Scripts\\python.exe -m pipeline.tools.analyze_cjk_quality_run --sfx-benchmark ..\\data\\sfx_benchmarks\\manhwa
```

## Testes

Use pytest para pipeline:

```bash
cd pipeline
.\\venv\\Scripts\\python.exe -m pytest -q
```
