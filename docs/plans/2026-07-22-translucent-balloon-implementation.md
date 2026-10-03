# Translucent Balloon Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Detectar automaticamente balões translúcidos e impedir preenchimento sólido que apaga a arte sob o texto.

**Architecture:** Promover `_looks_translucent_or_textured_background` a um perfil persistente `translucent_balloon`. O perfil vence `dark_panel`, restringe a máscara aos glifos dentro do balão e exige inpaint texturizado/local ou revisão segura.

**Tech Stack:** Python 3.12, OpenCV/NumPy, Pillow, pytest.

---

### Task 1: Persistir a classificação translúcida

**Files:** Modify `pipeline/inpainter/mask_builder.py`; test `pipeline/tests/test_inpaint_mask_geometry.py`.

1. Escrever o teste que cria um bloco com `image_white_bubble_mask`, arte atrás do texto e entorno escuro; exigir `layout_profile`, `block_profile` e métrica `translucent_balloon`.
2. Rodar `python -m pytest -q tests/test_inpaint_mask_geometry.py -k translucent_balloon_profile`; confirmar falha.
3. Criar helper que promove o perfil somente com máscara branca ancorada e evidência positiva; excluir SFX, cards, painéis escuros explícitos e máscara sem âncora.
4. Rodar o mesmo teste e confirmar sucesso.
5. Commit isolado: `feat: classify translucent balloons`.

### Task 2: Fazer o perfil vencer preenchimento sólido

**Files:** Modify `pipeline/inpainter/__init__.py` e `pipeline/inpainter/region_strategy.py`; tests `pipeline/tests/regression/test_fast_fill_evidence_gate.py` e `pipeline/tests/test_vision_stack_inpainter.py`.

1. Escrever testes para um `translucent_balloon` com entorno escuro; exigir que fast white, fast dark, fast solid e `dark_panel_fill` não sejam usados.
2. Rodar `python -m pytest -q tests/regression/test_fast_fill_evidence_gate.py tests/test_vision_stack_inpainter.py -k translucent`; confirmar falha.
3. Mapear `translucent_balloon` para `lama_required`, vetar fill antes das rotas de painel escuro e enviar somente máscara de glifo recortada ao inpaint local/texturizado. Registrar `solid_fill_blocked_reason: translucent_balloon` no debug.
4. Rodar os testes novamente e confirmar sucesso.
5. Commit isolado: `fix: route translucent balloons to textured inpaint`.

### Task 3: Garantir máscara e QA seguros

**Files:** Modify `pipeline/inpainter/mask_builder.py` e `pipeline/qa/export_gate.py`; tests `pipeline/tests/test_inpaint_mask_geometry.py` e `pipeline/tests/test_export_gate.py`.

1. Escrever testes exigindo máscara dentro do interior do balão e `dark_panel_fill_applied` crítico para perfil translúcido.
2. Rodar `python -m pytest -q tests/test_inpaint_mask_geometry.py tests/test_export_gate.py -k translucent_balloon`; confirmar falha.
3. Aplicar clip obrigatório pelo interior, métricas de área e regra de QA que preserve rastreabilidade de fallback sem ocultar dano visual.
4. Rodar os testes novamente e confirmar sucesso.
5. Commit isolado: `test: enforce translucent balloon cleanup contract`.

### Task 4: Regressão visual no Mythic ch39

**Files:** Create `docs/reports/2026-07-22-mythic-ch39-translucent-balloon-validation.md`; output `N:/TraduzAI/.codex-tmp/mythic_ch39_translucent_validation/`.

1. Rodar página 1 e conferir que `page_001_band_005` não usa `dark_panel_fill`; deve usar inpaint real ou revisão explícita.
2. Gerar recorte antes/depois e validar que não existe patch retangular.
3. Rodar capítulo completo em pasta temporária nova, comparar gate e documentar bloqueadores restantes.
4. Commit isolado: `docs: validate translucent balloon regression`.
