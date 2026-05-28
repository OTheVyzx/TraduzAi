# Debug E2E do Pipeline TraduzAi — Guia mestre

> **Status**: especificação executável **v2** — pós-primeira implementação do debug padrão.
> **Última atualização**: 2026-05-18.
> **Owners**: pipeline (Python sidecar).
> **Referências analisadas**:
> - `2026-05-17_chapter1_e2e_debug_151830.zip` — baseline antigo, antes das primeiras correções (P0-1 a P0-10, ver §0.3).
> - `2026-05-18_chapter1_e2e_debug_002011.zip` — debug novo com `DebugRecorder`/strict/skip/layout parcialmente corrigidos (DBG2-01 a DBG2-20, ver §0.2).

Este documento é o guia único do **debug E2E padrão** do pipeline automático do TraduzAi. Quando `debug=true` (config) ou `TRADUZAI_DEBUG_E2E=1` (env), o pipeline emite uma árvore `debug/e2e/...` rastreável do extract até o export gate, **com IDs estáveis** ligando cada bloco do começo ao fim.

## Manifestos de regressão visual

Os manifestos pequenos ficam em `pipeline/tests/regression/manifests/*.json` e registram apenas metadados do run, classes de issue esperadas e caminhos relativos de crops de amostra. Imagens e árvores `DEBUGM/runs` continuam locais e não devem ser versionadas.

Matriz inicial R1.4:

- `ci_visual_smoke` (fixture pequena versionada para CI)
- `articuno_ch61`
- `chapter_1`
- `chapter_39`
- `god_of_death_ch2`
- `one_second_ch1`
- `one_second_ch2`

Por padrão os seis manifestos da matriz real apontam para `N:/TraduzAI/DEBUGM/runs/R1_4_visual_regression_matrix/...` com `ci_fixture: false`; se o run local não existir, o pytest valida o shape do manifesto e skipa o run com aviso claro. Fixtures de CI usam `ci_fixture: true` e precisam conter `project.json` e `debug/e2e/`; a fixture `ci_visual_smoke` mantém esse caminho exercitado sem versionar outputs grandes de `DEBUGM/runs`.

O debug serve para responder, **olhando só os artefatos**, perguntas como:

- Por que o OCR aceitou ou rejeitou um bloco?
- Qual bbox foi escolhida no dedupe? Qual descartada? Por quê?
- O texto colado (`IGETBACK`, `TOWORK`, `CANYOUFINDAGOOD`) foi normalizado?
- Onde a máscara saiu do balão? Quanta arte ela invadiu?
- O `skip_inpaint=true` foi de fato honrado em cada banda?
- O strict/export_gate bloqueou e mesmo assim o processo terminou `exit_code=0`?
- O texto traduzido caiu fora do balão? Encostou no rosto?
- Por que apareceu `JÃ�` no balão em vez de `JÁ`?
- O `debug_report` agregado está dizendo a verdade ou está dobrando contagens?
- O `render_plan` rastreia o bloco do project.json ou perdeu os IDs?
- Cobertura por etapa: 13 estágios estão bem instrumentados ou alguns estão vazios?

---

## 0. Estado atual do debug E2E

### 0.1. Correções já validadas no pacote 2026-05-18

Estas correções **foram implementadas e validadas** comparando o ZIP `2026-05-17_chapter1_e2e_debug_151830.zip` (baseline) com `2026-05-18_chapter1_e2e_debug_002011.zip` (debug novo):

| ID histórico | Correção validada | Evidência no ZIP 2026-05-18 |
|---|---|---|
| **P0-1** | `skip_inpaint=true` agora é honrado | `B/runner_config.json.skip_inpaint=true`; `B/debug/e2e/08_inpaint/` vazio; `B/debug/e2e/06_mask_segmentation/` vazio; `B/debug_inpaint/` vazio (0 entries); `B/images/*.jpg` MD5 ≠ `A/images/*.jpg` MD5; `B.strip_process_bands_total=94.5s` vs A=157.9s. |
| **P0-2** | `strict + BLOCK` → `exit_code=2` | `D/run_status.json.exit_code=2`; `D/_stdout.jsonl` última linha = `{"type":"error","message":"Strict falhou: export gate bloqueou a exportacao"}`. |
| **P0-3** | `qa.summary` consistente com `export_gate` | A/C/D: `qa_summary_critical_count=7 == export_gate_critical_issue_count=7`; `qa_export_consistent=true` em `11_qa_export_gate/qa_export_gate_consistency.json`. |
| **P0-4** (parcial) | Mojibake **detectado** com flag automático | `04_text_normalization_router/mojibake_audit.jsonl` tem 1 entry para `ocr_005` (`mojibake_samples=["ân"]`, flag `mojibake_in_translation` propagado ao export gate). **Atenção**: detecção OK, mas `suggested_fix` ainda é igual ao input em alguns casos. |
| **P0-5** | Confidence preservada (audit interno) | `03_ocr/ocr_confidence_audit.json.blocks_with_confidence_zero=0` (de 88 blocos). **Atenção**: `debug_report.json` agregado ainda lê o campo errado e reporta `confianca_ocr_zero_count=88` (DBG2-01/DBG2-03). |
| **P0-6** (parcial) | `layout_bbox` principal remapeado para page-global | `05_layout_geometry/bbox_coordinate_audit.json`: `mixed_coordinate_space_count=0`, `findings=[]`. **Atenção**: audit não cobre `bbox` top-level (regressão local), `render_bbox`, `safe_text_box`, `position_bbox`, `capacity_bbox` (DBG2-06/DBG2-07). |
| **P0-9** | Env vars `TRADUZAI_STRIP_FAST_*` persistidas | `C/debug/e2e/00_run/env_snapshot.json.env_vars` contém `TRADUZAI_STRIP_FAST_WHITE_INPAINT=1` e `TRADUZAI_STRIP_FAST_LOCAL_INPAINT=1`. |
| **PR-novo F** | T/N e URL com `skip_processing=true` | A run tem `content_class=tn_note` e `content_class=url_watermark` com `skip_processing=true` registrado no project.json. |
| **PR-novo F** | SFX dentro de fala separado antes da tradução | `DON'T HIT SFX: KICK MY MOM!` agora vira `DON'T HIT MY MOM!` → `Não bata na minha mãe!`; SFX não entra na tradução do balão. |
| **PR-novo F** | Fast fill funcional | `C/06_mask_segmentation/mask_chain_summary.json`: `bands_with_mask=12` (vs 74 em A), `expanded_mask_pixels=167.507` (vs 2.089.995 em A, −92%), `outside_balloon_pixels=854` (vs 20.233 em A, −96%); 22 bands com fast_white_fill, 42 com fast_local_fill. |

### 0.2. Bugs e lacunas atuais da cobertura v2

Estes problemas foram observados na análise do ZIP `2026-05-18_chapter1_e2e_debug_002011.zip` e **precisam** ser endereçados pelos PRs v2 (§6.2). Cada bug tem evidência direta no ZIP novo.

| ID | Bug | Evidência no ZIP 2026-05-18 | Severidade |
|---|---|---|---|
| **DBG2-01** | `debug_report.json/md` raiz reporta `confianca_ocr_zero_count=88` (deveria ser 0) | `03_ocr/ocr_confidence_audit.json.blocks_with_confidence_zero=0`; `debug_report.json[0].metrics.confianca_ocr_zero_count=88`. O audit interno está certo; o agregador raiz mente. | crítica/debug |
| **DBG2-02** | `content_class_counts` no agregador raiz está **exatamente dobrado** | `debug_report.json` diz `dialogue=106, narration=60, tn_note=4, noise=2, sign=2, url_watermark=2` (=176); contagem real no `project.json` é `dialogue=53, narration=30, tn_note=2, noise=1, sign=1, url_watermark=1` (=88). Analyzer soma `text_layers` + `textos` — mas no project.json os dois têm os mesmos 88 itens. | crítica/debug |
| **DBG2-03** | Analyzer lê campo de confidence errado | Tenta `confidence` em vez de `confidence_raw` → `ocr_confidence` → `confianca_ocr`. Resultado: falsos alertas. | crítica/debug |
| **DBG2-04** | `render_plan.jsonl` tem 100% das entries com `page_id=null`, `band_id=null`, `coordinate_space=null` | Verificado em `A/09_typeset/render_plan.jsonl`: 168 entries, 168 com `page_id=null`, 168 com `band_id=null`, 168 com `coordinate_space=null`. Só `text_id` está preenchido. | crítica/debug |
| **DBG2-05** | `render_plan.jsonl` tem 168 entries para **apenas 5 distinct text_ids** | `ocr_001` aparece **101 vezes**, `ocr_002` 40×, `ocr_003` 19×, `ocr_005` 6×, `ocr_004` 2×. Está duplicando massivamente — provavelmente recorder chamado uma vez por banda em vez de uma vez por bloco final. | crítica/debug |
| **DBG2-06** | `target_bbox`, `position_bbox`, `safe_text_box`, `render_bbox`, `bbox` no project.json final estão **band-local** em vários blocos | `PLEASE!` no project.json final: `bbox=[492,65,641,84]` (band-local), `safe_text_box=[473,65,641,96]` (band-local), `render_bbox=[492,65,641,84]` (band-local), mas `source_bbox=[88,2716,775,3523]` (page). Regressão parcial: o `bbox` top-level virou band-local também. | crítica/layout |
| **DBG2-07** | `bbox_coordinate_audit.json` só valida `layout_bbox`; declara `findings=[]` mesmo com `bbox`/`render_bbox`/`safe_text_box` band-local | Audit interno `mixed_coordinate_space_count=0`, mas evidência DBG2-06 mostra coordenadas misturadas. | crítica/debug |
| **DBG2-08** | `07_translation/` totalmente **vazio** nas 4 runs | `ls A/B/C/D/debug/e2e/07_translation/` = 0 files. Sem `translation_inputs.jsonl`, `translation_outputs.jsonl`, `glossary_application.jsonl`, `translation_fallbacks.jsonl`. | alta |
| **DBG2-09** | Dedupe OCR ainda escolhe bbox gigante para `PLEASE!` | Decision trace: `drop_block reason=overlapping_duplicate_ocr_block`, dropped `[468,57,644,100]`, kept `[88,16,775,823]`. Project final mantém `bbox_overreach_critical`. | crítica/ocr |
| **DBG2-10** | `source_bbox == balloon_bbox` ainda aparece | `05_layout_geometry/source_bbox_balloon_overreach.jsonl` em A: 2 entries com `severity=critical`, `area_ratio` 14.58 e 4.5. Bug raiz P0-7 do baseline não foi totalmente corrigido. | crítica/layout |
| **DBG2-11** | Cover/logo/noise ainda viram `dialogue/fala` e são renderizados | `Shadow Erian Shadow` → `content_class=dialogue, tipo=fala, translated="Sombra erian sombra"`, flag `render_on_art_suspected`. `NTEEM` idêntico. Ambos `skip_processing=false`. | crítica/router |
| **DBG2-12** | Normalização de joined words funciona para alguns, mas deixa muitos passar | Normalizou: `SOPLEASE, ALITTLELONGER, VHEN, IGETBACK, TOWORK, HOSPITALBILLS, LOANFOR, REAL-LIFEINSURANCE`. Deixou passar: `WE'REFOOL'S, TOBELIEVE, CANYOUFINDAGOOD, THATGIVESINTERESTUP, TILLTHREEMONTHS, TOSHOWYOUR, AJUMMAYOU, THERE'SNO, GETMONEYFROM, EVENTHINK, PAYUSBACK, CANDIE, IDON'T, LET'SJUST`. | alta/ocr |
| **DBG2-13** | Normalização nem sempre propaga para o texto final/tradução | `normalization_trace.jsonl` mostra `changed=true` mas o `project.json` pode preservar o `raw_ocr` em alguns layers. Sem campo `normalized_text_final` canônico. | alta/ocr |
| **DBG2-14** | Flags QA do `render_plan` não propagam ao project/export gate | `render_plan` mostra `ocr_003: WHEN I GET BACK TO WORK...` com `qa_flags=[ocr_run_on_suspect, render_on_art_suspected]`, mas no `project.json` esse bloco pode aparecer sem as mesmas flags. `visual_blockers.jsonl` em A tem só 1 linha (mojibake) embora export_gate tenha 6 críticos. | crítica/qa |
| **DBG2-15** | `sign` ainda é renderizado como narração comum | `TEXT: DARLING KARAOKE` → `content_class=sign` mas `tipo=narracao`, render normal fora da placa. Sem política de render em região da placa. | alta/router |
| **DBG2-16** | `10_copyback_reassemble/` totalmente **vazio** nas 4 runs | `ls A/B/C/D/debug/e2e/10_copyback_reassemble/` = 0 files. | média |
| **DBG2-17** | `12_contact_sheets/` totalmente **vazio** nas 4 runs | `ls A/B/C/D/debug/e2e/12_contact_sheets/` = 0 files. Existe `debug/contact_sheets/` fora da árvore E2E, mas não no padrão. | média |
| **DBG2-18** | `page_cleanup_rerender` continua ~220s em todas as runs, sem breakdown | B (skip_inpaint) tem `page_cleanup_rerender=221.5s` mesmo sem inpaint. `debug_manifest.json.stage_durations_sec={}` — `recorder.time_stage()` não foi usado para sub-stages. | média |
| **DBG2-19** | Fast-fill efetivo não propaga para QA final em C | `C/06_mask_segmentation/mask_chain_summary.json` mostra 9 flagged_bands (vs 54 em A); mas `C/11_qa_export_gate/export_gate.json` mantém **idênticas** 11 `mask_density_high`, 2 `mask_outside_balloon`, 3 `bbox_overreach_critical`. QA final usa métricas antigas. | crítica/qa |
| **DBG2-20** | `01_input_extract/` vazio; `inpaint_blocks.jsonl` agregado ausente | `ls A/debug/e2e/01_input_extract/` = 0; `08_inpaint/` só tem subpastas por band com `inpaint_decision.json`, sem `inpaint_blocks.jsonl` agregado. | média |
| **DBG2-21** | `skip_inpaint_honored` reportado como `null` no debug_report mesmo em B | Todos os 4 runs têm `skip_inpaint_honored=null` no `debug_report.json`. Métrica não está sendo extraída/calculada. | alta/debug |
| **DBG2-22** | `stage_durations_sec={}` no `debug_manifest.json` | `recorder.time_stage()` foi adicionado à API mas não foi instrumentado em nenhum stage. | média/debug |
| **DBG2-23** | `source_bbox_equals_balloon_bbox_count=29` no debug_report mas `source_bbox_balloon_overreach.jsonl` tem só 2 linhas | Inconsistência entre o JSONL stage-level e a contagem agregada do analyzer. | alta/debug |

### 0.2b. Rodada Codex afterfix4 2026-05-18

Rodada executada no capitulo real usado nas auditorias anteriores:

```text
input: C:\Users\PICHAU\Downloads\Chapter 1
run_root: N:\TraduzAI\DEBUGM\runs\2026-05-18_chapter1_e2e_debug_afterfix4_2026-05-18_184848
artefatos: 6.785 arquivos, 189,94 MB
analyzer: python tools/analyze_e2e_debug.py <run_root> --write-report --strict-debug-audit
resultado historico da rodada: analyzer_exit_code=0, strict_audit.all_passed=true
reavaliacao com analyzer atual: analyzer_exit_code=3, strict_audit.all_passed=false
```

Resultado das 4 runs canonicas na reavaliacao atual:

| Run | Exit obtido | Gate | Analyzer strict atual |
|---|---:|---|---|
| `A_baseline_debug` | `0` | `BLOCK` | FAIL |
| `B_skip_inpaint` | `0` | `BLOCK` | FAIL |
| `C_fast_fill` | `0` | `BLOCK` | FAIL |
| `D_strict_export_gate` | `2` | `BLOCK` | FAIL |

Invariantes que continuaram validados em todas as runs:

```text
trace_id_null_count == 0
project_textos_trace_id_null_count == 0
project_trace_id_unique_ratio == 1.0
page_band_mismatch_count == 0
qa_export_consistent == true
translation_summary_mismatch == false
render_plan_final_incomplete == false
inpaint_debug_missing == false
inpaint_trace_id_missing_count == 0
copyback_trace_ids_missing_count == 0
source_bbox_equals_balloon_bbox_count == 0
detect_accepted_null_match_count == 0
debug_errors_count == 0
translated_comparison_present == true
```

Invariantes novos que o analyzer atual passou a bloquear corretamente no afterfix4:

```text
render_plan_final_matches_project == false
render_plan_trace_page_band_consistent == false
project_render_bbox_inside_balloon == false
qa_issues_are_traceable == false
```

Interpretação: o afterfix4 e um artefato historico util, mas nao e mais um pacote PASS. Ele deve ser usado como fixture negativa: `--strict-debug-audit` precisa retornar `3` ate que uma nova rodada gere artefatos coerentes com o contrato atual.

Correcoes implementadas nesta rodada:

- `layout_blocks.jsonl` passou a derivar `page_id` do proprio texto ou de `band_id`, eliminando falso `page_band_mismatch`.
- `project.json.paginas[].textos[]` recebeu `trace_id`, `text_id`, `page_id`, `band_id` e `text_instance_id`, mantendo o alias legado rastreavel.
- `source_bbox` deixou de cair para `balloon_bbox`; quando a origem vinha contaminada, o pipeline repara a partir de `text_pixel_bbox` e marca `source_bbox_origin`.
- `02_strip_detect/candidate_text_matching.jsonl` liga candidatos detectados a `trace_id`/`text_id`; o analyzer aceita candidatos sem texto apenas quando a banda nao tem texto.
- `08_inpaint/inpaint_blocks.jsonl` e `10_copyback_reassemble/copyback_decisions.jsonl` agora sao bloqueantes no strict quando faltam IDs rastreaveis.
- `qa.summary` e `export_gate` agora separam `critical_issue_count` de `critical_flag_count` e ignoram camadas `skip_processing`, resolvendo a divergencia do gate.
- O detector de mojibake foi ajustado para nao bloquear portugues valido enquanto ainda pega sequencias corrompidas.

### 0.3. Bugs históricos do baseline 2026-05-17 (já cobertos pelo guia v1)

Estes bugs foram detectados na análise do ZIP `2026-05-17_chapter1_e2e_debug_151830.zip` (baseline antigo) e estão **resolvidos ou em endereçamento** pelo pacote 2026-05-18 (ver §0.1). Mantidos aqui como histórico para devs que precisem entender a evolução.

| ID | Bug histórico | Status pós-2026-05-18 |
|---|---|---|
| **P0-1** | `skip_inpaint=true` não honrado | ✅ resolvido |
| **P0-2** | Strict `BLOCK` retornando `exit_code=0` | ✅ resolvido |
| **P0-3** | `qa.summary` mentindo severidade | ✅ resolvido |
| **P0-4** | Mojibake duplo PT-BR no traduzido | ⚠️ detecção OK, fix automático parcial |
| **P0-5** | `confianca_ocr=0.0` em todos os textos | ⚠️ audit interno OK, agregador raiz quebrado (DBG2-01) |
| **P0-6** | Mismatch de coord-space entre bboxes | ⚠️ `layout_bbox` principal OK, derivados ainda band-local (DBG2-06) |
| **P0-7** | `source_bbox = balloon_bbox` em `assign_balloon_bbox` | ⚠️ ainda aparece (DBG2-10) |
| **P0-8** | Cluster duplo no mesmo balão (VHEN bug) | ⚠️ normalizado, mas dedupe geometric pendente (DBG2-09) |
| **P0-9** | `C_fast_fill` não validado por env vars | ✅ resolvido |
| **P0-10** | Renderer warning vs report divergem em `balloon_bbox_missing` | ⚠️ `balloon_bbox_missing_audit.jsonl` existe (3 entries em A); falta cross-check |
Os PRs deste guia (seção 6 e 6.2) endereçam **cada** um destes bugs com instrumentação específica.

---

## 1. Regras de implementação

### 1.1. Não quebrar o pipeline normal

Debug ativa apenas quando:

```python
config.get("debug") is True or os.getenv("TRADUZAI_DEBUG_E2E", "").lower() in {"1", "true", "yes", "on"}
```

Com debug **off**, overhead deve ser ≤ 0.5% wall-clock e nenhuma escrita em disco fora de `pipeline.log` legacy.

### 1.2. Debug nunca derruba a run

Todo bloco do recorder fica dentro de `try/except`. Falhas registram em `debug/e2e/debug_errors.jsonl` com `traceback` curto. **Nenhuma exceção de debug pode propagar** para OCR, tradução, inpaint, render ou export gate.

### 1.3. Versionamento e cabeçalho

Todo JSON/JSONL principal começa com:

```json
{
  "schema_version": 1,
  "run_id": "2026-05-17T18-18-00Z_chapter1_a1b2c3",
  "stage": "ocr",
  "created_at": "2026-05-17T18:19:01.526242+00:00"
}
```

### 1.4. IDs estáveis e propagáveis

```text
run_id            ← gerado no início da run (timestamp + obra + 6-hex)
page_id           ← f"page_{page_number:03d}"
source_page_number ← número da página original (não da banda)
band_id           ← f"page_{page_number:03d}_band_{band_index:03d}"
balloon_id        ← f"{band_id}_balloon_{i:02d}"
candidate_id      ← f"{band_id}_cand_{i:03d}"      (proposta antes do accept/reject)
block_id          ← f"{band_id}_block_{i:03d}"     (após accept)
text_id           ← canônico em todo o pipeline; igual ao "id" usado hoje no project.json (ex: "ocr_017")
trace_id          ← f"{text_id}@{band_id}"         (chave única cross-stage)
```

`text_id` **nunca muda** entre OCR e export. Se houver merge/split, registrar `merge_into` / `split_from` no evento, mas manter o ID original como rastro.

### 1.5. Separar debug de correção

Este guia **instrumenta e expõe**. PRs marcados como "fix" são correções pequenas necessárias para registrar metadados corretos (ex: P0-5 confidence preservation). Correções grandes de qualidade visual ficam para PRs separados, **fora** deste guia.

---

## 2. Pacote `pipeline/debug_tools/`

```text
pipeline/debug_tools/
  __init__.py          — exporta DebugRecorder + helpers
  recorder.py          — classe central, IO seguro
  schemas.py           — schemas + validators (jsonschema opcional)
  ids.py               — geradores de run_id, band_id, text_id
  bbox.py              — bbox_coordinate_audit, area, overlap, contém
  masks.py             — render de mask chain (glyph→balloon→protection→final)
  overlays.py          — render de overlays por página (cv2.putText + cv2.polylines)
  contact_sheets.py    — montagem de contact sheets PIL/cv2
  text_diff.py         — diff token a token para normalization debug
  detectors.py         — heurísticas: joined_word_suspect, mojibake_regex, sfx_marker
  report.py            — gera debug_report.md + debug_report.json
```

### 2.1. `recorder.py` — interface obrigatória

```python
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class DebugLevel:
    STANDARD = "standard"   # JSON/JSONL + overlays só para bands com flags
    FULL = "full"           # tudo: overlays e masks para todas as bands
    MINIMAL = "minimal"     # só events.jsonl + bbox_audit + debug_report


class DebugRecorder:
    """Sistema central de debug E2E. Tolerante a falhas, versionado, rastreável."""

    def __init__(
        self,
        work_dir: Path,
        enabled: bool,
        run_id: str,
        *,
        level: str = DebugLevel.STANDARD,
        clock: callable = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.enabled = bool(enabled)
        self.run_id = run_id
        self.level = level
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self._root = self.work_dir / "debug" / "e2e"
        self._manifest_path = self._root / "debug_manifest.json"
        self._events_path = self._root / "events.jsonl"
        self._errors_path = self._root / "debug_errors.jsonl"
        self._artifacts_path = self._root / "artifacts.jsonl"
        self._artifacts: list[dict] = []
        self._stage_durations: dict[str, float] = {}
        if self.enabled:
            self._bootstrap_tree()

    # ---- public API ----

    def event(self, stage: str, action: str, payload: dict | None = None) -> None:
        if not self.enabled:
            return
        try:
            record = {
                "schema_version": 1,
                "run_id": self.run_id,
                "ts": self._clock(),
                "stage": stage,
                "action": action,
                **(payload or {}),
            }
            self._append_jsonl(self._events_path, record)
        except Exception as exc:
            self._record_error(stage=stage, action=action, exc=exc)

    def write_json(self, rel_path: str, payload: dict) -> None:
        if not self.enabled:
            return
        try:
            target = self._root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = self._header(payload, stage=self._stage_from_rel(rel_path))
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.register_artifact(stage=self._stage_from_rel(rel_path), rel_path=rel_path, kind="json")
        except Exception as exc:
            self._record_error(stage=self._stage_from_rel(rel_path), action="write_json", exc=exc, rel_path=rel_path)

    def write_jsonl(self, rel_path: str, payload: dict) -> None:
        if not self.enabled:
            return
        try:
            target = self._root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = self._header(payload, stage=self._stage_from_rel(rel_path))
            self._append_jsonl(target, payload)
        except Exception as exc:
            self._record_error(stage=self._stage_from_rel(rel_path), action="write_jsonl", exc=exc, rel_path=rel_path)

    def write_image(self, rel_path: str, image, *, quality: int = 88) -> None:
        if not self.enabled:
            return
        try:
            import cv2
            target = self._root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            ext = target.suffix.lower()
            if ext in {".jpg", ".jpeg"}:
                cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            else:
                cv2.imwrite(str(target), image)
            self.register_artifact(stage=self._stage_from_rel(rel_path), rel_path=rel_path, kind="image")
        except Exception as exc:
            self._record_error(stage=self._stage_from_rel(rel_path), action="write_image", exc=exc, rel_path=rel_path)

    def register_artifact(
        self, stage: str, rel_path: str, kind: str, meta: dict | None = None
    ) -> None:
        if not self.enabled:
            return
        entry = {
            "schema_version": 1,
            "stage": stage,
            "rel_path": rel_path,
            "kind": kind,
            "meta": meta or {},
        }
        self._artifacts.append(entry)
        try:
            self._append_jsonl(self._artifacts_path, entry)
        except Exception as exc:
            self._record_error(stage=stage, action="register_artifact", exc=exc, rel_path=rel_path)

    @contextmanager
    def time_stage(self, stage_name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        import time
        started = time.perf_counter()
        try:
            yield
        finally:
            self._stage_durations[stage_name] = round(
                float(self._stage_durations.get(stage_name, 0.0)) + (time.perf_counter() - started), 4
            )

    def finalize(self, *, config_snapshot: dict | None = None, extra: dict | None = None) -> None:
        if not self.enabled:
            return
        try:
            manifest = {
                "schema_version": 1,
                "run_id": self.run_id,
                "created_at": self._clock(),
                "level": self.level,
                "stage_durations_sec": self._stage_durations,
                "artifact_count": len(self._artifacts),
                "config_snapshot": config_snapshot or {},
                **(extra or {}),
            }
            self._root.mkdir(parents=True, exist_ok=True)
            self._manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("DebugRecorder.finalize falhou: %s", exc)

    # ---- internals ----

    def _bootstrap_tree(self) -> None:
        for sub in [
            "00_run", "01_input_extract", "02_strip_detect", "03_ocr",
            "04_text_normalization_router", "05_layout_geometry",
            "06_mask_segmentation", "07_translation", "08_inpaint",
            "09_typeset", "10_copyback_reassemble", "11_qa_export_gate",
            "12_contact_sheets", "13_report",
        ]:
            (self._root / sub).mkdir(parents=True, exist_ok=True)

    def _header(self, payload: dict, stage: str) -> dict:
        if not isinstance(payload, dict):
            return {"schema_version": 1, "run_id": self.run_id, "stage": stage, "value": payload}
        if "schema_version" in payload:
            return payload
        return {"schema_version": 1, "run_id": self.run_id, "stage": stage, **payload}

    def _stage_from_rel(self, rel_path: str) -> str:
        head = rel_path.split("/", 1)[0]
        return head[3:] if head[:2].isdigit() and head[2] == "_" else "misc"

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _record_error(self, **kwargs) -> None:
        try:
            import traceback
            exc = kwargs.pop("exc", None)
            payload = {
                "schema_version": 1,
                "ts": self._clock(),
                "run_id": self.run_id,
                "traceback": traceback.format_exc(limit=4) if exc else "",
                **kwargs,
            }
            with self._errors_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
```

### 2.2. Singleton/contexto

Para evitar passar `recorder` em cada função, expor um contextvar:

```python
# pipeline/debug_tools/__init__.py
from contextvars import ContextVar
from .recorder import DebugRecorder, DebugLevel

_current: ContextVar[DebugRecorder | None] = ContextVar("traduzai_debug_recorder", default=None)

def get_recorder() -> DebugRecorder | None:
    return _current.get()

def bind_recorder(recorder: DebugRecorder) -> None:
    _current.set(recorder)

def event(stage: str, action: str, **payload) -> None:
    r = get_recorder()
    if r and r.enabled:
        r.event(stage, action, payload)
```

Assim, módulos profundos (mask_builder, contextual_reviewer) podem fazer `from debug_tools import event` sem mudar assinaturas.

---

## 3. Estrutura final da árvore de debug

```text
work_dir/
└── debug/
    └── e2e/
        ├── debug_manifest.json
        ├── events.jsonl
        ├── artifacts.jsonl
        ├── debug_errors.jsonl
        │
        ├── 00_run/
        │   ├── config_snapshot.json
        │   ├── env_snapshot.json
        │   ├── pipeline_args.json
        │   ├── runner_config_snapshot.json     ← cópia do runner_config.json
        │   ├── performance_timing_snapshot.json
        │   └── stage_summary.json
        │
        ├── 01_input_extract/
        │   ├── input_manifest.json
        │   ├── original_pages_manifest.json
        │   └── original_contact_sheet.jpg
        │
        ├── 02_strip_detect/
        │   ├── strip_manifest.json
        │   ├── bands_manifest.json
        │   ├── detect_candidates.jsonl
        │   ├── detect_accept_reject.jsonl
        │   ├── strip_overlay.jpg
        │   └── page_overlays/
        │       └── page_{NNN}_detect_overlay.jpg
        │
        ├── 03_ocr/
        │   ├── ocr_raw_blocks.jsonl
        │   ├── ocr_accepted_blocks.jsonl
        │   ├── ocr_dedupe_decisions.jsonl
        │   ├── ocr_merge_decisions.jsonl
        │   ├── ocr_confidence_audit.json       ← P0-5
        │   └── page_overlays/
        │       └── page_{NNN}_ocr_overlay.jpg
        │
        ├── 04_text_normalization_router/
        │   ├── normalization_trace.jsonl
        │   ├── joined_word_splits.jsonl
        │   ├── text_router_decisions.jsonl
        │   ├── special_text_splits.jsonl
        │   └── mojibake_audit.jsonl            ← P0-4
        │
        ├── 05_layout_geometry/
        │   ├── layout_blocks.jsonl
        │   ├── bbox_coordinate_audit.json      ← P0-6
        │   ├── source_bbox_balloon_overreach.jsonl  ← P0-7
        │   ├── connected_subregions.jsonl
        │   └── page_overlays/
        │       └── page_{NNN}_layout_overlay.jpg
        │
        ├── 06_mask_segmentation/
        │   ├── mask_blocks.jsonl
        │   ├── mask_chain_summary.json
        │   └── {band_id}/
        │       ├── 01_glyph_mask.png
        │       ├── 02_line_polygon_mask.png
        │       ├── 03_detected_text_mask.png
        │       ├── 04_balloon_mask.png
        │       ├── 05_balloon_inner_mask.png
        │       ├── 06_protection_mask.png
        │       ├── 07_raw_text_mask.png
        │       ├── 08_expanded_text_mask.png
        │       ├── 09_final_inpaint_mask.png
        │       ├── 10_mask_overlay.jpg
        │       └── mask_decision.json
        │
        ├── 07_translation/                        ← DBG2-08 (PR 11)
        │   ├── translation_inputs.jsonl
        │   ├── translation_outputs.jsonl
        │   ├── glossary_application.jsonl
        │   ├── translation_fallbacks.jsonl
        │   └── translation_debug_summary.json     ← v2
        │
        ├── 08_inpaint/
        │   ├── inpaint_blocks.jsonl               ← agregado, falta hoje (DBG2-20)
        │   ├── skip_inpaint_audit.json            ← P0-1, DBG2-21
        │   └── {band_id}/
        │       ├── before.jpg
        │       ├── raw_mask.png
        │       ├── expanded_mask.png
        │       ├── effective_limit_mask.png
        │       ├── after.jpg
        │       ├── changed_outside_mask.png
        │       ├── changed_outside_overlay.jpg
        │       ├── residual_text_check.json
        │       └── inpaint_decision.json
        │
        ├── 09_typeset/                            ← DBG2-04/05/06 (PR 10)
        │   ├── render_plan_raw.jsonl              ← v2: band-local OK; obriga coordinate_space="band"
        │   ├── render_plan_final.jsonl            ← v2: SOMENTE page-global; sem duplicação
        │   ├── render_qa.jsonl
        │   ├── render_coordinate_audit.json       ← v2
        │   ├── balloon_bbox_missing_audit.jsonl   ← P0-10
        │   └── page_overlays/
        │       ├── page_{NNN}_typeset_overlay.jpg
        │       ├── page_{NNN}_safe_text_box_overlay.jpg
        │       └── page_{NNN}_render_bbox_overlay.jpg
        │
        ├── 10_copyback_reassemble/                ← DBG2-16 (PR 16)
        │   ├── copyback_decisions.jsonl
        │   ├── reassemble_manifest.json
        │   ├── page_maps.jsonl
        │   ├── page_cleanup_breakdown.json        ← v2 (DBG2-18)
        │   └── copyback_overlays/
        │       └── page_{NNN}_copyback_overlay.jpg
        │
        ├── 11_qa_export_gate/
        │   ├── qa_summary.json
        │   ├── qa_issues.jsonl
        │   ├── export_gate.json
        │   ├── qa_export_gate_consistency.json    ← P0-3
        │   ├── qa_flag_propagation_audit.json     ← v2 (DBG2-14)
        │   ├── strict_exit_audit.json             ← P0-2
        │   ├── visual_blockers.jsonl
        │   └── visual_blockers_overlay.jpg
        │
        ├── 12_contact_sheets/                     ← DBG2-17 (PR 16)
        │   ├── translated_comparison.jpg
        │   ├── problem_bands.jpg
        │   ├── mask_chain_top_issues.jpg
        │   └── typeset_top_issues.jpg
        │
        └── 13_report/
            ├── debug_report.md
            ├── debug_report.json
            └── debug_report_consistency.json      ← v2 (DBG2-01/02/03)
```

> **Regra de coordenadas (v2 obrigatório)**: todo arquivo `*_final` (ex: `render_plan_final.jsonl`, `layout_blocks.jsonl`) deve estar em **coordenada global de página**. Entradas band-local só podem existir em arquivos `*_raw` (ex: `render_plan_raw.jsonl`) **e** precisam carregar `coordinate_space="band"`, `band_y_top`, `page_id`, `band_id` e `text_id` preenchidos.

### Diagnosing Mixed Coordinate Space

If final `translated/*.jpg` shows text at the top of the page, over character art, or far from the target balloon while the intended balloon is blank, inspect these artifacts first:

- `debug/e2e/05_layout_geometry/bbox_coordinate_audit.json`
- `debug/e2e/09_typeset/render_plan_final.jsonl`
- `debug/e2e/11_qa_export_gate/visual_blockers.jsonl`
- `project.json`

After strip reassembly and final page-space typeset, these fields must be in the same page coordinate space:

- `bbox`
- `source_bbox`
- `text_pixel_bbox`
- `balloon_bbox`
- `bubble_mask_bbox`
- `bubble_inner_bbox`
- `balloon_inner_bbox`
- `safe_text_box`
- `_debug_safe_text_box`
- `layout_safe_bbox`
- `render_bbox`
- `_render_debug.*bbox`

Confirmed blockers:

- `layout_bbox_coordinate_mismatch`
- `bubble_inner_bbox_coordinate_mismatch`
- `page_space_rerender_mixed_coordinates`

`sync_final_page_space_typeset` must not overwrite a final page when these flags are detected. In that case the previous band-rendered output is preserved and the export gate receives the critical QA flags.

### Diagnosing Fast Solid Fill Residue

If `debug_inpaint/<band>/metadata.json` or `debug/e2e/08_inpaint/<band>/inpaint_decision.json` shows:

- `used_fast_solid_fill: true`
- `used_real_inpaint: false`
- `raw_mask_pixels: 0`
- `expanded_mask_pixels: 0`

then fast solid fill must also show verified metadata in `fast_solid_fill_samples`:

- `fast_fill_verified: true`
- `fast_fill_text_bbox_coverage`
- `fast_fill_residual_edge_ratio`

If verification fails, the text block must remain eligible for real inpaint or emit a blocking QA flag. Relevant flags:

- `fast_fill_insufficient_coverage`
- `fast_fill_unverified_residual`
- `text_residual_after_inpaint_suspected`
- `text_residual_after_inpaint_confirmed`

Em `level=minimal`, gerar apenas: `debug_manifest.json`, `events.jsonl`, `00_run/`, `05_layout_geometry/bbox_coordinate_audit.json`, `11_qa_export_gate/`, `13_report/`.

Em `level=full`, gerar **tudo** acima.

Em `level=standard`, **pular** overlays/masks para bands sem flags; manter todos os JSON/JSONL.

---

### 3.1. Minimo para o analyzer strict

Para `tools/analyze_e2e_debug.py <run_root> --strict-debug-audit`, o pacote precisa ter pelo menos:

| Escopo | Artefato | Uso |
|---|---|---|
| raiz da run | `project.json` | contagem final, layers, bboxes, `qa_flags`, `trace_id` |
| raiz da run | `runner_config.json` | detectar `strict`, `skip_inpaint`, modos A/B/C/D |
| raiz da run | `qa_report.json` ou `project.json.qa` | comparar summary/export gate |
| raiz da run | `_stdout.jsonl`, `run_status.json` ou equivalente | validar exit code e erro strict |
| raiz da run | `pipeline.log` | auditar erros operacionais |
| debug/e2e | `01_input_extract/input_manifest.json` | provar origem das paginas |
| debug/e2e | `07_translation/translation_inputs.jsonl` e `translation_outputs.jsonl` | ligar texto original/traduzido por identidade |
| debug/e2e | `08_inpaint/inpaint_blocks.jsonl` ou `08_inpaint/skip_inpaint_audit.json` | provar inpaint executado ou pulado explicitamente |
| debug/e2e | `09_typeset/render_plan_final.jsonl` | contrato canonico page-global do render |
| debug/e2e | `10_copyback_reassemble/copyback_decisions.jsonl` | provar retorno da banda para a pagina |
| debug/e2e | `11_qa_export_gate/qa_issues.jsonl` e `visual_blockers.jsonl` | blockers rastreaveis |
| debug/e2e | `12_contact_sheets/translated_comparison.jpg` | evidencia visual minima |
| debug/e2e | `13_report/debug_report.json` e `debug_report.md` | sumario humano e machine-readable |

`debug_report.md` nao pode esconder falhas de `strict_audit`: se `debug_report.json.strict_audit.all_passed=false`, o Markdown principal deve listar os invariantes falhos e os offenders. `Consistent=true` na tabela resumida nao substitui `strict_audit`.

## 4. Instrumentação por etapa

### 4.1. Run setup — `00_run/`

**Onde**: [pipeline/main.py:700](../../pipeline/main.py:700) (entrada de `_run_pipeline`).

```python
from debug_tools import DebugRecorder, DebugLevel, bind_recorder
from debug_tools.ids import generate_run_id

def _run_pipeline(config_path: str) -> int | None:
    ...
    debug_enabled = bool(config.get("debug")) or _env_truthy("TRADUZAI_DEBUG_E2E")
    debug_level = str(config.get("debug_level") or os.getenv("TRADUZAI_DEBUG_LEVEL") or "standard")
    run_id = generate_run_id(work_dir.name, config.get("obra", ""))
    recorder = DebugRecorder(work_dir, enabled=debug_enabled, run_id=run_id, level=debug_level)
    bind_recorder(recorder)

    recorder.write_json("00_run/config_snapshot.json", _redact_secrets(config))
    recorder.write_json("00_run/env_snapshot.json", _collect_env_snapshot())
    recorder.write_json("00_run/pipeline_args.json", {"argv": sys.argv})
    try:
        recorder.write_json("00_run/runner_config_snapshot.json",
                            json.loads((work_dir / "runner_config.json").read_text("utf-8")))
    except Exception:
        pass
```

#### `00_run/env_snapshot.json` (formato)

```json
{
  "schema_version": 1,
  "run_id": "...",
  "stage": "run",
  "created_at": "...",
  "python_version": "3.12.x",
  "platform": "Windows-11-...",
  "cuda_available": true,
  "torch_version": "2.x.y",
  "cv2_version": "4.x.y",
  "env_vars": {
    "TRADUZAI_DEBUG_E2E": "1",
    "TRADUZAI_DEBUG_LEVEL": "full",
    "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1",
    "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "1",
    "TRADUZAI_MACRO_OCR": "0",
    "TRADUZAI_STRIP_INPAINTER_PREWARM": "1",
    "STRIP_DEBUG": "1"
  },
  "config_flags": {
    "skip_inpaint": false,
    "skip_ocr": false,
    "strict": false,
    "export_mode": "with_warnings",
    "engine_preset_id": "",
    "runtime_profile": "balanced"
  }
}
```

**P0-9**: registrar **todas** as env vars `TRADUZAI_*` ativas no momento da run. Sem isto, "C_fast_fill" é indistinguível de baseline.

---

### 4.2. Input / extract — `01_input_extract/`

**Onde**: após `extract_source` em `pipeline/main.py`.

```json
{
  "schema_version": 1,
  "source_path": "C:\\Users\\PICHAU\\Downloads\\Chapter 1",
  "work_dir": "N:\\TraduzAI\\DEBUGM\\runs\\...",
  "page_count": 6,
  "pages": [
    {
      "page_id": "page_001",
      "page": 1,
      "filename": "001.jpeg",
      "width": 800,
      "height": 1079,
      "is_strip": false,
      "sha256": "aae2969508d1b324...",
      "size_bytes": 161276
    },
    {
      "page_id": "page_002",
      "page": 2,
      "filename": "002.jpeg",
      "width": 737,
      "height": 16383,
      "is_strip": true,
      "sha256": "ffedf62e972d3df2..."
    }
  ]
}
```

Regra v3: `input_manifest.json` precisa fechar o contrato por arquivo, nao apenas do strip. Cada item em `pages[]` deve ter `source_path` ou `filename`, `width`, `height`, `sha256` e `size_bytes`. Sem hash/tamanho, nao da para provar que duas runs A/B/C/D partiram exatamente do mesmo input.

`original_contact_sheet.jpg`: grid 3×2 com thumbnails das 6 páginas, label por página.

---

### 4.3. Strip / detect — `02_strip_detect/`

**Onde**: [pipeline/strip/run.py:2340](../../pipeline/strip/run.py:2340) (`build_strip`) e :2353 (`detect_strip_balloons`).

#### `strip_manifest.json`

```json
{
  "strip_width": 800,
  "strip_height": 82994,
  "source_page_breaks": [0, 1079, 17462, 33845, ...],
  "page_x_offsets": [0, 31, 31, 31, 31, 73],
  "band_margin_px": 16
}
```

#### `bands_manifest.json`

```json
{
  "band_count": 119,
  "bands": [
    {
      "band_id": "page_001_band_000",
      "source_page_number": 1,
      "band_index": 0,
      "y_top": 528,
      "y_bottom": 1030,
      "height": 502,
      "balloon_count": 6,
      "balloon_ids": ["page_001_band_000_balloon_00", ...]
    }
  ]
}
```

#### `detect_candidates.jsonl` (uma linha por candidato)

```json
{
  "candidate_id": "page_002_band_017_cand_002",
  "band_id": "page_002_band_017",
  "bbox_strip": [120, 2730, 350, 2780],
  "bbox_page": [120, 1651, 350, 1701],
  "confidence": 0.91,
  "source": "comic_text_detector",
  "accepted": true,
  "reject_reason": null,
  "matched_text_id": "ocr_017",
  "matched_text_ids": ["ocr_017"],
  "matched_trace_ids": ["ocr_017@page_002_band_017"],
  "match_count": 1,
  "match_method": "same_band_bbox_overlap",
  "band_text_count": 1
}
```

Regra v3: `detect_candidates.jsonl` continua sendo o artefato legado para consumers antigos, mas precisa ser enriquecido depois de `candidate_text_matching.jsonl`. Um candidato aceito so pode ficar sem `matched_trace_ids` quando `band_text_count == 0`; caso contrario o strict deve falhar em `detect_accepted_candidates_have_matches`.

#### Overlays
- `strip_overlay.jpg`: strip inteiro reduzido a 1200px de altura, balões em retângulos azuis, bands em linhas tracejadas amarelas, labels com `band_id`.
- `page_overlays/page_{NNN}_detect_overlay.jpg`: por página, balões aceitos em verde, rejeitados em vermelho com reject_reason em cima.

---

### 4.4. OCR — `03_ocr/`

**Onde**: [pipeline/strip/process_bands.py](../../pipeline/strip/process_bands.py) durante OCR; [pipeline/ocr/contextual_reviewer.py](../../pipeline/ocr/contextual_reviewer.py) durante dedupe.

#### `ocr_raw_blocks.jsonl`

```json
{
  "text_id": "ocr_017",
  "band_id": "page_002_band_017",
  "trace_id": "ocr_017@page_002_band_017",
  "raw_ocr": "VHEN IGETBACK TOWORK...",
  "confidence_raw": 0.56,
  "bbox_band": [137, 156, 726, 625],
  "bbox_page": [137, 11390, 726, 11859],
  "text_pixel_bbox_band": [137, 156, 726, 625],
  "source_bbox_band": [77, 73, 786, 683],
  "line_polygons_count": 0,
  "background_rgb": [212, 223, 240],
  "balloon_type": "white",
  "block_profile": "white_balloon",
  "accepted": true,
  "accept_reason": "ready_for_layout",
  "reject_reason": null,
  "ocr_backend": "vision-paddleocr"
}
```

#### `ocr_confidence_audit.json` — **P0-5**

```json
{
  "schema_version": 1,
  "summary": {
    "total_blocks": 89,
    "blocks_with_confidence_zero": 89,
    "blocks_with_confidence_lt_05": 89,
    "warning": "all_blocks_have_confidence_zero_likely_lost_in_metadata_flow"
  },
  "by_band": [
    {
      "band_id": "page_002_band_017",
      "text_id": "ocr_017",
      "confidence_at_accept": 0.56,
      "confidence_in_project_json": 0.0,
      "delta": -0.56,
      "lost_between": "_finalize_page_ocr_texts | _shift_text_geometry_y"
    }
  ]
}
```

Captura a confidence **no momento do `accept_block`** (decision_trace tem `confidence` em `details`) E depois lê do project.json final, registra divergências.

#### `ocr_dedupe_decisions.jsonl`

```json
{
  "action": "dedupe_blocks",
  "kept_text_id": "ocr_017_good",
  "dropped_text_ids": ["ocr_017_bad"],
  "reason": "geometry_quality_score",
  "kept_score": 42.1,
  "dropped_score": -18.5,
  "candidates": [
    {
      "text_id": "ocr_017_good", "text": "PLEASE!",
      "bbox": [468, 57, 644, 100], "area": 7568,
      "confidence": 0.92, "line_polygons_count": 2
    },
    {
      "text_id": "ocr_017_bad", "text": "PLEASE!",
      "bbox": [88, 16, 775, 823], "area": 554409,
      "confidence": 0.70, "line_polygons_count": 0
    }
  ],
  "debug_warning": "kept_bbox_area_much_larger_than_dropped"
}
```

> **Nota P0-7**: o caso PLEASE do baseline 2026-05-17 **não é** dedupe — é `source_bbox = balloon_bbox` propagado por `assign_balloon_bbox`. Mesmo assim instrumentar dedupe é necessário para casos futuros.

---

### 4.5. Normalização e router — `04_text_normalization_router/`

**Onde**: criar `pipeline/ocr/text_normalizer.py` e `pipeline/ocr/text_router.py`, chamados em [pipeline/ocr/contextual_reviewer.py](../../pipeline/ocr/contextual_reviewer.py) antes da tradução.

#### `normalization_trace.jsonl`

```json
{
  "text_id": "ocr_017",
  "raw": "VHEN IGETBACK TOWORK...",
  "normalized": "WHEN I GET BACK TO WORK...",
  "changed": true,
  "rules_applied": ["vhen_to_when", "split_joined_words", "ocr_punct_normalize"],
  "token_diff": [
    ["VHEN", "WHEN"],
    ["IGETBACK", "I GET BACK"],
    ["TOWORK", "TO WORK"]
  ],
  "confidence_before": 0.56,
  "confidence_after_estimate": 0.71,
  "needs_review": true,
  "review_reason": "low_initial_confidence_and_multiple_corrections"
}
```

#### `joined_word_splits.jsonl`

Tokens suspeitos obrigatórios para flagar (lista mínima):

```text
itsaexample  IGETBACK  TOWORK  CANYOUFIND  AISHIT'SNOT
VHEN  OWT  AU  SOPLEASE  ALITTLELONGER  AREWAITI
HOSPITALBILLS  REAL-LIFEINSURANCE  LOANFOR
```

```json
{
  "text_id": "ocr_017",
  "raw_text": "SOPLEASE WAIT ALITTLELONGER. WHENI GET BACK TO WORK...",
  "joined_word_suspect": true,
  "suspect_tokens": ["SOPLEASE", "ALITTLELONGER", "WHENI"],
  "suggested_split": ["SO PLEASE", "A LITTLE LONGER", "WHEN I"],
  "method": "dictionary|regex|heuristic|none",
  "applied": true,
  "after_split": "SO PLEASE WAIT A LITTLE LONGER. WHEN I GET BACK TO WORK..."
}
```

#### `text_router_decisions.jsonl`

Classes:

```text
speech              ← fala em balão
narration           ← caixa de narração
sfx                 ← onomatopeia
editorial_note      ← T/N, asterisco, nota do tradutor
sign                ← placa/menu/letreiro
url_watermark       ← URL, @handle, "Read at X"
scanlator_credit    ← "TL: X | TS: Y | PR: Z"
noise               ← OCR garble sem texto real
unknown_review      ← caiu em nada, review humano
```

```json
{
  "text_id": "ocr_031",
  "input": "DON'T HIT SFX: KICK My MOM!",
  "route": "speech_with_sfx_split",
  "parts": [
    {"class": "speech", "text": "DON'T HIT MY MOM!", "text_id_synthetic": "ocr_031_a"},
    {"class": "sfx", "text": "KICK", "text_id_synthetic": "ocr_031_b"}
  ],
  "rules_applied": ["split_sfx_marker_inside_dialogue"],
  "needs_review": false
}
```

```json
{
  "text_id": "ocr_022",
  "input": "T/N: HYUNGNIM SIGNIFICA IRMÃO MAIS VELHO",
  "route": "editorial_note",
  "render_policy": "preserve",
  "translate_policy": "skip_translation",
  "needs_review": true,
  "reason": "no_explicit_tn_render_layer_configured"
}
```

#### `mojibake_audit.jsonl` — **P0-4**

Regex de detecção:

```python
MOJIBAKE_PATTERN = re.compile(
    r"(?:Ã[-¿])"          # Ã + byte CP1252 0x80-0xBF → typical UTF-8-as-CP1252
    r"|(?:�)"                    # replacement char
    r"|(?:[\ud800-\udfff])"           # lone surrogate
    r"|(?:Ã[-ÿ])",    # composição broader
    re.UNICODE,
)
```

```json
{
  "text_id": "ocr_001",
  "stage": "translation_output",
  "translated": "VOCÃŠ SABE, GASTEI TODO O DINHEIRO...",
  "mojibake_match_count": 1,
  "mojibake_samples": ["ÃŠ"],
  "suggested_fix": "VOCÊ SABE, GASTEI TODO O DINHEIRO...",
  "fix_method": "decode_latin1_encode_utf8_safe"
}
```

Se `mojibake_match_count > 0` em qualquer texto, registrar **também** como blocker `mojibake_in_translation` em `11_qa_export_gate/visual_blockers.jsonl`.

---

### 4.6. Layout / geometria — `05_layout_geometry/`

**Onde**: [pipeline/layout/simple_text_geometry.py](../../pipeline/layout/simple_text_geometry.py), [pipeline/layout/balloon_layout.py](../../pipeline/layout/balloon_layout.py), [pipeline/strip/run.py:2587](../../pipeline/strip/run.py:2587).

#### Conceitos separados (obrigatório)

```text
source_bbox          ← bbox vinda do detector OCR original (não mexer)
ocr_text_bbox        ← bbox do bloco como retornado pelo OCR
text_pixel_bbox      ← bbox dos pixels reais de texto (apertada)
glyph_bbox           ← derivada de line_polygons
balloon_bbox         ← região do balão / placa / caixa
balloon_polygon      ← polígono real do balão (se houver)
balloon_inner_region ← interior do balão menos borda
layout_bbox          ← área que o typesetter pode usar
safe_text_box        ← área segura dentro do balão para texto
render_bbox          ← bbox final renderizada
```

> **P0-7**: `source_bbox` **nunca** pode receber `balloon_bbox` por cópia em `assign_balloon_bbox`. Manter `source_bbox` apenas como bbox crua do detector.

#### `layout_blocks.jsonl`

```json
{
  "text_id": "ocr_001",
  "page_id": "page_001",
  "band_id": "page_001_band_018",
  "coordinate_space": "page",
  "source_coordinate_space": "band",
  "band_y_top": 12824,
  "bboxes": {
    "source_bbox": {"value": [88, 2716, 775, 3523], "space": "page"},
    "bbox": {"value": [473, 2765, 641, 2796], "space": "page"},
    "text_pixel_bbox": {"value": [473, 2765, 641, 2796], "space": "page"},
    "balloon_bbox": {"value": [88, 2716, 775, 3523], "space": "page"},
    "layout_bbox": {"value": [473, 2765, 641, 2796], "space": "page"}
  },
  "polygons": {
    "line_polygons_count": 1,
    "balloon_polygon_present": false
  }
}
```

#### `bbox_coordinate_audit.json` — **P0-6**

```json
{
  "schema_version": 1,
  "summary": {
    "total_text_layers": 89,
    "all_consistent": false,
    "mixed_coordinate_space_count": 89,
    "band_local_in_page_context_count": 89,
    "page_global_in_band_context_count": 0
  },
  "findings": [
    {
      "text_id": "ocr_001",
      "page_id": "page_001",
      "band_id": "page_001_band_002",
      "band_y_top": 2700,
      "issue": "bbox_appears_band_local",
      "evidence": {
        "bbox_y_top": 65,
        "source_bbox_y_top": 2716,
        "text_pixel_bbox_y_top": 2765,
        "delta": 2651
      },
      "severity": "critical",
      "blocker": "layout_bbox_coordinate_mismatch"
    }
  ]
}
```

Heurística de detecção (não-trivial):
- Se a página é strip (`height > 4000`): comparar Y de `bbox`, `source_bbox`, `text_pixel_bbox`, `balloon_bbox` 2 a 2. Se delta > `band.height + band.y_top * 0.5`, suspeitar de space mismatch.
- Marcar `coordinate_space="band"` se `bbox.y_top < page.height * 0.2 AND source_bbox.y_top > bbox.y_top * 5`.

#### `source_bbox_balloon_overreach.jsonl` — **P0-7**

```json
{
  "text_id": "ocr_001",
  "issue": "source_bbox_equals_balloon_bbox",
  "source_bbox": [88, 2716, 775, 3523],
  "balloon_bbox": [88, 2716, 775, 3523],
  "text_pixel_bbox": [473, 2765, 641, 2796],
  "source_area": 554409,
  "text_pixel_area": 5208,
  "area_ratio": 106.45,
  "decision_trace_reason": "refined_same_as_cluster",
  "severity": "critical",
  "blocker": "source_bbox_assigned_from_balloon"
}
```

#### Overlays — paleta canônica

```text
vermelho (#FF0000) ─ source_bbox / bbox crua
amarelo  (#FFD000) ─ text_pixel_bbox / glyph_bbox
azul     (#0080FF) ─ balloon_bbox
verde    (#00C800) ─ safe_text_box
ciano    (#00C8C8) ─ layout_bbox
roxo     (#A000C0) ─ render_bbox
laranja  (#FF8000) ─ flagged (qa_flags ≠ vazio)
```

---

### 4.7. Mask chain — `06_mask_segmentation/`

**Onde**: [pipeline/inpainter/mask_builder.py](../../pipeline/inpainter/mask_builder.py).

#### Hierarquia obrigatória de PNGs por band

```text
01_glyph_mask.png            ← se line_polygons existe → preenchido; senão vazio
02_line_polygon_mask.png     ← line_polygons preenchidos
03_detected_text_mask.png    ← raw OCR-detected pixels
04_balloon_mask.png          ← balloon_bbox/polygon preenchido
05_balloon_inner_mask.png    ← balloon - borda (erode_px=2)
06_protection_mask.png       ← regiões que NUNCA podem ser inpaintadas
07_raw_text_mask.png         ← máscara inicial antes da expansão
08_expanded_text_mask.png    ← após dilate por _raw_text_search_expand_px
09_final_inpaint_mask.png    ← (expanded_text_mask ∩ balloon_inner_mask) - protection
10_mask_overlay.jpg          ← original image + final mask em vermelho 40% alpha
```

#### `mask_decision.json` (por band)

```json
{
  "schema_version": 1,
  "band_id": "page_002_band_017",
  "text_id": "ocr_017",
  "text_ids": ["ocr_017"],
  "trace_ids": ["ocr_017@page_002_band_017"],
  "trace_ids_in_band": ["ocr_017@page_002_band_017"],
  "text_instance_ids": ["page_002_band_017_ocr_017"],
  "mask_source": "line_polygons",
  "used_balloon_clip": true,
  "used_protection_mask": true,
  "raw_mask_pixels": 1234,
  "expanded_mask_pixels": 1820,
  "balloon_mask_pixels": 12000,
  "balloon_inner_mask_pixels": 11400,
  "outside_balloon_pixels": 0,
  "expanded_raw_ratio": 1.47,
  "mask_density_in_band": 0.031,
  "source_bbox_area": 9450,
  "glyph_bbox_area": 5208,
  "source_glyph_area_ratio": 1.81,
  "flags": [],
  "gates": {
    "mask_density_high": false,
    "mask_outside_balloon": false,
    "mask_outside_balloon_critical": false,
    "bbox_overreach": false,
    "bbox_overreach_critical": false,
    "expanded_ratio_review": false
  },
  "thresholds": {
    "mask_density_warn": 0.12,
    "expanded_raw_warn": 2.5,
    "source_glyph_critical": 8.0,
    "outside_balloon_critical_pixels": 50
  }
}
```

Regra v3: `mask_decision.json` deve carregar `trace_ids` e `text_instance_ids`. `text_id` sozinho nao e chave unica porque o strip reinicia IDs por band (`ocr_001`, `ocr_002`, ...). Se um artefato antigo trouxer apenas `band_id + text_ids`, o propagador pode derivar `trace_id = "{text_id}@{band_id}"`; se nem isso resolver de forma unica, a flag deve entrar em `missing_in_project`, nunca ser aplicada ao primeiro `ocr_001` encontrado.

#### `mask_chain_summary.json` (1 arquivo agregando todas as bands)

```json
{
  "band_count": 119,
  "bands_with_mask": 74,
  "bands_with_flags": 18,
  "totals": {
    "raw_mask_pixels": 152340,
    "expanded_mask_pixels": 218970,
    "outside_balloon_pixels": 0
  },
  "by_source": {
    "line_polygons": 60,
    "text_pixel_bbox": 11,
    "bbox": 3,
    "fallback": 0
  },
  "flagged_bands": ["page_002_band_017", "page_003_band_028"]
}
```

---

### 4.8. Translation — `07_translation/`

**Onde**: [pipeline/translator/translate.py](../../pipeline/translator/translate.py).

#### `translation_inputs.jsonl`

```json
{
  "text_id": "ocr_017",
  "route": "speech",
  "source_text_before_normalization": "VHEN IGETBACK TOWORK...",
  "source_text_sent_to_translator": "WHEN I GET BACK TO WORK...",
  "target_lang": "pt-BR",
  "backend": "google",
  "glossary_terms_applied": [],
  "context_used": true,
  "context_summary_chars": 124
}
```

#### `translation_outputs.jsonl`

```json
{
  "text_id": "ocr_017",
  "translated_raw": "QUANDO EU VOLTAR AO TRABALHO...",
  "translated_after_postprocess": "QUANDO EU VOLTAR AO TRABALHO...",
  "fallback_used": false,
  "mojibake_detected": false,
  "warnings": []
}
```

#### `translation_fallbacks.jsonl`

```json
{
  "text_id": "ocr_017",
  "primary_backend": "google",
  "primary_error": "HTTPSConnectionPool... timeout",
  "fallback_backend": "ollama",
  "fallback_model": "traduzai-translator",
  "fallback_success": true,
  "fallback_duration_sec": 2.14
}
```

#### Redaction obrigatória

- **Nunca** registrar Authorization headers, API keys, cookies.
- Para Ollama: registrar `prompt_hash` (sha1 dos primeiros 4096 chars), `prompt_preview` (≤ 256 chars), `raw_response_preview` (≤ 256 chars), `parse_status`.
- Truncar todos os strings para ≤ 4 KB.

---

### 4.9. Inpaint — `08_inpaint/`

**Onde**: [pipeline/inpainter/__init__.py](../../pipeline/inpainter/__init__.py), [pipeline/inpainter_legacy/classical.py](../../pipeline/inpainter_legacy/classical.py).

#### `inpaint_decision.json` (por band)

```json
{
  "schema_version": 1,
  "band_id": "page_002_band_017",
  "text_ids": ["ocr_017"],
  "trace_ids": ["ocr_017@page_002_band_017"],
  "trace_ids_in_band": ["ocr_017@page_002_band_017"],
  "used_real_inpaint": true,
  "used_fast_white_fill": false,
  "used_fast_local_fill": false,
  "skip_inpaint_requested": false,
  "skip_inpaint_honored": true,
  "remaining_inpaint_blocks": 2,
  "raw_mask_pixels": 1234,
  "expanded_mask_pixels": 1820,
  "changed_pixels_total": 1850,
  "changed_pixels_outside_expanded": 30,
  "changed_pixels_outside_effective_limit": 0,
  "raw_changed_outside_limit_mask": 4880,
  "cleanup_changed_outside_limit_mask": 2269,
  "residual_text_detected": false,
  "residual_score": 0.18,
  "duration_sec": 1.42,
  "backend": "lama|classical",
  "flags": []
}
```

Regra v3: flags vindas do inpaint, em especial `text_residual_after_inpaint`, precisam ser propagadas para `project.json.paginas[].text_layers[].qa_flags` antes de `evaluate_export_gate()`. O audit `11_qa_export_gate/qa_flag_propagation_audit.json.summary.inpaint_decision_flags` deve contar essas flags, e `qa_flag_not_propagated_count` precisa continuar `0`.

#### `residual_text_check.json` (por band)

```json
{
  "schema_version": 1,
  "band_id": "page_002_band_017",
  "search_region": "line_polygons_padded_5px",
  "dark_pixel_ratio_after": 0.012,
  "component_count_text_like": 1,
  "residual_score": 0.18,
  "has_residual": false,
  "threshold_used": 0.6,
  "comparison": {
    "before_dark_ratio": 0.18,
    "after_dark_ratio": 0.012,
    "delta": -0.168
  }
}
```

Ação quando `has_residual=true`:
1. tentar máscara mais agressiva local;
2. fallback para real inpaint local restrito ao `balloon_inner_region`;
3. se ainda falhar, emitir flag `text_residual_after_inpaint` e bloquear export gate.

#### `skip_inpaint_audit.json` — **P0-1**

```json
{
  "schema_version": 1,
  "config_skip_inpaint": true,
  "summary": {
    "total_bands": 119,
    "bands_with_skip_honored": 0,
    "bands_with_real_inpaint": 74,
    "bands_with_fast_white_fill": 0,
    "bands_with_fast_local_fill": 0,
    "blocker_emitted": "skip_inpaint_not_honored"
  },
  "evidence": {
    "images_diff_vs_originals_md5_changed_count": 6,
    "images_diff_vs_baseline_md5_identical_count": 6,
    "implication": "skip_inpaint flag was not propagated through run_chapter"
  },
  "fix_pointer": "pipeline/main.py:933 — wrap inpainter when config.skip_inpaint=true"
}
```

#### Pseudocódigo da correção (incluir no PR)

```python
# pipeline/main.py — substituir linha 939
def _build_inpainter_for_strip(config):
    real = SimpleNamespace(inpaint_band_image=inpaint_band_image)
    if not config.get("skip_inpaint"):
        return real
    def _noop(slice_img, page):
        page["_skip_inpaint_honored"] = True
        return slice_img  # devolve original sem inpaint
    return SimpleNamespace(inpaint_band_image=_noop)

# uso:
inpainter=_build_inpainter_for_strip(config),
```

---

### 4.10. Typesetting — `09_typeset/`

**Onde**: [pipeline/typesetter/renderer.py](../../pipeline/typesetter/renderer.py).

#### `render_plan_raw.jsonl` e `render_plan_final.jsonl`

`render_plan_raw.jsonl` e diagnostico band-local. `render_plan_final.jsonl` e o contrato canonico page-global usado pelo analyzer, pelo `project.json` e pelo export gate. O arquivo legado `render_plan.jsonl` nao deve ser usado como fonte primaria em checks novos.

```json
{
  "text_id": "ocr_017",
  "trace_id": "ocr_017@page_002_band_017",
  "text_instance_id": "page_002_band_017_ocr_017",
  "page_id": "page_002",
  "band_id": "page_002_band_017",
  "coordinate_space": "page",
  "original": "SO PLEASE WAIT A LITTLE LONGER. WHEN I GET BACK TO WORK...",
  "translated": "POR FAVOR, ESPERE MAIS UM POUCO. QUANDO EU VOLTAR AO TRABALHO...",
  "target_bbox": [137, 11390, 726, 11859],
  "position_bbox": [137, 11390, 726, 11859],
  "capacity_bbox": [120, 11380, 740, 11870],
  "safe_text_box": [148, 11400, 715, 11849],
  "render_bbox": [148, 11410, 715, 11800],
  "balloon_bbox": [77, 11293, 786, 11903],
  "font_name": "ComicNeue-Bold.ttf",
  "font_size_seed": 28,
  "font_size_final": 22,
  "line_height": 27,
  "wrapped_lines": ["POR FAVOR, ESPERE", "MAIS UM POUCO.", "QUANDO EU VOLTAR", "AO TRABALHO..."],
  "fit_status": "PASS",
  "qa_flags": [],
  "warnings": []
}
```

Regra v3: `render_plan_final.jsonl` deve bater com o layer final do `project.json` por `trace_id`; fallback aceitavel e `text_instance_id`, depois `text_id + band_id`. `text_id` isolado nao e suficiente. O strict deve falhar se `render_bbox`, `safe_text_box`, `balloon_bbox`, `page_id`, `band_id` ou `coordinate_space` divergem entre render plan e project.

#### `render_qa.jsonl`

Flags obrigatórias:

```text
TEXT_CLIPPED              ← render_bbox > balloon_bbox por > 4px
TEXT_OVERFLOW             ← font_size_final == minimum E ainda não cabe
render_outside_balloon    ← containment(render_bbox, balloon_bbox) < 0.85
render_on_art_suspected   ← background sample não-branco em balão branco
safe_text_box_missing     ← campo ausente
render_bbox_missing       ← campo ausente
balloon_bbox_missing      ← campo ausente (P0-10)
font_size_at_minimum      ← font_size_final == minimum E ainda passa
```

#### `balloon_bbox_missing_audit.jsonl` — **P0-10**

```json
{
  "text_id": "ocr_022",
  "band_id": "page_002_band_023",
  "warning_in_pipeline_log": "render_band_image: 1 text(s) sem balloon_bbox — RISCO DE OVERFLOW",
  "captured_at": "renderer.render_band_image",
  "fallback_used": "bbox_as_balloon_bbox",
  "consistency_with_debug_report_md": "INCONSISTENT_debug_report_says_zero"
}
```

#### Overlays por página
- `page_{NNN}_typeset_overlay.jpg`: imagem final + render_bbox em roxo, balloon_bbox em azul, safe_text_box em verde.
- `page_{NNN}_safe_text_box_overlay.jpg`: só safe_text_box, para diagnóstico de área disponível.
- `page_{NNN}_render_bbox_overlay.jpg`: render_bbox e check `containment(render_bbox, balloon_bbox)`.

---

### 4.11. Copy-back / reassemble — `10_copyback_reassemble/`

**Onde**: [pipeline/strip/process_bands.py](../../pipeline/strip/process_bands.py) (`_apply_copy_back_outside_balloons`).

```json
{
  "page_id": "page_002",
  "band_id": "page_002_band_017",
  "copyback_mask_pixels": 120000,
  "changed_pixels_before_copyback": 90000,
  "changed_pixels_after_copyback": 45000,
  "rendered_pixels_preserved": true,
  "rendered_pixels_overwritten": 0,
  "flags": []
}
```

#### `reassemble_manifest.json`

```json
{
  "input_pages": 6,
  "output_pages": 6,
  "strip_height": 82994,
  "page_layout": [
    {"page_id": "page_001", "y_top": 0, "y_bottom": 1079, "source_page_number": 1},
    {"page_id": "page_002", "y_top": 1079, "y_bottom": 17462, "source_page_number": 2}
  ],
  "page_cleanup_rerender_sec": 227.24,
  "page_cleanup_rerender_breakdown": {
    "_cleanup_page_inpaint_and_rerender": 198.5,
    "typesetter.render_band_image": 28.7
  }
}
```

> Instrumentar `page_cleanup_rerender` em sub-stages para confirmar onde o tempo vai (suspeita: render_band_image é o gargalo, não cleanup).

---

### 4.12. QA / export gate — `11_qa_export_gate/`

**Onde**: [pipeline/qa/export_gate.py](../../pipeline/qa/export_gate.py), [pipeline/main.py:1056](../../pipeline/main.py:1056).

#### `qa_export_gate_consistency.json` — **P0-3**

```json
{
  "schema_version": 1,
  "qa_summary": {
    "highest_severity": "low",
    "critical_count": 0,
    "total": 36,
    "flags": ["bbox_overreach_critical", "mask_density_high", ...]
  },
  "export_gate": {
    "status": "BLOCK",
    "critical_issue_count": 3,
    "review_issue_count": 16,
    "issue_count": 19
  },
  "consistency": {
    "critical_count_match": false,
    "highest_severity_consistent": false,
    "blocker": "qa_summary_and_export_gate_diverge"
  },
  "root_cause": {
    "file_a": "pipeline/qa/translation_qa.py",
    "list_a": "FLAG_SEVERITY",
    "file_b": "pipeline/qa/export_gate.py",
    "list_b": "P0_FLAGS",
    "missing_flags_in_a": [
      "bbox_overreach_critical",
      "mask_density_high",
      "mask_outside_balloon",
      "mask_outside_balloon_critical",
      "bbox_overreach"
    ],
    "fix": "unify into FLAG_SEVERITY; export_gate.py imports severity_for_flag"
  }
}
```

#### `strict_exit_audit.json` — **P0-2**

```json
{
  "schema_version": 1,
  "strict_mode_active": true,
  "export_gate_status": "BLOCK",
  "exit_code_observed": 0,
  "exit_code_expected": 2,
  "blocker": "strict_gate_not_enforced",
  "stdout_last_event_type": "complete",
  "stdout_last_event_expected": "error",
  "fix_pointer": {
    "file": "pipeline/main.py",
    "line": 1106,
    "snippet": "emit('complete', ...) without checking export_gate.status",
    "patch": "if strict and gate.status=='BLOCK': emit('error', ...); sys.exit(2)"
  }
}
```

#### `qa_issues.jsonl`

Uma linha por issue. Inclui **todos** os campos do `export_gate.issues` + cross-references:

```json
{
  "text_id": "ocr_001",
  "page_id": "page_001",
  "band_id": "page_001_band_002",
  "trace_id": "ocr_001@page_001_band_002",
  "type": "p0_render_blocker",
  "severity": "critical",
  "flags": ["bbox_overreach_critical"],
  "text_excerpt": "POR FAVOR!",
  "bbox": [88, 2716, 775, 3523],
  "linked_artifacts": [
    "05_layout_geometry/source_bbox_balloon_overreach.jsonl#ocr_001",
    "06_mask_segmentation/page_001_band_002/mask_decision.json",
    "09_typeset/render_plan_final.jsonl#ocr_001@page_001_band_002"
  ]
}
```

#### `visual_blockers.jsonl`

Apenas issues que **bloqueiam** o export em strict:

```text
bbox_overreach_critical
mask_outside_balloon_critical
layout_bbox_coordinate_mismatch
text_residual_after_inpaint
render_outside_balloon
render_on_art_suspected
skip_inpaint_not_honored
strict_gate_not_enforced
TEXT_CLIPPED
TEXT_OVERFLOW
source_script_leak
special_class_rendered_as_dialogue
mojibake_in_translation
qa_summary_and_export_gate_diverge
```

`visual_blockers_overlay.jpg`: contact sheet 4×N com snapshot 320×320 de cada blocker (recorte centrado em `bbox`), label `text_id + flag` por baixo.

---

## 5. Contact sheets — `12_contact_sheets/`

### 5.1. `translated_comparison.jpg`

Grid `N_pages × 4_runs` (A/B/C/D). Cada célula 240px de largura, altura proporcional. Header com `run_id` + `export_gate.status` + `exit_code`.

### 5.2. `problem_bands.jpg`

Top 20 bands com mais flags (ordenado por `severity_rank × flag_count`). Cada célula: thumbnail da band original + thumbnail após inpaint + lista de flags por baixo.

### 5.3. `mask_chain_top_issues.jpg`

Top 12 bands com `mask_outside_balloon` ou `mask_density_high`. Mostra `04_balloon_mask.png` ao lado de `09_final_inpaint_mask.png` com diff em vermelho.

### 5.4. `typeset_top_issues.jpg`

Top 12 textos com `render_outside_balloon`, `TEXT_CLIPPED`, `TEXT_OVERFLOW`. Mostra `render_bbox` + `balloon_bbox` em overlay.

---

## 6. Plano de implementação por PR

Cada PR é independente e mergeable. O `DebugRecorder` precisa existir antes; os outros são paralelizáveis após PR 1.

### PR 1 — Infraestrutura do DebugRecorder + P0 fixes mínimos

**Arquivos novos**:
- `pipeline/debug_tools/__init__.py`
- `pipeline/debug_tools/recorder.py`
- `pipeline/debug_tools/ids.py`
- `pipeline/debug_tools/schemas.py`
- `pipeline/tests/test_debug_recorder.py`

**Arquivos modificados**:
- `pipeline/main.py` — bootstrap do recorder, snapshots iniciais; **P0-1** wrapper de inpainter; **P0-2** exit_code em strict; redirect emit para usar recorder.

**Critérios de aceite**:
- `pipeline roda com debug=false` → 0 arquivos novos em `work_dir`, overhead ≤ 0.5%.
- `pipeline roda com debug=true` → existe `debug/e2e/debug_manifest.json` e `00_run/config_snapshot.json`.
- Erro forçado no recorder vira linha em `debug_errors.jsonl`, run continua.
- `B_skip_inpaint`: `B/images/*.jpg` MD5 == `B/originals/*.jpg` MD5 (ou clean copy).
- `D_strict + BLOCK` → `_stdout.jsonl` última linha tem `"type":"error"`, processo retorna 2.

---

### PR 2 — Strip / detect / OCR debug

**Arquivos modificados**:
- `pipeline/strip/run.py` — `bands_manifest`, `detect_candidates`, `strip_overlay`.
- `pipeline/strip/process_bands.py` — propagar `band_id` e `text_id` no OCR.
- `pipeline/vision_stack/runtime.py` — capturar `confidence` antes de `_finalize_page_ocr_texts` (**P0-5**).
- `pipeline/ocr/contextual_reviewer.py` — `ocr_dedupe_decisions.jsonl`.

**Critérios**:
- `bands_manifest.json` lista 119 bands com `band_id` estável.
- `ocr_raw_blocks.jsonl` tem 1 linha por bloco aceito, com `confidence_raw`.
- `ocr_confidence_audit.json` mostra `blocks_with_confidence_zero=0` (após fix).
- Overlay `page_001_ocr_overlay.jpg` mostra cada bloco com label `text_id + conf`.

---

### PR 3 — Normalização e router

**Arquivos novos**:
- `pipeline/ocr/text_normalizer.py`
- `pipeline/ocr/text_router.py`
- `pipeline/debug_tools/detectors.py` (mojibake regex, joined-word heuristic, sfx marker)

**Arquivos modificados**:
- `pipeline/ocr/contextual_reviewer.py` — chamar normalizer antes do envio à tradução.
- `pipeline/translator/translate.py` — detectar mojibake na saída e tentar fix automático (**P0-4**).

**Critérios**:
- `IGETBACK TOWORK` gera `joined_word_suspect` em `joined_word_splits.jsonl`.
- `DON'T HIT SFX: KICK MY MOM!` gera evento `speech_with_sfx_split`.
- `T/N:` → route `editorial_note` + `translate_policy=skip_translation`.
- URL/handle → `url_watermark`.
- Mojibake na saída do Google → `mojibake_audit.jsonl` + tentativa de fix automático com flag `mojibake_in_translation`.

---

### PR 4 — Layout audit (**P0-6 + P0-7**)

**Arquivos modificados**:
- `pipeline/strip/run.py:168` — adicionar `layout_bbox` à lista de keys em `_shift_text_geometry_y`.
- `pipeline/strip/run.py:240` — auditar `_finalize_output_page_ocr_metadata` e onde `bbox` é sobrescrito.
- `pipeline/layout/balloon_layout.py` — não propagar `balloon_bbox` para `source_bbox` em `assign_balloon_bbox`.
- `pipeline/layout/simple_text_geometry.py` — emitir `coordinate_space` em cada bloco.
- `pipeline/debug_tools/bbox.py` — `audit_bbox_coordinate_space()`.

**Critérios**:
- `bbox_coordinate_audit.json` mostra `mixed_coordinate_space_count=0` após fix.
- `source_bbox_balloon_overreach.jsonl` lista os casos restantes (PLEASE, etc.).
- `layout_blocks.jsonl` tem `coordinate_space="page"` em **todos** os blocos do project.json final.
- Overlay `page_001_layout_overlay.jpg` mostra todas as 7 cores por bbox.

---

### PR 5 — Mask chain debug

**Arquivos modificados**:
- `pipeline/inpainter/mask_builder.py` — exportar máscaras intermediárias.
- `pipeline/inpainter/__init__.py` — chamar recorder após cada bloco da máscara.
- `pipeline/debug_tools/masks.py` — utilitários de rendering.

**Critérios**:
- Para cada band com inpaint: existem 10 PNGs + `mask_decision.json`.
- `mask_chain_summary.json` agrega 119 bands.
- `mask_density_in_band > 0.12` gera flag em `qa_issues.jsonl`.
- `outside_balloon_pixels > 50` gera flag `mask_outside_balloon_critical`.

---

### PR 6 — Inpaint debug padronizado + residual

**Arquivos modificados**:
- `pipeline/inpainter/__init__.py` — gerar `inpaint_decision.json` por band com flag `skip_inpaint_honored`.
- `pipeline/inpainter_legacy/classical.py` — emitir `before.jpg`/`after.jpg` quando level=full.
- novo `pipeline/qa/inpaint_residual.py` — `detect_residual_text()`.

**Critérios**:
- `skip_inpaint=true` → 100% das bands com `skip_inpaint_honored=true` e nenhum byte alterado.
- Fast fill com texto residual → `residual_text_check.json.has_residual=true` e flag `text_residual_after_inpaint`.
- `changed_outside_expanded_pixels` é registrado por band.

---

### PR 7 — Typesetting render plan

**Arquivos modificados**:
- `pipeline/typesetter/renderer.py` — emitir `render_plan_raw.jsonl` e `render_plan_final.jsonl`; expor `safe_text_box` e `render_bbox` no `project.json`; emitir warning `balloon_bbox_missing_audit` (**P0-10**).
- `pipeline/qa/render_geometry.py` (novo) — `check_render_inside_balloon`, `check_render_background`.

**Critérios**:
- Cada bloco renderizado tem 1 linha final em `render_plan_final.jsonl` com `fit_status`.
- `render_outside_balloon` aparece quando containment < 0.85.
- `render_on_art_suspected` aparece quando bg luma < 215 em balão branco.
- `balloon_bbox_missing_audit.jsonl` casa com `pipeline.log` warnings.

---

### PR 8 — QA/export gate consistente + report (**P0-3**)

**Arquivos modificados**:
- `pipeline/qa/translation_qa.py` — unificar `FLAG_SEVERITY` com `P0_FLAGS` e `HIGH_LAYOUT_FLAGS`.
- `pipeline/qa/export_gate.py` — importar `severity_for_flag` em vez de manter listas locais.
- `pipeline/main.py:1056` — após `evaluate_export_gate`, gerar `qa_export_gate_consistency.json`.
- `pipeline/debug_tools/report.py` — gerar `debug_report.md` + `debug_report.json`.

**Critérios**:
- `qa.summary.critical_count == qa.export_gate.critical_issue_count` em todas as runs.
- `qa.summary.highest_severity == "critical"` quando há flags P0.
- `debug_report.md` lista top 10 issues com link relativo a cada artefato.
- Strict + BLOCK → exit_code=2 + `_stdout.jsonl` termina com `{"type":"error"}`.

---

## 7. Testes obrigatórios

```text
pipeline/tests/test_debug_recorder.py               — IO seguro, schema_version, tolerância a falhas
pipeline/tests/test_debug_bbox_audit.py             — detecta band-local vs page-global
pipeline/tests/test_text_normalization_debug.py     — IGETBACK → I GET BACK
pipeline/tests/test_text_router_debug.py            — T/N, sign, SFX split
pipeline/tests/test_mask_chain_debug.py             — chain completa + outside_balloon
pipeline/tests/test_typeset_render_plan_debug.py    — render_plan_final.jsonl + flags
pipeline/tests/test_export_gate_debug_consistency.py — qa.summary == export_gate
pipeline/tests/test_skip_inpaint_debug_contract.py  — skip_inpaint_honored em todas as bands
pipeline/tests/test_mojibake_audit.py               — regex + fix automático
pipeline/tests/test_confidence_preservation.py      — confidence > 0 após _finalize
pipeline/tests/regression/test_e2e_debug_20260517.py — fixture mínima do ZIP baseline
```

### Casos mínimos por teste

| Teste | Caso | Resultado esperado |
|---|---|---|
| `skip_inpaint_debug_contract` | `config.skip_inpaint=true`, 1 band com balão branco | `inpaint_decision.skip_inpaint_honored=true`, image == original |
| `bbox_audit` | Texto com bbox.y=65, source_bbox.y=2716 | `findings` contém `layout_bbox_coordinate_mismatch` severity critical |
| `text_normalization` | `"VHEN IGETBACK TOWORK"` | `normalized="WHEN I GET BACK TO WORK"`, token_diff registrado |
| `text_router` | `"DON'T HIT SFX: KICK MY MOM!"` | route=`speech_with_sfx_split`, parts=2 |
| `text_router` | `"T/N: HYUNGNIM..."` | route=`editorial_note`, translate_policy=`skip_translation` |
| `text_router` | `"https://lagoonscans.com"` | route=`url_watermark` |
| `mask_chain` | Máscara cruzando borda do balão por 200px | flag `mask_outside_balloon`, severity warning |
| `typeset` | `render_bbox` 30% fora do balão | flag `render_outside_balloon`, severity critical |
| `export_gate_consistency` | Flag `bbox_overreach_critical` em 1 bloco | `summary.critical_count=1 == export_gate.critical_issue_count=1` |
| `strict_exit` | strict=true + BLOCK | `sys.exit(2)`, `_stdout.jsonl` última linha `type=error` |
| `mojibake` | `"VOCÃŠ SABE"` na saída do tradutor | `mojibake_audit` com sample, fix `"VOCÊ SABE"` |
| `confidence_preservation` | Block com conf 0.951 no accept | project.json `confianca_ocr == 0.951` |

---

## 8. Comandos de validação

Após implementar, rodar **as 4 runs canônicas** + analisador:

```bash
# A — baseline debug
python pipeline/main.py --input <chapter_dir> --work "Chapter 1" \
  --source-lang en --target pt-BR --debug \
  --output debug/e2e/A_baseline_debug

# B — sem inpaint (validar P0-1)
python pipeline/main.py --input <chapter_dir> --work "Chapter 1" \
  --source-lang en --target pt-BR --debug --skip-inpaint \
  --output debug/e2e/B_skip_inpaint

# C — fast fill ligado
$env:TRADUZAI_STRIP_FAST_WHITE_INPAINT = "1"
$env:TRADUZAI_STRIP_FAST_LOCAL_INPAINT = "1"
python pipeline/main.py --input <chapter_dir> --work "Chapter 1" \
  --source-lang en --target pt-BR --debug \
  --output debug/e2e/C_fast_fill

# D — strict (validar P0-2)
python pipeline/main.py --input <chapter_dir> --work "Chapter 1" \
  --source-lang en --target pt-BR --debug --strict --export-mode strict \
  --output debug/e2e/D_strict_export_gate

# Análise
python tools/analyze_e2e_debug.py <run_root> --write-report --strict-debug-audit
echo $LASTEXITCODE
```

Codigo de saida esperado: `0` quando o pacote passa todos os invariantes do analyzer; `3` quando o pacote foi lido mas algum invariante strict falhou; `2` pertence ao pipeline em run `--strict` bloqueada pelo export gate, nao ao analyzer.

`<run_root>` deve ser a pasta que contem as runs A/B/C/D ou uma run unica com `project.json`, `runner_config.json` e `debug/e2e/`. Nao apontar diretamente para `debug/e2e/` enquanto o analyzer nao aceitar esse formato como alias; isso pode produzir `run_count=0`.

### Métricas que o analisador deve reportar

```text
- exit_code por run                                     ← P0-2
- export_gate.status por run
- needs_review por run
- qa.summary.critical_count vs export_gate.critical    ← P0-3
- skip_inpaint_honored_bands / total_bands             ← P0-1; `0/0` e valido quando `skip_inpaint` pula a etapa inteira
- images MD5 vs originals MD5 (A vs B)                  ← P0-1
- mojibake_match_count em translated                    ← P0-4
- confianca_ocr_zero_count                              ← P0-5
- mixed_coordinate_space_count                          ← P0-6
- source_bbox_equals_balloon_bbox_count                 ← P0-7
- balloon_bbox_missing_count (parse pipeline.log)       ← P0-10
- env vars TRADUZAI_* persistidas no runner_config      ← P0-9
- page_cleanup_rerender breakdown sub-stages
- per-band: raw / expanded / changed / residual pixels
- render outside balloon count
- render on art count
- bbox overreach count
- content class counts
```

---

## 9. Critérios finais de aceite global

A implementação inteira do debug E2E está **pronta** quando, olhando **apenas** os artefatos em `debug/e2e/`, é possível responder estas 15 perguntas em ≤ 60 segundos:

1. Por que `ocr_017` foi aceito ou rejeitado? → `03_ocr/ocr_raw_blocks.jsonl#ocr_017`
2. Qual bbox foi escolhida e qual descartada no dedupe? → `03_ocr/ocr_dedupe_decisions.jsonl#ocr_017`
3. O texto foi normalizado? Quais tokens mudaram? → `04_text_normalization_router/normalization_trace.jsonl#ocr_017`
4. Houve texto colado suspeito? → `04_text_normalization_router/joined_word_splits.jsonl`
5. Houve SFX/T/N/sign/URL misturado com fala? → `04_text_normalization_router/text_router_decisions.jsonl`
6. Qual área foi usada para layout? → `05_layout_geometry/layout_blocks.jsonl#ocr_017.bboxes.layout_bbox`
7. Qual área foi usada para máscara? → `06_mask_segmentation/{band_id}/mask_decision.json.mask_source`
8. A máscara ficou dentro do balão? → `06_mask_segmentation/{band_id}/mask_decision.json.gates.mask_outside_balloon`
9. Qual backend traduziu? → `07_translation/translation_inputs.jsonl#ocr_017.backend`
10. Fast fill, local fill, real inpaint, ou skip? → `08_inpaint/{band_id}/inpaint_decision.json.{used_*}`
11. Sobrou texto fantasma após inpaint? → `08_inpaint/{band_id}/residual_text_check.json.has_residual`
12. Qual fonte/tamanho/quebra de linha o typesetter escolheu? → `09_typeset/render_plan_final.jsonl#trace_id`
13. O texto ficou dentro do safe_text_box e do balão? → `09_typeset/render_qa.jsonl#ocr_017.qa_flags`
14. O copy-back sobrescreveu algo importante? → `10_copyback_reassemble/copyback_decisions.jsonl`
15. Por que o export gate bloqueou ou passou? → `11_qa_export_gate/export_gate.json` + `qa_export_gate_consistency.json` + `strict_exit_audit.json`

### Aceite final (checklist)

- [ ] **P0-1** `B_skip_inpaint`: 100% bands com `skip_inpaint_honored=true`; imagens iguais aos originais.
- [ ] **P0-2** `D_strict + BLOCK` → `exit_code=2`, último evento stdout = `error`.
- [ ] **P0-3** `qa.summary.critical_count == qa.export_gate.critical_issue_count` em todas as runs.
- [ ] **P0-4** `mojibake_audit.jsonl` vazio na run final; nenhuma sequência `Ã[-¿]` em `translated`.
- [ ] **P0-5** `confianca_ocr > 0` em ≥ 80% dos textos do project.json.
- [ ] **P0-6** `bbox_coordinate_audit.json.mixed_coordinate_space_count == 0`.
- [ ] **P0-7** Para PLEASE: `qa_flags` **sem** `bbox_overreach_critical`.
- [ ] **P0-8** Para VHEN: `skip_processing=true reason="duplicate_in_same_balloon"`.
- [ ] **P0-9** `00_run/env_snapshot.json.env_vars.TRADUZAI_STRIP_FAST_WHITE_INPAINT` presente quando `C_fast_fill`.
- [ ] **P0-10** `09_typeset/balloon_bbox_missing_audit.jsonl` casa com `pipeline.log` warns.
- [ ] Todas as 13 etapas geram seus arquivos obrigatórios.
- [ ] Erro forçado em qualquer escrita de debug gera linha em `debug_errors.jsonl` sem interromper run.
- [ ] `debug_report.md` automático lista top 10 issues com links relativos funcionais.
- [ ] Run com `debug=false` produz **zero** arquivos em `debug/e2e/`.

---

## 10. Tabela de flags e severidade canônica

A unificação **P0-3** deve resultar nesta tabela como fonte única em `pipeline/qa/translation_qa.py`:

| Flag | Severidade | Bloqueia strict? | Origem |
|---|---|---|---|
| `bbox_overreach_critical` | critical | sim | mask_builder |
| `layout_bbox_coordinate_mismatch` | critical | sim | debug bbox_audit (P0-6) |
| `source_bbox_assigned_from_balloon` | critical | sim | debug layout (P0-7) |
| `mask_outside_balloon_critical` | critical | sim | mask_builder |
| `render_outside_balloon` | critical | sim | typesetter QA |
| `render_on_art_suspected` | critical | sim | typesetter QA |
| `text_residual_after_inpaint` | critical | sim | inpaint residual |
| `source_script_leak` | critical | sim | export_gate |
| `special_class_rendered_as_dialogue` | critical | sim | text_router |
| `glossary_violation` | critical | sim | translation_qa |
| `placeholder_lost` | critical | sim | translation_qa |
| `mojibake_in_translation` | critical | sim | text normalizer (P0-4) |
| `skip_inpaint_not_honored` | critical | sim | debug skip audit (P0-1) |
| `strict_gate_not_enforced` | critical | sim | debug strict audit (P0-2) |
| `qa_summary_and_export_gate_diverge` | critical | sim | debug consistency (P0-3) |
| `speech_cjk_preserved_inside_balloon` | critical | sim | export_gate |
| `TEXT_CLIPPED` | high | sim | typesetter |
| `TEXT_OVERFLOW` | high | sim | typesetter |
| `text_overflow_high` | high | sim | typesetter |
| `outline_damage_high` | high | sim | typesetter |
| `balloon_bbox_collapsed_to_text` | high | sim | layout |
| `balloon_bbox_missing` | high | sim | typesetter (P0-10) |
| `mask_outside_balloon` | high | sim | mask_builder |
| `mask_density_high` | high | sim | mask_builder |
| `bbox_overreach` | medium | warn | mask_builder |
| `safe_text_box_outside_balloon` | medium | warn | typesetter QA |
| `sign_render_outside_region` | medium | warn | text_router |
| `tn_note_rendered_as_speech` | medium | warn | text_router |
| `url_watermark_inpainted` | medium | warn | text_router |
| `ocr_run_on_suspect` | medium | warn | OCR |
| `ocr_false_positive_review` | medium | warn | OCR (P0-8) |
| `ocr_duplicate_garble_review` | medium | warn | dedupe |
| `low_ocr_confidence` | medium | warn | OCR |
| `top_narration` | low | não | layout |

---

## 11. Riscos e mitigações

### 11.1. Risco: debug full = 500 MB+ por run

Mitigação:
- `level=standard` pula PNGs/JPEGs para bands sem flags.
- Documentar `TRADUZAI_DEBUG_LEVEL=minimal` para CI.
- Adicionar `debug_compress=true` que tarball-zipa `debug/e2e/` no final.

### 11.2. Risco: instrumentar typesetter quebra render por overhead serial

Mitigação:
- Recorder usa `append` simples, sem fsync.
- Overlays JPEG só para bands com `qa_flags ≠ []`.
- Benchmark contra baseline antes de merge final.

### 11.3. Risco: ContextVar não funciona em ThreadPool

Mitigação:
- `_start_inpainter_prewarm` usa ThreadPoolExecutor — passar recorder explicitamente nessas branchs.
- `bind_recorder` em qualquer thread filha.

### 11.4. Risco: `_stdout.jsonl` em UTF-16 LE atrapalha o analisador

Mitigação:
- O runner (PowerShell) **deve** definir `$ProcessStartInfo.StandardOutputEncoding = [Text.Encoding]::UTF8`.
- Analisador detecta BOM `FF FE` e re-decodifica automaticamente.

### 11.5. Risco: o fix de `assign_balloon_bbox` (P0-7) quebra `connected_text_groups`

Mitigação:
- `connected_lobe_bboxes` continua sendo populado pelo cluster, **separado** de `source_bbox`.
- Regressão: `pipeline/tests/regression/test_connected_text_cut.py`.

### 11.6. Risco: unificar `FLAG_SEVERITY` quebra UI/store

Mitigação:
- `src/lib/stores/appStore.ts` e tipos TS importam de `pipeline/qa/flag_catalog.json` (gerado pelo Python como source of truth).
- Manter backward compat de nomes de flag em pelo menos 2 versões.

---

## 12. Quick reference para devs

### Adicionar instrumentação em um lugar novo

```python
# Em qualquer lugar do pipeline:
from debug_tools import event, get_recorder

event("ocr", "accept_block", text_id="ocr_017", confidence=0.91)

r = get_recorder()
if r:
    r.write_jsonl("03_ocr/ocr_raw_blocks.jsonl", {"text_id": "ocr_017", ...})
    r.write_image(f"06_mask_segmentation/{band_id}/09_final_inpaint_mask.png", mask_uint8)
```

### Adicionar uma flag P0 nova

1. Adicionar entrada em `pipeline/qa/translation_qa.py::FLAG_SEVERITY` com severity correta.
2. Adicionar entrada na tabela §10 deste documento.
3. Garantir que `pipeline/qa/export_gate.py` reusa `severity_for_flag(flag)`.
4. Adicionar caso em `pipeline/tests/test_export_gate_debug_consistency.py`.

### Adicionar uma sub-stage de timing

```python
with recorder.time_stage("page_cleanup_rerender.typesetter_render"):
    typesetter.render_band_image(...)
```

Aparece em `debug_manifest.json.stage_durations_sec`.

### Investigar uma run falha

1. Abrir `debug/e2e/13_report/debug_report.md` — overview.
2. Para cada blocker, seguir `linked_artifacts` em `11_qa_export_gate/qa_issues.jsonl`.
3. Comecar sempre pelo `trace_id`. Se ele faltar, usar `text_instance_id`; se ambos faltarem, a issue nao e rastreavel e deve falhar em `qa_issues_are_traceable`.
4. Cruzar a mesma identidade em `project.json`, `07_translation/translation_inputs.jsonl`, `07_translation/translation_outputs.jsonl`, `08_inpaint/inpaint_blocks.jsonl`, `09_typeset/render_plan_final.jsonl`, `10_copyback_reassemble/copyback_decisions.jsonl`, `11_qa_export_gate/qa_issues.jsonl` e `visual_blockers.jsonl`.
5. Para entender geometria: `05_layout_geometry/layout_blocks.jsonl#trace_id`.
6. Para entender máscara: `06_mask_segmentation/{band_id}/10_mask_overlay.jpg`.
7. Para entender render: `09_typeset/page_overlays/page_{NNN}_render_bbox_overlay.jpg`.

---

## 5b. Invariantes obrigatórios do debug v2

Após o ZIP 2026-05-18, ficou claro que ter a árvore criada **não basta**: arquivos vazios, IDs nulos, contagens dobradas e flags não-propagadas corroem a confiança no debug. Os invariantes abaixo são **dura** e devem virar testes automáticos (PR 17) e gate do `analyze_e2e_debug.py --strict-debug-audit`.

### 5b.1. Integridade de IDs e coordenadas

1. **`render_plan_final.jsonl` não pode ter `page_id`, `band_id`, `text_id` ou `coordinate_space` nulos**. Verificado no analyzer; conta como `render_plan_null_id_count`.
2. **`render_plan_final.jsonl` não pode ter mais entries que o número de blocos finais**. Hoje há 168 entries para 5 distinct text_ids (DBG2-05). Conta como `render_plan_duplicate_final_entry_count`. Tolerância: 1 entry final por `trace_id` ou `text_instance_id` (versões intermediárias vão em `render_plan_raw.jsonl`). `text_id` isolado nao e chave unica.
3. **Todos os bboxes finais em `project.json`, `render_plan_final.jsonl`, `qa_issues.jsonl` e `export_gate.json` devem estar em coordenada global de página**. Auditado por `bbox_coordinate_audit.json` para **todas** as keys derivadas (não apenas `layout_bbox`).
4. **`coordinate_space` é obrigatório em todo bloco geometric** — valores permitidos: `"band"`, `"page"`, `"strip"`. Conta como `render_plan_null_coordinate_space_count` e `derived_bbox_coordinate_mismatch_count`.

### 5b.2. Integridade do agregador

5. **`debug_report.json` deve recomputar contagens a partir de uma única fonte canônica**. Não pode somar `text_layers + textos` (DBG2-02). Fonte canônica: `page.text_layers` se existir, senão `page.textos`.
6. **Confidence canônica por layer**:
```python
confidence = first_present(
    layer.get("confidence_raw"),
    layer.get("ocr_confidence"),
    layer.get("confianca_ocr"),
    layer.get("confidence"),
)
# Se ausente, contar como "missing", NÃO como zero.
```
Resolve DBG2-01/03.
7. **`debug_report_consistency.json` deve comparar** e marcar inconsistência entre: `project.json`, `events.jsonl`, `09_typeset/render_plan_final.jsonl`, `11_qa_export_gate/qa_issues.jsonl`, `pipeline.log`. Output: `all_consistent: true|false` + lista de divergências.

### 5b.3. Cobertura por etapa

8. **`07_translation/` não pode ficar vazio** quando há textos traduzíveis. Pelo menos `translation_inputs.jsonl` e `translation_outputs.jsonl` precisam existir, com contagem ≥ número de textos com `skip_processing=false` E `translate_policy != skip_translation`. Resolve DBG2-08.
9. **`10_copyback_reassemble/` não pode ficar vazio** quando há `page_cleanup_rerender > 0`. Resolve DBG2-16.
10. **`12_contact_sheets/`** deve ter no mínimo `translated_comparison.jpg` e `problem_bands.jpg` em qualquer run com `debug=true`. Resolve DBG2-17.

### 5b.4. Propagação de flags e classes

11. **Toda flag visual** detectada em mask/typeset/router deve aparecer no **layer final** do `project.json` E no `export_gate` (quando blocker). Auditado por `qa_flag_propagation_audit.json`. Resolve DBG2-14.
12. **Classes especiais não podem virar `dialogue`**: `cover_credit`, `logo`, `noise`, `sign`, `editorial_note`, `url_watermark`, `scanlator_credit` precisam ter `content_class != "dialogue"` **OU** `needs_review=true` **OU** `skip_processing=true`. Resolve DBG2-11/15.
13. **`skip_inpaint_honored`** deve ser `true` ou `false` (nunca `null`) em **toda** band quando `config.skip_inpaint=true` E todas precisam ser `true`. Resolve DBG2-21.

### 5b.5. Comportamento base

14. **`debug=false` continua sem gerar `debug/e2e/`**. Validar com test de não-regressão.
15. **`debug_manifest.json.stage_durations_sec` não pode ser `{}`** quando `debug=true`. `recorder.time_stage()` precisa estar instrumentado em pelo menos: `extract`, `strip_detect`, `ocr`, `text_normalization_router`, `layout`, `mask`, `translation`, `inpaint`, `typeset`, `copyback`, `qa`. Resolve DBG2-22.
16. **Arquivos `stage`-level e agregadores devem bater em contagens**: `source_bbox_balloon_overreach.jsonl` line_count == `debug_report.metrics.source_bbox_equals_balloon_bbox_count`. Resolve DBG2-23.

---

## 6.2. Plano de correção v2 após o debug 2026-05-18

Os PRs 1-8 (§6) atacavam P0-1 a P0-10 do baseline. Os PRs 9-17 abaixo atacam DBG2-01 a DBG2-23 do pacote 2026-05-18. **Ordem recomendada**: PR 9 + PR 10 primeiro (analyzer e coordenadas confiáveis são pré-requisito de tudo); depois PR 11, 12, 13, 14 em paralelo; PR 15 e 16 depois; PR 17 fecha como gate de regressão.

### PR 9 — Analyzer/report correctness (DBG2-01/02/03/23)

**Objetivo**: corrigir `debug_report.md/json` para não mentir métricas.

**Arquivos-alvo**:
```text
tools/analyze_e2e_debug.py
pipeline/debug_tools/report.py
pipeline/tests/test_e2e_debug_report_consistency.py
```

**Regras**:
- Não contar `text_layers` e `textos` ao mesmo tempo. Fonte canônica: `page.text_layers` se existir, senão `page.textos`.
- Confidence canônica via `first_present(confidence_raw, ocr_confidence, confianca_ocr, confidence)`. Ausente = `missing`, **não** zero.
- `content_class_counts` calculado uma vez por `text_id`.
- Cross-check stage-level files vs métricas agregadas:
  - `source_bbox_balloon_overreach.jsonl` line_count == `metrics.source_bbox_equals_balloon_bbox_count`
  - `09_typeset/balloon_bbox_missing_audit.jsonl` line_count == `metrics.balloon_bbox_missing_count`
  - `08_inpaint/*/inpaint_decision.json.skip_inpaint_honored` aggregado == `metrics.skip_inpaint_honored_bands`
- Gerar `13_report/debug_report_consistency.json`.

**Aceite**:
```text
debug_report.text_count == project.estatisticas.total_textos
confidence_zero_count == ocr_confidence_audit.blocks_with_confidence_zero
content_class_counts não dobra dialogue/narration
skip_inpaint_honored ≠ null quando config.skip_inpaint=true
```

---

### PR 10 — Trace integrity e coordinate audit v2 (DBG2-04/05/06/07)

**Objetivo**: render_plan rastreável + coordenadas auditadas em **todas** as keys derivadas.

**Arquivos-alvo**:
```text
pipeline/typesetter/renderer.py
pipeline/strip/run.py
pipeline/debug_tools/bbox.py
pipeline/tests/test_render_plan_trace_integrity.py
pipeline/tests/test_derived_bbox_coordinate_audit.py
```

**Regras**:
- Separar `render_plan_raw.jsonl` (band-local, com `coordinate_space="band"`, `band_y_top`, `page_id`, `band_id`, `text_id`) de `render_plan_final.jsonl` (apenas page-global, **1 entry por `trace_id` ou `text_instance_id`**).
- Renderer chamado N vezes por band não pode multiplicar entries no `_final`. Solução: deduplicar por `text_id` mantendo a última versão page-global, ou emitir só após o `copyback`.
- `_shift_text_geometry_y` em [pipeline/strip/run.py:168](../../pipeline/strip/run.py:168) precisa cobrir **também**: `render_bbox`, `safe_text_box`, `_debug_safe_text_box`, `layout_safe_bbox`, `position_bbox`, `capacity_bbox`, `target_bbox`, `connected_position_bboxes` (quando band-local), `qa_metrics.*bbox`, `_render_debug.*bbox`.
- `bbox_coordinate_audit.json` deve auditar **todas** estas keys, não só `layout_bbox`. Schema:
```json
{
  "summary": {
    "total_text_layers": 88,
    "all_consistent": false,
    "mixed_coordinate_space_count": 0,
    "derived_bbox_coordinate_mismatch_count": 12,
    "by_key": {
      "bbox":            {"page": 88, "band": 0, "mismatch": 0},
      "render_bbox":     {"page": 76, "band": 12, "mismatch": 12},
      "safe_text_box":   {"page": 76, "band": 12, "mismatch": 12},
      "position_bbox":   {"page": 76, "band": 12, "mismatch": 12},
      "capacity_bbox":   {"page": 76, "band": 12, "mismatch": 12}
    }
  },
  "findings": [
    {"text_id": "ocr_001", "key": "render_bbox", "value": [492,65,641,84], "expected_space": "page", "band_y_top": 2700, "severity": "critical"}
  ]
}
```

**Aceite**:
```text
render_plan_null_id_count = 0
render_plan_null_coordinate_space_count = 0
render_plan_duplicate_final_entry_count = 0
derived_bbox_coordinate_mismatch_count = 0
project.render_bbox e project.safe_text_box em page-global
```

---

### PR 11 — Translation debug completo (DBG2-08)

**Objetivo**: preencher `07_translation/`.

**Arquivos-alvo**:
```text
pipeline/translator/translate.py
pipeline/translator/context.py
pipeline/debug_tools/text_diff.py
pipeline/tests/test_translation_debug_outputs.py
```

**Artefatos obrigatórios**:
```text
07_translation/translation_inputs.jsonl
07_translation/translation_outputs.jsonl
07_translation/glossary_application.jsonl
07_translation/translation_fallbacks.jsonl
07_translation/translation_debug_summary.json
```

**Regras**:
- Registrar `source_text_before_normalization` E `source_text_sent_to_translator` separadamente (resolve DBG2-13).
- Registrar `backend`, `model`, `fallback_used`, `duration_ms`, `prompt_hash` (sha1 truncado), `raw_response_preview` (≤ 256 chars), `final_translation_after_postprocess`.
- Redigir secrets/API keys (Authorization, cookie, x-api-key).
- `translation_inputs_count` deve bater com quantidade de textos traduzíveis (excluindo `skip_processing=true` e `translate_policy=skip_translation`).
- `translation_debug_summary.json` agrega: total inputs, total outputs, fallback rate, backend distribution, mojibake count, glossary application count.

**Aceite**:
```text
translation_debug_entry_count > 0 quando text_count > 0
translation_outputs_count == translation_inputs_count
nenhuma linha contém API key/cookie/header Authorization
```

---

### PR 12 — OCR geometry dedupe e cover/noise router (DBG2-09/10/11)

**Objetivo**: corrigir `PLEASE!`, `Shadow Erian Shadow`, `NTEEM`.

**Arquivos-alvo**:
```text
pipeline/ocr/contextual_reviewer.py
pipeline/ocr/text_router.py
pipeline/vision_stack/runtime.py
pipeline/tests/test_ocr_geometry_dedupe.py
pipeline/tests/test_cover_noise_router.py
```

**Regras de dedupe** (em `contextual_reviewer.py`):
- Em duplicatas ou blocos no mesmo `balloon_bbox`, calcular `geometry_quality_score`. Penalizar:
  - `area(source_bbox) / area(text_pixel_bbox) > 8` → −45
  - `line_polygons = []` → −25
  - `confidence < 0.75` → −20
  - `background_rgb` não-branco em `balloon_type=white` → −15
  - `source_bbox == balloon_bbox` → −50
- Preferir bbox menor/apertada com `line_polygons` E maior confidence.
- Registrar decisão em `03_ocr/ocr_dedupe_decisions.jsonl` com `kept_score`/`dropped_score`.

**Regras de router de cover/noise** (em `text_router.py`):
- Classificar antes da página principal: top 15% da altura da page 1 (cover) → `content_class` candidatos: `cover_credit`, `logo`, `noise`, `scanlator_credit`.
- Heurísticas: `Shadow Erian Shadow` (palavras repetidas + cover region) → `noise`; `NTEEM` (≤ 5 chars + ornamental) → `noise`; `TL/PR/TS/CL Kiki/Mars/etc.` (palavras conhecidas de scanlator) → `scanlator_credit`.
- Default: `skip_processing=true` ou `needs_review=true`.

**Aceite**:
```text
PLEASE: bbox apertada vence dedupe; sem bbox_overreach_critical
Shadow Erian Shadow: content_class=noise|cover_credit; skip_processing=true
NTEEM: content_class=noise; skip_processing=true
```

---

### PR 13 — Text normalization v2 e propagação (DBG2-12/13)

**Objetivo**: fortalecer normalização de joined words E garantir que chega à tradução/project.

**Arquivos-alvo**:
```text
pipeline/ocr/text_normalizer.py
pipeline/ocr/contextual_reviewer.py
pipeline/translator/translate.py
pipeline/tests/test_joined_word_normalization_v2.py
pipeline/tests/test_normalized_text_propagates_to_translation.py
```

**Tokens mínimos para cobertura** (devem virar testes):
```text
CANYOUFINDAGOOD     → CAN YOU FIND A GOOD
THATGIVESINTERESTUP → THAT GIVES INTEREST UP
TILLTHREEMONTHS     → TILL THREE MONTHS
TOSHOWYOUR          → TO SHOW YOUR
TOBELIEVE           → TO BELIEVE
WE'REFOOL'S         → WE'RE FOOLS
AJUMMAYOU           → AJUMMA, YOU
THERE'SNO           → THERE'S NO
GETMONEYFROM        → GET MONEY FROM
EVENTHINK           → EVEN THINK
PAYUSBACK           → PAY US BACK
CANDIE              → CAN DIE
IDON'T              → I DON'T
LET'SJUST           → LET'S JUST
```

**Regras**:
- Heurística de split: dicionário PyEnchant/wordfreq + regex greedy (camelCase artificial via consoante+vogal repetida).
- Adicionar `normalized_text_final` em **cada layer** do `project.json` (campo canônico).
- Tradutor usa `normalized_text_final` quando `normalization.changed=true` E `confidence_after_estimate >= 0.7`.
- Se normalização suspeita mas incerta: `qa_flags += ["ocr_joined_word_review"]`, `needs_review=true`.

**Aceite**:
```text
joined_word_suspect_count cobre os 14 tokens acima
source_text_sent_to_translator usa texto normalizado quando changed=true
project.json preserva raw_ocr e normalized_text_final lado a lado
normalized_text_not_propagated_count = 0
```

---

### PR 14 — Router de conteúdo especial v2: sign/cover/TN/URL/SFX (DBG2-15 + reforço DBG2-11)

**Objetivo**: impedir sign/capa/ruído como fala/narração comum.

**Arquivos-alvo**:
```text
pipeline/ocr/text_router.py
pipeline/layout/balloon_layout.py
pipeline/typesetter/renderer.py
pipeline/tests/test_special_content_router_v2.py
```

**Políticas canônicas**:
```text
speech              → traduz/renderiza em balão
sfx                 → preservar OU traduzir separado conforme config; nunca juntar com fala
editorial_note/TN   → skip OU nota editorial; nunca fala
url_watermark       → preservar OU remover conforme config; nunca fala
scanlator_credit    → preservar OU remover conforme config; nunca fala
cover_credit/logo   → preservar OU remover OU review; nunca fala
sign                → render dentro da região da placa/sign OU review; nunca narração solta
noise               → skip_processing=true
```

**Para `sign` especificamente**:
- Adicionar `sign_bbox` no project.json (= região da placa).
- Renderer pega `render_bbox` clampado a `sign_bbox`.
- Se sem `sign_bbox` confiável → `render_policy="preserve_original"` OU `needs_review=true`.

**Aceite**:
```text
TEXT: DARLING KARAOKE não pode virar narração comum fora da placa
T/N não pode renderizar como speech
URL não pode entrar em inpaint/render de fala
SFX dentro de fala deve ser separado antes da tradução
sign_rendered_as_narration_count = 0 (exceto needs_review explícito)
cover_noise_rendered_as_dialogue_count = 0
```

---

### PR 15 — QA propagation e export gate sync v2 (DBG2-14/19)

**Objetivo**: toda flag visual gerada em mask/typeset/router precisa chegar ao project.json final E ao export gate.

**Arquivos-alvo**:
```text
pipeline/typesetter/renderer.py
pipeline/qa/export_gate.py
pipeline/qa/translation_qa.py
pipeline/main.py
pipeline/tests/test_qa_flag_propagation_v2.py
```

**Regras**:
- `qa_flags` geradas no `render_plan`, `mask_decision` ou `inpaint_decision` voltam ao mesmo layer do `project.json` por `trace_id`; fallback permitido: `text_instance_id`, depois `text_id + band_id`. `text_id` isolado e ambiguo e nao pode ser join canonico.
- `qa_issues.jsonl` aponta para `render_plan_final` e `layout_blocks` via `linked_artifacts`.
- `visual_blockers.jsonl` precisa conter **todos** os críticos do `export_gate.issues`, não só mojibake (hoje em A só tem 1 entry).
- C_fast_fill: QA final usa as métricas da `mask_chain_summary.json` da própria run, **não** as do baseline. Resolve DBG2-19.
- Criar `11_qa_export_gate/qa_flag_propagation_audit.json` com:
```json
{
  "summary": {
    "render_plan_flags": 17,
    "project_layer_flags": 14,
    "export_gate_flags": 11,
    "qa_flag_not_propagated_count": 3
  },
  "missing_in_project": [
    {"text_id": "ocr_003", "flag": "ocr_run_on_suspect", "in_render_plan": true, "in_project": false}
  ]
}
```

**Aceite**:
```text
render_on_art_suspected no render_plan ⇒ mesma flag no project layer e no qa_issues/export_gate
qa_flag_not_propagated_count = 0
qa.summary e export_gate consistentes em A/B/C/D
C_fast_fill: mask_density_high count em export_gate refere a máscara efetiva da run
```

---

### PR 16 — Copyback/reassemble + cleanup breakdown + contact sheets (DBG2-16/17/18/20)

**Objetivo**: preencher etapas vazias/fracas.

**Arquivos-alvo**:
```text
pipeline/strip/process_bands.py
pipeline/strip/run.py
pipeline/debug_tools/contact_sheets.py
tools/analyze_e2e_debug.py
pipeline/tests/test_copyback_reassemble_debug.py
pipeline/tests/test_contact_sheets_debug.py
pipeline/tests/test_page_cleanup_breakdown.py
```

**Regras**:
- Sempre gerar `10_copyback_reassemble/copyback_decisions.jsonl` para cada band processada.
- Sempre gerar `reassemble_manifest.json` por página final.
- `page_cleanup_breakdown.json` com sub-stages instrumentados via `recorder.time_stage()`:
```text
cleanup_inpaint   ← _cleanup_page_inpaint_and_rerender chamada interna
cleanup_typeset   ← typesetter.render_band_image
cleanup_copyback  ← _apply_copy_back_outside_balloons
cleanup_save      ← cv2.imwrite final
cleanup_total
```
- Sempre gerar `01_input_extract/input_manifest.json` (hoje vazio) — esquema já está no §4.2.
- Sempre gerar `08_inpaint/inpaint_blocks.jsonl` agregado a partir das `inpaint_decision.json` por band.
- `12_contact_sheets/` sempre tem no mínimo `translated_comparison.jpg` e `problem_bands.jpg` quando `debug=true`.
- Adicionar flag de debug rápido: `TRADUZAI_DEBUG_SKIP_PAGE_CLEANUP_RERENDER=1` torna cleanup opcional.

**Aceite**:
```text
copyback_debug_missing_count = 0
contact_sheet_missing_count = 0
page_cleanup_breakdown_total ≈ performance.page_cleanup_rerender (±5%)
01_input_extract/input_manifest.json existe em toda run com debug=true
08_inpaint/inpaint_blocks.jsonl agregado existe em toda run com inpaint ativo
```

---

### PR 17 — Invariantes do analyzer e CI smoke (todas as DBG2-*)

**Objetivo**: impedir regressões do debug padrão.

**Arquivos-alvo**:
```text
tools/analyze_e2e_debug.py
pipeline/tests/regression/test_e2e_debug_20260518.py
```

**Invariantes obrigatórios** (= testes):
```text
debug_report_metric_mismatch_count = 0
render_plan_null_id_count = 0
render_plan_duplicate_final_entry_count = 0
render_plan_null_coordinate_space_count = 0
derived_bbox_coordinate_mismatch_count = 0
translation_debug_missing = false
copyback_debug_missing = false
contact_sheets_missing = false
qa_flag_not_propagated_count = 0
normalized_text_not_propagated_count = 0
cover_noise_rendered_as_dialogue_count = 0
sign_rendered_as_narration_count = 0
page_cleanup_breakdown_missing = false
skip_inpaint_honored_consistent = true
```

**Aceite**:
```text
python tools/analyze_e2e_debug.py <run_dir> --strict-debug-audit
  → exit_code=0 só quando TODOS invariantes passam
  → exit_code=3 quando algum invariante falha (separado de export_gate exit_code=2)
```

---

## 10b. Tabela de flags v2 (atualizar §10 com estas novas)

Adicionar à tabela canônica de `pipeline/qa/translation_qa.py::FLAG_SEVERITY`:

| Flag | Severidade | Categoria | Bloqueia strict? | Origem | Resolve |
|---|---|---|---|---|---|
| `debug_report_metric_mismatch` | critical | debug | sim | analyzer | DBG2-01/02/03/23 |
| `render_plan_missing_ids` | critical | debug | sim | analyzer | DBG2-04 |
| `render_plan_duplicate_final_entry` | high | debug | sim | analyzer | DBG2-05 |
| `derived_bbox_coordinate_mismatch` | critical | layout | sim | bbox_audit v2 | DBG2-06/07 |
| `translation_debug_missing` | high | debug | sim | analyzer | DBG2-08 |
| `cover_noise_rendered_as_dialogue` | critical | router | sim | text_router v2 | DBG2-11 |
| `sign_rendered_as_narration` | high | router | sim | text_router v2 | DBG2-15 |
| `qa_flag_not_propagated` | critical | qa | sim | qa_flag_propagation_audit | DBG2-14 |
| `copyback_debug_missing` | medium | debug | warn | analyzer | DBG2-16 |
| `contact_sheet_missing` | medium | debug | warn | analyzer | DBG2-17 |
| `page_cleanup_breakdown_missing` | medium | debug | warn | analyzer | DBG2-18 |
| `normalized_text_not_propagated` | high | ocr | sim | normalizer v2 | DBG2-13 |
| `ocr_joined_word_review` | medium | ocr | warn | normalizer v2 | DBG2-12 |
| `skip_inpaint_honored_inconsistent` | high | inpaint | sim | analyzer | DBG2-21 |
| `stage_duration_missing` | medium | debug | warn | analyzer | DBG2-22 |
| `stage_file_aggregate_mismatch` | high | debug | sim | analyzer | DBG2-23 |

---

## 8b. Comandos de validação v2 (atualizar §8)

Manter as 4 runs A/B/C/D do §8, e **adicionar** ao analisador:

```bash
python tools/analyze_e2e_debug.py <run_root> --write-report --strict-debug-audit
```

`--strict-debug-audit` retorna `exit_code=3` quando qualquer invariante do §5b falha.

### Codigos de saida do analyzer

```text
0 = analyzer executou e todos os invariantes strict passaram
3 = analyzer executou, mas pelo menos um invariante strict falhou
outros = erro operacional do analyzer ou falha de execucao Python
```

Nao confundir `exit_code=3` do analyzer com `exit_code=2` do pipeline em uma run `strict + export_gate BLOCK`. O primeiro valida a qualidade do pacote de debug; o segundo valida o comportamento do pipeline durante a exportacao strict.

### Métricas novas que o analyzer precisa reportar (em adição às do §8)

```text
- debug_report_metric_mismatch_count
- render_plan_null_id_count
- render_plan_duplicate_final_entry_count
- render_plan_null_coordinate_space_count
- render_plan_project_mismatch_count
- render_plan_project_field_mismatch_count
- render_plan_trace_page_mismatch_count
- render_plan_trace_band_mismatch_count
- project_render_outside_count
- qa_issue_traceability_missing_count
- derived_bbox_coordinate_mismatch_count (por key derivada)
- translation_debug_entry_count
- translation_debug_missing (bool)
- normalized_text_not_propagated_count
- cover_noise_rendered_as_dialogue_count
- sign_rendered_as_narration_count
- qa_flag_not_propagated_count
- copyback_debug_missing_count
- contact_sheet_missing_count
- page_cleanup_breakdown_missing (bool)
- skip_inpaint_honored_consistent (bool)
- stage_file_aggregate_mismatch_count (cross-check stage-level vs agregador)
```

---

## 9b. Checklist final v2 (atualizar §9)

Em adição ao checklist v1 do §9, adicionar:

- [ ] `debug_report_consistency.json.all_consistent=true`.
- [ ] `render_plan_final.jsonl`: 0 entries com `page_id`, `band_id`, `text_id` ou `coordinate_space` nulos.
- [ ] `render_plan_final.jsonl`: 1 entry por identidade final (`trace_id` preferencial; fallback `text_instance_id` ou `text_id+band_id`), sem duplicacao massiva por `text_id` isolado.
- [ ] `render_plan_final_matches_project=true`: campos geometricos e IDs finais batem com `project.json`.
- [ ] `render_plan_trace_page_band_consistent=true`: `trace_id` nao contradiz `page_id`/`band_id`.
- [ ] `project_render_bbox_inside_balloon=true`: `render_bbox` final fica contido em `balloon_bbox` final.
- [ ] `qa_issues_are_traceable=true`: cada issue critica aponta para `trace_id`/`text_instance_id` e artefatos linkados.
- [ ] `mask_decision.json`: `trace_ids` e `text_instance_ids` preenchidos; sem fallback ambiguo por `text_id` isolado.
- [ ] Todas as bboxes derivadas (`bbox`, `render_bbox`, `safe_text_box`, `position_bbox`, `capacity_bbox`, `target_bbox`, `_debug_safe_text_box`) em page-global no `project.json` final.
- [ ] `07_translation/translation_inputs.jsonl` e `translation_outputs.jsonl` com contagem ≈ textos traduzíveis (excl. skip).
- [ ] `qa_flag_not_propagated_count=0` em todas as runs.
- [ ] `copyback_debug_missing_count=0` em todas as runs.
- [ ] `contact_sheet_missing_count=0` em todas as runs.
- [ ] `cover_noise_rendered_as_dialogue_count=0` (Shadow Erian Shadow, NTEEM bloqueados antes do render).
- [ ] `sign_rendered_as_narration_count=0`, exceto se `needs_review=true` + render_policy explícito.
- [ ] `page_cleanup_breakdown.json` existe e `page_cleanup_breakdown_total ≈ performance.page_cleanup_rerender (±5%)`.
- [ ] `normalized_text_not_propagated_count=0`: WE'REFOOL'S, TOBELIEVE, CANYOUFINDAGOOD, TOSHOWYOUR, AJUMMAYOU, THERE'SNO, GETMONEYFROM, EVENTHINK, PAYUSBACK, CANDIE, IDON'T, LET'SJUST corrigidos no project.json final.
- [ ] PLEASE: dedupe mantém bbox apertada `[473,57,644,100]` (sem `bbox_overreach_critical`).
- [ ] `skip_inpaint` com `config.skip_inpaint=true`: modo por band aceita `skip_inpaint_honored_bands == strip.band_count`; modo wholesale aceita `skip_inpaint_honored_bands/total_bands == 0/0` somente se existir `skip_inpaint_audit.json` ou evidencia explicita de etapa pulada.
- [ ] `debug_manifest.json.stage_durations_sec` populado com pelo menos 11 sub-stages.
- [ ] `01_input_extract/input_manifest.json` e `08_inpaint/inpaint_blocks.jsonl` existem em runs com `debug=true`.

---

## 13. Melhorias tecnicas pendentes

Estas melhorias nao bloqueiam a leitura do guia, mas devem entrar nos proximos PRs para reduzir falso positivo/falso negativo:

1. **Alias `debug/e2e` no analyzer**: `tools/analyze_e2e_debug.py` deve aceitar a propria pasta `debug/e2e` como entrada e resolver a run pai automaticamente. Enquanto isso nao existir, usar `<run_root>`.
2. **Relatorio runtime com `project.json`**: `pipeline/debug_tools/report.py::generate_debug_report()` deve receber `project_path` ou `project_data` no fim do pipeline. O relatorio gerado durante a run precisa ter a mesma consistencia do analyzer offline.
3. **Skip inpaint explicito**: quando `skip_inpaint=true` pula a etapa inteira, gerar `08_inpaint/skip_inpaint_audit.json`. O analyzer deve separar `honored=true` de `evidence_missing`.
4. **JSONL valido, nao apenas linha nao vazia**: o analyzer deve reportar `invalid_jsonl_line_count` por artefato critico e usar contagem de JSON valido em checks de consistencia.
5. **Identidade de origem vs destino no render**: se um render final muda de contexto, separar `source_page_id/source_band_id/source_trace_id` de `render_page_id/render_band_id`. O analyzer nao deve confundir trace de origem com destino rebaseado.
6. **Escopo de traceability por issue**: issue de texto exige `trace_id` ou `text_instance_id`, `page_id`, `band_id` e `linked_artifacts`; issue de pagina exige `page_id`; issue `chapter/run/global` exige `type`, `severity` e artefato/summary que explique o escopo, mas nao deve exigir `page_id`.
7. **Markdown principal completo**: `debug_report.md` da raiz do pacote deve listar invariantes strict falhos e contagens como `qa_issue_traceability_missing_count`, nao apenas a tabela resumida.
8. **Contact sheets por categoria**: manter `translated_comparison.jpg` como minimo, mas gerar tambem `mask_chain_top_issues.jpg`, `typeset_top_issues.jpg` e overlays especificos quando houver blockers.

## 14. Referências internas

- Análise do baseline 2026-05-17: `2026-05-17_chapter1_e2e_debug_151830.zip` + relatório de auditoria do plano original.
- Plano original (com correções aplicadas neste guia): `traduzai_e2e_remediation_plan_2026_05_17.md`.
- Análise da v1 implementada 2026-05-18: `2026-05-18_chapter1_e2e_debug_002011.zip` + `e2e_new_full_audit_report.md` (cobre §0.1, §0.2, §6.2).
- Prompt de atualização v2: `prompt_claude_atualizar_guia_debug_e2e_v2.md`.
- Arquitetura geral: [docs/architecture.md](../architecture.md), [docs/pipeline.md](../pipeline.md).
- QA system: [docs/qa-system.md](../qa-system.md).
- Project schema: [docs/project-schema.md](../project-schema.md).

---

**Para começar agora (v2)**: implementar **PR 9 + PR 10** juntos — sem analyzer correto e coordenadas auditadas em todas as keys, os outros 7 PRs vão produzir métricas em que não dá para confiar. Em seguida PR 11 (tradução), PR 12 (cover/dedupe), PR 13 (normalização), PR 14 (sign/router) em paralelo. PR 15 e 16 depois. PR 17 fecha como gate de regressão CI.
