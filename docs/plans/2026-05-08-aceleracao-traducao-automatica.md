# Aceleracao Da Traducao Automatica Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduzir o tempo total do pipeline automatico do TraduzAi atacando OCR, inpaint e orquestracao, mantendo a qualidade visual e a compatibilidade do `project.json`.

**Architecture:** A estrategia e implementar primeiro medicao e modo sombra, depois ativar otimizacoes de baixo risco. O ganho principal deve vir de Smart Skip para trabalho visual inutil, Macro OCR para reduzir chamadas repetidas de OCR e, depois, uma reorganizacao em fases/DAG com um worker GPU unico.

**Tech Stack:** Python 3.12, PaddleOCR, OpenCV, ONNX Runtime/LaMA, Google Translate, Tauri v2, Rust, React/TypeScript, JSON lines do sidecar.

---

## Contexto E Evidencia Atual

O ultimo capitulo analisado em `D:\TraduzAi\AAAAAAA\traduzido2` indicou que a traducao textual nao e o gargalo principal.

- Tempo total: `130.4s`
- OCR: `52.7s`
- Inpaint: `52.8s`
- Typeset: `5.0s`
- Traducao textual: `0.67s`

O alvo inicial e ficar abaixo de `113.65s`, que corresponde a metade do primeiro benchmark de `227.3s`. O capitulo atual precisa economizar aproximadamente `16.75s`.

O melhor primeiro corte esta em evitar processamento visual de textos que nao precisam ser traduzidos, especialmente creditos, avisos de site, timers, logos e textos decorativos.

---

## Principios De Implementacao

1. Comecar sempre por medicao.
2. Implementar toda heuristica primeiro em modo sombra.
3. Nao alterar `project.json` sem manter compatibilidade.
4. Nao paralelizar multiplos modelos GPU de forma agressiva na RTX 4060 8GB.
5. Preservar Google-only como caminho padrao de traducao.
6. Validar qualidade por artefato, nao apenas por tempo.
7. Separar perfis de execucao: Performance e Eco.

---

## Fase 0: Baseline Automatizado

**Objetivo:** criar uma base confiavel de comparacao antes de otimizar.

**Files:**
- Create: `pipeline/tools/analyze_pipeline_run.py`
- Create: `pipeline/tests/test_analyze_pipeline_run.py`
- Read: `pipeline/main.py`
- Read: `pipeline/strip/run.py`
- Read: `pipeline/strip/process_bands.py`
- Read: `pipeline/vision_stack/runtime.py`

### Task 0.1: Criar parser de metricas do capitulo

**Step 1: Write the failing test**

Criar `pipeline/tests/test_analyze_pipeline_run.py` com um fixture minimo que simule um diretorio de saida contendo metricas por banda.

O teste deve validar:

- tempo total
- soma por estagio
- top bandas por OCR
- top bandas por inpaint
- contagem de textos
- contagem de blocos inpaintados

**Step 2: Run test to verify it fails**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_analyze_pipeline_run.py -q
```

Expected: fail porque `pipeline/tools/analyze_pipeline_run.py` ainda nao existe.

**Step 3: Implement minimal parser**

Implementar funcoes puras:

- `load_run_metrics(output_dir: Path) -> RunMetrics`
- `summarize_stages(metrics: RunMetrics) -> dict[str, float]`
- `rank_bands(metrics: RunMetrics, stage: str, limit: int = 10) -> list[BandMetric]`

**Step 4: Run test to verify it passes**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_analyze_pipeline_run.py -q
```

Expected: pass.

**Step 5: Run on real outputs**

```powershell
pipeline\venv\Scripts\python.exe pipeline\tools\analyze_pipeline_run.py D:\TraduzAi\AAAAAAA\traduzido2 --json-out D:\TraduzAi\debug\performance_baselines\traduzido2.json
```

Expected: gerar relatorio sem modificar o capitulo.

---

## Fase 1: Smart Skip Em Modo Sombra

**Objetivo:** identificar textos que poderiam ser pulados sem alterar a saida final.

**Files:**
- Create: `pipeline/strip/smart_skip.py`
- Create: `pipeline/tests/test_strip_smart_skip.py`
- Modify: `pipeline/strip/process_bands.py`

### Task 1.1: Definir tipos de skip

Criar enum ou constantes para:

- `credit_or_watermark`
- `timer_or_ui`
- `decorative_logo`
- `sfx_keep_original`
- `low_value_noise`
- `not_safe_to_skip`

### Task 1.2: Escrever testes com textos reais

Casos que devem ser classificados como candidatos:

```python
[
    "All comics on this website are just previews...",
    "For the original version, please buy the comic...",
    "READ On",
    "FOR FASTER UPDATE",
    "00:00:05",
    "oo:oo:os",
]
```

Casos que nao devem ser pulados automaticamente:

```python
[
    "AH, AH, MIC TEST.",
    "IS THIS RECORDING?",
    "I can't go back now.",
]
```

**Run:**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py -q
```

Expected: fail antes da implementacao, pass depois.

### Task 1.3: Implementar classificador puro

Criar funcao:

```python
def classify_text_for_skip(text: str, *, page_number: int | None, confidence: float | None, bbox: tuple[int, int, int, int] | None) -> SmartSkipDecision:
    ...
```

Regras iniciais:

- forte padrao de site/update/preview em pagina inicial ou final
- timer/UI com padrao numerico ou OCR corrompido equivalente
- logos/decorativos apenas quando baixa confianca e fora de balao narrativo
- nunca pular fala comum so por estar em caixa pequena

### Task 1.4: Integrar modo sombra

Em `pipeline/strip/process_bands.py`, calcular a decisao e registrar:

- texto original
- bbox
- pagina/banda
- motivo
- tempo de inpaint da banda
- economia estimada

Nao alterar:

- `skip_processing`
- `_vision_blocks`
- imagem final
- traducao final

### Task 1.5: Validar em output real

Rodar o capitulo com flag de sombra:

```powershell
$env:TRADUZAI_SMART_SKIP_SHADOW="1"
pipeline\venv\Scripts\python.exe pipeline\main.py config.json
```

Expected:

- saida visual igual
- relatorio com candidatos a skip
- economia simulada >= `16.75s` no capitulo analisado

---

## Fase 2: Smart Skip Real

**Objetivo:** ativar o skip real somente para decisoes seguras.

**Files:**
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/strip/types.py`
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_strip_process_bands.py`
- Test: `pipeline/tests/test_main_strip_config.py`

### Task 2.1: Adicionar flag de runtime

Adicionar:

```text
TRADUZAI_SMART_SKIP=0|1
```

Default inicial: `0`.

### Task 2.2: Aplicar skip seguro

Quando ativo:

- marcar textos seguros como `skip_processing=True`
- remover blocos correspondentes de `_vision_blocks` usados pelo inpaint
- nao enviar esses textos para traducao
- nao renderizar overlay traduzido
- preservar auditoria no resultado

### Task 2.3: Testar contrato

Validar:

- texto seguro nao vai para inpaint
- texto duvidoso continua no pipeline normal
- `project.json` continua contendo informacao suficiente para auditoria
- nao ha perda de textos narrativos nos fixtures

**Run:**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_main_strip_config.py -q
```

### Task 2.4: Validar no capitulo real

Rodar com:

```powershell
$env:TRADUZAI_SMART_SKIP="1"
pipeline\venv\Scripts\python.exe pipeline\main.py config.json
```

Expected:

- tempo total <= `113.65s` no capitulo de referencia, ou economia justificada por relatorio
- imagens finais sem perda obvia de fala/narracao
- editor abre o projeto

---

## Fase 3: Macro OCR Em Modo Sombra

**Objetivo:** testar OCR por pagina ou macro-janela sem substituir a saida atual.

**Files:**
- Create: `pipeline/ocr/macro_ocr.py`
- Create: `pipeline/tests/test_macro_ocr_mapping.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/strip/run.py`

### Task 3.1: Criar modelo de resultado macro

Criar tipos:

- `MacroOcrLine`
- `MacroOcrPageResult`
- `MappedBandOcr`

Campos minimos:

- texto
- bbox absoluto na pagina
- confidence
- source page
- mapped band id
- mapping confidence

### Task 3.2: Escrever testes de remapeamento

Casos:

- texto dentro da banda
- texto na borda entre bandas
- texto grande atravessando corte
- falso positivo fora de regiao detectada

### Task 3.3: Rodar OCR macro em sombra

Em modo sombra:

- OCR antigo continua sendo fonte da verdade
- OCR macro roda e compara
- relatorio mostra missing/extra/different

Flag:

```text
TRADUZAI_MACRO_OCR_SHADOW=1
```

### Task 3.4: Medir beneficio esperado

Comparar:

- numero de chamadas OCR atual
- numero de chamadas OCR macro
- tempo estimado economizado
- diferenca de texto
- diferenca de bbox

Expected:

- queda material de tempo de OCR estimado
- diferencas auditaveis antes de ativar

---

## Fase 4: Macro OCR Real Com Fallback

**Objetivo:** usar OCR macro como fonte principal quando o mapeamento for confiavel.

**Files:**
- Modify: `pipeline/ocr/macro_ocr.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/strip/process_bands.py`
- Test: `pipeline/tests/test_macro_ocr_mapping.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

### Task 4.1: Adicionar flag real

```text
TRADUZAI_MACRO_OCR=0|1
```

Default inicial: `0`.

### Task 4.2: Precomputar OCR por pagina

Antes do loop serial de bandas:

- agrupar bandas por pagina
- rodar OCR por pagina ou macro-janela
- criar mapa `band_id -> ocr_lines`

### Task 4.3: Fallback seguro

Usar OCR antigo quando:

- mapeamento ambiguo
- texto cruza borda de banda
- confianca macro baixa
- OCR macro retorna vazio em banda com deteccao forte

### Task 4.4: Validar

**Run:**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_vision_stack_runtime.py pipeline\tests\test_strip_process_bands.py -q
```

Expected:

- fallback coberto por testes
- contadores de fallback aparecem no relatorio

Meta:

- OCR total sair de `~52s` para algo proximo de `20-30s` em capitulos parecidos.

---

## Fase 5: Traducao Em Lote Por Capitulo

**Objetivo:** reduzir variancia de rede e melhorar contexto, sem depender de LLM local.

**Files:**
- Modify: `pipeline/translator/translate.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/strip/process_bands.py`
- Test: `pipeline/tests/test_translate_context.py`
- Test: `pipeline/tests/test_strip_run.py`

### Task 5.1: Coletar textos aceitos

Depois de OCR e Smart Skip:

- coletar todos os textos que precisam traducao
- preservar ordem por pagina/banda/bloco
- manter chave de retorno para aplicar a traducao correta

### Task 5.2: Traduzir em lotes maiores

Usar cache persistente existente e Google-only como padrao.

Nao reativar semantic review local por padrao.

### Task 5.3: Reaplicar traducao

Garantir:

- cada bloco recebe sua traducao correta
- textos pulados nao entram no batch
- erro parcial nao derruba capitulo inteiro sem fallback

Meta:

- manter traducao textual abaixo de `1-2s` com cache quente
- reduzir chamadas e instabilidade

---

## Fase 6: Scheduler DAG Com Worker GPU Unico

**Objetivo:** reduzir tempo ocioso sem estourar VRAM.

**Files:**
- Create: `pipeline/strip/scheduler.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/vision_stack/inpainter.py`
- Test: `pipeline/tests/test_strip_scheduler.py`
- Test: `pipeline/tests/test_strip_run.py`

### Novo fluxo proposto

1. Detectar bandas.
2. Precomputar Macro OCR.
3. Aplicar Smart Skip.
4. Traduzir em lote.
5. Enfileirar apenas inpaint necessario.
6. Rodar um worker GPU unico.
7. Rodar typeset/reassemble quando a pagina ficar pronta.

### Task 6.1: Criar scheduler simples

O scheduler deve manter:

- fila de CPU tasks
- fila de GPU tasks
- limite de memoria
- progresso por pagina/banda

### Task 6.2: Preservar eventos Tauri

O sidecar ainda deve emitir progresso por JSON lines de forma compativel.

Nao mudar shape publico sem atualizar Rust/TS.

### Task 6.3: Validar uso de recursos

Medir:

- pico de RAM
- pico de VRAM
- uso medio de CPU
- tempo total

Risco:

- paralelismo excessivo pode deixar a RTX 4060 8GB pior por thrash de VRAM.

Regra:

- apenas um worker GPU por padrao.

---

## Fase 7: Perfis Performance E Eco

**Objetivo:** permitir escolha clara entre velocidade e baixo consumo.

**Files:**
- Modify: `pipeline/main.py`
- Modify: `src-tauri/src/commands/pipeline.rs`
- Modify: `src/lib/tauri.ts`
- Optional UI: `src/pages/Setup.tsx`

### Perfil Performance

- Smart Skip ativo
- Fast-fill ativo
- Macro OCR ativo
- modelos aquecidos
- ONNX CUDA
- TensorRT opcional e opt-in
- Google-only por padrao

### Perfil Eco

- Smart Skip ativo
- Fast-fill ativo
- sem LLM local
- sem prewarm pesado quando nao necessario
- liberar OCR/detector antes de LaMA quando possivel
- limitar threads de CPU
- evitar TensorRT
- priorizar menos pico de RAM/VRAM

### Task 7.1: Adicionar preset no config

Campos sugeridos:

```json
{
  "preset": "performance"
}
```

Valores:

- `performance`
- `eco`
- `balanced`

### Task 7.2: Atualizar Rust/TS

Se exposto na UI, atualizar:

- serde em Rust
- tipo TS em `src/lib/tauri.ts`
- store/config no frontend

### Task 7.3: Testar contrato

```powershell
npm run check
cd src-tauri; cargo check
cd D:\TraduzAi
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_main_strip_config.py -q
```

---

## Fase 8: Validacao De Qualidade

**Objetivo:** impedir que o ganho de tempo esconda perda de qualidade.

**Files:**
- Create: `pipeline/tools/compare_pipeline_outputs.py`
- Create: `pipeline/tests/test_compare_pipeline_outputs.py`

### Comparacoes obrigatorias

Comparar antes/depois:

- tempo total
- tempo por estagio
- quantidade de textos
- quantidade de `_vision_blocks`
- quantidade de blocos inpaintados
- textos pulados por Smart Skip
- paginas com fallback OCR
- imagens finais
- `project.json`

### Gate de qualidade

Falhar a validacao se:

- sumir texto narrativo
- houver queda grande de confidence sem fallback
- `project.json` quebrar importacao
- inpaint remover balao real indevidamente
- editor nao abrir o projeto

---

## Ordem Recomendada De Execucao

1. Fase 0: Baseline automatizado.
2. Fase 1: Smart Skip sombra.
3. Fase 2: Smart Skip real.
4. Fase 3: Macro OCR sombra.
5. Fase 4: Macro OCR real.
6. Fase 5: Traducao em lote.
7. Fase 6: Scheduler DAG.
8. Fase 7: Perfis Performance/Eco.
9. Fase 8: Validacao final.

---

## Metas Por Marco

### Marco A: Smart Skip

Meta:

- reduzir `traduzido2` de `130.4s` para <= `113.65s`

Validacao:

- nenhum texto narrativo perdido
- auditoria de skips disponivel
- editor abre `project.json`

### Marco B: Macro OCR

Meta:

- reduzir OCR de `~52.7s` para `20-30s`

Validacao:

- textos equivalentes ou melhores
- fallback quando mapeamento for incerto

### Marco C: DAG + Perfis

Meta:

- mirar `80-100s` em capitulos parecidos
- reduzir pico de recursos no modo Eco

Validacao:

- sem OOM
- sem aumento grave de RAM/VRAM
- eventos Tauri preservados

---

## Riscos E Mitigacoes

| Risco | Impacto | Mitigacao |
| --- | --- | --- |
| Smart Skip pular texto real | Alto | modo sombra, regras conservadoras, auditoria visual |
| Macro OCR mapear texto para banda errada | Alto | fallback por confianca, testes de borda, comparacao por IoU |
| GPU ficar instavel com paralelismo | Alto | worker GPU unico, filas limitadas |
| `project.json` quebrar editor | Alto | testes de schema/importacao, compatibilidade defensiva |
| TensorRT aumentar complexidade | Medio | opt-in apenas no perfil Performance |
| QA atual nao detectar erro visual | Medio | criar comparador de artefatos e amostras visuais |
| Traducao em lote separar texto errado | Medio | chaves estaveis por pagina/banda/bloco |

---

## Testes Que O Agente Deve Rodar E Julgar

**Objetivo:** o executor do plano deve conseguir medir, decidir e bloquear regressao sem depender de avaliacao manual do usuario a cada etapa.

Esses testes nao substituem a revisao visual humana final, mas devem ser suficientes para o proprio agente dizer: "passou", "falhou" ou "precisa voltar uma fase".

### Gate 1: Baseline Reproduzivel

**Files:**
- Create: `pipeline/tools/run_performance_gate.py`
- Create: `pipeline/tests/test_performance_gate.py`
- Output: `debug/performance_gates/<run_id>/summary.json`

**O que o agente deve testar:**

1. Ler um output existente, como `D:\TraduzAi\AAAAAAA\traduzido2`.
2. Extrair tempo total e tempo por estagio.
3. Confirmar que OCR e inpaint somados representam o gargalo principal.
4. Gerar um JSON com:
   - `total_seconds`
   - `stage_seconds`
   - `top_ocr_bands`
   - `top_inpaint_bands`
   - `text_count`
   - `inpaint_block_count`
   - `skip_candidate_count`

**Comando:**

```powershell
pipeline\venv\Scripts\python.exe pipeline\tools\run_performance_gate.py D:\TraduzAi\AAAAAAA\traduzido2 --out debug\performance_gates\traduzido2_baseline
```

**Criterio de julgamento:**

- PASS se o JSON existir e `stage_seconds.ocr + stage_seconds.inpaint` for maior que 60% do tempo medido por estagios.
- FAIL se o parser nao encontrar metricas suficientes.
- BLOCK se o output analisado nao tiver `project.json` ou imagens finais.

### Gate 2: Smart Skip Sombra

**Files:**
- Create: `pipeline/tests/test_smart_skip_shadow_gate.py`
- Output: `debug/performance_gates/<run_id>/smart_skip_shadow.json`

**O que o agente deve testar:**

1. Rodar o pipeline ou simulador com `TRADUZAI_SMART_SKIP_SHADOW=1`.
2. Confirmar que a saida final nao mudou.
3. Confirmar que ha candidatos de skip auditaveis.
4. Estimar economia por banda usando o tempo real de inpaint das bandas candidatas.

**Comando:**

```powershell
$env:TRADUZAI_SMART_SKIP_SHADOW="1"
pipeline\venv\Scripts\python.exe pipeline\main.py config.json
```

**Criterio de julgamento:**

- PASS se a economia estimada for >= `16.75s` no capitulo de referencia e nenhum texto classificado como dialogo/narracao for candidato a skip automatico.
- FAIL se o modo sombra alterar imagem final, `project.json`, contagem de textos ou traducao.
- BLOCK se mais de 5% dos candidatos forem classificados como `not_safe_to_skip`.

### Gate 3: Smart Skip Real

**Files:**
- Create: `pipeline/tests/test_smart_skip_real_gate.py`
- Output: `debug/performance_gates/<run_id>/smart_skip_real.json`

**O que o agente deve testar:**

1. Rodar antes/depois no mesmo capitulo.
2. Comparar tempo total.
3. Comparar `project.json`.
4. Comparar contagem de textos narrativos.
5. Gerar lista de regioes realmente puladas.

**Comando:**

```powershell
$env:TRADUZAI_SMART_SKIP="1"
pipeline\venv\Scripts\python.exe pipeline\main.py config.json
```

**Criterio de julgamento:**

- PASS se o tempo total ficar <= `113.65s` ou se a economia real for >= 90% da economia estimada no modo sombra.
- PASS somente se nenhum texto narrativo dos fixtures conhecidos sumir.
- FAIL se qualquer texto de fala/narracao for removido sem motivo auditavel.
- FAIL se o editor nao conseguir abrir o projeto gerado.
- BLOCK se a economia for menor que `8s`, porque o risco nao compensa ativar por padrao.

### Gate 4: Macro OCR Sombra

**Files:**
- Create: `pipeline/tests/test_macro_ocr_shadow_gate.py`
- Output: `debug/performance_gates/<run_id>/macro_ocr_shadow.json`

**O que o agente deve testar:**

1. Rodar OCR atual e Macro OCR no mesmo capitulo.
2. Comparar textos por pagina/banda.
3. Comparar bboxes por IoU/centro.
4. Medir quantos blocos precisariam de fallback.

**Comando:**

```powershell
$env:TRADUZAI_MACRO_OCR_SHADOW="1"
pipeline\venv\Scripts\python.exe pipeline\main.py config.json
```

**Criterio de julgamento:**

- PASS se `missing_text_rate <= 2%`, `wrong_band_rate <= 1%` e `fallback_rate <= 15%`.
- FAIL se textos narrativos forem mapeados para a banda errada.
- BLOCK se o ganho estimado de OCR for menor que 10s no capitulo de referencia.

### Gate 5: Macro OCR Real

**Files:**
- Create: `pipeline/tests/test_macro_ocr_real_gate.py`
- Output: `debug/performance_gates/<run_id>/macro_ocr_real.json`

**O que o agente deve testar:**

1. Ativar `TRADUZAI_MACRO_OCR=1`.
2. Rodar o capitulo completo.
3. Confirmar que o fallback foi usado nos casos incertos.
4. Comparar OCR total contra baseline.

**Comando:**

```powershell
$env:TRADUZAI_MACRO_OCR="1"
pipeline\venv\Scripts\python.exe pipeline\main.py config.json
```

**Criterio de julgamento:**

- PASS se OCR total cair para `20-30s` ou economizar pelo menos 35% do tempo de OCR.
- FAIL se a contagem de textos narrativos cair sem fallback.
- FAIL se a comparacao visual apontar texto duplicado, faltante ou em pagina errada.
- BLOCK se a taxa de fallback passar de 25%, porque nesse caso o modo ainda nao esta maduro.

### Gate 6: Comparacao Estrutural De Outputs

**Files:**
- Create: `pipeline/tools/compare_pipeline_outputs.py`
- Create: `pipeline/tests/test_compare_pipeline_outputs.py`
- Output: `debug/performance_gates/<run_id>/output_compare.json`

**O que o agente deve testar:**

Comparar baseline e output otimizado:

- `project.json`
- numero de paginas
- numero de textos
- numero de regioes traduzidas
- numero de `_vision_blocks`
- textos com `skip_processing`
- imagens finais existentes
- dimensoes das imagens finais

**Comando:**

```powershell
pipeline\venv\Scripts\python.exe pipeline\tools\compare_pipeline_outputs.py D:\TraduzAi\AAAAAAA\traduzido2 D:\TraduzAi\AAAAAAA\traduzido_optimized --out debug\performance_gates\compare_optimized.json
```

**Criterio de julgamento:**

- PASS se paginas e dimensoes forem identicas, e diferencas de texto estiverem explicadas por Smart Skip auditado.
- FAIL se faltar pagina, imagem final ou campo essencial do `project.json`.
- FAIL se `_vision_blocks` desaparecerem de texto que ainda precisa inpaint/typeset.

### Gate 7: Comparacao Visual Amostrada

**Files:**
- Create: `pipeline/tools/export_visual_review_sheet.py`
- Output: `debug/performance_gates/<run_id>/visual_review_sheet.html`

**O que o agente deve testar:**

Gerar uma pagina HTML ou pasta com crops lado a lado:

- baseline
- output otimizado
- mascara/inpaint quando existir
- bbox/texto original
- motivo de skip/fallback

Incluir obrigatoriamente:

- top 10 bandas por economia
- todas as bandas com Smart Skip real
- todas as bandas com fallback Macro OCR
- qualquer pagina com diferenca estrutural

**Comando:**

```powershell
pipeline\venv\Scripts\python.exe pipeline\tools\export_visual_review_sheet.py --baseline D:\TraduzAi\AAAAAAA\traduzido2 --candidate D:\TraduzAi\AAAAAAA\traduzido_optimized --out debug\performance_gates\visual_review_sheet.html
```

**Criterio de julgamento:**

- PASS se o agente nao identificar texto narrativo faltante, duplicado, cortado ou inpaint indevido nos crops.
- FAIL se qualquer crop mostrar fala/narracao removida.
- BLOCK se o HTML nao puder ser gerado, porque a validacao visual fica fraca.

### Gate 8: Recursos Do Sistema

**Files:**
- Create: `pipeline/tools/measure_resource_profile.py`
- Create: `pipeline/tests/test_resource_profile.py`
- Output: `debug/performance_gates/<run_id>/resources.json`

**O que o agente deve testar:**

Medir durante uma execucao:

- pico de RAM
- pico aproximado de VRAM, quando disponivel via `nvidia-smi`
- uso medio de CPU
- tempo total
- quantidade de workers

**Comando:**

```powershell
pipeline\venv\Scripts\python.exe pipeline\tools\measure_resource_profile.py -- pipeline\venv\Scripts\python.exe pipeline\main.py config.json
```

**Criterio de julgamento:**

- PASS se o modo Performance reduzir tempo sem aumentar VRAM de forma instavel.
- PASS se o modo Eco reduzir pico de RAM/VRAM ou preloads pesados, mesmo que seja um pouco mais lento.
- FAIL se houver OOM, crash de CUDA/ONNX ou aumento grande de memoria sem ganho de tempo.

### Gate 9: Decisao Final Do Agente

**Files:**
- Output: `debug/performance_gates/<run_id>/decision.md`

**O agente deve escrever uma decisao curta com:**

- resultado: `aprovar`, `reprovar` ou `voltar para sombra`
- tempo baseline
- tempo candidato
- economia real
- riscos encontrados
- paginas que precisam revisao humana
- recomendacao de default: `off`, `shadow`, `performance` ou `eco`

**Criterio de julgamento:**

- `aprovar` somente se performance melhorou e os gates estruturais/visuais passaram.
- `voltar para sombra` se a performance melhorou, mas houve duvida visual ou estrutural.
- `reprovar` se a economia for baixa ou se houver perda de texto narrativo.

---

## Definition Of Done

O plano so deve ser considerado concluido quando:

- benchmarks antes/depois estiverem salvos
- Smart Skip tiver auditoria
- Macro OCR tiver fallback
- `project.json` continuar reimportavel
- editor abrir o projeto gerado
- testes focados passarem
- gates de julgamento do agente tiverem `decision.md`
- tempo total melhorar sem perda visual importante
- modo Performance e modo Eco tiverem comportamento documentado

---

## Comandos De Verificacao Sugeridos

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py -q
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_macro_ocr_mapping.py -q
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_process_bands.py pipeline\tests\test_strip_run.py -q
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_main_strip_config.py -q
npm run check
cd src-tauri; cargo check
```

---

## Notas De Implementacao

- Nao mexer primeiro em modelo de traducao. O gargalo atual nao e traducao textual.
- Nao ativar LLM local por padrao. Isso aumenta custo e variancia.
- Nao usar paralelismo GPU amplo. A maquina atual tem RTX 4060 8GB.
- Preferir skips conservadores que economizam inpaint.
- Toda heuristica nova deve ter modo sombra antes do modo real.
- Todo ganho de tempo deve vir acompanhado de comparacao visual e estrutural.
