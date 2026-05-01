# Fase 6 - Reading Order e Region Grouping

Data: 2026-05-01

## Resultado
- Fase 6 concluida.
- Criados modulos `pipeline/layout/reading_order.py` e `pipeline/layout/region_grouping.py`.
- Ordem de leitura padrao: linhas de cima para baixo, direita para esquerda dentro da linha.
- Agrupamento separa SFX/narracao e agrupa falas compativeis por mesmo balao, conexao visual simples, alinhamento e proximidade.
- Overlay de debug disponivel via `write_debug_overlay(...)`, gerando arquivo como `debug/region_grouping/page_001_overlay.png`.
- `build_project_json()` agora aplica `group_regions()` nos text layers antes de persistir `project.json`.

## Arquivos alterados
- `pipeline/layout/reading_order.py`
- `pipeline/layout/region_grouping.py`
- `pipeline/tests/test_reading_order.py`
- `pipeline/tests/test_region_grouping.py`
- `pipeline/main.py`

## Testes e comandos
- `.\\venv\\Scripts\\python.exe -m pytest tests/test_region_grouping.py tests/test_reading_order.py -q` passou com 7 testes.
- `.\\venv\\Scripts\\python.exe -m pytest tests/test_region_grouping.py tests/test_reading_order.py tests/test_main_emit.py -q` passou com 27 testes.

## Falhas e correcoes
- O primeiro comando foi executado a partir de `D:\TraduzAi\pipeline` com prefixo `pipeline\venv\...`, o que o PowerShell interpretou como modulo. Corrigido usando `.\\venv\\Scripts\\python.exe`.

## Observacoes
- A etapa foi integrada no wrap-up do `project.json`, preservando o pipeline atual.
- O agrupamento visual profundo por borda/mascara continua no `balloon_layout.py`; esta fase adiciona o contrato leve entre detect e traducao.

## Proximo ponto
Avancar para a Fase 7: OCR Normalizer.
