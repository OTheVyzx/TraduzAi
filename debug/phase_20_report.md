# Fase 20 - Pipeline Runner e CLI debug

Data: 2026-05-01

## Resultado
- Fase 20 concluida.
- `pipeline/main.py` agora aceita runner por flags sem quebrar o modo legado `config.json`.
- CLI suporta `--input`, `--work`, `--target`, `--mode mock|real`, `--debug`, `--skip-inpaint`, `--skip-ocr`, `--strict`, `--export-mode` e `--output`.
- Modo `mock` roda offline, copia paginas para `originals/images/translated`, cria `project.json` e gera relatórios (`qa_report.json`, `qa_report.md`, `issues.csv`, `glossary_used.json`, `ocr_corrections.json`, `structured_log.jsonl`).
- `--strict` retorna codigo diferente de zero quando ha flag critical.

## Arquivos alterados
- `pipeline/main.py`
- `pipeline/tests/test_main_emit.py`
- `debug/runs/tiny_chapter/`

## Testes e comandos
- `.\\venv\\Scripts\\python.exe -m pytest tests/test_main_emit.py -q` passou com 23 testes.
- `.\\pipeline\\venv\\Scripts\\python.exe pipeline\\main.py --input fixtures\\tiny_chapter\\original --work "The Regressed Mercenary Has a Plan" --target pt-BR --mode mock --debug --strict --export-mode clean --output debug\\runs\\tiny_chapter` passou e gerou `project.json`/reports.

## Falhas e correcoes
- RED esperado: testes falharam por falta de parser, runner e ajuda `--input`.
- Corrigido com parser dedicado e modo mock offline.

## Proximo ponto
Avancar para a Fase 21: Teste final completo.
