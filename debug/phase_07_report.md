# Fase 7 - OCR Normalizer

Data: 2026-05-01

## Resultado
- Fase 7 concluida.
- Criado `pipeline/ocr/ocr_normalizer.py`.
- Correcoes obrigatorias implementadas: `RAID SOUAD`, `DRCS`, `RDC`, `CARBAGE`, `TRAe`, `FENRISNOW`.
- Gagueira com glossario implementada, por exemplo `Y-YOUNG MASTER?` vira `J-Jovem mestre?`.
- Gibberish recebe `qa_flags=["ocr_gibberish"]` e `skip_processing=True`.
- `raw_ocr`, `normalized_ocr` e `normalization` sao persistidos no text layer.
- Google e Ollama normalizam registros antes da traducao.

## Arquivos alterados
- `pipeline/ocr/ocr_normalizer.py`
- `pipeline/tests/test_ocr_normalizer.py`
- `pipeline/main.py`
- `pipeline/translator/translate.py`

## Testes e comandos
- `.\\venv\\Scripts\\python.exe -m pytest tests/test_ocr_normalizer.py tests/test_main_emit.py -q` passou com 25 testes.
- `.\\venv\\Scripts\\python.exe -m pytest tests/test_ocr_normalizer.py tests/test_translate_context.py tests/test_main_emit.py -q` passou com 71 testes.

## Falhas e correcoes
- Sem falhas apos implementacao.

## Proximo ponto
Avancar para a Fase 8: Entity Detector e Term Protection.
