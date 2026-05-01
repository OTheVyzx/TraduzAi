# Fase 14 - Mascaras reais e validacao

Data: 2026-05-01

## Resultado
- Fase 14 concluida.
- Criado `pipeline/inpainter/mask_validator.py`.
- Valida existencia, leitura, tamanho 1x1, transparencia, vazio e mismatch com bbox.
- Checa se export/projeto contem mascaras esperadas por regiao.

## Arquivos alterados
- `pipeline/inpainter/mask_validator.py`
- `pipeline/tests/test_mask_validator.py`

## Testes e comandos
- `.\\venv\\Scripts\\python.exe -m pytest tests/test_mask_validator.py -q` passou com 5 testes.

## Proximo ponto
Avancar para a Fase 15: Inpaint por tipo de regiao.
