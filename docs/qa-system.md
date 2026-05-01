# QA system

## Flags

O QA detecta:

- Ingles restante.
- Termos fora do glossario.
- Texto grande demais.
- Paginas suspeitas.
- Artefatos de export.

## Severidade

- `critical`: bloqueia export limpo.
- `high`: bloqueia export limpo.
- `medium`: aviso.
- `low`: aviso informativo.

## Revisao

O usuario pode ir para a pagina da flag, corrigir, ou ignorar com motivo. Motivos devem aparecer no relatorio.

## Export

O modo `Clean` exige que flags criticas/altas estejam corrigidas ou ignoradas com motivo.
