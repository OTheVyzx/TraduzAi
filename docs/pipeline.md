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

## Testes

Use pytest para pipeline:

```bash
cd pipeline
.\\venv\\Scripts\\python.exe -m pytest -q
```
