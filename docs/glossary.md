# Glossario

## Estados

- `reviewed`: confirmado pelo usuario.
- `candidate`: sugerido por contexto ou deteccao.
- `auto`: criado automaticamente.
- `rejected`: rejeitado.
- `conflict`: conflito entre traducoes.

## Prioridade

```text
reviewed > protected names > memory > candidate > auto
```

## Acoes

- Confirmar candidato.
- Editar termo.
- Rejeitar.
- Aplicar em todas as ocorrencias.
- Adicionar forbidden.
- Transformar em nome protegido.

## Integracao

O Setup centraliza revisao de glossario antes do processamento. O Preview e o Editor usam QA para detectar divergencias.
