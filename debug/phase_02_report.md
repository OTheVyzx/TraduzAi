# Phase 02 Report

## Implementado
- Contrato v12 minimo criado em Python e JSON Schema.
- Migrador v1/v2-like para v12 criado sem plugar ainda no pipeline principal.
- Tipos TypeScript v12 adicionados.
- Validacao de consistencia entre `qa.summary` e `qa.flags`.
- Compatibilidade com `legacy.paginas` preservada.
- Correcoes de typesetting feitas para liberar o gate amplo de pytest.

## Arquivos alterados
- `pipeline/schema/__init__.py`
- `pipeline/schema/project_schema_v12.py`
- `pipeline/schema/project_schema_v12.json`
- `pipeline/schema/migrate_project.py`
- `pipeline/tests/test_project_schema_v12.py`
- `pipeline/tests/test_project_migration.py`
- `src/lib/projectSchema.ts`
- `pipeline/typesetter/renderer.py`

## Testes adicionados
- `pipeline/tests/test_project_schema_v12.py`
- `pipeline/tests/test_project_migration.py`

## Comandos rodados
- `pipeline/venv/Scripts/python.exe -m pytest tests/test_project_schema_v12.py tests/test_project_migration.py -q`
- `pipeline/venv/Scripts/python.exe -m pytest tests/test_typesetting_layout.py -q`
- `pipeline/venv/Scripts/python.exe -m pytest -q`
- `npm run build`

## Falhas encontradas
- RED inicial: `ModuleNotFoundError: No module named 'schema'`, esperado antes da implementacao.
- Gate amplo de pytest falhou inicialmente em 3 testes de typesetting por bbox/altura/posicionamento de fonte.
- Apos o primeiro ajuste, a suite completa ainda expunha um caso order-dependent em baloes conectados.

## Correcoes aplicadas
- `SafeTextPathFont.getbbox()` passou a tratar o pixel maximo como limite exclusivo.
- `measure_text_width()` passou a usar a largura real da mascara para `SafeTextPathFont`.
- `white_balloon` passa a centralizar verticalmente e usa tolerancia de altura mais estrita.
- Baloes conectados left-right ganharam deslocamento diagonal mais claro.
- Scoring de split conectado reforcado para preservar fronteiras de frase e casos de discurso com `MAS SEUS EFEITOS JA`.

## Evidencias
- Testes v12 focados: 5 passed.
- `pipeline/venv/Scripts/python.exe -m pytest -q`: 483 passed, 1 skipped.
- `npm run build`: passou.

## Status
Aprovado.
