# Project schema

## `project.json`

Arquivo aberto e reimportavel com:

- Identidade do projeto.
- Obra e capitulo.
- Paginas.
- Camadas de texto.
- Camadas de imagem/mascara.
- Flags de QA.
- Preset aplicado.
- Referencias de export.

## Compatibilidade

Migradores devem preservar projetos antigos sempre que possivel. Campos novos precisam ter defaults seguros.

## Editor

Edicoes manuais devem atualizar o projeto sem quebrar schema. Camadas criadas, removidas, ordenadas ou alteradas precisam ser salvas de forma explicita.

## Export

Exports devem incluir o estado final e, quando aplicavel, relatorios de QA.
