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

## Ownership textual page-global

O formato operacional (`versao`/`paginas`) e o schema v12
(`schema_version`/`pages`) compartilham um envelope aditivo de ownership:

```json
{
  "owner_graph_schema_version": 1,
  "owner_graph_status": "verified",
  "page_owner_graphs": [],
  "owner_invariant_summary": {}
}
```

`page_owner_graphs` guarda snapshots completos e serializaveis de `OwnerGraph`.
Os IDs de pagina, component, observation e owner precisam ser unicos e todas as
referencias devem permanecer dentro do mesmo graph de pagina. O summary e um
cache derivado; ele e recalculado e conferido antes da gravacao, nunca usado
sozinho para conceder confianca.

Uma camada ligada ao control plane persiste `owner_id`, `component_ids`,
`observation_ids`, `semantic_role`, `route_action`, `action_mask_ref` e
`layout_region_ids`. O writer rejeita owner ausente, referencia cruzada ou
identidade divergente do graph verificado.

## Compatibilidade

Migradores devem preservar projetos antigos sempre que possivel. Campos novos precisam ter defaults seguros.
Projetos sem graph autoritativo recebem `owner_graph_status="legacy_unverified"`,
`page_owner_graphs=[]` e continuam editaveis. Esse estado nao pode entrar no
caminho automatico verificado: a pagina precisa ser reprocessada para construir
um graph novo. A migracao nunca cria owners ou marca `verified` a partir de
layers, regioes ou alegacoes de formatos legados.

## Editor

Edicoes manuais devem atualizar o projeto sem quebrar schema. Camadas criadas, removidas, ordenadas ou alteradas precisam ser salvas de forma explicita.

## Export

Exports devem incluir o estado final e, quando aplicavel, relatorios de QA.
