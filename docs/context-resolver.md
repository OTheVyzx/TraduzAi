# Context resolver

## Objetivo

Resolver contexto de obra com fontes online e cache local, produzindo candidatos seguros para o glossario.

## Fontes

- AniList.
- MyAnimeList.
- MangaUpdates.
- Wikipedia.
- Wikidata.
- Fandom.

## Saida

O resolvedor normaliza:

- Titulo.
- Sinopse.
- Personagens.
- Termos de lore.
- Fontes.
- Confianca.

## Regras de seguranca

- Nao enviar imagens para fontes de contexto.
- Nao marcar termos online como reviewed automaticamente.
- Permitir continuar sem contexto.
- Exibir risco quando obra ou glossario estiverem vazios.

## Testes

Testes devem usar mocks/fixtures e nao depender de internet real.
