# Suporte Dinamico a Idiomas Google - Design

**Data:** 2026-04-17

**Objetivo**

Permitir que o TraduzAi use os idiomas suportados pelo backend atual do Google Translate tanto como idioma de origem quanto de destino, removendo a restricao antiga de fluxo focado apenas em ingles.

**Escopo aprovado**

- Origem: todos os idiomas retornados pelo backend atual do Google Translate.
- Destino: todos os idiomas retornados pelo backend atual do Google Translate.
- A lista de idiomas deve vir do runtime Python, nao de tabela manual fixa no frontend.
- O OCR passa a operar em modo melhor esforco para idiomas sem suporte forte no backend OCR atual.

**Arquitetura**

- O sidecar Python passa a expor um modo para listar idiomas suportados pelo Google Translate via `deep-translator`.
- O backend Rust adiciona um command Tauri para consultar essa lista e repassa os dados ao frontend.
- O frontend usa essa lista dinamica em `Setup` e `Settings`, tanto para origem quanto para destino.
- O pipeline Python normaliza codigos de idioma em um ponto central para:
  - Google Translate
  - OCR/PaddleOCR
  - EasyOCR fallback
- Para idiomas sem mapeamento OCR dedicado, o pipeline cai para um modelo OCR generico de melhor esforco em vez de travar em ingles de forma silenciosa.

**Decisoes principais**

- `en-GB` e variantes semelhantes serao normalizados para `en` no OCR, mas o codigo original pode ser preservado para traducao quando o backend aceitar.
- O destino nao sera mais limitado a `pt-BR`; qualquer idioma suportado pelo Google podera ser salvo em projeto e settings.
- A origem sera totalmente liberada na UI, com comportamento experimental para OCR onde nao houver suporte dedicado.

**Riscos conhecidos**

- O Google Translate suporta mais idiomas do que o OCR instalado no projeto reconhece bem hoje.
- Idiomas fora dos grupos principais do OCR atual podem ter reconhecimento inferior mesmo quando a traducao em si estiver disponivel.
- Alguns codigos do Google e do OCR usam nomes diferentes e exigem normalizacao cuidadosa.

**Mitigacoes**

- Centralizar normalizacao de idioma no pipeline.
- Exibir rotulos claros no frontend e evitar fallback silencioso incorreto.
- Cobrir com testes a listagem de idiomas, normalizacao e fluxo de traducao para idiomas adicionais.
