# TraduzAI Studio Master Editor Design

**Status:** aprovado pela direção registrada no board de auditoria do Studio.

**Board:** https://www.figma.com/design/2sbwORZPp50cK5aUTRpO7j?node-id=2-2

## Objetivo

Transformar o TraduzAI Studio no editor mestre de pós-tradução para scanlation. O TraduzAI Central continua responsável por detecção, OCR, tradução e limpeza automática; o Studio abre o `project.json` resultante e concentra lettering, camadas, máscaras, retoque, revisão, FLUX e exportação.

## Escopo e limites

- O app central não muda de papel nem perde as ações automáticas existentes.
- O Studio não expõe Detectar, OCR, Traduzir, idioma de origem ou o processo regional automático.
- `paginas[] + image_layers + text_layers` continua sendo o contrato compatível na primeira fase.
- O futuro `studio_scene` será aditivo e projetará de volta os aliases legados nas fronteiras de importação e exportação.
- O FLUX entra como ferramenta de edição por seleção/máscara, nunca como pipeline de capítulo.
- Toda UI continua em português brasileiro e dark-only.

## Alternativas consideradas

### 1. Modo Studio no editor compartilhado — recomendado

Adicionar um contrato explícito de experiência ao editor reutilizado. O modo padrão mantém o TraduzAI Central intacto; o modo Studio oculta superfícies automáticas e oferece somente ferramentas de edição. É a menor mudança segura e evita duplicação de canvas, histórico e exportação.

### 2. Copiar o editor atual para `studio/`

Permite liberdade visual imediata, mas cria duas implementações de seleção, texto, brush, undo/redo e exportação. A divergência e o custo de correção dobrariam rapidamente.

### 3. Reativar `StudioEditor.tsx` como reescrita independente

Produz uma arquitetura isolada, mas perde a paridade já alcançada no editor compartilhado e reabre problemas resolvidos de Konva, estilos, bitmap e PSD.

## Arquitetura escolhida

O componente compartilhado recebe `mode="traduzai" | "studio"`. Um resolvedor puro converte o modo em capacidades:

- ações de pipeline;
- seletor de idioma de origem;
- ações automáticas por bloco;
- ferramentas automáticas na barra lateral;
- apresentação profissional do painel de camadas.

O modo padrão é `traduzai`, garantindo compatibilidade para callers existentes. `StudioSharedEditor` passa explicitamente `mode="studio"`.

## Primeiro corte implementável

1. Trocar a entrada falsa “Novo projeto” por “Abrir projeto TraduzAI”.
2. Remover a criação do projeto de demonstração da ação principal.
3. Ocultar no Studio a lateral Detectar/OCR/Traduzir/Inpaint.
4. Ocultar seletor de idioma e mensagens de pipeline.
5. Ocultar ações OCR/Traduzir/Limpar de cada camada textual.
6. Ocultar ferramentas `repairBrush`, `reinpaintBrush` e `process` até existirem implementações editoriais próprias.
7. Renomear a superfície lateral do Studio de “Textos” para “Camadas” e usar nomes legíveis para layers raster.

## Evolução do documento

Depois do primeiro corte, cada página ganhará `studio_scene` com nós ordenáveis:

- `raster`;
- `text`;
- `group`;
- `mask`;
- `generated`;
- `adjustment`;
- `fill`.

Os nós terão `id`, `name`, `visible`, `locked`, `opacity`, `blendMode`, `parentId`, `order`, máscaras e metadados específicos. Durante a migração, `image_layers` e `text_layers` permanecem como projeção compatível.

## Estilos

O motor de estilos será não destrutivo e dividido em:

- tipografia: fonte, tamanho, tracking, leading, escala, baseline, alinhamento, orientação e warp;
- preenchimentos: sólido e gradientes múltiplos;
- contornos: múltiplos strokes com posição e opacidade;
- efeitos: sombra, glow e sombras internas;
- presets: globais, por projeto, por capítulo e por função semântica.

O mesmo resolvedor visual deve alimentar canvas, exportação raster e PSD.

## FLUX no Studio

O fluxo será seleção/máscara → prompt opcional → crop local → 2–4 variantes → novas camadas `generated`. Pixels fora da máscara não podem mudar. Cada variante registra modelo, seed, prompt, dimensões e origem; aceitar/rejeitar precisa ser reversível no histórico.

## Erros e segurança

- Projeto incompatível deve falhar antes de substituir o documento aberto.
- Operações longas devem gerar comandos transacionais no histórico.
- Arquivos legados nunca perdem aliases durante round-trip.
- FLUX deve manter o original e nunca gravar sobre a camada-base.
- Autosave deve usar escrita atômica e recuperação de sessão.

## Estratégia de testes

- testes puros do contrato de capacidades;
- render estático da home e componentes condicionais;
- round-trip de `project.json` e futuro `studio_scene`;
- histórico/undo para operações estruturais;
- paridade visual canvas/exportação;
- PSD com ordem e texto editável;
- FLUX garantindo invariância fora da máscara.

