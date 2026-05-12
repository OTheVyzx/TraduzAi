# Contexto Atual do Projeto

Ultima atualizacao: 2026-05-09

## Resumo
- Branch ativa: `feat/editor-brush-mask-typesetting`
- HEAD atual: `f60e80d`
- Foco recente: estabilizacao do editor Konva, ferramentas bitmap/recovery, preview/export fiel e padrao automatico de typesetting.
- Estado geral: o editor esta em uma fase de ajustes finos. O workspace esta bem sujo com alteracoes reais e muitos artefatos de runtime/validacao, entao qualquer commit deve ser feito por caminhos explicitos.
- Contexto migrado para o Untera em `N:\TraduzAI\TraduzAi`. Resumo consolidado dos chats antigos: `docs/plans/2026-05-09-untera-chat-transfer.md`.

## Migracao Untera
- Raiz atual do workspace: `N:\TraduzAI`.
- App principal: `N:\TraduzAI\TraduzAi`.
- Pasta Koharu vista no workspace: `N:\TraduzAI\koharu`.
- O checkout veio muito sujo depois da transferencia; nao usar `git add .`.
- Ao retomar trabalho com Koharu, verificar primeiro a versao/API do Koharu em `N:\TraduzAI\koharu`, porque antes havia divergencia entre binario antigo `0.41.4` e repo atualizado `0.58.0`.

## Mudancas Recentes Importantes

### Editor bitmap/recovery
- O pincel de recuperacao voltou a ser instantaneo no preview, sem exigir alternar visibilidade de camadas tecnicas.
- `recovery`, `brush` e `eraser` foram ajustados para evitar conflito de cache/estado entre ferramentas.
- Ferramentas bitmap nao devem selecionar textos traduzidos enquanto a ferramenta ativa nao for `select`.
- O ponteiro de brush/eraser/recovery passou a usar overlay visual proprio, com cursor circular mais responsivo e contraste por `mix-blend-mode: difference`.

### Historico e salvamento
- Undo/redo foi ampliado para cobrir edicoes bitmap relevantes.
- O save do editor e `Ctrl+S` devem representar o caminho principal de salvar/renderizar quando necessario, evitando depender de um botao separado de "Salvar+Render".
- Ainda vale revisar regressao em operacoes muito rapidas consecutivas de recovery/brush/eraser, porque esse fluxo foi sensivel a concorrencia e cache.

### Typesetting automatico
- O automatico agora deve concluir com:
  - fonte `ComicNeue-Bold.ttf`;
  - sem contorno;
  - sem sombra;
  - sem brilho;
  - cor por contraste contra o fundo do balao/painel.
- A normalizacao fica no pipeline automatico e no renderer como defesa final, mas o editor continua livre: quando o usuario altera estilo, a camada passa a `style_origin: "editor"`.
- Arquivo principal novo: `pipeline/typesetter/style_policy.py`.
- Pontos integrados:
  - `pipeline/main.py::build_text_layer`
  - `pipeline/vision_stack/runtime.py`
  - `pipeline/typesetter/renderer.py`
  - `src/lib/tauri.ts`
  - `src/lib/stores/editorStore.ts`
  - `src-tauri/src/commands/project_schema.rs`

### Layout de texto
- Foi criado suporte para reduzir tamanho do texto quando a caixa esta correta mas o texto fica cortado.
- Continuar tratando como requisito: OCR/traducao podem estar corretos, mas o layout deve caber no balao sem cortar.

## Validacao Recente
- `npm run check`: passou.
- Vitest focado em estilo/hidratacao/historico: passou (`11 passed`).
- Pytest focado em politica de estilo/typesetting: passou (`8 passed`).
- `cargo test create_patch_delete_text_layer_updates_legacy_aliases`: passou.
- `cargo check`: passou.
- Playwright focado do editor:
  - `editor Konva usa fundo limpo e layers editaveis`: passou.
  - `pincel de recuperacao mantem textos visiveis sem alternar camadas tecnicas`: passou.
- Playwright amplo em `e2e/editor-rebuild.spec.ts --grep "editor|typesetting"` pegou quase o arquivo inteiro e falhou fora do escopo atual:
  - `setup revisa candidatos no glossario central`;
  - `processing mostra progresso percebido e metricas`;
  - `processing final mostra tempo total e capa da obra`.

## Pendencias Conhecidas
- Investigar os 3 testes Playwright acima antes de considerar a suite ampla verde.
- Verificar manualmente um capitulo automatico novo para confirmar contraste de texto em baloes brancos e fundos escuros.
- Monitorar concorrencia de strokes quando recovery/brush/eraser sao usados em sequencia muito rapida.
- Manter cuidado com projetos antigos: `style_origin` e opcional e deve preservar compatibilidade.

## Convencao Operacional
- `cntbk` significa:
  - atualizar este `context.md`;
  - criar um novo backup versionado do projeto;
  - excluir o backup versionado anterior.
