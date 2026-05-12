# Untera Chat Transfer - 2026-05-09

Este arquivo consolida os chats informados pelo usuario para continuar no workspace do Untera.

## Escopo Local

- Workspace recebido: `N:\TraduzAI`.
- App principal: `N:\TraduzAI\TraduzAi`.
- Pasta Koharu presente: `N:\TraduzAI\koharu`.
- O antigo contexto vinha de `D:\TraduzAi` e `D:\koharu`.
- O checkout atual esta sujo com muitas mudancas reais e muitos artefatos de runtime. Evitar `git add .`; usar caminhos explicitos.

## Sessoes Migradas

- `019e0ab9-ca62-77a2-8c9d-ba671e88b062`: Koharu, videos, API/headless, plano de integracao seletiva e licenca GPL.
- `019e0a97-1242-7ce0-a286-ec9429968851`: correcao de inpaint ultrapassando balao e falso positivo de balao conectado.
- `019e04a5-8388-7ec2-aea3-3afed789953b`: site/landing/dashboard e ajustes visuais do produto.
- `019e05b7-8a9a-7fa1-8621-d862078ff7ec`: aceleracao do pipeline, OCR/inpaint, Smart Skip, Macro OCR, Fast Fill e benchmarks.
- `019e04a8-f854-7663-aefa-e980d2c1d983`: editor, recovery brush, lasso, acoes regionais, camadas, estilo e ajustes de Setup.
- `019e089d-82ae-7181-97ec-ecb316b81273`: auditoria somente leitura de tamanho do projeto.
- `019e0570-8f42-7b52-96f3-655a8c05b5a2`: landing/login/signup e base de Google OAuth no site/backend.

## Decisoes Sobre Koharu

- Nao copiar codigo do Koharu para dentro do TraduzAi agora.
- Usar Koharu, se aprovado, como backend externo/opcional via headless/API.
- Antes de qualquer integracao, verificar a versao local no Untera. A analise anterior encontrou:
  - repo atualizado em `D:\koharu`: `0.58.0`, com API nova `POST /api/v1/pipelines`, `/operations`, `/events` e MCP;
  - binario antigo em `N:\koharu`: `0.41.4`, com contrato antigo, apesar de aceitar `--headless --port`.
- Integracao proposta: `KoharuVisionAdapter` isolado, sem misturar com o pipeline principal.
- Roteamento sugerido:
  - paginas com altura `<= 4000px`: tentar Koharu;
  - paginas maiores: testar bandas de `3200px` a `3900px`, com overlap de `256px` a `384px`;
  - se detectar pouco texto, errar OCR, falhar inpaint ou demorar demais: fallback para pipeline TraduzAi.
- Comecar somente por detect/OCR/inpaint. Manter Google como traducao padrao e manter editor/renderer/schema do TraduzAi como donos do resultado final.
- Licenca: Koharu e GPL-3.0. Para monetizacao, o caminho menos arriscado e processo externo/opcional via API/CLI, sem copiar codigo e sem embutir binario no desktop ate haver revisao de licencas do Koharu e dos modelos.

## Pipeline E Performance

Estado final do trabalho de aceleracao:

- Fast Fill default validado como ganho parcial.
- Comparacao pareada:
  - baseline antigo: `181.9548s`;
  - default novo: `139.7025s`;
  - ganho: `42.2523s`.
- Meta do plano era `<=113.65s`; nao foi atingida.
- Gargalo restante: OCR + inpaint, `99.6763s`, `93.02%` do tempo medido por estagios.
- Manter desligado por default:
  - `TRADUZAI_SMART_SKIP`;
  - `TRADUZAI_MACRO_OCR`;
  - `TRADUZAI_STRIP_SCHEDULER_EXECUTOR=overlap`.
- Status detalhado ja existe em `docs/plans/2026-05-08-aceleracao-traducao-automatica-status.md`.

## Correcoes Visuais Recentes

Sessao `019e0a97` corrigiu dois pontos:

- Inpaint ultrapassando balao: ajuste em `pipeline/inpainter/__init__.py`, usando `text_pixel_bbox`/`bbox` como semente e aplicando mascara a partir da banda original.
- Falso positivo de balao conectado:
  - layout rejeita caso de texto unico com linhas OCR empilhadas/compactas quando grupos nao sao separaveis;
  - renderer limpa `balloon_subregions` se projeto antigo vier com `layout_group_size=1` + `connected_balloon`.
- Validacoes relatadas: suite de layout `72 passed`; pacote final focado `89 passed`; rerun real confirmou `layout_profile=white_balloon`, `layout_group_size=1`, `balloon_subregions=[]`.

## Editor E Setup

Estado relevante vindo das sessoes:

- Recovery brush virou camada editavel e deve restaurar pixels originais no preview/render/export.
- Lasso virou selecao persistente com menu contextual para detect/OCR/translate/inpaint regional.
- Acoes regionais aceitam `bbox` ou mascara externa ate o Python sidecar.
- `brush` preserva RGBA; `mask` e `recovery` sao camadas tecnicas.
- Politica canonica de texto automatico: `ComicNeue-Bold`, sem contorno/sombra/brilho, preservando cor/tamanho quando o usuario edita.
- `src/lib/editorTextStylePolicy.ts` e `pipeline/typesetter/style_policy.py` sao pontos centrais.
- Blocos de UI de contexto online/glossario avancado foram escondidos da tela de Setup sem remover estado/funcoes por baixo; `npm run check` passou naquela rodada.
- Status detalhado do plano de editor: `docs/plans/editor-bugfixes-e-features-status.md`.

## Site, Landing E Login

- O site continua isolado em `site/`.
- Dashboard foi simplificado:
  - removeu textos tecnicos como GPU/LLM pronto/reiniciar app;
  - sidebar usa logo da landing;
  - mostra conta e `1000 creditos`;
  - headline mudou para `Um capitulo em instantes`.
- Landing teve varias iteracoes de copy/rolagem/header/login.
- Fluxo base de Google OAuth foi criado no backend:
  - `server/auth_api.py`;
  - `server/config.py`;
  - variaveis em `.env.example`;
  - botoes `Continuar com Google` em `/login` e `/signup`.
- Sem `TRADUZAI_GOOGLE_CLIENT_ID` e `TRADUZAI_GOOGLE_CLIENT_SECRET`, o botao deve responder como indisponivel.

## Auditoria De Tamanho

Sessao `019e089d` foi somente leitura e nao editou nada.

Maiores alvos encontrados no checkout antigo:

- `.git`: muitos loose objects, cerca de dezenas de GiB.
- `src-tauri/target`: build cache Rust.
- `pipeline/venv`: pesado por torch/nvidia/tensorrt.
- `vision-worker/target`: build cache.
- `exemplos`, `data`, `debug`, `pipeline/scratch`, `test-results`: artefatos/dados de validacao.

Nao apagar caches, dados ou backups sem pedido explicito do usuario.

## Proximo Passo Recomendado No Untera

1. Confirmar versao e contrato do Koharu em `N:\TraduzAI\koharu`.
2. Confirmar que `N:\TraduzAI\TraduzAi` abre/testa com o ambiente local.
3. Se o pedido for Koharu: fazer spike sem mexer no pipeline principal, comparando 3 grupos de paginas:
   - pagina pequena boa (`<=4000px`);
   - pagina grande em bandas;
   - pagina em que Koharu falha.
4. Medir tempo, textos detectados, OCR, qualidade do inpaint e imagem final comparativa.
5. So promover para integracao se passar em qualidade e tiver fallback automatico.
