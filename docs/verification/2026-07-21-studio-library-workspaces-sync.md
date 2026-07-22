# Verificação — biblioteca e áreas de trabalho do Studio

- Plano verificado: `2026-07-21-studio-library-workspaces-sync-implementation.md`
- Execução final: 22 de julho de 2026
- Branch: `codex/studio-library-workspaces`

## Resultado

O marco está funcional para o fluxo específico de mangá/manhwa: organizar obras
e capítulos locais, abrir projetos do TraduzAI Central, criar capítulos a partir
de imagens, traduzir/revisar manualmente e continuar a edição no mesmo documento.
AniList e MangaDex são opcionais e limitados a metadados. O pipeline automático
não foi alterado nem chamado, e FLUX permanece fora deste marco.

Isto não representa paridade geral com Photoshop. O objetivo verificado é o
conjunto de operações editoriais necessárias ao fluxo TraduzAI.

## Ambiente

| Item | Valor |
| --- | --- |
| SO | Microsoft Windows NT 10.0.26200.0 |
| Janela/viewport | 1360 × 820 |
| WebView | WebView2 do aplicativo Tauri |
| Node.js | v22.19.0 |
| npm | 10.9.3 |
| Rust | rustc 1.94.1 / cargo 1.94.1 |
| App | Tauri v2, identificador `com.traduzai.studio` |

A aceitação foi executada no Tauri real, com `__TAURI_INTERNALS__` presente. A
interface foi inspecionada pelo CDP do WebView2 e os diálogos de arquivo pelo
Windows UI Automation. O conector nativo de Computer Use não estava disponível
nesta máquina, por isso ele não foi usado.

## Verificação automatizada final

| Comando | Resultado |
| --- | --- |
| `npm --prefix studio test` | PASS — 42 arquivos, 190 testes |
| `npm --prefix studio run build` | PASS — TypeScript e Vite, 3.378 módulos |
| `cargo fmt --manifest-path studio/src-tauri/Cargo.toml -- --check` | PASS |
| `cargo test --manifest-path studio/src-tauri/Cargo.toml` | PASS — 40 testes |
| `cargo clippy --manifest-path studio/src-tauri/Cargo.toml --all-targets -- -D warnings` | PASS, zero warnings |
| `cargo check --manifest-path studio/src-tauri/Cargo.toml` | PASS |
| `npm run build` | PASS — app principal, 1.775 módulos |

## Fixtures

As fixtures locais ficaram em `.tmp/task14/fixtures-r2/` dentro da worktree:

- projeto Central existente com uma página;
- pasta com `page-2` e `page-10`, para conferir ordenação natural;
- ZIP com duas imagens;
- CBZ com duas imagens e subpasta segura;
- cópia movida de `project.json`, para relocalização.

Os dados editoriais usados no teste de persistência foram o bloco original
`BURNED`, a tradução `TEXTO VALIDADO TASK 14`, tipo `pensamento`, status
`Aprovado` e a nota `Nota editorial persistida no teste Tauri.`.

## Matriz de aceitação no Tauri

| # | Cenário | Resultado e evidência observada |
| --- | --- | --- |
| 1 | Biblioteca vazia | PASS — home abriu sem catálogo, com obras à esquerda, capítulos ao centro e ações de criação/anexo. |
| 2 | Criar obra manual | PASS — `Obra Piloto Task 14` foi criada e selecionada pela interface. |
| 3 | Anexar projeto Central | PASS combinado — o `project.json` real foi carregado, registrado, exibido e aberto no editor; validação de formulário/duplicata está coberta pelos testes. O seletor nativo usa o mesmo caminho exercitado integralmente em Relocalizar. |
| 4 | Criar por pasta | PASS no runtime — duas páginas foram copiadas e ordenadas como 2, 10; formulário e escolha de origem estão cobertos pelos testes. |
| 5 | Criar por ZIP/CBZ | PASS no runtime — ambos geraram capítulos de duas páginas; extração segura e limites têm testes Rust. |
| 6 | Alternar Tradução/Edição | PASS — alternância manteve projeto, URL e seleção; a área Tradução mostrou fila, inspector e glossário. |
| 7 | Salvar, fechar e reabrir | PASS — tradução, tipo, nota, status aprovado, camada e workspace foram persistidos após duas confirmações e reabertura. |
| 8 | Relocalizar capítulo movido | PASS — caminho ausente foi detectado, o seletor nativo apontou para a cópia movida, o catálogo foi atualizado e o capítulo abriu. |
| 9 | Vincular AniList/MangaDex | PASS — os dois provedores apareceram na obra; atualização real pelo comando Rust retornou metadados válidos. |
| 10 | Online e cache offline | PASS — atualização online concluiu; com cache e erro offline, os dados anteriores continuaram visíveis com a indicação `Offline`. |
| 11 | Conflito de status manual | PASS — o override manual permaneceu e a interface exibiu `Conflito`, sem sobrescrever silenciosamente. |
| 12 | Sem páginas remotas/pipeline | PASS — tracking trafegou somente metadados, nenhuma página foi baixada e nenhuma ação do pipeline Central foi disparada. |

Para pasta, ZIP e CBZ, a preparação foi invocada pelo comando Rust real através
do WebView, com as fixtures acima; os três seletores nativos de origem não foram
repetidos individualmente pela automação. O fluxo nativo de seleção foi
exercitado de ponta a ponta no relink, e os contratos dos diálogos têm cobertura
automatizada.

## Recuperação

O catálogo principal foi corrompido de propósito após existir uma cópia válida.
Na reabertura, o Studio carregou o `.bak`, exibiu o banner de recuperação e a
ação **Salvar cópia recuperada** regravou o catálogo principal. A biblioteca
continuou com uma obra e cinco referências de capítulo.

Também foi confirmada a permissão Tauri `fs:allow-exists`: um caminho existente
retornou `true`, um caminho ausente retornou `false` e o card mudou para o estado
relocalizável.

O catálogo criado exclusivamente para o ensaio foi removido de
`%APPDATA%/com.traduzai.studio/` ao fim da execução. As fixtures permanecem na
pasta `.tmp`, ignorada pelo Git.

## Defeitos encontrados e corrigidos durante a aceitação

1. `translation_status` e `translation_notes` eram removidos quando a edição
   pendente era confirmada. Os dois caminhos de commit agora preservam os campos
   e há testes de regressão.
2. Um projeto anexado cujo título interno diferia da obra selecionada podia
   criar uma obra duplicada ao abrir. O registro agora procura primeiro pelo
   caminho normalizado do capítulo e só depois pelo título.
3. A checagem de caminho ausente era negada pela capability do Tauri e o erro
   acabava tratado como arquivo existente. `fs:allow-exists` foi autorizado nos
   mesmos escopos de leitura/escrita.

## Build desktop

`npm --prefix studio run tauri:build` concluiu em modo release e gerou:

`studio/src-tauri/target/release/traduzai_studio.exe`

- tamanho: 18.139.136 bytes;
- SHA-256: `93119206F2C7A60EBF38065221DFEEAB84C85FFCEEC3D9703FFE3955A6A77142`;
- inicialização isolada: PASS — processo `traduzai_studio` responsivo após cinco
  segundos, sem servidor ouvindo na porta Vite 1430.

O harness precisou encerrar o processo de verificação com força depois que o
primeiro pedido de término não finalizou em dez segundos. Não houve erro de
inicialização. Como `bundle.active` está desativado, esta etapa produz o
executável, mas ainda não produz MSI/NSIS.

## Limitações abertas

- O chunk `StudioWorkspaceShell` tem aproximadamente 977,56 kB minificado; o
  Vite conclui o build, mas recomenda divisão adicional.
- Se o usuário relocalizar deliberadamente para outro projeto cujo campo
  interno de capítulo difira da referência, a reabertura adota o metadado do
  projeto escolhido. O relink normal do mesmo projeto movido não é afetado.
- Empacotamento com instalador e assinatura ainda não estão ativos.
- FLUX/ControlNet, escolha de modelo, requisitos de VRAM e inpainting por prompt
  continuam adiados.
- Tradução automática, OCR automático, download de capítulos, nuvem e
  colaboração permanecem fora do Studio.

## Conclusão

Os critérios funcionais deste marco foram atendidos no runtime real e nas suítes
automatizadas. A biblioteca é independente dos `project.json`, os capítulos
continuam compatíveis com o Central e a troca Tradução/Edição preserva o mesmo
documento. O próximo marco pode tratar distribuição e desempenho antes de
retomar FLUX.
