# TraduzAI Studio

O TraduzAI Studio é o editor desktop do ecossistema TraduzAI para tradução
manual, revisão e pós-produção de capítulos de mangá, manhwa e manhua. Ele abre
os projetos produzidos pelo TraduzAI Central, mas não executa OCR, tradução
automática nem o pipeline automático.

O escopo é substituir, nesse fluxo editorial específico, as operações que
normalmente exigiriam um editor de imagem generalista. Não há promessa de
paridade completa com o Photoshop.

## Executar e compilar

```powershell
npm --prefix studio run tauri:dev
npm --prefix studio test
npm --prefix studio run build
npm --prefix studio run tauri:build
```

O modo de navegador (`npm --prefix studio run dev`) é útil para desenvolvimento,
mas diálogos nativos, persistência no disco e provedores externos dependem do
Tauri. Atualmente `bundle.active` está desativado: `tauri:build` produz o
executável release, não um instalador MSI/NSIS.

## Biblioteca, obras e capítulos

A tela inicial segue o modelo de uma biblioteca de produção:

- a coluna esquerda contém as **obras**;
- a área principal contém os **capítulos** da obra selecionada;
- cada capítulo referencia um `project.json` local e pode ser pesquisado,
  aberto ou relocalizado.

O catálogo é armazenado logicamente em
`app_data_dir()/studio-library.json`, com backup em
`app_data_dir()/studio-library.json.bak`. No Windows, para o identificador
atual do aplicativo, isso corresponde a
`%APPDATA%/com.traduzai.studio/studio-library.json`.

O catálogo e os projetos têm responsabilidades diferentes:

- `studio-library.json` guarda obras, referências de capítulos, preferência de
  workspace e cache de acompanhamento;
- cada `project.json` continua sendo o documento editável do capítulo, com
  páginas, camadas, textos, estilos e metadados compatíveis com o TraduzAI;
- remover uma referência da biblioteca não apaga o projeto do disco.

Projetos TraduzAI v1/v2 e projetos de análise v12 são adaptados para o modelo
editável do Studio sem descartar os aliases usados pelo app Central.

## Adicionar conteúdo

Há duas formas de adicionar capítulos:

1. **Anexar projeto existente:** seleciona um `project.json` já criado pelo
   TraduzAI Central.
2. **Criar capítulo manual:** seleciona uma pasta, ZIP ou CBZ com imagens e cria
   um novo `project.json`.

A criação manual aceita PNG, JPEG e WebP, preserva subpastas seguras e aplica
ordenação natural dos nomes. A importação é local e transacional; rejeita
caminhos inseguros, links simbólicos e imagens inválidas. Os limites atuais são
2.000 páginas, 100 MiB por arquivo, 2 GiB por importação e 10.000 entradas por
arquivo compactado.

## Áreas de trabalho

O seletor no canto superior direito alterna o mesmo capítulo, sem recarregar ou
duplicar o documento:

- **Tradução:** fila de blocos por estado, original somente para leitura,
  tradução editável, tipo do texto, notas editoriais, status
  (`Pendente`, `Traduzido`, `Revisão`, `Aprovado`) e glossário local da obra.
  `Alt+↑/↓` navega pelos blocos e `Ctrl+Enter` confirma e avança.
- **Edição:** canvas, seleção e transformação, texto, camadas raster, máscara,
  pincel, borracha, laço, retoque, ferramentas de capítulo, undo/redo e
  exportações existentes.

A tradução desta área é manual. A ação de tradução automática permanece
desconectada no Studio para não criar um segundo pipeline concorrente com o
TraduzAI Central.

## Acompanhamento de obras

O vínculo com **AniList** e **MangaDex** é opcional e consulta somente
metadados por comandos Rust. O Studio não faz scraping de leitores, não baixa
páginas e não adiciona capítulos remotos à biblioteca.

- o cache de atualizações tem TTL de 30 minutos;
- dados em cache continuam visíveis sem conexão, com indicação de defasagem e
  do último erro;
- a atualização manual respeita espera e backoff para falhas transitórias;
- um status editorial definido manualmente não é sobrescrito em silêncio:
  divergências com o provedor são exibidas como conflito.

## Recuperação e caminhos movidos

O catálogo é gravado de forma atômica. Se a cópia principal estiver corrompida,
o Studio pode carregar o `.bak`, sinaliza a recuperação e oferece **Salvar
cópia recuperada**. Falhas de gravação mantêm o estado em memória para que o
usuário possa tentar novamente.

Um capítulo cujo `project.json` foi movido permanece na biblioteca como caminho
ausente. A ação **Relocalizar** troca apenas a referência, sem apagar dados nem
criar uma obra duplicada.

Os projetos abertos também usam autosave incremental e snapshots locais em
`.traduzai-studio/recovery/`, ao lado do `project.json`.

## Privacidade e funcionamento offline

Imagens, máscaras e projetos permanecem locais. Somente os identificadores e
metadados necessários ao acompanhamento opcional são consultados nos
provedores externos. Sem conexão, edição, tradução manual, biblioteca e
projetos locais continuam disponíveis; apenas a atualização externa fica
pendente.

## FLUX

FLUX/ControlNet e geração por prompt estão explicitamente adiados neste marco.
O protótipo local existente permanece isolado e opcional, mas não integra o
fluxo suportado da biblioteca ou da tradução manual. Escolha do modelo,
empacotamento, requisitos de VRAM e validação de inpainting serão tratados em
uma etapa própria, depois das funções editoriais.

## Licença e política de reutilização

O Studio é GPL-3.0-only. Reutilizamos primeiro contratos e componentes do
TraduzAI; código externo só pode ser incorporado quando a licença e a
compatibilidade forem verificadas. Não são copiados código, assets ou modelos
de repositórios sem licença compatível.
