# Blindagem Pública do Produto

> Design - 2026-04-17

## Contexto

O produto foi concebido com foco forte em OCR, substituição de texto, composição visual e processamento local de imagens. Hoje, porém, a apresentação pública ainda expõe diretamente um caso de uso sensível por meio de nome, copy, descrições, fluxos e artefatos visíveis ao usuário.

As superfícies mais críticas são:

- branding e descrições públicas com linguagem explícita de tradução;
- UI com termos como `obra`, `capítulo`, `páginas traduzidas`, `imagem traduzida`;
- marketing e documentação pública associando o produto diretamente a mangá, manhwa e manhua;
- recursos visíveis de nicho, como referências externas especializadas e formatos/exportações que denunciam o caso de uso.

O objetivo desta fase não é descaracterizar o motor interno nem mudar o foco real do desenvolvimento. O objetivo é blindar o enquadramento público do produto para que ele seja apresentado e distribuído como um software local de edição e processamento visual.

---

## Goal

Reposicionar o produto público como um `editor inteligente de imagens`, removendo de todas as superfícies visíveis ao usuário, distribuição, marketing e suporte qualquer associação explícita com tradução de mangá, manhwa, manhua ou capítulos.

O nome público ficará temporariamente fixado como `PUBLIC_NAME` até a marca final ser escolhida.

---

## Decisões já aprovadas

- O nome público não será `TraduzAi`.
- O produto público usará um `nome neutro`, com `PUBLIC_NAME` como placeholder oficial.
- A categoria pública aprovada é `editor inteligente de imagens`.
- A linguagem visível ao usuário será totalmente neutralizada, não apenas o marketing.
- Recursos de nicho continuarão podendo existir internamente, mas devem ficar escondidos ou relabelados no produto público.
- O escopo desta rodada é `scrub público completo`, não limpeza total do repositório.

---

## Arquitetura de posicionamento público

### 1. Camada de identidade

Tudo que o usuário, o checkout, o instalador, o README público ou screenshots virem deve apontar para `PUBLIC_NAME`, nunca para uma proposta explícita de tradução de obras seriadas.

Isso inclui:

- nome do app;
- título da janela;
- descrição curta do produto;
- tagline;
- descrição em `package.json`, `Cargo.toml`, manifestos e superfícies de distribuição.

### 2. Camada de narrativa

A narrativa pública deve ser única e consistente:

- editor inteligente de imagens;
- OCR local;
- substituição e composição de texto;
- automação em lote;
- processamento 100% local;
- controle total do usuário sobre os arquivos.

Devem sair da narrativa pública:

- mangá, manhwa, manhua;
- tradutor, tradução automática, EN→PT-BR;
- scanlator, capítulos grátis, páginas traduzidas;
- qualquer copy que sugira finalidade principal de adaptação de obras protegidas.

### 3. Camada de fluxo do app

O fluxo visível ao usuário não pode continuar denunciando o caso de uso por vocabulário de domínio.

Substituições-base:

- `obra` -> `projeto` ou `coleção`
- `capítulo` -> `lote`, `item` ou `arquivo`
- `páginas traduzidas` -> `páginas editadas`
- `imagem traduzida` -> `imagem processada`
- `tradução em lote` -> `processamento em lote`

O objetivo não é maquiar superficialmente uma ou duas telas. O objetivo é que a experiência inteira, do boot ao export, mantenha o mesmo enquadramento neutro.

### 4. Camada de recursos sensíveis

Recursos específicos de nicho não devem desaparecer do motor se ainda forem importantes para o produto real, mas precisam deixar de ser expostos de forma óbvia.

Diretriz:

- esconder ou neutralizar referências a `AniList`, `Webnovel`, `Fandom` e equivalentes;
- remover ou renomear opções de exportação que denunciem o nicho;
- evitar nomes de modelos, botões ou descrições que revelem o caso de uso;
- impedir que mensagens visíveis de erro, progresso ou configuração tragam branding ou copy sensível.

### 5. Camada de artefatos públicos

Os arquivos e metadados gerados por exportação também fazem parte do produto público.

Portanto:

- nomes de saída devem ser neutros;
- descrições de exportação devem ser neutras;
- estruturas públicas como `project.json` devem evitar campos e rótulos que exponham a finalidade sensível;
- a compatibilidade com projetos antigos deve ser mantida, mas novos projetos devem nascer com nomenclatura neutra.

---

## Áreas de mudança

### UI principal

Revisar e neutralizar copy em:

- `src/pages/Home.tsx`
- `src/pages/Setup.tsx`
- `src/pages/Processing.tsx`
- `src/pages/Preview.tsx`
- `src/pages/Settings.tsx`
- `src/components/ui/Layout.tsx`
- `src/App.tsx`

### Branding e distribuição

Revisar e neutralizar:

- `README.md`
- `package.json`
- `src-tauri/Cargo.toml`
- `src-tauri/tauri.conf.json`
- `index.html`

### Exportações e interfaces visíveis

Revisar:

- labels de exportação;
- nomes padrão de arquivos gerados;
- descrições de output;
- metadados visíveis do projeto exportado.

### Linguagem legal e comercial

Adotar copy estável e segura:

- software local de edição/processamento visual;
- OCR e composição visual automatizada;
- usuário responsável pelos direitos e permissões sobre os arquivos utilizados;
- sem promessas comerciais baseadas em obras protegidas ou redistribuição.

---

## Fora de escopo desta fase

- reescrever o core do pipeline para abandonar o foco interno atual;
- limpar todo o histórico do repositório;
- renomear todo símbolo interno, todo comentário técnico ou todo artefato legível apenas por desenvolvimento;
- redefinir neste momento a marca final em vez do placeholder `PUBLIC_NAME`.

---

## Critérios de verificação

1. O app pode ser aberto e navegado sem exibir `mangá`, `manhwa`, `manhua`, `tradutor`, `tradução`, `capítulo`, `scanlator`, `CBZ` ou equivalentes em superfícies públicas.
2. README, descrições de pacote e manifestos públicos sustentam a mesma narrativa de `editor inteligente de imagens`.
3. O fluxo de uso continua funcional após a neutralização de copy e labels.
4. Exportações novas não expõem nomenclatura sensível em nomes de arquivo, rótulos ou descrições visíveis.
5. Projetos antigos continuam podendo ser importados.
6. O scrub público não altera o comportamento do pipeline além do necessário para evitar vazamentos de nomenclatura.

---

## Riscos

- neutralização parcial, deixando o produto incoerente entre UI, docs e distribuição;
- trocar só o marketing e manter o app denunciando o uso real;
- quebrar compatibilidade de import/export ao renomear superfícies sem mapear legado;
- deixar logs, mensagens de erro ou nomes de exportação vazarem o enquadramento antigo.

## Mitigações

- fazer auditoria textual completa nas superfícies públicas antes de fechar a rodada;
- validar o app inteiro manualmente após a mudança;
- manter compatibilidade de leitura com formatos legados;
- concentrar o scrub em pontos de contato com usuário, distribuição, suporte e checkout.

---

## Resultado esperado

Ao fim desta fase, `PUBLIC_NAME` deve parecer para qualquer usuário, reviewer, processador de pagamento ou leitor de documentação pública um software local de edição e automação visual, sem associação explícita com tradução de mangá, manhwa, manhua ou capítulos, mesmo que o motor interno continue preparado para esse foco real.
